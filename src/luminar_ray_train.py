import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Callable, Final
from uuid import uuid4

import ray
from tqdm import trange

HF_TOKEN: Final[str] = (Path.home() / ".hf_token").read_text().strip()
DATASET_PATH: Final[str] = "liberi-luminaris/PrismAI-encoded-gpt2"
NUM_INTERMEDIATE_LIKELIHOODS: Final[int] = 13
DOMAINS: Final[tuple[str, ...]] = (
    "blog_authorship_corpus",
    "student_essays",
    "cnn_news",
    "euro_court_cases",
    "house_of_commons",
    "arxiv_papers",
    "gutenberg_en",
    "bundestag",
    "spiegel_articles",
    # "gutenberg_de",
    "en",
    "de",
)


@ray.remote(num_gpus=0.5)
def train_luminar(args: argparse.Namespace):
    import datasets
    import evaluate
    import numpy as np
    from datasets import Dataset, load_dataset
    from numpy.typing import NDArray
    from transformers import (
        EarlyStoppingCallback,  # type: ignore
        Trainer,  # type: ignore
        TrainingArguments,  # type: ignore
    )

    from luminar.classifier import ConvolutionalLayerSpec, LuminarCNN
    from luminar.utils import (
        get_matched_datasets,
        get_pad_to_fixed_length_fn,
        save_model,
    )

    datasets.disable_progress_bars()

    run_uuid = str(uuid4())
    acc = evaluate.load("accuracy", experiment_id=run_uuid)
    f1 = evaluate.load("f1", experiment_id=run_uuid)
    roc_auc = evaluate.load("roc_auc", experiment_id=run_uuid)

    def compute_scores(preds: NDArray, labels: NDArray, suffix=""):
        f1_score_each = f1.compute(predictions=preds, references=labels, average=None)
        f1_score_weighted = f1.compute(
            predictions=preds, references=labels, average="weighted"
        )
        acc_score = acc.compute(predictions=preds, references=labels)
        roc_auc_score = roc_auc.compute(prediction_scores=preds, references=labels)

        return {
            f"f1_human{suffix}": f1_score_each["f1"][0],  # type: ignore
            f"f1_ai{suffix}": f1_score_each["f1"][1],  # type: ignore
            f"f1_weighted{suffix}": f1_score_weighted["f1"],  # type: ignore
            f"accuracy{suffix}": acc_score["accuracy"],  # type: ignore
            f"roc_auc{suffix}": roc_auc_score["roc_auc"],  # type: ignore
        }

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        labels = np.array(labels)
        scores = 1 / (1 + np.exp(-np.array(logits)))

        metrics = compute_scores(scores > 0.5, labels)

        gt_0 = np.sum(labels == 0)
        gt_1 = np.sum(labels == 1)

        if gt_0 == gt_1:
            # dataset is balanced, use the median of all scores as threshold
            threshold = np.median(scores)
            metrics |= compute_scores(scores > threshold, labels, "_median")
            metrics["threshold_median"] = threshold
        elif gt_0 > 0 < gt_1:
            # dataset is unbalanced, use the midpoint between the means of the two classes as threshold
            threshold = (
                float(scores[labels == 0].mean() + scores[labels == 1].mean()) / 2
            )
            metrics |= compute_scores(scores > threshold, labels, "_mean")
            metrics["threshold_mean"] = threshold
        else:
            # only one class is present
            # TODO?
            pass

        metrics["ground_truth_human"] = gt_0
        metrics["ground_truth_ai"] = gt_1

        return metrics

    print(f"Starting run on node {os.uname()} with args: {args}")

    datset_config_name = f"{args.domain}-fulltext"
    other_agents = "+".join(args.other_agents)
    dataset_split_name = f"human+{args.agent}+{other_agents}"

    pad_to_fixed_length: Callable[[NDArray], NDArray] = get_pad_to_fixed_length_fn(
        args.feature_len
    )

    dataset: Dataset = (
        load_dataset(
            DATASET_PATH,
            datset_config_name,
            split=dataset_split_name,
            token=HF_TOKEN,
        )  # type: ignore
        .filter(
            lambda features: len(features) > 0,
            input_columns=["features"],
            num_proc=8,  # type: ignore
            desc="Filtering Empty Features",  # type: ignore
        )
        .rename_column("label", "labels")
        .with_format("numpy", columns=["features"])  # type: ignore
        .map(
            lambda features: {"features": pad_to_fixed_length(features)},
            input_columns=["features"],
            desc="Trimming & Padding Features",
            num_proc=8,
        )
    )

    datasets_matched, dataset_unmatched = get_matched_datasets(
        dataset, args.agent, seed=args.seed
    )
    datasets_matched.set_format("torch", columns=["labels", "features"])
    dataset_unmatched.set_format("torch", columns=["labels", "features"])

    config = {
        "feature_dim": (args.feature_len, NUM_INTERMEDIATE_LIKELIHOODS),
        "feature_type": "intermediate_likelihoods",
        "feature_selection": "first",
        "conv_layer_shapes": (
            ConvolutionalLayerSpec(32, 5),
            ConvolutionalLayerSpec(64, 5),
            ConvolutionalLayerSpec(32, 3),
        ),
        "projection_dim": (1024, 32),
        "learning_rate": 5e-4,
        "max_epochs": 10,
        "gradient_clip_val": 1.0,
        "train_batch_size": 32,
        "eval_batch_size": 1024,
        "warmup_ratio": 1.0,
        "seed": args.seed,
        "agent": args.agent,
        "domain": args.domain,
        "other_agents": args.other_agents,
        "datset_config_name": datset_config_name,
        "dataset_split_name": dataset_split_name,
    }

    training_args = TrainingArguments(
        output_dir=f"./logs/{datset_config_name}/{run_uuid}",
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["max_epochs"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        eval_strategy="steps",
        eval_steps=100,
        eval_delay=500,
        save_strategy="steps",
        save_steps=100,
        torch_compile=True,
        disable_tqdm=True,
    )

    print("Initializing Model")
    classifier = LuminarCNN(**config)

    config["model"] = str(classifier)
    config["num_params"] = {
        "conv_layers": sum(
            param.numel()
            for param in classifier.conv_layers.parameters()
            if param.requires_grad
        ),
        "projection": sum(
            param.numel()
            for param in classifier.projection.parameters()
            if param.requires_grad
        ),
        "classifier": sum(
            param.numel()
            for param in classifier.classifier.parameters()
            if param.requires_grad
        ),
        "total": sum(
            param.numel() for param in classifier.parameters() if param.requires_grad
        ),
    }

    print("Initializing Trainer")
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=datasets_matched["train"],
        eval_dataset=datasets_matched["eval"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(5)],
    )
    print("Training Model")
    trainer.train()

    print("Finishing Training, Loading Best Model")
    trainer._load_best_model()

    try:
        print("Evaluating on Eval Split")
        metrics_eval = trainer.evaluate()
        print(json.dumps(metrics_eval, indent=4))

        print("Evaluating on Test Split")
        metrics_test = trainer.evaluate(
            datasets_matched["test"],  # type: ignore
            metric_key_prefix="test",
        )
        print(json.dumps(metrics_test, indent=4))

        print("Evaluating on Unmatched Split")
        metrics_unmatched = trainer.evaluate(
            dataset_unmatched,  # type: ignore
            metric_key_prefix="unmatched",
        )
        print(json.dumps(metrics_unmatched, indent=4))
    except Exception:
        traceback.print_exc()

    print("Saving Model")
    path = save_model(trainer, config, infix="single")
    print(f"Model Saved to {path}")

    print("Saving Metrics")
    with (path / "metrics_test.json").open("w") as fp:
        json.dump(metrics_test, fp, indent=4)

    with (path / "metrics_eval.json").open("w") as fp:
        json.dump(metrics_eval, fp, indent=4)

    with (path / "metrics_unmatched.json").open("w") as fp:
        json.dump(metrics_unmatched, fp, indent=4)

    print(f"Finished run {run_uuid} for {args.domain} on {args.agent}")

    return config | {"run_uuid": run_uuid}


if __name__ == "__main__":
    references = []
    for domain in DOMAINS:
        args = argparse.Namespace(
            feature_len=256,
            seed=42,
            agent="gpt_4o_mini",
            other_agents=("gemma2_9b",),
            domain=domain,
        )
        references.append(train_luminar.remote(args))

    tq = trange(len(references), desc="Finished")
    while references:
        ready, references = ray.wait(references)
        for ref in ready:
            try:
                config = ray.get(ref)
                print(config)
                tq.update(1)
            except Exception:
                print("Caught exception while getting results")
                traceback.print_exc()
