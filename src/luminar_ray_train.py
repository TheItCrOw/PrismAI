import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Callable, Final
from uuid import uuid4

import datasets
import numpy as np
import ray
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from numpy.typing import NDArray
from tqdm import trange
from transformers import (
    EarlyStoppingCallback,  # type: ignore
    Trainer,  # type: ignore
    TrainingArguments,  # type: ignore
)
from transformers.trainer_utils import EvalPrediction

from luminar.classifier import LuminarCNN
from luminar.utils import (
    ConvolutionalLayerSpec,
    save_model,
)
from luminar.utils.data import get_pad_to_fixed_length_fn
from luminar.utils.evaluation import run_evaluation
from luminar.utils.training import LuminarTrainingConfig

HF_TOKEN: Final[str] = (Path.home() / ".hf_token").read_text().strip()
DATASET_PATH: Final[Path] = Path("/storage/projects/stoeckel/prismai/encoded/fulltext/")
DEFAULT_OUTPUT_DIR: Final[Path] = Path.home() / "Projects/PrismAI/models/luminar_cnn"
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

ray.init()


def compute_metrics(
    eval_pred: tuple[NDArray, NDArray] | EvalPrediction,
) -> dict[str, float]:
    y_scores, labels = eval_pred
    return run_evaluation(
        np.array(labels),
        np.array(y_scores),
        sigmoid=True,
        threshold=0.5,
    )


@ray.remote(num_gpus=0.5)
def train_luminar(args: argparse.Namespace):
    datasets.disable_progress_bars()

    run_uuid = str(uuid4())

    print(f"Starting run on node {os.uname()} with args: {args}")

    dataset_config_name = f"{args.domain}_{args.dataset.stem}"
    other_agents = "+".join(args.other_agents)
    dataset_split_name = f"human+{args.agent}+{other_agents}"

    datasets_matched: DatasetDict = (
        load_from_disk(str(args.dataset))  # type: ignore
        .filter(
            lambda features: len(features) > 0,
            input_columns=["features"],
            num_proc=8,  # type: ignore
            desc="Filtering Empty Features",  # type: ignore
        )
        .with_format("numpy", columns=["features"])  # type: ignore
    )

    if args.feature_len != 512:
        if args.feature_len > 512:
            raise ValueError(
                f"Feature length {args.feature_len} is greater than 512, which is not supported."
            )

        pad_to_fixed_length: Callable[[NDArray], NDArray] = get_pad_to_fixed_length_fn(
            args.feature_len
        )
        datasets_matched = datasets_matched.map(
            lambda features: {"features": pad_to_fixed_length(features)},
            input_columns=["features"],
            desc="Trimming & Padding Features",
            num_proc=8,
        )

    datasets_matched.set_format("torch", columns=["labels", "features"])
    dataset_unmatched: Dataset = datasets_matched.pop("unmatched")
    dataset_unmatched.set_format("torch", columns=["labels", "features"])

    config = (
        {
            "feature_dim": (args.feature_len, args.num_il),
            "feature_type": "intermediate_likelihoods",
            "feature_selection": "first",
            "conv_layer_shapes": (
                ConvolutionalLayerSpec(64, 5),
                ConvolutionalLayerSpec(128, 3),
                ConvolutionalLayerSpec(128, 3),
                ConvolutionalLayerSpec(128, 3),
                ConvolutionalLayerSpec(64, 3),
                # ConvolutionalLayerSpec(32, 5),
                # ConvolutionalLayerSpec(64, 5),
                # ConvolutionalLayerSpec(32, 3),
            ),
            "feed_forward_dim": (1024, 32),
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
            "datset_config_name": dataset_config_name,
            "dataset_split_name": dataset_split_name,
        }
    )

    training_args = TrainingArguments(
        output_dir=f"./logs/{dataset_config_name}/{run_uuid}",
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
            for param in classifier.cnn.parameters()
            if param.requires_grad
        ),
        "projection": sum(
            param.numel()
            for param in classifier.feed_forward.parameters()
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

    path = Path(DEFAULT_OUTPUT_DIR) / args.model / config["domain"]  # / suffix
    try:
        save_model(path, trainer, config)
        print(f"Model Saved to {path}")
    except Exception:
        print("Failed to save model")
        traceback.print_exc()

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
        agent = "gpt_4o_mini"
        other_agents = ("gemma2_9b",)
        agent_str = "_".join([agent, "_".join(other_agents)])
        # model, num_il = "gpt2", 13
        model, num_il = "gpt_j_6b", 29
        # model, num_il = "falcon_7b", 33
        args = argparse.Namespace(
            dataset=DATASET_PATH / agent_str / f"{model}_512" / domain,
            model=model,
            num_il=num_il,
            feature_len=256,
            seed=42,
            agent=agent,
            other_agents=other_agents,
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
