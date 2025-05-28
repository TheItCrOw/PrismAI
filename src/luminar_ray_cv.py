import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Final

import datasets
import ray
from datasets import Dataset, load_dataset
from numpy.typing import NDArray
from transformers import (
    EarlyStoppingCallback,  # type: ignore
    Trainer,  # type: ignore
    TrainingArguments,  # type: ignore
)
from ulid import ulid

from luminar.classifier import LuminarCNN
from luminar.utils import (
    compute_metrics,
    get_matched_cross_validation_datasets,
    get_pad_to_fixed_length_fn,
    save_model,
)
from luminar.utils.data import DatasetDictTrainEvalTest, DatasetUnmatched
from luminar.utils.training import ConvolutionalLayerSpec, LuminarTrainingConfig

HF_TOKEN: Final[str] = (Path.home() / ".hf_token").read_text().strip()
DATASET_PATH: Final[str] = "liberi-luminaris/PrismAI-encoded-gpt2"


@ray.remote(num_cpus=1)
def luminar_cv_start_run(config: LuminarTrainingConfig):
    datasets.disable_progress_bars()
    cv_run_str = f"[CvRun::{config.run_ulid}::{config.hash(10)}]"
    print(
        f"{cv_run_str} Starting Cross Validation Run on {os.uname().nodename} with args: {config}"
    )

    print(f"{cv_run_str} Loading Datasets")
    pad_to_fixed_length: Callable[[NDArray], NDArray] = get_pad_to_fixed_length_fn(
        config.feature_len
    )
    dataset: Dataset = (
        load_dataset(
            DATASET_PATH,
            config.datset_config_name,
            split=config.dataset_split_name,
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

    print(f"{cv_run_str} Preparing Cross Validation Datasets")
    cv_datasets_matched, dataset_unmatched = get_matched_cross_validation_datasets(
        dataset, config.agent, seed=config.seed
    )

    print(f"{cv_run_str} Starting Cross Validation Experiments")
    references = []
    for cv_idx, datasets_matched in enumerate(cv_datasets_matched):
        ref = luminar_cv_train.remote(
            config, cv_idx, datasets_matched, dataset_unmatched, ulid()
        )
        references.append(ref)
    len_refs = len(references)

    results = []
    while references:
        ready, references = ray.wait(references)
        for ref in ready:
            try:
                result = ray.get(ref)
                print(result)
                results.append(result)
            except Exception:
                print(
                    f"Caught exception while getting results of CV experiment for {config.domain} on {config.agent}",
                    file=sys.stderr,
                )
                traceback.print_exc()

    print(
        f"{cv_run_str} Sucessfully finished {len(results)}/{len_refs} CV experiments for {config.domain} on {config.agent}"
    )

    return results


@ray.remote(num_gpus=0.25)
def luminar_cv_train(
    config: LuminarTrainingConfig,
    cv_idx: int,
    dataset_matched: DatasetDictTrainEvalTest,
    dataset_unmatched: DatasetUnmatched,
    experiment_ulid: str,
):
    print(
        f"[Exp::{experiment_ulid}] Starting Cross Validation Experiment {config.domain}-{config.agent}-cv_idx_{cv_idx} on {os.uname().nodename}"
    )

    train_batch_size = config["train_batch_size"]
    steps_per_epoch = len(dataset_matched["train"]) // train_batch_size
    eval_steps = steps_per_epoch // 5

    training_args = TrainingArguments(
        output_dir=f"./logs/{config.datset_config_name}/{experiment_ulid}",
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["max_epochs"],
        warmup_steps=int(config["warmup_ratio"] * steps_per_epoch),
        logging_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_delay=steps_per_epoch,
        save_strategy="steps",
        save_steps=eval_steps,
        torch_compile=True,
        disable_tqdm=True,
    )

    print(f"[Exp::{experiment_ulid}] Initializing Model")
    classifier = LuminarCNN(**config.asdict())

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

    print(f"[Exp::{experiment_ulid}] Moving Data to Device")
    dataset_matched.set_format("torch", columns=["labels", "features"])
    dataset_unmatched.set_format("torch", columns=["labels", "features"])

    print(f"[Exp::{experiment_ulid}] Initializing Trainer")
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=dataset_matched["train"],
        eval_dataset=dataset_matched["eval"],
        compute_metrics=compute_metrics,  # type: ignore
        callbacks=[EarlyStoppingCallback(10)],
    )
    print(f"[Exp::{experiment_ulid}] Training Model")
    trainer.train()

    print(f"[Exp::{experiment_ulid}] Finishing Training, Loading Best Model")
    trainer._load_best_model()

    try:
        print(f"[Exp::{experiment_ulid}] Evaluating on Eval Split")
        metrics_eval = trainer.evaluate()
        print(json.dumps(metrics_eval, indent=4))

        print(f"[Exp::{experiment_ulid}] Evaluating on Test Split")
        metrics_test = trainer.evaluate(
            dataset_matched["test"],  # type: ignore
            metric_key_prefix="test",
        )
        print(json.dumps(metrics_test, indent=4))

        print(f"[Exp::{experiment_ulid}] Evaluating on Unmatched Split")
        metrics_unmatched = trainer.evaluate(
            dataset_unmatched,  # type: ignore
            metric_key_prefix="unmatched",
        )
        print(json.dumps(metrics_unmatched, indent=4))
    except Exception:
        traceback.print_exc()

    print(f"[Exp::{experiment_ulid}] Saving Model")

    path_infix = f"cross_validation_{config.run_ulid}"
    path_suffix = f"cv_idx_{cv_idx}"
    path = save_model(trainer, config, infix=path_infix, suffix=path_suffix)

    print(f"[Exp::{experiment_ulid}] Model Saved to {path}")

    print(f"[Exp::{experiment_ulid}] Saving Metrics")
    with (path / "metrics_eval.json").open("w") as fp:
        json.dump(metrics_eval, fp, indent=4)

    with (path / "metrics_test.json").open("w") as fp:
        json.dump(metrics_test, fp, indent=4)

    with (path / "metrics_unmatched.json").open("w") as fp:
        json.dump(metrics_unmatched, fp, indent=4)

    print(
        f"[Exp::{experiment_ulid}] Finished cv_idx_{cv_idx} for {config.domain} on {config.agent}"
    )

    return {
        "config": config.asdict(),
        "cv_idx": cv_idx,
        "experiment_ulid": experiment_ulid,
        "metrics_eval": metrics_eval,
        "metrics_test": metrics_test,
        "metrics_unmatched": metrics_unmatched,
    }


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
FEATURE_MODEL: Final[str] = "gpt2"
FEATURE_TYPE: Final[str] = "intermediate_likelihoods"
FEATURE_SELECTION: Final[str] = "first"
NUM_INTERMEDIATE_LIKELIHOODS: Final[int] = 13

if __name__ == "__main__":
    RUN_ULID: Final[str] = ulid()

    references = []
    for domain in DOMAINS:
        agent = "gpt_4o_mini"
        other_agents = ("gemma2_9b",)
        config = LuminarTrainingConfig(
            feature_len=256,
            feature_dim=(256, NUM_INTERMEDIATE_LIKELIHOODS),
            feature_type=FEATURE_TYPE,
            feature_model=FEATURE_MODEL,
            feature_selection=FEATURE_SELECTION,
            agent=agent,
            domain=domain,
            other_agents=other_agents,
            datset_config_name=f"{domain}-fulltext",
            dataset_split_name="+".join(("human", agent, *other_agents)),
            conv_layer_shapes=(
                ConvolutionalLayerSpec(32, 5),
                ConvolutionalLayerSpec(64, 5),
                ConvolutionalLayerSpec(32, 3),
            ),
            projection_dim=(1024, 32),
            max_epochs=25,
            learning_rate=5e-4,
            gradient_clip_val=1.0,
            train_batch_size=32,
            eval_batch_size=1024,
            warmup_ratio=1.0,
            seed=42,
            run_ulid=RUN_ULID,
        )
        references.append(luminar_cv_start_run.remote(config))

    while references:
        ready, references = ray.wait(references)
        for ref in ready:
            try:
                results = ray.get(ref)
                print(results)
            except Exception:
                print("Caught exception while getting results")
                traceback.print_exc()
