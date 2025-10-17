import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Callable, Final
from uuid import uuid4

from luminar.utils import ConvolutionalLayerSpec
from luminar.utils.data import get_pad_to_fixed_length_fn
from luminar.utils.evaluation import run_evaluation

HF_TOKEN: Final[str] = (Path.home() / ".hf_token").read_text().strip()

DATASET_PATH: Final[str] = "liberi-luminaris/PrismAI-encoded-gpt2"
NUM_INTERMEDIATE_LIKELIHOODS: Final[int] = 13

UUID = str(uuid4())


def train_luminar(args: argparse.Namespace):
    import torch
    from datasets import Dataset, load_dataset
    from numpy.typing import NDArray
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    from luminar.classifier import LuminarCNN
    from luminar.utils import (
        get_matched_datasets,
        save_model,
    )

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
        .rename_column("label", "labels")
        .with_format("numpy", columns=["features"])  # type: ignore
        .map(
            lambda features: {"features": pad_to_fixed_length(features)},
            input_columns=["features"],
            desc="Trimming & Padding Features",
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
        "other_agents": other_agents,
        "datset_config_name": datset_config_name,
        "dataset_split_name": dataset_split_name,
    }

    training_args = TrainingArguments(
        output_dir=f"/tmp/luminar/{datset_config_name}/{UUID}",
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
        # torch_compile_mode="reduce-overhead",
    )

    print("Initializing model")
    classifier = LuminarCNN(**config)
    classifier.forward(
        torch.randn(config["train_batch_size"], *config["feature_dim"]),
        labels=torch.randint(0, 2, (config["train_batch_size"],)),
    )
    print("Model initialized")

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

    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=datasets_matched["train"],
        eval_dataset=datasets_matched["eval"],
        # data_collator=PaddingDataCollator(config["feature_dim"]),
        compute_metrics=run_evaluation,
        callbacks=[EarlyStoppingCallback(5)],
    )
    trainer.train()

    print("Finishing training, loading best model")
    trainer._load_best_model()

    try:
        print("Evaluating on test set")
        metrics_test = trainer.evaluate(
            datasets_matched["test"],  # type: ignore
            metric_key_prefix="test",
        )

        trainer.create_model_card(
            language="en",
            license="cc-by-nc-sa-4.0",
            tags="luminar",
            model_name=f"luminar-cnn-gpt2_256-{args.domain}",
            tasks="text-classification",
            dataset="liberi-luminaris/PrismAI-encoded-gpt2",
        )

        print("Evaluating on eval set")
        metrics_eval = trainer.evaluate()

        print("Evaluating on unmatched set")
        metrics_unmatched = trainer.evaluate(
            dataset_unmatched,  # type: ignore
            metric_key_prefix="unmatched",
        )
    except Exception:
        traceback.print_exc()

    print("Saving model")
    path = save_model(trainer, config)

    print("Saving metrics")
    with (path / "metrics_test.json").open("w") as fp:
        json.dump(metrics_test, fp, indent=4)

    with (path / "metrics_eval.json").open("w") as fp:
        json.dump(metrics_eval, fp, indent=4)

    with (path / "metrics_unmatched.json").open("w") as fp:
        json.dump(metrics_unmatched, fp, indent=4)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device_ids",
        type=str,
        default="0",
        help="Devices to use for training, e.g. 0,1,2,3",
    )

    parser.add_argument(
        "-l",
        "--feature_len",
        type=int,
        default=256,
        help="Length of the features to be used for training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="gpt_4o_mini",
    )
    parser.add_argument(
        "--other_agents",
        type=str,
        nargs="*",
        default=("gemma2_9b",),
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Domain to use for training",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    train_luminar(args)
