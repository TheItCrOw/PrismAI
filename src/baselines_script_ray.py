import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import ray
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments

from luminar.baselines.core import run_detector_tokenized
from luminar.baselines.trainable import AutoClassifier
from luminar.baselines.trainable.roberta import RoBERTaClassifier
from luminar.baselines.trainable.gpt2 import GPT2Classifier
from luminar.utils import get_matched_cross_validation_datasets

HF_TOKEN = (Path.home() / ".hf_token").read_text().strip()


@ray.remote(num_gpus=1)
def run_training_ray(
    config: str,
    dataset: DatasetDict,
    model_class: type[AutoClassifier],
    model_name: str,
    model_args: dict,
    output_path: Path,
    logs_path: Path,
    eval_dataset: dict[str, DatasetDict],
):
    model = model_class(model_name, **model_args)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        seed=42,
        #
        learning_rate=1e-5,
        num_train_epochs=3,
        #
        per_device_train_batch_size=15,
        per_device_eval_batch_size=30,
        #
        logging_steps=50,
        logging_strategy="steps",
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="epoch",
        save_total_limit=2,
    )

    model.train(
        dataset,
        training_args,
        save_path=str(output_path / "final"),
    )

    scores = run_detector_tokenized(
        model,
        eval_dataset,
        sigmoid=False,
    )

    logs_path.mkdir(parents=True, exist_ok=True)
    with (logs_path / "scores.json").open("w") as fp:
        json.dump(scores, fp, indent=4)

    return config, scores


@ray.remote(num_gpus=0.5)
def run_eval_ray(
    config: str,
    eval_dataset: dict[str, DatasetDict],
    model_class: type[AutoClassifier],
    model_name: str,
    model_args: dict,
    final_model_path: Path,
):
    model = model_class(
        str(final_model_path),
        tokenizer_name=model_name,
        **model_args,
    )

    return config, run_detector_tokenized(
        model,
        eval_dataset,
        sigmoid=False,
    )


DOMAINS = [
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
]

AGENT = "gpt_4o_mini"
OTHER_AGENTS = "gemma2_9b"
NUM_PROC = 32
NUM_SPLITS = 10

ROOT: Path = Path.home() / "Projects/PrismAI"
MAX_TASKS = 6


@ray.remote
def run_finetuning_ray(
    model_class: type[AutoClassifier],
    model_name: str,
    model_args: dict,
    cross_eval: bool = False,
):
    model_str = model_name.replace("/", "--").replace(":", "--")
    if model_args.get("freeze_lm"):
        model_str += "-frozen"

    model_path = ROOT / "models/finetuning/cross_validation" / model_str
    model_path.mkdir(parents=True, exist_ok=True)

    logs_path = ROOT / "logs/finetuning/cross_validation" / model_str
    logs_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    dataset_splits = [{} for _ in range(NUM_SPLITS)]
    for domain in tqdm(DOMAINS):
        datset_config_name = f"{domain}-fulltext"
        dataset_split_name = f"human+{AGENT}+{OTHER_AGENTS}"
        dataset: Dataset = (
            load_dataset(
                "liberi-luminaris/PrismAI",
                datset_config_name,
                split=dataset_split_name,
                token=HF_TOKEN,
            )  # type: ignore
            .rename_column("label", "labels")
            .filter(
                lambda text: len(text.strip()) > 0,
                input_columns=["text"],
                num_proc=NUM_PROC,  # type: ignore
            )
            .map(
                lambda texts: tokenizer(
                    texts,
                    padding=False,
                    truncation=True,
                    max_length=512,
                    return_length=True,
                ),
                input_columns=["text"],
                remove_columns=["text"],
                batched=True,
                batch_size=1024,
                desc="Tokenizing",
                keep_in_memory=True,
                num_proc=8,
            )
        )
        datasets, dataset_unmatched = get_matched_cross_validation_datasets(
            dataset, AGENT, num_proc=NUM_PROC, num_splits=NUM_SPLITS
        )

        for i, dataset in enumerate(datasets):
            dataset_splits[i][domain] = dataset
            dataset_splits[i][domain]["unmatched"] = dataset_unmatched

    futures = []
    for i, dataset_split in enumerate(dataset_splits):
        for domain in DOMAINS:
            dataset = dataset_split[domain]

            for config, dataset in dataset_split.items():
                if len(futures) > MAX_TASKS:
                    ready_refs, result_refs = ray.wait(futures, num_returns=1)
                    ray.get(ready_refs)

                futures.append(
                    run_training_ray.remote(
                        config,
                        dataset,
                        model_class,
                        model_name,
                        model_args,
                        model_path / config / f"cv-{i}",
                        logs_path / config / f"cv-{i}",
                        dataset_split if cross_eval else {config: dataset},
                    )
                )

    tq = tqdm(
        total=len(futures),
        desc="Running Cross Validation",
    )
    scores_fine_tuning = defaultdict(list)
    while futures:
        ready, futures = ray.wait(futures)
        for ref in ready:
            try:
                tq.update(1)
                config, results = ray.get(ref)
                scores_fine_tuning[config].append(results)
                with (logs_path / f"{config}.json").open("w") as fp:
                    json.dump(scores_fine_tuning[config], fp, indent=4)
            except Exception:
                print("Caught exception while getting results")

    with open(logs_path / f"{model_name}-ft-frozen.json", "w") as fp:
        json.dump(scores_fine_tuning, fp, indent=4)


if __name__ == "__main__":
    futures = []
    for model_class, model_name, model_args in [
        (GPT2Classifier, "gpt2", dict(freeze_lm=True)),
        (RoBERTaClassifier, "roberta-base", dict(freeze_lm=False)),
    ]:
        ray.get(
            run_finetuning_ray.remote(
                model_class,
                model_name,
                model_args,
                True,  # cross_eval
            )
        )
