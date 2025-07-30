import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments

from luminar.baselines.core import run_detector_tokenized
from luminar.baselines.trainable import AutoClassifier
from luminar.baselines.trainable.gpt2 import GPT2Classifier
from luminar.utils import get_matched_datasets

HF_TOKEN = (Path.home() / ".hf_token").read_text().strip()


def run_training(
    config: str,
    dataset: DatasetDict,
    model_class: type[AutoClassifier],
    model_name: str,
    model_args: dict,
    output_dir: Path,
    eval_dataset: dict[str, DatasetDict],
):
    model = model_class(model_name, **model_args)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        save_path=str(output_dir / "final"),
    )

    results = run_detector_tokenized(
        model,
        eval_dataset,
        sigmoid=False,
    )

    model.model.to("cpu")
    del model

    return config, results


def run_eval(
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
    results = run_detector_tokenized(
        model,
        eval_dataset,
        sigmoid=False,
    )

    model.model.to("cpu")
    del model

    return config, results


def run_finetuning(
    datasets: dict[str, DatasetDict],
    model_class: type[AutoClassifier],
    model_name: str,
    model_args: dict,
    only_eval: bool = False,
    cross_eval: bool = False,
):
    root: Path = Path.home() / "PrismAI"

    model_str = model_name.replace("/", "--").replace(":", "--")
    if model_args.get("freeze_lm"):
        model_str += "-frozen"

    output_path = root / "models/finetuning/" / model_str

    logs_path = root / "logs/finetuning/" / model_str
    logs_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    datasets_tokenized = {
        config: dataset.map(
            lambda texts: tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=512,
                return_length=True,
            ),
            input_columns=["text"],
            batched=True,
            batch_size=1024,
            desc="Tokenizing",
            keep_in_memory=True,
            num_proc=8,
        ).sort("length")
        for config, dataset in (
            datasets.items()
            if not only_eval
            else (
                # We can omit training & dev splits if we are only evaluating
                (config, DatasetDict({"test": dataset["test"]}))
                for config, dataset in datasets.items()
            )
        )
    }

    tq = tqdm(
        datasets_tokenized.items(),
        desc="Evaluating" if only_eval else "Finetuning",
    )
    scores_fine_tuning = {}
    if not only_eval:
        for config, dataset in tq:
            scores_fine_tuning[config] = run_training(
                config,
                dataset,
                model_class,
                model_name,
                model_args,
                output_path / config,
                datasets_tokenized if cross_eval else {config: dataset},
            )
    else:
        for config, dataset in tq:
            scores_fine_tuning[config] = run_eval(
                config,
                datasets_tokenized if cross_eval else {config: dataset},
                model_class,
                model_name,
                model_args,
                output_path / config / "final",
            )

    for config, scores in scores_fine_tuning.items():
        with (logs_path / f"{config}.json").open("w") as fp:
            json.dump(scores, fp, indent=4)

    return scores_fine_tuning


if __name__ == "__main__":
    agent = "gpt_4o_mini"
    other_agents = "gemma2_9b"
    datasets = {}
    num_proc = 32
    for domain in tqdm(
        [
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
    ):
        datset_config_name = f"{domain}-fulltext"
        dataset_split_name = f"human+{agent}+{other_agents}"
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
                num_proc=num_proc,  # type: ignore
            )
        )
        datasets_matched, dataset_unmatched = get_matched_datasets(
            dataset, agent, num_proc=num_proc
        )
        datasets_matched["unmatched"] = dataset_unmatched
        datasets[domain] = datasets_matched
        del dataset

    scores_gpt2_ft = run_finetuning(
        datasets,
        GPT2Classifier,
        "gpt2",
        dict(freeze_lm=True),
        only_eval=False,
        cross_eval=True,
    )

    print(json.dumps(scores_gpt2_ft, indent=4))
    with open("../logs/finetuning/gpt2-ft-frozen.json", "w") as fp:
        json.dump(scores_gpt2_ft, fp, indent=4)
