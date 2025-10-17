from luminar.utils import get_matched_datasets
from luminar.baselines.trainable.roberta import RoBERTaClassifier
from luminar.baselines.trainable import AutoClassifier
from luminar.baselines.core import run_detector, run_detector_tokenized
from transformers.training_args import TrainingArguments
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, load_dataset
import torch
import numpy as np
from pathlib import Path
import json
import gc
import warnings

warnings.filterwarnings(
    "ignore", message=r".*Please note that with a fast tokenizer.*")
warnings.filterwarnings(
    "ignore",
    message=r".*Using the `WANDB_DISABLED` environment variable is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Was asked to gather along dimension \d+, but all input tensors were scalars.*",
)


HF_TOKEN = (Path.home() / ".hf_token").read_text().strip()

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
            num_proc=num_proc,
        )
    )
    datasets_matched, dataset_unmatched = get_matched_datasets(
        dataset, agent, num_proc=num_proc
    )
    datasets_matched["unmatched"] = dataset_unmatched
    datasets[domain] = datasets_matched
del dataset


def run_finetuning(model: AutoClassifier, only_eval=False, cross_eval=False):
    model_name_or_path = model.model_name_or_path

    try:
        model_str = model_name_or_path.replace("/", "--").replace(":", "--")
        if model.freeze_lm:
            model_str += "-frozen"

        output_path = Path("../models/finetuning/") / model_str

        logs_path = Path("../logs/finetuning/") / model_str
        logs_path.mkdir(parents=True, exist_ok=True)

        datasets_tokenized = {
            config: dataset.map(
                model.tokenize,
                input_columns=["text"],
                batched=True,
                batch_size=1024,
                desc="Tokenizing",
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
        for config, dataset in tq:
            tq.set_postfix_str(config)

            output_dir = output_path / config
            final_model_path = output_dir / "final"
            if not only_eval:
                model.reset()
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
                    save_path=str(final_model_path),
                )
            else:
                cls: type[AutoClassifier] = type(model)

                model.model.to("cpu")
                del model

                model = cls(
                    final_model_path,
                    tokenizer_name=model_name_or_path,
                    device="cuda:0",
                )

            scores_fine_tuning[config] = run_detector_tokenized(
                model,
                datasets_tokenized if cross_eval else {config: dataset},
            )
            with (logs_path / f"{config}.json").open("w") as fp:
                json.dump(scores_fine_tuning[config], fp, indent=4)

        return scores_fine_tuning
    finally:
        model.model.to("cpu")
        del model
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


scores_roberta_base_ft = run_finetuning(
    RoBERTaClassifier("roberta-base", freeze_lm=True),
    only_eval=False,
    cross_eval=True,
)
print(json.dumps(scores_roberta_base_ft, indent=4))
with open("../logs/finetuning/roberta-base-ft-frozen.json", "w") as fp:
    json.dump(scores_roberta_base_ft, fp, indent=4)
