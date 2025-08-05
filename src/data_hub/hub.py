import os
from collections import Counter

import datasets
from pathlib import Path

from datasets import Dataset, DatasetDict, ClassLabel
from unicodedata import normalize


def normalize_text(batch: dict) -> dict:
    return {
        "text": [
            " ".join(normalize("NFC", text).strip().split()) for text in batch["text"]
        ]
    }


class DataHub:
    """
    A DataHub class that handles easy access and processing of the various data source from the data hub HF.
    """

    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        pass

    def get_many_splits(self, hf_datasets: list[str], num_proc: int = 16, seed: int = 42) -> list[
        DatasetDict]:
        """
        Given a list of Hugging Face dataset names, returns a list of train/eval/test splits for each dataset.
        """
        all_splits = []
        for hf_dataset in hf_datasets:
            print(f"\n📦 Processing dataset: {hf_dataset}")
            splits = self.get_splits(hf_dataset=hf_dataset, num_proc=num_proc, seed=seed)
            all_splits.append(splits)
        return all_splits

    def get_splits(self, hf_dataset: str, num_proc: int = 16, seed: int = 42):
        """
        Fetches the given sources from the Hugging Face Hub or.
        :param hf_dataset: Dataset names to be included in the pipeline as outlined in the
        :param num_proc: Number of processes to use for parallel processing.
        :seed: Random seed for reproducibility.
        """
        raw_dataset = (
            datasets
            .load_dataset(hf_dataset, token=self.hf_token)
            .filter(
                lambda text: len(text.strip()) > 0,
                input_columns=["text"],
                num_proc=num_proc
            )
            .map(
                normalize_text,
                batched=True,
                num_proc=num_proc,
                desc="Normalizing text"
            )
        )

        if isinstance(raw_dataset, DatasetDict):
            dataset = raw_dataset[next(iter(raw_dataset))]
        else:
            dataset = raw_dataset

        # Identify unique label names
        unique_labels = list(set(dataset["label"]))
        unique_labels.sort()
        class_label = ClassLabel(names=unique_labels)
        print("Unique labels: ", class_label)
        dataset = dataset.cast_column("label", class_label)

        # split off test set (20%)
        split_1 = dataset.train_test_split(
            test_size=0.2,
            seed=seed,
            stratify_by_column="label"
        )

        # from remaining 80%, split eval as 1/8 = 10% of total
        split_2 = split_1["train"].train_test_split(
            test_size=0.125,  # 0.125 of 80% = 10% of total
            seed=seed,
            stratify_by_column="label"
        )

        # Final dataset dict
        dataset_split = DatasetDict({
            "train": split_2["train"].shuffle(seed=seed),
            "eval": split_2["test"].shuffle(seed=seed),
            "test": split_1["test"].shuffle(seed=seed),
        })

        for split_name in ["train", "eval", "test"]:
            labels = dataset_split[split_name]["label"]
            label_counts = Counter(labels)
            total = sum(label_counts.values())
            print(f"{split_name} distribution:")
            for label_id, count in label_counts.items():
                label_name = class_label.int2str(label_id)
                pct = 100 * count / total
                print(f"  {label_name}: {count} ({pct:.1f}%)")

        return dataset_split


if __name__ == "__main__":
    # Example Usage
    hub = DataHub((Path.home() / ".hf_token").read_text().strip())
    dataset = hub.get_splits("liberi-luminaris/Ghostbuster-encoded-gpt2")
    print(dataset)
