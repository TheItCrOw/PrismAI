import nltk
import torch
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from numpy._typing import NDArray
from typing import Final, Callable
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from nltk.tokenize import sent_tokenize
from luminar.encoder import LuminarEncoder
from luminar.utils import (
    get_pad_to_fixed_length_fn,
    get_matched_datasets,
    LuminarSequenceDataset
)


class SequentialDataService:

    def __init__(self,
                 luminar_encoder: LuminarEncoder,
                 batch_size: int = 256,
                 feature_len: int = 512):
        nltk.download("punkt")
        self.luminar_encoder = luminar_encoder
        self.feature_len = feature_len
        self.batch_size = batch_size

    def dataset_to_luminar_sequence_dataset(self, dataset: Dataset) -> tuple[
        LuminarSequenceDataset, LuminarSequenceDataset, DataLoader]:
        # Merge train and eval for cross-validation
        train_dataset = LuminarSequenceDataset(
            ConcatDataset([dataset["train"], dataset["eval"]])
        )
        test_dataset = LuminarSequenceDataset(dataset['test'])
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn
        )
        return (train_dataset, test_dataset, test_loader)

    def load_dataset_as_luminar_sequence_dataset(self, dataset_path: str):
        return self.dataset_to_luminar_sequence_dataset(self.load_dataset(dataset_path))

    def load_multiple_datasets(self, dataset_paths: list[str]) -> DatasetDict:
        if not dataset_paths:
            raise ValueError("No dataset paths provided.")

        # Load individual datasets
        dataset_dicts = [self.load_dataset(path) for path in dataset_paths]

        merged_splits = {}
        for split in dataset_dicts[0].keys():
            merged_splits[split] = concatenate_datasets([ds[split] for ds in dataset_dicts if split in ds])

        return DatasetDict(merged_splits)

    def load_dataset(self, dataset_path: str) -> DatasetDict:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        print(f'Loading dataset from: {dataset_path}')

        data_files = {
            "train": sorted(str(f) for f in dataset_path.glob("train/*.arrow")),
            "test": sorted(str(f) for f in dataset_path.glob("test/*.arrow")),
            "eval": sorted(str(f) for f in dataset_path.glob("eval/*.arrow")),
        }
        data_files = {k: v for k, v in data_files.items() if v}
        dataset = load_dataset(
            "arrow",
            data_files=data_files,
        )
        return self.process_dataset(dataset)

    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        pad_to_fixed_length: Callable[[NDArray], NDArray] = get_pad_to_fixed_length_fn(self.feature_len)

        # We need the tokenized text since we label sequences based on sentences.
        return (
            dataset
            .map(
                lambda batch: {
                    "tokenized_text": [
                        pad_to_fixed_length(
                            np.array(self.luminar_encoder.tokenize(t)["input_ids"]).reshape(-1, 1)
                        )
                        for t in batch["text"]
                    ],
                    "sentence_token_spans": [
                        self._sentence_to_token_spans(t)
                        for t in batch["text"]
                    ],
                },
                batched=True,
                desc="Tokenizing, padding, and aligning sentences"
            )
            .map(
                lambda batch: {
                    "span_labels": [
                        [label] * len(spans)
                        for label, spans in zip(batch["labels"], batch["sentence_token_spans"])
                    ]
                },
                batched=True,
                desc="Assigning labels to sentence spans"
            )
        )

    def _collate_fn(self, batch):
        features = [item["features"] for item in batch]
        sentence_spans = [item["sentence_spans"] for item in batch]
        span_labels = [item["span_labels"] for item in batch]
        features = torch.stack(features)
        return {
            "features": features,
            "sentence_spans": sentence_spans,
            "span_labels": span_labels
        }

    def _sentence_to_token_spans(self, text: str, span_min_length: int = -1) -> list[tuple[int, int]]:
        """Return a list of (start_token_idx, end_token_idx) for each sentence in the text."""
        sentences = sent_tokenize(text)
        spans = []
        current_token_idx = 0

        for sent in sentences:
            tokens = self.luminar_encoder.tokenize(sent)["input_ids"]
            token_count = len(tokens)

            start = current_token_idx
            end = min(current_token_idx + token_count, self.feature_len)

            if start >= self.feature_len:
                break  # Stop if padding region or overflow
            # Only take spans of a minimal length
            if end - start > span_min_length:
                spans.append((start, end))
            current_token_idx = end

        return spans
