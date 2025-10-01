from pathlib import Path

import numpy as np
import nltk
import torch

from typing import Callable
from datasets import DatasetDict
from numpy._typing import NDArray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, ConcatDataset
from unicodedata import normalize
from data_hub.hub import DataHub
from nltk.tokenize import sent_tokenize
from luminar.encoder import LuminarEncoder
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from luminar.utils import LuminarSequenceDataset

class SequentialDataProcessor:

    def __init__(self, luminar_encoder: LuminarEncoder):
        self.luminar_encoder = luminar_encoder
        nltk.download("punkt")
        nltk.download("punkt_tab")

    def dataset_to_luminar_sequence_dataset(self, dataset: Dataset, batch_size: int = 256) -> tuple[
        LuminarSequenceDataset, LuminarSequenceDataset, DataLoader]:
        # Merge train and eval for cross-validation
        train_dataset = LuminarSequenceDataset(
            ConcatDataset([dataset["train"], dataset["eval"]])
        )
        test_dataset = LuminarSequenceDataset(dataset['test'])
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn
        )
        return (train_dataset, test_dataset, test_loader)

    def process_for_training(self, dataset : DatasetDict, split_by: str | None = None) -> (
            DatasetDict | dict[str, DatasetDict]):
        if split_by is None:
            return self._process(dataset)

        # If split_by is set: group datasets by unique values of `split_by`, e.g. "agent"
        unique_values = set()
        for split_name in dataset.keys():
            unique_values.update(set(dataset[split_name].unique(split_by)))

        grouped_datasets: dict[str, dict[str, Dataset]] = {val: {} for val in unique_values}

        for split_name, split_dataset in dataset.items():
            for val in unique_values:
                filtered = split_dataset.filter(lambda ex: ex[split_by] == val)
                if len(filtered) > 0:
                    grouped_datasets[val][split_name] = filtered

        # Convert to DatasetDicts and process
        result: dict[str, DatasetDict] = {}
        for val, splits in grouped_datasets.items():
            result[val] = self._process(DatasetDict(splits))

        return result

    def process_for_detector(self, document : str, features_len: int):
        """
        Process a single document to prepare as LuminarSequence input.
        """
        return self._sentence_to_token_spans(document, features_len)

    @classmethod
    def map_token_spans_to_character_spans(cls, spans, offset_mapping):
        char_spans = []
        for start_token, end_token in spans:
            if start_token >= len(offset_mapping) or end_token > len(offset_mapping) or start_token >= end_token:
                continue  # skip invalid spans

            start_char = offset_mapping[start_token][0]
            end_char = offset_mapping[end_token - 1][1]
            char_spans.append((start_char, end_char))
        return char_spans

    @classmethod
    def normalize_text(cls, batch: dict) -> dict:
        return {
            "text": [
                " ".join(
                    normalize("NFC", text)  # Unicode NFC normalization
                    .replace("\n", " ")
                    .strip()
                    .split()
                )
                for text in batch["text"]
            ]
        }

    def _process(self, dataset: DatasetDict) -> DatasetDict:
        return (dataset
            #.map(
            #    self.normalize_text,
            #    batched=True,
            #    desc="Normalizing text",
            #    num_proc=32
            #)
            .map(
                self._tokenize_batch,
                batched=True,
                desc="Tokenizing text and extracting offsets"
            )
            .map(
                lambda batch: {
                    "sentence_token_spans": [
                        self._sentence_to_token_spans(text, feature_len)
                        for text, feature_len in zip(batch["text"], batch["feature_length"])
                    ]
                },
                batched=True,
                desc="Assigning sentence spans to tokenized text"
                # Dont num_proc here, as it causes issues with encoder and cuda parallelization
            )
            .map(
                lambda batch: {
                    "span_labels": [
                        self._compute_span_labels(label, spans, source, offsets)
                        for label, spans, source, offsets in zip(
                            batch["label"],
                            batch["sentence_token_spans"],
                            batch["source"],
                            batch["offset_mapping"],
                        )
                    ]
                },
                batched=True,
                desc="Assigning labels to sentence spans"
            )
            .remove_columns(["offset_mapping"])
        )

    def _tokenize_batch(self, batch):
        tokenized_text = []
        offset_mappings = []

        for text in batch["text"]:
            encoding = self.luminar_encoder.tokenize(text, truncate=False)
            tokenized_text.append(np.array(encoding["input_ids"]))
            offset_mappings.append(encoding["offset_mapping"])

        return {
            "tokenized_text": tokenized_text,
            "offset_mapping": offset_mappings
        }

    def _sentence_to_token_spans(self,
                                 text: str,
                                 feature_len: int,
                                 span_min_length: int = -1) -> list[tuple[int, int]]:
        """Return a list of (start_token_idx, end_token_idx) for each sentence in the text."""
        sentences = sent_tokenize(text)
        spans = []
        current_token_idx = 0

        for sent in sentences:
            tokens = self.luminar_encoder.tokenize(sent)["input_ids"]
            token_count = len(tokens)

            start = current_token_idx
            end = min(current_token_idx + token_count, feature_len)

            if start >= feature_len:
                break  # Stop if padding region or overflow

            if end - start > span_min_length:
                spans.append((start, end))
            current_token_idx = end

        return spans

    def _compute_span_labels(
            self,
            label: str,
            spans: list[tuple[int, int]],
            source: str,
            offset_mapping: list[tuple[int, int]]
    ):
        # If this item is not fusion, we can return the same label foreach span
        if label != 2:
            return [label] * len(spans)

        # If this is fusion, and its PrismAI, then we can set the labels to each span finegrained
        if "__" in source:
            try:
                range_str = source.split("__")[-1].strip("[]")
                fusion_char_start, fusion_char_end = map(int, range_str.split(":"))
            except Exception:
                return [0] * len(spans)

            def classify_span(start_tok: int, end_tok: int) -> int:
                token_offsets = offset_mapping[start_tok:end_tok]

                # TODO: What about those labels that are not fully inside the fusion range?
                # Right now, we label them as fusion, but maybe we should label them as human?
                if all(start >= fusion_char_start and end <= fusion_char_end for start, end in token_offsets):
                    return 1  # Fully inside fusion (its AI)
                elif any(
                        (start < fusion_char_end and end > fusion_char_start)
                        for start, end in token_offsets
                ):
                    return 2  # Partial overlap with fusion, then we keep labeling it fusion
                else:
                    return 0  # Outside fusion, its human

            return [classify_span(start, end) for (start, end) in spans]

        # If this is fusion, but not PrismAI, we can only label it as fusion.
        return [2] * len(spans)

    @classmethod
    def collate_fn(cls, batch):
        features = [item["features"] for item in batch]
        sentence_spans = [item["sentence_spans"] for item in batch]
        span_labels = [item["span_labels"] for item in batch]

        return {
            "features": features,
            "sentence_spans": sentence_spans,
            "span_labels": span_labels
        }


if __name__ == "__main__":
    # Example Usage
    hub = DataHub((Path.home() / ".hf_token").read_text().strip())
    dataset = hub.get_splits("liberi-luminaris/MAGE-encoded-gpt2")
    #for split in dataset:
    #    dataset[split] = dataset[split].select(range(min(len(dataset[split]), 1000)))
    print(dataset)

    processor = SequentialDataProcessor(LuminarEncoder("gpt2"))
    dataset = processor.process_for_training(dataset)
    print(dataset)

    train_dataset, test_dataset, test_loader = processor.dataset_to_luminar_sequence_dataset(dataset)
