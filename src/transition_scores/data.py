from abc import ABC, abstractmethod
from collections import UserDict
from hashlib import sha256
from typing import Any, NamedTuple

from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from transition_scores.utils import transpose_dict_of_lists


class EncodedSequence(NamedTuple):
    input_ids: list[int]
    attention_mask: list[int]


class TransitionScores(UserDict):
    def __init__(
        self,
        target_id: int,
        target_prob: float,
        top_k_ids: list[int],
        top_k_scores: list[float],
    ):
        super().__init__(
            {
                "target_id": target_id,
                "target_prob": target_prob,
                "top_k_ids": top_k_ids,
                "top_k_scores": top_k_scores,
            }
        )


def infer_max_length(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    if hasattr(config, "n_positions"):
        return config.n_positions
    raise ValueError(f"Could not infer max length from {model_name_or_path}")


def chunks_to_text(chunks: list[str]) -> str:
    return " ".join(chunk.strip() for chunk in chunks)


class PreProcessor(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int | None = None):
        """Pre-Processor base class wrapping a `PreTrainedTokenizer`.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to wrap.
            max_length (int, optional): The max_length for each tokenized sequence.
                Will try to infer max_length from the tokenizer if not specified.

        Raises:
            ValueError: If max_length is not given and we could not infer it from the tokenizer.
        """
        self._tokenizer = tokenizer
        max_length = max_length or tokenizer.model_max_length
        if max_length is None:
            try:
                max_length = infer_max_length(tokenizer.name_or_path)
            except ValueError as e:
                raise ValueError(
                    f"max_length was not given and we could not infer the max_length from {type(tokenizer)}({tokenizer.name_or_path}). Please provide a max_length to the {type(self).__name__} constructor."
                ) from e
        self.max_length = max_length

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        return cls(AutoTokenizer.from_pretrained(model_name_or_path), **kwargs)

    @abstractmethod
    def process(self, batch: dict[str, list]) -> BatchEncoding: ...

    @abstractmethod
    def prepare_dataset(self, dataset: Dataset) -> Dataset: ...


class TextPreProcessor(PreProcessor):
    """
    Simple `text` pre-processor.
    Sequences are tokenized and truncated to `max_length`.
    """

    def process(self, batch: dict[str, list]) -> BatchEncoding:
        """Process a *batch* of samples.
        Expects a dictionary with a "text" field containing a list of strings.

        Note:
            Effectively calls:
            ```py
            >>> tokenizer(
            >>>     batch["text"],
            >>>     truncation=True,
            >>>     return_length=True,
            >>>     add_special_tokens=True,
            >>> )
            ```

        Args:
            batch (dict[str, list]): Batch of samples to tokenize.

        Returns:
            BatchEncoding | dict[str, list]: Tokenized batch.
        """
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            add_special_tokens=True,
        )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare the `text` of the samples in a dataset.
        Adds `text_sha256` field to the dataset.

        Args:
            dataset (Dataset): A dataset containing with fields: `text: str` and `chunks: list[str]`.

        Returns:
            Dataset: Tokenized dataset. The `text` and `chunks` fields are removed.
        """
        return (
            dataset.map(
                lambda row: {"text_sha256": sha256(row["text"].encode()).hexdigest()},
            )
            .map(self.process, batched=True, remove_columns=["text", "chunks"])
            .sort("length")
            .remove_columns("length")
        )


class RollingWindowChunkPreProcessor(TextPreProcessor):
    """
    Rolling-Window chunk pre-processor.
    Creates prefix-windows of chunks that fit within the max_length.
    """

    def process(self, row: dict[str, Any]) -> BatchEncoding:
        """Process a list of chunks from a single document.
        Will apply a rolling-window to create prefix-windows of chunks that fit within the max_length.

        Args:
            row (dict[str, Any]): A dictionary with a "chunks" field containing a list of strings.

        Raises:
            ValueError: If the start token of a chunk could not be found in the encoding.

        Returns:
            BatchEncoding: Tokenized batch. Has the following additional fields:
              - `text`: The entire text, including prefix.
              - `prefix_idx`: The index of the first chunk in the prefix.
              - `start_idx`/`end_idx`: The index of the first/last chunk in the window.
                  Here, `end_idx` is always `start_idx + 1`, but we add it for compatibility to synthezied chunks that may cover more than one chunk.
              - `start_token_idx`: The index of the first token in the `input_ids` that belongs to the first chunk in the window.
        """
        chunks = row["chunks"]
        batch_encoding = BatchEncoding(
            {
                "input_ids": [],
                "attention_mask": [],
                "length": [],
                "text": [],
                "prefix_idx": [],
                "start_idx": [],
                "end_idx": [],
                "start_token_idx": [],
            }
        )

        # For each span, try to find the largest possible prefix-span that fits within the max_length.
        prefix_start = 0
        for chunk_idx in range(len(chunks)):
            while True:
                buffer = chunks[prefix_start : chunk_idx + 1]
                text = chunks_to_text(buffer)
                encoding = self._tokenizer(
                    text,
                    return_length=True,
                    add_special_tokens=True,
                )

                if (
                    prefix_start == chunk_idx
                    or len(encoding["input_ids"]) <= self.max_length
                ):
                    break

                prefix_start += 1

            try:
                char_idx = text.index(chunks[chunk_idx].strip())
                word_idx = encoding.char_to_word(char_idx)
                token_idx, _ = encoding.word_to_tokens(word_idx)
            except Exception as e:
                raise ValueError(
                    f"Could not find the start token of chunk {chunk_idx}:'{chunks[chunk_idx]}' in the encoding of '{text}'."
                ) from e

            if encoding["length"][0] > self.max_length:
                encoding = self._tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    return_length=True,
                    add_special_tokens=True,
                )

            batch_encoding["input_ids"].append(encoding["input_ids"])
            batch_encoding["attention_mask"].append(encoding["attention_mask"])
            batch_encoding["length"].extend(encoding["length"])
            batch_encoding["text"].append(text)
            batch_encoding["prefix_idx"].append(prefix_start)
            batch_encoding["start_idx"].append(chunk_idx)
            batch_encoding["end_idx"].append(chunk_idx + 1)
            batch_encoding["start_token_idx"].append(token_idx)

        return {"encoding": batch_encoding}

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize a dataset of chunks from multiple documents.
        Will return a new dataset of different length, where each row contains a single prefix-window.
        Other fields are duplicated for each prefix-window.

        Args:
            dataset (Dataset): A dataset containing with fields: `text: str` and `chunks: list[str]`.

        Returns:
            Dataset: Tokenized dataset. Each row contains a single prefix-window.
                The `text` and `chunks` fields are removed.
        """
        dataset = dataset.map(
            lambda row: {"text_sha256": sha256(row["text"].encode()).hexdigest()},
            remove_columns=["text"],
        ).map(
            self.process,
            remove_columns=["chunks"],
        )
        return (
            dataset.map(
                flatten_batch_encoding_of_one,
                batched=True,
                batch_size=1,
                remove_columns=dataset.column_names,
            )
            .sort("length")
            .remove_columns("length")
        )


def flatten_batch_encoding_of_one(row: dict[str, list]) -> dict[str, list]:
    encoding = row.pop("encoding")[0]
    flattend = {key: [] for key in row.keys() | encoding.keys()}
    for transposed in transpose_dict_of_lists(encoding, iter=True):
        for key, [value] in row.items():
            flattend[key].append(value)
        for key, value in transposed.items():
            flattend[key].append(value)
    return flattend
