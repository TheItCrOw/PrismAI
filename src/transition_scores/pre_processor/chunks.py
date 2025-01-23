from datasets import Dataset
from transformers import BatchEncoding

from transition_scores.pre_processor.abc import text_sha256
from transition_scores.pre_processor.text import TextPreProcessor
from transition_scores.utils import chunks_to_text, transpose_dict_of_lists


def flatten_batch_encoding_of_one(row: dict[str, list]) -> dict[str, list]:
    encoding = row.pop("encoding")[0]
    flattend = {key: [] for key in row.keys() | encoding.keys()}
    for transposed in transpose_dict_of_lists(encoding, iter=True):
        for key, [value] in row.items():
            flattend[key].append(value)
        for key, value in transposed.items():
            flattend[key].append(value)
    return flattend


class RollingWindowChunkPreProcessor(TextPreProcessor):
    """
    Rolling-Window chunk pre-processor.
    Creates prefix-windows of chunks that fit within the max_length.
    """

    @property
    def additional_fields(self) -> tuple[str, ...]:
        return (
            "text",
            "prefix_idx",
            "start_idx",
            "end_idx",
            "start_token_idx",
        )

    def process(self, row: dict[str, list[str]]) -> BatchEncoding:
        """
        Process a list of chunks from a single document.
        Will apply a rolling-window to create prefix-windows of chunks that fit within the max_length.

        Args:
            row (dict[str, Any]): A dictionary with a "chunks" field containing a list of strings.

        Raises:
            ValueError: If the start token of a chunk could not be found in the encoding.

        Returns:
            BatchEncoding: Tokenized batch. Has the following additional fields:
              - `text`: The chunk text, including prefix.
              - `prefix_idx`: The index of the first chunk in the prefix.
              - `start_idx`/`end_idx`: The index of the first/last chunk in the window.
                  Here, `end_idx` is always `start_idx + 1`, but we add it for compatibility to synthezied chunks that may cover more than one chunk.
              - `start_token_idx`: The index of the first token in the `input_ids` that belongs to the first chunk in the window.
        """
        chunks = row["chunks"]
        batch_encoding = BatchEncoding(
            {
                key: []
                for key in (
                    "input_ids",
                    "attention_mask",
                    "length",
                    *self.additional_fields,
                )
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

        Raises:
            ValueError: If the start token of a chunk could not be found in the encoding.

        Returns:
            Dataset: Tokenized dataset. Each row contains a single prefix-window.
                The original `text` and `chunks` fields are removed.
                New fields are added:
                  - `text`: The chunk text, including prefix.
                  - `prefix_idx`: The index of the first chunk in the prefix.
                  - `start_idx`/`end_idx`: The index of the first/last chunk in the window.
                      Here, `end_idx` is always `start_idx + 1`, but we add it for compatibility to synthezied chunks that may cover more than one chunk.
                  - `start_token_idx`: The index of the first token in the `input_ids` that belongs to the first chunk in the window.
        """
        return (
            dataset.map(text_sha256, remove_columns=["text"])
            .map(self.process, remove_columns=["chunks"])
            .map(
                flatten_batch_encoding_of_one,
                batched=True,
                batch_size=1,
            )
            .sort("length")
            .remove_columns("length")
        )
