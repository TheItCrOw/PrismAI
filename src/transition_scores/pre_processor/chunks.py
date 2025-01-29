from hashlib import sha256
from typing import Any

from tqdm import tqdm
from transformers import BatchEncoding

from transition_scores.data import OutputProbabilities, PreProcessorMetadata
from transition_scores.pre_processor.abc import PreProcessor
from transition_scores.utils import (
    chunks_to_text,
    group_by_column,
    sort_by_column,
    transpose_dict_of_lists,
)


class RollingWindowChunkPreProcessor(PreProcessor):
    """
    Rolling-Window chunk pre-processor.
    Creates prefix-windows of chunks that fit within the max_length.
    """

    @property
    def required_fields(self) -> tuple[str, ...]:
        return {
            "text": str,
            "chunks": list[str],
        }

    @property
    def additional_fields(self) -> tuple[str, ...]:
        return {
            "text": str,
            "prefix": str,
            "prefix_chunk_idx": int,
            "start_chunk_idx": int,
            "end_chunk_idx": int,
            "start_token_idx": int,
        }

    def get_metadata(self) -> PreProcessorMetadata:
        return PreProcessorMetadata.new(
            "chunk-in-context",
            max_length=self.max_length,
        )

    def _process(self, chunks: list[str]) -> BatchEncoding:
        """
        Process a list of chunks from a single document.
        Will apply a rolling-window to create prefix-windows of chunks that fit within the max_length.

        Args:
            chunks (list[str]): List of chunks from a single document.

        Raises:
            ValueError: If the start token of a chunk could not be found in the encoding.

        Returns:
            BatchEncoding: Tokenized batch. Has the following additional fields:
              - `text`: The chunk text, excluding the prefix.
              - `prefix`: The chunk text prefix.
              - `prefix_chunk_idx`: The index of the first chunk in the prefix.
              - `start_chunk_idx`/`end_chunk_idx`: The index of the first/last chunk in the window.
                  Here, `end_chunk_idx` is always `start_chunk_idx + 1`, but we add it for compatibility to synthezied chunks that may cover more than one chunk.
              - `start_token_idx`: The index of the first token in the `input_ids` that belongs to the first chunk in the window.
        """
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
            if not chunks[chunk_idx].strip():
                continue

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
            batch_encoding["text"].append(text[char_idx:])
            batch_encoding["prefix"].append(text[:char_idx])
            batch_encoding["prefix_chunk_idx"].append(prefix_start)
            batch_encoding["start_chunk_idx"].append(chunk_idx)
            batch_encoding["end_chunk_idx"].append(chunk_idx + 1)
            batch_encoding["start_token_idx"].append(token_idx)

        return batch_encoding

    def pre_process(self, dataset: list[dict]) -> list[dict]:
        """Tokenize a dataset of chunks from multiple documents.
        Will return a new dataset of different length, where each row contains a single prefix-window.
        Other fields are duplicated for each prefix-window.

        Args:
            dataset (list[dict]): A dataset containing with fields: `text: str` and `chunks: list[str]`.

        Raises:
            ValueError: If the start token of a chunk could not be found in the encoding.

        Returns:
            list[dict]: Tokenized dataset. Each row contains a single prefix-window.
                The original `text` and `chunks` fields are removed.
                New fields are added:
                  - `text`: The chunk text, excluding the prefix.
                  - `prefix`: The chunk text prefix.
                  - `prefix_chunk_idx`: The index of the first chunk in the prefix.
                  - `start_chunk_idx`/`end_chunk_idx`: The index of the first/last chunk in the window.
                      Here, `end_chunk_idx` is always `start_chunk_idx + 1`, but we add it for compatibility to synthezied chunks that may cover more than one chunk.
                  - `start_token_idx`: The index of the first token in the `input_ids` that belongs to the first chunk in the window.
        """
        with tqdm(
            total=4, desc="Pre-Processing Dataset", position=1, leave=False
        ) as tq:
            tq.set_postfix_str("Calculating Text Hash")
            text_hashes = [
                sha256(row.pop("text").encode()).hexdigest() for row in dataset
            ]
            tq.update(1)

            tq.set_postfix_str("Tokenizing Rolling Windows")
            encodings = [
                self._process(row.pop("chunks"))
                for row in tqdm(dataset, position=2, leave=False)
            ]
            tq.update(1)

            tq.set_postfix_str("Exploding Samples from Encoding")
            dataset = (
                dict(
                    **row,
                    **transposed,
                    text_sha256=txt_hsh,
                )
                for row, txt_hsh, encoding in zip(
                    tqdm(dataset, position=2, leave=False),
                    text_hashes,
                    encodings,
                )
                for transposed in transpose_dict_of_lists(encoding, iter=True)
            )
            tq.update(1)

            tq.set_postfix_str("Sorting Dataset by Length")
            dataset = self._sort_dataset_by_length(dataset)
            tq.update(1)

        return dataset

    def post_process(
        self,
        dataset: list[dict[str, Any]],
        output_probabilities: list[OutputProbabilities],
    ) -> list[dict]:
        with tqdm(total=4, position=1, leave=False, desc="Post-Processing") as tq:
            dataset = super().post_process(dataset, output_probabilities)
            tq.update(1)

            tq.set_postfix_str("Truncating Transition Scores")
            dataset = [
                row
                | {
                    "transition_scores": row.pop("transition_scores")[
                        row["start_token_idx"] :
                    ]
                }
                for row in dataset
            ]
            tq.update(1)

            tq.set_postfix_str("Grouping Transition Scores")
            dataset = group_by_column(
                dataset,
                "_ref_id",
                deduplicate=(
                    "_ref_id",
                    "ref_id",
                    "_orig_ref_id",
                    "orig_ref_id",
                    "text_sha256",
                ),
                aggregate=tuple(self.additional_fields) + ("transition_scores",),
            )
            tq.update(1)

            tq.set_postfix_str("Sorting Transition Scores")
            dataset = sort_by_column(
                dataset,
                "start_chunk_idx",
                tuple(self.additional_fields) + ("transition_scores",),
            )
            tq.update(1)

        return dataset
