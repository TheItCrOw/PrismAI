from typing import Any

from tqdm import tqdm
from transformers import BatchEncoding

from simple_dataset.dataset import Dataset
from transition_scores.data import (
    OutputProbabilities,
    PreProcessorMetadata,
)
from transition_scores.pre_processor.text import TextPreProcessor
from transition_scores.utils import (
    _explode_encodings,
    _pop_or_calc_length,
)


class SlidingWindowTextPreProcessor(TextPreProcessor):
    def __init__(
        self,
        tokenizer,
        stride: int | None = None,
        max_length: int | None = None,
        include_text: bool = False,
    ):
        """
        Sliding-Window text pre-processor.
        Creates sliding text windows of a fixed length with configurable stride/overlap.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to wrap.
            stride (int, optional): The stride for the sliding window.
                Defaults to 1/4 of the max_length.
            max_length (int, optional): The max_length for each tokenized sequence.
                Will try to infer max_length from the tokenizer if not specified.
        """
        super().__init__(tokenizer, max_length, include_text)
        self.stride = stride or (self.max_length // 4)

    @property
    def required_fields(self) -> dict[str, type]:
        return {"text": str}

    @property
    def additional_fields(self) -> dict[str, type]:
        fields = {
            "start_token_idx": int,
            "prefix_token_offset": int,
            "window_size": int,
        }
        if self.include_text:
            fields.update({"text": str, "prefix": str})

        return fields

    def get_metadata(self) -> PreProcessorMetadata:
        return PreProcessorMetadata.new(
            "sliding-window",
            stride=self.stride,
            max_length=self.max_length,
        )

    @property
    def num_special_tokens_to_add(self) -> int:
        return self._tokenizer.num_special_tokens_to_add(False)

    def _process(self, text: str) -> BatchEncoding:
        """
        Process a single text sample.
        Will apply a sliding-window to create prefix-windows of chunks that fit within the max_length.

        Args:
            text (str): The text to tokenize.

        Returns:
            BatchEncoding: Tokenized batch. Has the following additional fields:
              - `text`: The sliding window text.
              - `prefix`: The sliding window text prefix.
              - `start_token_idx`: The index of the first token in the window.
              - `prefix_token_offset`: The offset of the first token from the entire text.
              - `window_size`: The number of tokens in the window.
        """
        batch_encoding = self._tokenizer(
            text,
            stride=self.max_length - self.stride,
            max_length=self.max_length,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )
        del batch_encoding["overflow_to_sample_mapping"]

        # As the first window is not a sliding window, it covers all tokens,
        # so we set the `start_token_idx` to 0.
        # The rest of the windows cover only the last `stride` tokens.
        batch_encoding["start_token_idx"] = [0] + [
            max(0, self.max_length - self.stride)
            for _ in range(len(batch_encoding.input_ids[1:]))
        ]
        batch_encoding["window_size"] = [len(batch_encoding.input_ids[0])] + [
            len(input_ids) - self.max_length + self.stride
            for input_ids in batch_encoding.input_ids[1:]
        ]
        batch_encoding["prefix_token_offset"] = [
            i * self.stride for i, _ in enumerate(batch_encoding.input_ids)
        ]

        if self.include_text:
            batch_encoding["text"] = self._tokenizer.batch_decode(
                [
                    input_ids[start_token_idx:]
                    for input_ids, start_token_idx in zip(
                        batch_encoding.input_ids, batch_encoding.start_token_idx
                    )
                ],
                skip_special_tokens=True,
            )
            batch_encoding["prefix"] = self._tokenizer.batch_decode(
                [
                    input_ids[:start_token_idx]
                    for input_ids, start_token_idx in zip(
                        batch_encoding.input_ids, batch_encoding.start_token_idx
                    )
                ],
                skip_special_tokens=True,
            )

        return batch_encoding

    def pre_process(self, dataset: Dataset[str, Any]) -> Dataset[str, Any]:
        """Tokenize a dataset of chunks from multiple documents.
        Will return a new dataset of different length, where each row contains a single prefix-window.
        Other fields are duplicated for each prefix-window.

        Args:
            dataset (Dataset[str, Any]): A dataset containing with fields: `text: str` and `chunks: list[str]`.

        Raises:
            ValueError: If the start token of a chunk could not be found in the encoding.

        Returns:
            Dataset[str, Any]: Tokenized dataset. Each row contains a single prefix-window.
                The original `text` and `chunks` fields are removed.
                New fields are added:
                  - `text`: The chunk text, excluding the prefix.
                  - `prefix`: The chunk text prefix.
                  - `start_token_idx`: The index of the first token in the window.
                  - `prefix_token_offset`: The offset of the first token from the entire text.
                  - `window_size`: The number of tokens in the window.
        """
        with tqdm(total=3, position=2, leave=False, desc="Pre-Processing") as tq:
            try:
                tq.set_postfix_str("Preparing Dataset")
                dataset = self._prepare(dataset)
                tq.update(1)

                tq.set_postfix_str("Tokenizing Sliding Windows")
                encodings = [
                    self._process(document.pop("text")) for document in dataset
                ]
                tq.update(1)

                tq.set_postfix_str("Exploding Documents from Encoding")
                dataset.flat_map_zip(_explode_encodings, encodings)
                tq.update(1)

                tq.set_postfix_str("Sorting Dataset by Length")
                dataset.sort_by(_pop_or_calc_length, in_place=True)
                tq.update(1)
            except KeyError as e:
                raise KeyError(
                    f"{type(self).__name__} requires the fields: {self.required_fields}."
                ) from e

        return dataset

    def post_process(
        self,
        dataset: Dataset[str, Any],
        output_probabilities: list[OutputProbabilities],
    ) -> Dataset[str, Any]:
        with tqdm(total=4, position=2, leave=False, desc="Post-Processing") as tq:
            dataset = super().post_process(dataset, output_probabilities)
            tq.update(1)

            # TODO: Extract the transition scores for the first couple of windows from the output probabilities.
            tq.set_postfix_str("Truncating Transition Scores")
            dataset.apply(
                lambda ts, idx: ts[idx:],
                "transition_scores",
                "start_token_idx",
            )
            tq.update(1)

            tq.set_postfix_str("Grouping Transition Scores")
            dataset.group_documents_by(
                by="_id",
                deduplicate=("_id", "id", "text_sha256"),
                aggregate=tuple(self.additional_fields) + ("transition_scores",),
            )
            tq.update(1)

            tq.set_postfix_str("Sorting Transition Scores")
            dataset.sort_documents_by(
                "prefix_token_offset",
                *self.additional_fields.keys(),
                "transition_scores",
            )
            tq.update(1)

        return dataset
