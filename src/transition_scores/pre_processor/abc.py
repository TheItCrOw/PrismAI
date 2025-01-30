from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Any, Iterable

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from transition_scores.data import (
    OutputProbabilities,
    PreProcessorMetadata,
)
from transition_scores.utils import infer_max_length


class PreProcessor(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None = None,
        include_text: bool = False,
    ):
        """Pre-Processor base class wrapping a `PreTrainedTokenizer`.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to wrap.
            max_length (int, optional): The max_length for each tokenized sequence.
                Will try to infer max_length from the tokenizer if not specified.
            include_text (bool, optional): Whether to include the text in the output.

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
        self.include_text = include_text

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        return cls(AutoTokenizer.from_pretrained(model_name_or_path), **kwargs)

    @property
    def required_fields(self) -> dict[str, type]:
        """The fields that this pre-processor requires."""
        raise NotImplementedError

    @property
    def additional_fields(self) -> None | dict[str, type]:
        """The additional fields that this pre-processor adds to the dataset, if any."""
        return None

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    @property
    def all_special_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids)

    @abstractmethod
    def get_metadata(self) -> PreProcessorMetadata: ...

    def _prepare(
        self,
        dataset: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        dataset = [
            # Skip rows that do not have all required fields
            document
            for document in dataset
            if all(bool(document.get(field, False)) for field in self.required_fields)
        ]
        for document in dataset:
            document["text_sha256"] = sha256(document["text"].encode()).hexdigest()
        return dataset

    @abstractmethod
    def _process(self, batch: dict[str, list]) -> BatchEncoding: ...

    @abstractmethod
    def pre_process(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def post_process(
        self,
        dataset: list[dict[str, Any]],
        output_probabilities: list[OutputProbabilities],
    ) -> list[dict[str, Any]]:
        transition_scores = []
        for row, (target_probs, top_k_indices, top_k_probs) in zip(
            dataset, output_probabilities
        ):
            seq_scores = {
                "target_ids": [],
                "target_probs": [],
                "top_k_indices": [],
                "top_k_probs": [],
            }

            # If this model does not use a BOS token, the first token is not predicted,
            # so we add a dummy result with a zero probability
            target_ids = row.pop("input_ids")
            if (first_token := target_ids[0]) not in self.all_special_ids:
                seq_scores["target_ids"].append(first_token)
                seq_scores["target_probs"].append(0.0)
                seq_scores["top_k_indices"].append(None)
                seq_scores["top_k_probs"].append(None)

            # Omit the last token if it is a special token, e.g. <|endoftext|>
            seq_end = -1 if target_ids[-1] in self.all_special_ids else None
            target_ids = target_ids[1:seq_end]

            seq_scores["target_ids"].extend(target_ids)
            seq_scores["target_probs"].extend(target_probs[: len(target_ids)])
            seq_scores["top_k_indices"].extend(top_k_indices[: len(target_ids)])
            seq_scores["top_k_probs"].extend(top_k_probs[: len(target_ids)])

            if "attention_mask" in row:
                del row["attention_mask"]

            transition_scores.append(row | {"transition_scores": seq_scores})
        return transition_scores

    @staticmethod
    def _sort_dataset_by_length(
        dataset: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort the dataset by the length of the input_ids.
        If present, will use and remove the `length` field to sort the dataset.
        Otherwise the length is calculated from the `input_ids`.

        Args:
            dataset (Iterable[dict[str, Any]]): The dataset to sort.

        Returns:
            list[dict[str, Any]]: The sorted dataset.
        """

        def _pop_or_calc_length(row: dict) -> int:
            length = row.pop("length", None)
            return length if length is not None else len(row["input_ids"])

        return list(sorted(dataset, key=_pop_or_calc_length))


def text_sha256(text: str) -> dict[str, str]:
    return {"text_sha256": sha256(text.encode()).hexdigest()}
