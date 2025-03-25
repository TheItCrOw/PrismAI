from abc import ABC, abstractmethod
from typing import Any

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from prismai_features.data import (
    PreProcessorMetadata,
)
from prismai_features.utils import infer_max_length
from simple_dataset.dataset import Dataset


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
        self.max_length: int = max_length
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
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if isinstance(pad_token_id, int):
            return pad_token_id

        raise ValueError(
            f"Could not infer pad_token_id from {type(self.tokenizer)}({self.tokenizer.name_or_path})."
        )

    @property
    def all_special_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids)

    @abstractmethod
    def get_metadata(self) -> PreProcessorMetadata: ...

    def _prepare(
        self,
        dataset: Dataset[str, Any],
    ) -> Dataset[str, Any]:
        dataset.filter(self._filter_required_fields, in_place=True)

        return dataset

    def _filter_required_fields(self, document: dict[str, Any]) -> bool:
        return all((field in document) for field in self.required_fields)

    @abstractmethod
    def _process(self, batch: dict[str, list]) -> BatchEncoding: ...

    @abstractmethod
    def pre_process(self, dataset: Dataset[str, Any]) -> Dataset[str, Any]: ...

    def post_process(
        self,
        dataset: Dataset[str, Any],
    ) -> Dataset[str, Any]:
        return dataset.remove_columns("input_ids", "attention_mask")
