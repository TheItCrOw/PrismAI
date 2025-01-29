from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Any

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from transition_scores.data import PreProcessorMetadata
from transition_scores.utils import infer_max_length


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

    @property
    def required_fields(self) -> tuple[str, ...]:
        """The fields that this pre-processor requires."""
        return ()

    @property
    def additional_fields(self) -> None | tuple[str, ...]:
        """The additional fields that this pre-processor adds to the dataset, if any."""
        return None

    @abstractmethod
    def get_metadata(self) -> PreProcessorMetadata: ...

    @abstractmethod
    def _process(self, batch: dict[str, list]) -> BatchEncoding: ...

    @abstractmethod
    def pre_process(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


def text_sha256(text: str) -> dict[str, str]:
    return {"text_sha256": sha256(text.encode()).hexdigest()}
