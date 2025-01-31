from abc import ABC, abstractmethod
from typing import Any

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from simple_dataset.dataset import Dataset
from transition_scores.data import (
    OutputProbabilities,
    PreProcessorMetadata,
    TransitionScores,
)
from transition_scores.utils import add_text_sha256, infer_max_length


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
        dataset: Dataset[str, Any],
    ) -> Dataset[str, Any]:
        dataset.filter(self._filter_required_fields, in_place=True)
        dataset.modify(add_text_sha256)

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
        output_probabilities: list[OutputProbabilities],
    ) -> Dataset[str, Any]:
        dataset.modify_zip(
            self._add_transition_scores,
            output_probabilities,
        )
        return dataset

    def _add_transition_scores(
        self,
        document: dict[str, Any],
        output_probabilities: OutputProbabilities,
    ) -> None:
        document["transition_scores"] = TransitionScores()

        # If this model does not use a BOS token, the first token is not predicted,
        # so we add a dummy result with a zero probability
        target_ids = document.pop("input_ids")
        if (first_token := target_ids[0]) not in self.all_special_ids:
            document["transition_scores"].append(first_token, 0.0, None, None)

        # Omit the last token if it is a special token, e.g. <|endoftext|>
        seq_end = -1 if target_ids[-1] in self.all_special_ids else None
        target_ids = target_ids[1:seq_end]

        target_probs, top_k_indices, top_k_probs = output_probabilities
        document["transition_scores"].extend(
            target_ids,
            target_probs[: len(target_ids)],
            top_k_indices[: len(target_ids)],
            top_k_probs[: len(target_ids)],
        )

        if "attention_mask" in document:
            del document["attention_mask"]
