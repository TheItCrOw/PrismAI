from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Self

from bson import ObjectId

from transition_scores.utils import DataClassMappingMixin


class OutputProbabilities(NamedTuple):
    target_probs: list[float]
    top_k_indices: list[list[int]]
    top_k_probs: list[list[float]]


@dataclass
class TransitionScores(DataClassMappingMixin):
    target_ids: list[int] = field(default_factory=list)
    target_probs: list[float] = field(default_factory=list)
    top_k_indices: list[list[int]] = field(default_factory=list)
    top_k_probs: list[list[float]] = field(default_factory=list)

    def __iter__(self):
        yield from zip(
            self.target_ids,
            self.target_probs,
            self.top_k_indices,
            self.top_k_probs,
        )

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return TransitionScores(
                self.target_ids[key],
                self.target_probs[key],
                self.top_k_indices[key],
                self.top_k_probs[key],
            )
        return super().__getitem__(key)

    def append(
        self,
        target_id: int,
        target_probs: float,
        top_k_indices: list[int],
        top_k_probs: list[float],
    ):
        self.target_ids.append(target_id)
        self.target_probs.append(target_probs)
        self.top_k_indices.append(top_k_indices)
        self.top_k_probs.append(top_k_probs)

    def extend(
        self,
        target_ids: list[int],
        target_probs: list[float],
        top_k_indices: list[list[int]],
        top_k_probs: list[list[float]],
    ):
        self.target_ids.extend(target_ids)
        self.target_probs.extend(target_probs)
        self.top_k_indices.extend(top_k_indices)
        self.top_k_probs.extend(top_k_probs)

    @classmethod
    def merge(cls, others: list[Self]) -> Self:
        self = cls()

        for other in others:
            self.extend(
                other.target_ids,
                other.target_probs,
                other.top_k_indices,
                other.top_k_probs,
            )
        return self


class ModelMetadata(dict):
    @classmethod
    def new(
        cls,
        name: str,
        provider: Literal["onnx", "transformers"],
        variant: Literal["default", "8bit", "o1", "o2", "o3", "o4"] = "default",
        **metadata,
    ) -> Self:
        return cls(
            {
                "name": name,
                "provider": provider,
                "variant": variant,
                **metadata,
            }
        )

    @classmethod
    def from_tuple(cls, tup: tuple) -> Self:
        return cls.new(*tup)


class PreProcessorMetadata(dict):
    @classmethod
    def new(
        cls,
        type: Literal["text", "chunk", "chunk-in-context"],
        **metadata,
    ) -> Self:
        return cls(
            {
                "type": type,
                **metadata,
            }
        )

    @classmethod
    def from_tuple(cls, tup: tuple) -> Self:
        return cls.new(*tup)


class FeaturesDict(dict):
    @classmethod
    def new(
        cls,
        refs: dict,
        text_sha256: str,
        model: ModelMetadata | dict,
        pre_processor: PreProcessorMetadata | dict,
        transition_scores: list[TransitionScores],
        **metadata,
    ) -> Self:
        return cls(
            {
                "_id": ObjectId(),
                "refs": refs,
                "text_sha256": text_sha256,
                "model": model,
                "pre_processor": pre_processor,
                "transition_scores": transition_scores,
                "metadata": metadata,
            }
        )

    @classmethod
    def from_tuple(cls, tup: tuple) -> Self:
        return cls.new(*tup)

    def split(self) -> tuple[Self, Self]:
        _split = self.get("_split", str(self["_id"]))
        ts = self["transition_scores"]
        tsa, tsb = (
            ts[: len(ts) // 2],
            ts[len(ts) // 2 :],
        )
        ma, mb = {}, {}
        for key, value in self["metadata"].items():
            if isinstance(value, list):
                ma[key], mb[key] = (
                    value[: len(value) // 2],
                    value[len(value) // 2 :],
                )
            else:
                ma[key], mb[key] = value, value
        return (
            FeaturesDict(
                {
                    "_id": ObjectId(),
                    "_split": _split + ".0",
                    "refs": self["refs"],
                    "text_sha256": self["text_sha256"],
                    "model": self["model"],
                    "pre_processor": self["pre_processor"],
                    "transition_scores": tsa,
                    "metadata": ma,
                }
            ),
            FeaturesDict(
                {
                    "_id": ObjectId(),
                    "_split": _split + ".1",
                    "refs": self["refs"],
                    "text_sha256": self["text_sha256"],
                    "model": self["model"],
                    "pre_processor": self["pre_processor"],
                    "transition_scores": tsb,
                    "metadata": mb,
                }
            ),
        )


def remove_columns(
    dataset: list[dict[str, Any]],
    *columns: str,
    in_place: bool = False,
) -> list[dict[str, Any]]:
    """
    Remove the specified columns from the dataset.

    Args:
        dataset (list[dict[str, Any]]): The dataset to remove columns from.
        *columns (str): A sequence of column names to remove.
        in_place (bool): Apply the operation **in-place**, modifying the original dataset.
            Default: `False`.

    Returns:
        list[dict[str, Any]]: The dataset with the specified columns removed.
    """
    columns = set(columns)
    if in_place:
        for document in dataset:
            for column in columns:
                document.pop(column, None)
        return dataset
    else:
        return [
            {key: value for key, value in document.items() if key not in columns}
            for document in dataset
        ]
