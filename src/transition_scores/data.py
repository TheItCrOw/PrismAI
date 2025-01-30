from typing import Any, Literal, NamedTuple, Self

from bson import ObjectId


class OutputProbabilities(NamedTuple):
    target_probs: list[float]
    top_k_indices: list[list[int]]
    top_k_probs: list[list[float]]


class TransitionScores(dict):
    @classmethod
    def new(
        cls,
        target_id: int,
        target_prob: float,
        top_k_ids: list[int],
        top_k_scores: list[float],
    ) -> Self:
        return cls(
            {
                "target_id": target_id,
                "target_prob": target_prob,
                "top_k_ids": top_k_ids,
                "top_k_scores": top_k_scores,
            }
        )

    @classmethod
    def from_tuple(cls, tup: tuple[int, float, list[int], list[float]]) -> Self:
        return cls.new(*tup)


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
