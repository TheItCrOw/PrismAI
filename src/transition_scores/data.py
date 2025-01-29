from typing import Any, Literal, NamedTuple, Self

from bson.dbref import DBRef


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


class ScoresDict(dict):
    @classmethod
    def new(
        cls,
        _ref_id: DBRef | dict,
        ref_id: str,
        text_sha256: str,
        transition_scores: list[TransitionScores],
        **metadata,
    ):
        return cls(
            {
                "_ref_id": _ref_id,
                "ref_id": ref_id,
                "text_sha256": text_sha256,
                "transition_scores": transition_scores,
                "metadata": metadata,
            }
        )

    @classmethod
    def from_raw(
        cls,
        scores: dict[str, Any],
        collection: str = "collected_items",
        database: str | None = None,
    ) -> Self:
        return cls.new(
            DBRef(collection, scores.pop("_id"), database=database),
            scores.pop("text_sha256"),
            scores.pop("transition_scores"),
            **scores,
        )


class FeaturesDict(dict):
    @classmethod
    def new(
        cls,
        _ref_id: DBRef | dict,
        ref_id: str,
        text_sha256: str,
        model: ModelMetadata | dict,
        pre_processor: PreProcessorMetadata | dict,
        transition_scores: list[TransitionScores],
        **metadata,
    ) -> Self:
        return cls(
            {
                "_ref_id": _ref_id,
                "ref_id": ref_id,
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

    @classmethod
    def from_scores(
        cls,
        scores: "ScoresDict",
        model_metadata: ModelMetadata,
        pre_processor_metadata: PreProcessorMetadata,
    ) -> Self:
        metadata = scores.pop("metadata")
        return cls.new(
            **scores,
            model=model_metadata,
            pre_processor=pre_processor_metadata,
            **metadata,
        )
