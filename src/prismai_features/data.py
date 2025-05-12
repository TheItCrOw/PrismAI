from dataclasses import dataclass, field
from hashlib import sha256
from typing import (
    Any,
    Generator,
    Literal,
    NamedTuple,
    Self,
)

from bson import DBRef, ObjectId

from prismai_features.utils import DataClassMappingMixin


@dataclass
class FeatureValues(DataClassMappingMixin):
    """
    `dataclass` for storing feature values.
    Supports slicing with the same semantics as a list for `int` and `slice` keys,
    while also implementing `Mapping` for MongoDB compatibility.

    As such, `__iter__` returns the names of the fields.

    `__len__`, however, returns the number of target probabilities.

    To iterate over a zipped representation of the feature values, use the `zipped` method.
    """

    target_ids: list[int] = field(default_factory=list)
    target_ranks: list[int] = field(default_factory=list)
    top_k_ids: list[list[int]] = field(default_factory=list)
    top_k_probs: list[list[float]] = field(default_factory=list)
    intermediate_likelihoods: list[list[float]] = field(default_factory=list)
    metrics: list[dict[str, float]] = field(default_factory=list)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key + 1)

        if isinstance(key, slice):
            return FeatureValues(
                self.target_ids[key],
                self.target_ranks[key],
                self.top_k_ids[key],
                self.top_k_probs[key],
                self.intermediate_likelihoods[key],
            )
        return super().__getitem__(key)

    @property
    def target_probs(self):
        return [probs[-1] for probs in self.intermediate_likelihoods]

    def __len__(self):
        return len(self.intermediate_likelihoods)

    def append(
        self,
        target_id: int,
        target_ranks: int,
        top_k_ids: list[int],
        top_k_probs: list[float],
        intermediate_probs: list[float],
        metrics: dict[str, float],
    ):
        self.target_ids.append(target_id)
        self.target_ranks.append(target_ranks)
        self.top_k_ids.append(top_k_ids)
        self.top_k_probs.append(top_k_probs)
        self.intermediate_likelihoods.append(intermediate_probs)
        self.metrics.append(metrics)

    def extend(
        self,
        target_ids: list[int],
        target_ranks: list[int],
        top_k_ids: list[list[int]],
        top_k_probs: list[list[float]],
        intermediate_probs: list[list[float]],
        metrics: list[dict[str, float]],
    ):
        self.target_ids.extend(target_ids)
        self.target_ranks.extend(target_ranks)
        self.top_k_ids.extend(top_k_ids)
        self.top_k_probs.extend(top_k_probs)
        self.intermediate_likelihoods.extend(intermediate_probs)
        self.metrics.extend(metrics)

    @classmethod
    def merge(cls, others: list[Self]) -> Self:
        self = cls()

        for other in others:
            self.extend(
                other.target_ids,
                other.target_ranks,
                other.top_k_ids,
                other.top_k_probs,
                other.get(
                    "intermediate_likelihoods", other.get("intermediate_probs", [])
                ),
                other.get("metrics", []),
            )
        return self

    class Item(NamedTuple):
        target_id: int
        target_prob: float
        target_rank: int
        top_k_index: list[int]
        top_k_prob: list[float]
        intermediate_likelihoods: list[float]

    def zipped(self) -> Generator[Item, None, None]:
        """
        Iterate over a zipped representation of the transition scores.
        Roughly equivalent to:
        >>> yield from zip(ts.target_ids, ts.target_ranks, ts.intermediate_likelihoods, ts.top_k_ids, ts.top_k_probs)  # doctest: +SKIP

        Yields:
            Item: A named tuple containing the target_id, target_prob, intermediate_likelihoods, top_k_ids, and top_k_probs.
        """
        yield from (
            self.Item(*item)
            for item in zip(
                self["target_ids"],
                self["target_ranks"],
                self["top_k_ids"],
                self["top_k_probs"],
                self["intermediate_likelihoods"],
            )
        )

    def unpack(self) -> tuple:
        return (
            self.target_ids,
            self.target_ranks,
            self.top_k_ids,
            self.top_k_probs,
            self.intermediate_likelihoods,
        )


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


class PreProcessorMetadata(dict):
    @classmethod
    def new(
        cls,
        type: Literal[
            "text", "chunk", "chunk-in-context", "sliding-window", "truncate"
        ],
        **metadata,
    ) -> Self:
        return cls(
            {
                "type": type,
                **metadata,
            }
        )


@dataclass
class DocumentMetadata(DataClassMappingMixin):
    _id: DBRef
    id: str
    agent: str
    domain: str
    label: Literal["human", "ai", "fusion"]
    lang: str
    source: str
    text_sha256: str
    type: Literal["source", "chunk", "fulltext"]

    def __hash__(self):
        return self._id.__hash__()

    @classmethod
    def add_metadata_to_document(
        cls,
        document: dict[str, Any],
        source_collection: str,
    ) -> None:
        agent = document.pop("agent", None)
        document["document"] = cls(
            _id=DBRef(source_collection, document.pop("_id")),  # type: ignore
            id=document.pop("id"),
            agent=agent,
            domain=document.pop("domain"),
            label=document.pop("label", "human" if not agent else "ai"),
            lang=document.pop("lang"),
            source=document.pop("source", None),
            text_sha256=sha256(document["text"].encode()).hexdigest(),
            type=document.pop("type", "source"),
        )


class FeaturesDict(dict):
    @classmethod
    def new(
        cls,
        document: DocumentMetadata | dict,
        model: ModelMetadata | dict,
        pre_processor: PreProcessorMetadata | dict,
        features: FeatureValues | None = None,
        metrics: list[dict] | None = None,
        split: str | None = None,
        _id: ObjectId | None = None,
        **metadata,
    ) -> Self:
        doc = {
            "_id": _id or ObjectId(),
            "document": document,
            "model": model,
            "pre_processor": pre_processor,
            "split": split,
            "metadata": metadata,
        }
        if features:
            doc["features"] = features
        if metrics:
            doc["metrics"] = metrics

        return cls(doc)

    @classmethod
    def from_tuple(cls, tup: tuple) -> Self:
        return cls.new(*tup)

    def split(self) -> tuple[Self, Self]:
        _split = self.get("_split", str(self["_id"]))
        features = self["features"]
        idx = len(features) // 2
        features_a, features_b = features[:idx], features[idx:]
        ma, mb = {}, {}
        for key, value in self["metadata"].items():
            if isinstance(value, (tuple, list)):
                idx = len(value) // 2
                ma[key], mb[key] = value[:idx], value[idx:]
            else:
                ma[key], mb[key] = value, value
        return (
            type(self)(
                {
                    "_id": ObjectId(),
                    "_split": _split + ".0",
                    "document": self["document"],
                    "model": self["model"],
                    "pre_processor": self["pre_processor"],
                    "features": features_a,
                    "split": self["split"],
                    "metadata": ma,
                }
            ),
            type(self)(
                {
                    "_id": ObjectId(),
                    "_split": _split + ".1",
                    "document": self["document"],
                    "model": self["model"],
                    "pre_processor": self["pre_processor"],
                    "features": features_b,
                    "split": self["split"],
                    "metadata": mb,
                }
            ),
        )


def convert_to_mongo(
    document: dict,
    model_metadata: ModelMetadata,
    pre_processor_metadata: PreProcessorMetadata,
) -> FeaturesDict:
    document_metadata = document.pop("document")
    return FeaturesDict.new(
        document=document_metadata,
        model=model_metadata,
        pre_processor=pre_processor_metadata,
        split=document.pop("split", None),
        **document,
    )
