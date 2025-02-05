from dataclasses import dataclass, field
from hashlib import sha256
from typing import (
    Generator,
    Literal,
    NamedTuple,
    Self,
)

from bson import DBRef, ObjectId

from transition_scores.utils import DataClassMappingMixin


class OutputProbabilities(NamedTuple):
    target_probs: list[float]
    target_ranks: list[int]
    top_k_indices: list[list[int]]
    top_k_probs: list[list[float]]


@dataclass
class TransitionScores(DataClassMappingMixin):
    """
    `dataclass` for storing transition scores.
    Supports slicing with the same semantics as a list for `int` and `slice` keys,
    while also implementing `Mapping` for MongoDB compatibility.

    As such, `__iter__` returns the names of the fields.

    `__len__`, however, returns the number of target probabilities.

    To iterate over a zipped representation of the transition scores, use the `zipped` method.
    """

    target_ids: list[int] = field(default_factory=list)
    target_probs: list[float] = field(default_factory=list)
    target_ranks: list[int] = field(default_factory=list)
    top_k_indices: list[list[int]] = field(default_factory=list)
    top_k_probs: list[list[float]] = field(default_factory=list)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return TransitionScores(
                self.target_ids[key],
                self.target_probs[key],
                self.target_ranks[key],
                self.top_k_indices[key],
                self.top_k_probs[key],
            )
        return super().__getitem__(key)

    def __len__(self):
        return len(self.target_probs)

    def append(
        self,
        target_id: int,
        target_probs: float,
        target_ranks: float,
        top_k_indices: list[int],
        top_k_probs: list[float],
    ):
        self.target_ids.append(target_id)
        self.target_probs.append(target_probs)
        self.target_ranks.append(target_ranks)
        self.top_k_indices.append(top_k_indices)
        self.top_k_probs.append(top_k_probs)

    def extend(
        self,
        target_ids: list[int],
        target_probs: list[float],
        target_ranks: list[float],
        top_k_indices: list[list[int]],
        top_k_probs: list[list[float]],
    ):
        self.target_ids.extend(target_ids)
        self.target_probs.extend(target_probs)
        self.target_ranks.extend(target_ranks)
        self.top_k_indices.extend(top_k_indices)
        self.top_k_probs.extend(top_k_probs)

    @classmethod
    def merge(cls, others: list[Self]) -> Self:
        self = cls()

        for other in others:
            self.extend(
                other["target_ids"],
                other["target_probs"],
                other["target_ranks"],
                other["top_k_indices"],
                other["top_k_probs"],
            )
        return self

    class Item(NamedTuple):
        target_id: int
        target_prob: float
        target_rank: int
        top_k_index: list[int]
        top_k_prob: list[float]

    def zipped(self) -> Generator[Item, None, None]:
        """
        Iterate over a zipped representation of the transition scores.
        Roughly equivalent to:
        >>> yield from zip(ts.target_ids, ts.target_probs, ts.target_ranks, ts.top_k_indices, ts.top_k_probs)  # doctest: +SKIP

        Yields:
            Item: A named tuple containing the target_id, target_prob, top_k_index, and top_k_prob.
        """
        yield from (
            self.Item(item)
            for item in zip(
                self["target_ids"],
                self["target_probs"],
                self["target_ranks"],
                self["top_k_indices"],
                self["top_k_probs"],
            )
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
        type: Literal["text", "chunk", "chunk-in-context", "sliding-window"],
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
    _id: ObjectId
    domain: str
    lang: str
    text_sha256: str
    type: Literal["source", "chunk", "fulltext"] = "source"
    label: Literal["human", "ai", "fusion"] = "human"
    agent: str | None = None
    _synth_id: ObjectId | None = None

    def __hash__(self):
        return self._id.__hash__()

    @classmethod
    def add_metadata_to_document(
        cls,
        document: dict[str, str],
        source_collection: str = "collected_items",
    ) -> None:
        _id: ObjectId | DBRef = document.pop("_id")
        if (_ref_id := document.pop("_ref_id", None)) is None:
            _id = DBRef(source_collection, _id)
            _synth_id = None
        else:
            _synth_id = DBRef(source_collection, _id)
            _id = _ref_id

        document["document"] = cls(
            _id=_id,
            domain=document.pop("domain"),
            lang=document.pop("lang"),
            text_sha256=sha256(document["text"].encode()).hexdigest(),
            type=document.pop("type", "source"),
            label=document.pop("label", None),
            agent=document.pop("agent", None),
            _synth_id=_synth_id,
        )


class FeaturesDict(dict):
    @classmethod
    def new(
        cls,
        document: DocumentMetadata | dict,
        model: ModelMetadata | dict,
        pre_processor: PreProcessorMetadata | dict,
        transition_scores: list[TransitionScores],
        _id: ObjectId | None = None,
        **metadata,
    ) -> Self:
        if not isinstance(next(iter(transition_scores)), TransitionScores):
            transition_scores = [TransitionScores(**ts) for ts in transition_scores]
        return cls(
            {
                "_id": _id or ObjectId(),
                "document": document,
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
        idx = len(ts) // 2
        tsa, tsb = ts[:idx], ts[idx:]
        ma, mb = {}, {}
        for key, value in self["metadata"].items():
            if isinstance(value, (tuple, list)):
                idx = len(value) // 2
                ma[key], mb[key] = value[:idx], value[idx:]
            else:
                ma[key], mb[key] = value, value
        return (
            FeaturesDict(
                {
                    "_id": ObjectId(),
                    "_split": _split + ".0",
                    "document": self["document"],
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
                    "document": self["document"],
                    "model": self["model"],
                    "pre_processor": self["pre_processor"],
                    "transition_scores": tsb,
                    "metadata": mb,
                }
            ),
        )
