from dataclasses import dataclass, field
from typing import Literal

from bson.dbref import DBRef

from transition_scores.data import TransitionScores
from transition_scores.utils import DataClassMappingMixin

# @dataclass
# class SynthetizedObject:
#     synth_chunks: str
#     synth_fulltext: str


# @dataclass
# class SynthetizedText:
#     agent: str


@dataclass
class BaseTransitionScore(DataClassMappingMixin):
    features: list[TransitionScores]
    type: str = field(init=False)

    def __post_init__(self):
        self.features = [dict(scores) for scores in self.features]


@dataclass
class TextTransitionScore(BaseTransitionScore):
    type: str = field(default="text", init=False)


@dataclass
class ChunkTransitionScore(BaseTransitionScore):
    type: str = field(default="chunk", init=False)
    start_idx: int
    end_idx: int


@dataclass
class ChunkInContextTransitionScore(ChunkTransitionScore):
    type: str = field(default="chunk-in-context", init=False)
    context_start_idx: int


@dataclass
class TransitionScoreItem(DataClassMappingMixin):
    ref: DBRef
    model: str
    provider: Literal["onnx", "transformers"]
    transition_scores: BaseTransitionScore
    variant: Literal["default", "8bit", "o1", "o2", "o3", "o4"] = "default"
