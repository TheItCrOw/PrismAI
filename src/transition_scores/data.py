from collections import namedtuple
from dataclasses import dataclass

EncodedSequence = namedtuple("EncodedSequence", ["input_ids", "attention_mask"])


@dataclass
class LogProbs:
    target_id: int
    target_prob: float
    top_k_id: list[int]
    top_k_probs: list[float]

    def __iter__(self):
        yield from (self.target_id, self.target_prob, self.top_k_id, self.top_k_probs)
