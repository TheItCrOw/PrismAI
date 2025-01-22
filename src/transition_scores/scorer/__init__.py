from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM

from transition_scores.scorer.abc import TransitionScorerABC


class OnnxTransitionScorer(TransitionScorerABC):
    def _init_model(self, model: str | Path):
        self.model = ORTModelForCausalLM.from_pretrained(
            model,
            provider=(
                "CUDAExecutionProvider"
                if self.device.type == "cuda"
                else "CPUExecutionProvider"
            ),
        )


class TransformersTransitionScorer(TransitionScorerABC):
    def __init__(self, *args, load_in_8bit=False, **kwargs):
        self._load_in_8bit = load_in_8bit
        super().__init__(*args, **kwargs)

    def _init_model(self, model: str | Path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            load_in_8bit=self._load_in_8bit,
        ).to(self.device)


__all__ = ["OnnxTransitionScorer", "TransformersTransitionScorer"]
