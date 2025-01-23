from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM

from transition_scores.data import PreProcessor
from transition_scores.scorer.abc import TransitionScorerABC


class OnnxTransitionScorer(TransitionScorerABC):
    def __init__(
        self,
        model: str | Path,
        pre_processor: PreProcessor | None = None,
        batch_size: int = 128,
        top_k: int = 100,
        skip_prefix_tokens: int = 0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        provider: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model,
            pre_processor=pre_processor,
            batch_size=batch_size,
            top_k=top_k,
            skip_prefix_tokens=skip_prefix_tokens,
            device=device,
        )
        self.model = ORTModelForCausalLM.from_pretrained(
            model,
            provider=(
                provider
                or (
                    "CUDAExecutionProvider"
                    if self.device.type == "cuda"
                    else "CPUExecutionProvider"
                )
            ),
            **kwargs,
        )


class TransformersTransitionScorer(TransitionScorerABC):
    def __init__(
        self,
        model: str | Path,
        pre_processor: PreProcessor | None = None,
        batch_size: int = 128,
        top_k: int = 100,
        skip_prefix_tokens: int = 0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=False,
        **kwargs,
    ):
        super().__init__(
            model,
            pre_processor=pre_processor,
            batch_size=batch_size,
            top_k=top_k,
            skip_prefix_tokens=skip_prefix_tokens,
            device=device,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            load_in_8bit=load_in_8bit,
            **kwargs,
        ).to(self.device)


__all__ = ["OnnxTransitionScorer", "TransformersTransitionScorer"]
