from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM

from transition_scores.data import ModelMetadata
from transition_scores.scorer.abc import TransitionScorer


class OnnxTransitionScorer(TransitionScorer):
    def __init__(
        self,
        model: str | Path,
        batch_size: int = 128,
        top_k: int = 16,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        provider: str | None = None,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            top_k=top_k,
            device=device,
        )

        self.model_name_or_path = self.model_name = model  # type: ignore
        if Path(model).exists():
            self.model_name: str = Path(model).stem

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

    def get_metadata(self) -> ModelMetadata:
        match str(self.model_name_or_path).strip("/").split("_"):
            case [*_, variant] if variant.startswith("o") and variant[1:].isdigit():
                return ModelMetadata.new(
                    name=self.model_name,
                    provider="onnx",
                    variant=variant,  # type: ignore
                )
            case _:
                return ModelMetadata.new(
                    name=self.model_name,
                    provider="onnx",
                    variant="default",
                )


class TransformersTransitionScorer(TransitionScorer):
    def __init__(
        self,
        model: str | Path,
        batch_size: int = 128,
        top_k: int = 16,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=False,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            top_k=top_k,
            device=device,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            load_in_8bit=load_in_8bit,
            **kwargs,
        )
        if not load_in_8bit:
            self.model = self.model.to(self.device)
        self._load_in_8bit = load_in_8bit

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata.new(
            name=self._model.name_or_path,
            provider="transformers",
            variant="8bit" if self._load_in_8bit else "default",
        )


__all__ = ["OnnxTransitionScorer", "TransformersTransitionScorer"]
