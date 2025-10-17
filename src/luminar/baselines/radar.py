# Modified from: RAID, Dugan et al. 2024
# > https://github.com/liamdugan/raid/blob/main/detectors/models/radar/radar.py

from typing import Self

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from luminar.baselines.core import DetectorABC


class Radar(DetectorABC):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(
            AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B"),
            device=device,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "TrustSafeAI/RADAR-Vicuna-7B",
        )
        self.model.eval()
        self.to(device)

    def to(self, device: str | torch.device) -> Self:
        self.device = device
        self.model.to(self.device)
        return self

    def tokenize(self, texts: list[str]) -> BatchEncoding:
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=512,
            return_length=True,
        )

    def _forward(self, encoding: BatchEncoding) -> torch.Tensor:
        outputs = self.model(**encoding)
        # RADAR returns logits in shape (batch_size, 2)
        # where the indices correspond to {0: "AI-generated", 1: "human-authored"}
        logits: torch.Tensor = outputs.logits
        return logits.softmax(-1)[:, 0]

    @torch.inference_mode()
    def predict(self, inputs: dict) -> list[float]:
        encoding = self.tokenizer.pad(inputs, return_tensors="pt").to(self.device)
        return self._forward(encoding).tolist()

    def process(self, inputs: dict) -> dict[str, list[float]]:
        y_scores = self.predict(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        )
        y_preds = (np.array(y_scores) > 0.5).astype(int).tolist()

        return {"y_scores": y_scores, "y_preds": y_preds}

    @torch.inference_mode()
    def process_texts(self, texts: list[str]) -> list[float]:
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        return self._forward(encoding).tolist()
