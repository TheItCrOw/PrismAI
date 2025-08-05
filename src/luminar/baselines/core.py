# Modified from: RAID, Dugan et al. 2024
# > https://github.com/liamdugan/raid/blob/main/detectors/models/chatgpt_roberta_detector/chatgpt_detector.py
from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from luminar.utils import run_evaluation


class PredictionResults(TypedDict):
    y_scores: list[float]
    y_preds: list[int] | None


class DetectorABC(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        device: str | torch.device = ("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def tokenize(self, texts: list[str]) -> BatchEncoding: ...

    @abstractmethod
    def process(self, inputs: dict) -> PredictionResults: ...


def run_detector(
    detector: DetectorABC,
    datasets: dict[str, DatasetDict],
    batch_size=32,
    threshold: float = 0.5,
    sigmoid: bool = False,
    less_than: bool = False,
):
    scores = {}
    for domain, dataset in tqdm(datasets.items(), desc="Predicting on Datasets"):
        test_dataset: Dataset = dataset["test"].map(
            detector.tokenize,
            input_columns=["text"],
            batched=True,
            batch_size=1024,
            desc="Tokenizing",
        )
        test_dataset = test_dataset.sort("length")

        labels = []
        y_scores = []
        y_preds = []
        for batch in tqdm(  # type: ignore
            test_dataset.batch(batch_size),
            desc=f"Processing {domain}",
            position=1,
        ):
            batch: dict[str, list]
            labels.extend(batch["labels"])
            preds = detector.process(batch)
            y_scores.extend(preds["y_scores"])
            if (yp := preds.get("y_preds")) is not None:
                y_preds.extend(yp)

        scores[domain] = run_evaluation(
            np.array(labels),
            np.array(y_scores),
            threshold=threshold,
            sigmoid=sigmoid,
            less_than=less_than,
            y_preds=np.array(y_preds) if y_preds else None,
        )
    return scores


def run_detector_tokenized(
    detector: DetectorABC,
    datasets: dict[str, DatasetDict],
    batch_size=32,
    threshold: float = 0.5,
    sigmoid: bool = False,
    less_than: bool = False,
):
    scores = {}
    for domain, dataset in tqdm(datasets.items(), desc="Predicting on Datasets"):
        labels = []
        y_scores = []
        y_preds = []
        for batch in tqdm(  # type: ignore
            dataset["test"].batch(batch_size),
            desc=f"Processing {domain}",
            position=1,
        ):
            batch: dict[str, list]
            labels.extend(batch["labels"])
            preds = detector.process(batch)
            y_scores.extend(preds["y_scores"])
            if (yp := preds.get("y_preds")) is not None:
                y_preds.extend(yp)

        scores[domain] = run_evaluation(
            np.array(labels),
            np.array(y_scores),
            threshold=threshold,
            sigmoid=sigmoid,
            less_than=less_than,
            y_preds=np.array(y_preds) if y_preds else None,
        )
    return scores
