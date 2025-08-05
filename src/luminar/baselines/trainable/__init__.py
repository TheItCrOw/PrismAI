# Modified from: RAID, Dugan et al. 2024
# > https://github.com/liamdugan/raid/blob/main/detectors/models/chatgpt_roberta_detector/chatgpt_detector.py

import traceback
from abc import abstractmethod
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from luminar.baselines.core import DetectorABC

accuracy = evaluate.load("accuracy")


def compute_metrics_acc(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


class AutoClassifier(DetectorABC):
    def __init__(
        self,
        model_name: str,
        tokenizer_name=None,
        freeze_lm: bool = False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            AutoTokenizer.from_pretrained(tokenizer_name or model_name),
            device=device,
        )
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)

        self.freeze_lm = freeze_lm
        if self.freeze_lm:
            self._freeze_lm()

    @property
    def model_name_or_path(self) -> str:
        return self.model.name_or_path  # type: ignore

    @abstractmethod
    def _freeze_lm(self): ...

    def reset(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path
        )
        self.model = self.model.to(self.device)

        if self.freeze_lm:
            self._freeze_lm()

    def tokenize(self, texts: list[str]) -> BatchEncoding:
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=512,
            return_length=True,
        )

    @torch.inference_mode()
    def predict(self, inputs: dict) -> dict[str, list[float]]:
        encoding = self.tokenizer.pad(inputs, return_tensors="pt").to(self.device)
        logits = self.model(**encoding).logits
        return {
            "y_scores": logits.softmax(dim=-1)[:, 1].cpu().flatten().tolist(),
            "y_preds": logits.argmax(dim=-1).cpu().flatten().tolist(),
        }

    def process(self, inputs: dict) -> dict[str, list[float]]:
        self.model.eval()
        return self.predict(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        )

    @torch.inference_mode()
    def process_texts(self, texts: list[str]) -> list[float]:
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self.model(**encoding)
        probs = outputs.logits.softmax(dim=-1)
        return probs[:, 1].detach().cpu().flatten().tolist()

    def train(
        self,
        dataset: DatasetDict,
        training_args: TrainingArguments,
        save_path: str | Path | None = None,
    ):
        tokenizer = self.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            data_collator=data_collator,
            compute_metrics=compute_metrics_acc,  # type: ignore
        )

        trainer.train()
        if save_path:
            try:
                trainer.save_model(str(save_path))
            except Exception:
                traceback.print_exc()

        self.model = trainer.model.to(self.device)  # type: ignore

        del trainer
