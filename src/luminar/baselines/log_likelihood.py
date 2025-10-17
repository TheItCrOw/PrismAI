import inspect
from abc import abstractmethod
from typing import Self, TypedDict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.tokenization_utils_base import BatchEncoding

from luminar.baselines.core import DetectorABC, PreTrainedModel


class Sample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]


class LogLikelihoodABC(DetectorABC):
    def __init__(
        self,
        model: str,
        batch_size: int = 128,
        max_length: int = 512,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(AutoTokenizer.from_pretrained(model), device=device)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.eval()
        self.pad_token_id: int = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id  # type: ignore
        )
        self.tokenizer.pad_token_id = self.pad_token_id
        self.batch_size = batch_size
        self.max_length = max_length

        tokenizer = self.tokenizer

        def _tokenize(texts: list[str]) -> BatchEncoding:
            # We can just pad to the right (not left), because we do not need to generate anything.
            # Padding left would work too (given correct attention mask and position IDs),
            # but slicing the outputs is a little bit more complicated.
            return tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_length=True,
                return_token_type_ids=False,
            )

        self.tokenize = _tokenize  # type: ignore

    def to(self, device: str | torch.device) -> Self:
        self.device = device
        self.model.to(self.device)  # type: ignore
        return self

    def tokenize(self):
        pass

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @model.setter
    def model(self, model: PreTrainedModel):
        self._model = model
        self._requires_position_ids = "position_ids" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )
        self._model.to(self.device)  # type: ignore

    def process(self, inputs: dict):
        return {
            "y_scores": self.predict(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        }

    @torch.inference_mode()
    def predict(self, inputs: dict) -> list[float]:
        """
        Calculate metrics for the given pre-processed dataset.

        Args:
            dataset (list[Sample]): A sequence of pre-processed documents to be processed.
            pad_token_id (int): The token ID to use for padding.

        Returns:
            list[Metrics]: A list of calculated metrics.
        """
        encoding = self.tokenizer.pad(inputs, return_tensors="pt").to(self.device)
        return self._process_batch(
            encoding.input_ids,
            encoding.attention_mask,
            self.pad_token_id,
        )

    @torch.inference_mode()
    def _process_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_token_id: int,
    ) -> list[float]:
        """
        Process the a batch of input sequences and calculate transition scores.
        Runs a forward pass on the model and extracts the top k probabilities.

        Args:
            input_ids (torch.Tensor): A list of input sequences, each represented as a list of token IDs.
            attention_mask (torch.Tensor): A list of attention masks for each input sequence.
            pad_token_id (int): The token ID that has been used for padding.

        Returns:
            list[float]: A list output probability tuples.
        """
        (
            batch_probabilities,
            batch_log_probabilities,
        ) = self._forward(input_ids, attention_mask)

        results = []
        for (
            target_ids,
            probabilities,
            log_probabilities,
        ) in zip(
            input_ids.to(self.device),
            batch_probabilities,
            batch_log_probabilities,
        ):
            # Truncate the sequence to the last non-pad token
            labels = target_ids[1:].view(-1, 1)
            labels = labels[: labels.ne(pad_token_id).sum()]
            labels = labels.to(log_probabilities.device)

            probabilities: torch.Tensor = probabilities[: labels.size(0)]
            log_probabilities: torch.Tensor = log_probabilities[: labels.size(0)]

            log_likelihood = log_probabilities.gather(-1, labels).squeeze(-1)

            # Get target probabilities and ranks
            _, sorted_indices = torch.sort(probabilities, descending=True)
            _, target_ranks = torch.where(sorted_indices.eq(labels))

            score = self._calculate_score(
                probabilities,
                log_probabilities,
                log_likelihood,
                target_ranks,
                device=self.device,
            )

            results.append(score)

        return results

    @torch.inference_mode()
    def _forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414:L415
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.to(self.device)

        outputs = self._model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            position_ids=position_ids,
        )

        probabilities: torch.Tensor = outputs.logits.softmax(-1)
        log_probabilities: torch.Tensor = outputs.logits.log_softmax(-1)

        return (
            probabilities,
            log_probabilities,
        )

    @abstractmethod
    def _calculate_score(
        self,
        probabilities: torch.Tensor,
        log_probabilities: torch.Tensor,
        log_likelihoods: torch.Tensor,
        target_ranks: torch.Tensor,
        device: torch.device | None = None,
    ) -> float: ...

    @torch.inference_mode()
    def process_texts(self, texts: list[str]) -> list[float]:
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        ).to(self.device)
        return self._process_batch(
            encoding.input_ids,
            encoding.attention_mask,
            self.pad_token_id,
        )
