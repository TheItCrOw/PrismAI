import inspect
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Generator, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from prismai_features.data import (
    FeatureValues,
    ModelMetadata,
)
from prismai_features.utils import PYTORCH_GC_LEVEL, PytorchGcLevel, free_memory
from simple_dataset.dataset import Dataset

type _ModelOutput = dict[str, torch.Tensor | list[torch.Tensor]]


class TransformersModelABC(ABC):
    def __init__(
        self,
        model: str | Path,
        batch_size: int = 128,
        top_k: int = 16,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=False,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = torch.device(device)
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

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self.set_model(model)

    def set_model(self, model):
        self._model = model
        self._model_lm_head: torch.nn.Linear = (
            model.lm_head if hasattr(self.model, "lm_head") else model.model.lm_head
        )
        self._requires_position_ids = "position_ids" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def process(
        self,
        dataset: Dataset | list[dict],
        pad_token_id: int,
    ) -> Dataset:
        """
        Calculate transition scores for the given pre-processed dataset.

        Args:
            dataset (Dataset): A sequence of pre-processed documents to be processed.
            pad_token_id (int): The token ID to use for padding.

        Raises:
            KeyError: If the pre-processor requires a field that is not present in the given dataset.

        Returns:
            Dataset: The processed dataset with a new "transition_scores" field.
        """
        if not isinstance(dataset, Dataset):
            dataset = Dataset(dataset)

        return dataset.update(self._process(dataset, pad_token_id))

    def _process(
        self, dataset: Dataset, pad_token_id: int
    ) -> Generator[dict[str, FeatureValues], None, None]:
        _collate_fn = partial(collate_fn, pad_token_id=pad_token_id)
        for input_ids, attention_mask in tqdm(
            DataLoader(
                dataset,  # type: ignore
                shuffle=False,
                collate_fn=_collate_fn,
                batch_size=self.batch_size,
            ),
            position=2,
            leave=False,
            desc="Processing Sequences",
        ):
            yield from self._process_batch(input_ids, attention_mask, pad_token_id)

            if PYTORCH_GC_LEVEL == PytorchGcLevel.BATCH:
                free_memory()

        if PYTORCH_GC_LEVEL == PytorchGcLevel.DATASET:
            free_memory()

    @abstractmethod
    def _process_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_token_id: int,
    ) -> Generator[dict[str, Any], None, None]: ...


def collate_fn(
    batch: list[dict],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = zip(
        *[(row["input_ids"], row.pop("attention_mask")) for row in batch]
    )
    input_ids = pad_sequence(
        [torch.tensor(seq_ids) for seq_ids in input_ids],
        batch_first=True,
        padding_value=pad_token_id,
    ).long()
    attention_mask = pad_sequence(
        [torch.tensor(mask) for mask in attention_mask],
        batch_first=True,
        padding_value=0,
    ).long()
    return input_ids, attention_mask


class TransformersFeatureModel(TransformersModelABC):
    def _process_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_token_id: int,
    ) -> Generator[dict[str, FeatureValues], None, None]:
        """
        Process the a batch of input sequences and calculate features.
        Runs a forward pass on the model and extracts likelihoods.

        Args:
            input_ids (torch.Tensor): A list of input sequences, each represented as a list of token IDs.
            attention_mask (torch.Tensor): A list of attention masks for each input sequence.
            pad_token_id (int): The token ID that has been used for padding.

        Yields:
            One feature dictionary for each input sequence.
        """
        batch_likelihoods, batch_hidden_states = self._forward(
            input_ids, attention_mask
        )

        for target_ids, likelihoods, hidden_states in zip(
            input_ids.to(self.device), batch_likelihoods, batch_hidden_states
        ):
            # Truncate the sequence to the last non-pad token
            labels: torch.Tensor = target_ids[1:].view(-1, 1)
            labels = labels[: labels.ne(pad_token_id).sum()]
            likelihoods: torch.Tensor = likelihoods[: labels.size(0)]

            # Get target likelihoods and ranks
            target_probs = likelihoods.gather(-1, labels).flatten().cpu()
            target_ranks, top_k_indices, top_k_probs = self._get_ranks_and_top_k(
                likelihoods, labels
            )
            del likelihoods

            # Calculate intermediate likelihoods from hidden states
            intermediate_likelihoods = self._calculate_intermediate_probs(
                hidden_states, labels
            )
            del hidden_states

            yield {
                "features": FeatureValues(
                    target_ids.tolist(),
                    target_probs.tolist(),
                    target_ranks.tolist(),
                    top_k_indices.tolist(),
                    top_k_probs.tolist(),
                    intermediate_likelihoods.tolist(),
                )
            }

            if PYTORCH_GC_LEVEL == PytorchGcLevel.INNER:
                free_memory()

    def _forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, Iterable[tuple[torch.Tensor]]]:
        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.to(self.device)

        with torch.no_grad():
            outputs: _ModelOutput = self._model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids,
                output_hidden_states=True,
            )

            likelihoods: torch.Tensor = outputs.logits.softmax(-1)  # type: ignore

            # Unpack hidden states to get one list of tensors per input sequence,
            # instead of one hidden state per layer in the model
            hidden_states = zip(*[hs.cpu() for hs in outputs.hidden_states])  # type: ignore

            del outputs
        return likelihoods, hidden_states

    def _get_ranks_and_top_k(
        self, likelihood: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sort likelihoods and get ranks of target labels
        sorted_probs, sorted_indices = torch.sort(likelihood, descending=True)
        _, target_ranks = torch.where(sorted_indices.eq(labels))

        # Get top-k probabilities and indices
        top_k_indices = sorted_indices[:, : self.top_k + 1]
        top_k_probs = sorted_probs[:, : self.top_k + 1]

        return target_ranks, top_k_indices, top_k_probs

    def _calculate_intermediate_probs(
        self, intermediate_probs: tuple[torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        seq_length = labels.size(0)

        results = []
        # Calculate likelihoods using intermediate representations
        # We need do this in a loop here to avoid running out of memory
        for probs in intermediate_probs:
            results.append(
                self._model_lm_head(probs[:seq_length].to(self.device))
                .softmax(-1)
                .gather(-1, labels)
                .squeeze(-1)
                .cpu()
            )
            del probs

        # transpose to get shape (seq_length, num_layers)
        return torch.stack(results).T


class TransformersMetricModel(TransformersModelABC):
    def _process_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_token_id: int,
    ) -> Generator[dict[str, list[dict[str, float]]], None, None]:
        """
        Process the a batch of input sequences and calculate transition scores.
        Runs a forward pass on the model and extracts the top k probabilities.

        Args:
            input_ids (torch.Tensor): A list of input sequences, each represented as a list of token IDs.
            attention_mask (torch.Tensor): A list of attention masks for each input sequence.
            pad_token_id (int): The token ID that has been used for padding.

        Yields:
            One metrics dictionary for each input sequence.
        """
        likelihoods, log_likelihoods = self._forward(input_ids, attention_mask)

        for target_ids, likelihood, log_likelihood in zip(
            input_ids.to(self.device), likelihoods, log_likelihoods
        ):
            # Truncate the sequence to the last non-pad token
            labels: torch.Tensor = target_ids[1:].view(-1, 1)
            labels = labels[: labels.ne(pad_token_id).sum()]
            likelihood: torch.Tensor = likelihood[: labels.size(0)]
            log_likelihood: torch.Tensor = log_likelihood[: labels.size(0)]

            # Get DetectLLM-LLR criterion
            target_ranks = self._get_ranks(likelihood, labels)
            target_log_probs = log_likelihood.gather(-1, labels).squeeze(-1)

            llr = self._calculate_log_likelihood_ratio(target_ranks, target_log_probs)

            # Get Fast-DetectGPT criterion
            fast_detect_gpt = self._calculate_fast_detect_gpt(
                likelihood, log_likelihood, target_log_probs
            )

            del likelihood
            del log_likelihood
            del target_log_probs

            yield {
                "metrics": [
                    {
                        "fast_detect_gpt": fast_detect_gpt,
                        "llr": llr,
                    }
                ]
            }

            if PYTORCH_GC_LEVEL == PytorchGcLevel.INNER:
                free_memory()

    def _forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.to(self.device)

        with torch.no_grad():
            outputs: _ModelOutput = self._model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids,
                output_hidden_states=True,
            )

            likelihoods: torch.Tensor = outputs.logits.softmax(-1)  # type: ignore
            log_likelihoods: torch.Tensor = outputs.logits.log_softmax(-1)  # type: ignore

            del outputs

        return likelihoods, log_likelihoods

    def _get_ranks(
        self, likelihood: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Sort likelihoods and get ranks of target labels."""
        sorted_probs, sorted_indices = torch.sort(likelihood, descending=True)
        _, target_ranks = torch.where(sorted_indices.eq(labels))

        return target_ranks

    def _calculate_log_likelihood_ratio(
        self,
        target_ranks: torch.Tensor,
        target_log_probs: torch.Tensor,
        device: torch.device | None = None,
    ) -> float:
        """Calculate the DetectLLM-LLR score (Su et al., 2023)."""
        device = device or self.device
        return (
            -torch.div(
                target_log_probs.to(device).sum(),
                target_ranks.to(device).log1p().sum(),
            )
            .cpu()
            .item()
        )

    def _calculate_fast_detect_gpt(
        self,
        likelihoods: torch.Tensor,
        log_likelihoods: torch.Tensor,
        target_log_likelihoods: torch.Tensor,
        device: torch.device | None = None,
    ) -> float:
        """Calculate the Fast-DetectGPT score (Bao et al., 2023) using the analytical solution."""

        device = device or self.device
        expectation = (likelihoods.to(device) * log_likelihoods.to(device)).sum(-1)
        variance = likelihoods.to(device) * log_likelihoods.to(device).square()
        variance = variance.sum(-1) - expectation.square()

        fast_detect_gpt = (
            target_log_likelihoods.to(device).sum(-1) - expectation.sum(-1)
        ) / variance.sum(-1).sqrt()

        return fast_detect_gpt.cpu().item()
