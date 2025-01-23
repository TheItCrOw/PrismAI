import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Iterable

import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from transition_scores.data import CustomTokenizer, TransitionScores


class TransitionScorerABC(ABC):
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        model: str | Path,
        tokenizer: CustomTokenizer | None = None,
        batch_size: int = 128,
        top_k: int = 100,
        skip_prefix_tokens: int = 0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.batch_size = batch_size
        self.top_k = top_k
        self.skip_prefix_tokens = skip_prefix_tokens
        self._device = torch.device(device)
        self._allocated = False

        self.tokenizer: CustomTokenizer = tokenizer or CustomTokenizer.from_pretrained(
            model
        )
        self._init_model(model)

        self._requires_position_ids = "position_ids" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )
        self._all_special_id_set = set(self.tokenizer.tokenizer.all_special_ids)

        self.pad_token_id = (
            self.tokenizer.tokenizer.pad_token_id
            or self.tokenizer.tokenizer.eos_token_id
        )

    @abstractmethod
    def _init_model(self, model: str | Path): ...

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: str | torch.device):
        if self._allocated:
            self._free()
        self._device = torch.device(value)
        if self._allocated:
            self._allocate()

    def to(self, device: str | torch.device):
        self.device = device
        return self

    def process(
        self,
        sequences: Iterable[str | list[str]] | None = None,
        dataset: Dataset | None = None,
    ) -> Generator[list[TransitionScores], None, None]:
        """
        Calculate transition scores for a batch of sequences.
        Yields one list of LogProbs for each sequence in the batch.
        Each list contains the target token and top-k token
        probabilities for each prediction step.

        Note:
            You must give either `sequences` or `dataset` but not both.

        Args:
            sequences: A batch of text sequences to process.
            dataset: A dataset containing the sequences to process.
                Should have a single column named "text".
            top_k: The number of top-k predictions to return.

        Yields:
            list[LogProbs]: A list of LogProbs for each sequence in the batch.
        """
        if sequences is None and dataset is None:
            raise ValueError("Either sequences or dataset must be provided")
        if sequences is not None and dataset is not None:
            raise ValueError("Only one of sequences or dataset may be provided")

        if sequences is not None:
            dataset = Dataset.from_dict({"text": sequences})

        dataset = self.tokenizer.tokenize_dataset(dataset)

        # sort by input_id length for efficient batching
        dataset = dataset.sort("length").remove_columns("length")

        return dataset.with_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            output_all_columns=True,
        ).map(
            self._process_batch,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["input_ids", "attention_mask"],
        )

    def _process_batch(self, batch: dict[str, list]) -> dict[str, list]:
        input_ids = pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = pad_sequence(
            batch["attention_mask"], batch_first=True, padding_value=0
        )

        # calculate position_ids if required (source: huggingface transformers)
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids.to(self.device),
            )
            logits: torch.Tensor = outputs.logits.softmax(-1).cpu()
            del outputs

        transition_scores = []
        for seq_probs, target_ids in zip(
            logits,
            input_ids,
        ):
            batch_top_k_probs, batch_top_k_indices = seq_probs.topk(self.top_k)

            # calculate length of the sequence and get corresponding target probabilities
            seq_len = target_ids.ne(self.pad_token_id).long().sum() - 1
            target_probs = seq_probs[:seq_len][
                torch.arange(seq_len), target_ids[1 : seq_len + 1]
            ].flatten()

            seq_results = []
            first_token = target_ids[0].item()
            if first_token not in self._all_special_id_set:
                seq_results.append(TransitionScores(first_token, 0.0, [], []))

            # omit the last token, if it is a special token, e.g. <|endoftext|>
            seq_end = (
                seq_len + 1
                if target_ids[seq_len] not in self._all_special_id_set
                else seq_len
            )

            # always skip the first token, as we do not get predictions for it
            for target_idx, target_prob, top_k_indices, top_k_probs in zip(
                target_ids[1:seq_end].tolist(),
                target_probs.tolist(),
                batch_top_k_indices.tolist(),
                batch_top_k_probs.tolist(),
            ):
                seq_results.append(
                    TransitionScores(
                        target_idx,
                        target_prob,
                        top_k_indices,
                        top_k_probs,
                    )
                )
            transition_scores.append(seq_results)
        return {"transition_scores": transition_scores}
