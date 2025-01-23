import inspect
from abc import ABC
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence

from transition_scores.data import PreProcessor, TextPreProcessor, TransitionScores


class TransitionScorerABC(ABC):
    def __init__(
        self,
        model: str | Path,
        pre_processor: PreProcessor | None = None,
        batch_size: int = 128,
        top_k: int = 100,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = torch.device(device)

        self.pre_processor: PreProcessor = (
            pre_processor or TextPreProcessor.from_pretrained(model)
        )

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self.set_model(model)

    def set_model(self, model):
        self._model = model
        self._requires_position_ids = "position_ids" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )

    @property
    def pre_processor(self) -> PreProcessor:
        return self._pre_processor

    @pre_processor.setter
    def pre_processor(self, pre_processor: PreProcessor):
        self.set_pre_processor(pre_processor)

    def set_pre_processor(self, pre_processor: PreProcessor):
        self._pre_processor = pre_processor
        self._all_special_id_set = set(pre_processor.tokenizer.all_special_ids)
        self._pad_token_id = (
            pre_processor.tokenizer.pad_token_id or pre_processor.tokenizer.eos_token_id
        )

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def process(
        self,
        dataset: Iterable[str] | Dataset,
    ) -> Dataset:
        """
        Calculate transition scores for a batch of sequences.
        Yields one list of LogProbs for each sequence in the batch.
        Each list contains the target token and top-k token
        probabilities for each prediction step.

        Note:
            You must give either `sequences` or `dataset` but not both.

        Args:
            sequences: A batch of text sequences to process.
                May

        Yields:
            list[LogProbs]: A list of LogProbs for each sequence in the batch.
        """
        if not isinstance(dataset, (Dataset, DatasetDict)):
            dataset = Dataset.from_dict({"text": dataset})

        try:
            dataset = self._pre_processor.prepare_dataset(dataset)
        except KeyError as e:
            raise KeyError(
                f"Pre-processor {type(self._pre_processor).__name__} requires the field {e.args[0]} in the dataset."
            ) from e

        return (
            dataset.with_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
                output_all_columns=True,
            )
            .map(
                self._process_batch,
                batched=True,
                batch_size=self.batch_size,
                remove_columns=["input_ids", "attention_mask"],
            )
            .with_format(None)
        )

    def _process_batch(
        self, batch: dict[str, list]
    ) -> dict[str, list[list[TransitionScores]]]:
        input_ids: list[torch.Tensor] = batch["input_ids"]

        output_probs = self._forward(input_ids, batch["attention_mask"])

        transition_scores = []
        for seq_probs, target_ids in zip(output_probs, input_ids):
            # Truncate the sequence to the last non-pad token
            seq_len = target_ids.ne(self._pad_token_id).long().sum() - 1
            seq_probs = seq_probs[:seq_len]

            # Get target token and top k probabilities
            target_probs = seq_probs[
                torch.arange(seq_len), target_ids[1 : seq_len + 1]
            ].flatten()
            batch_top_k_probs, batch_top_k_indices = seq_probs.topk(self.top_k)

            seq_scores = []

            # If this model does not use a BOS token, the first token is not predicted,
            # so we add a dummy result with a zero probability
            if (first_token := target_ids[0].item()) not in self._all_special_id_set:
                seq_scores.append(TransitionScores(first_token, 0.0, [], []))

            # Omit the last token if it is a special token, e.g. <|endoftext|>
            seq_end = -1 if target_ids[-1] in self._all_special_id_set else None
            seq_scores.extend(
                map(
                    TransitionScores.new,
                    zip(
                        # We skip the first token,
                        # as we will not get predictions for it
                        target_ids[1:seq_end],
                        target_probs,
                        batch_top_k_indices,
                        batch_top_k_probs,
                    ),
                )
            )
            transition_scores.append(seq_scores)
        return {"transition_scores": transition_scores}

    def _forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the model forward pass and return the output SoftMax probabilities.

        Returns:
            torch.Tensor: SoftMax probabilities (on CPU).
        """
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self._pad_token_id
        )
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            outputs = self._model.forward(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids.to(self.device),
            )
            return outputs.logits.softmax(-1).cpu()
