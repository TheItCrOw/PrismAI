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
        skip_prefix_tokens: int = 0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.batch_size = batch_size
        self.top_k = top_k
        self.skip_prefix_tokens = skip_prefix_tokens
        self.device = torch.device(device)

        self.pre_processor: PreProcessor = (
            pre_processor or TextPreProcessor.from_pretrained(model)
        )

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self._requires_position_ids = "position_ids" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )

    @property
    def pre_processor(self) -> PreProcessor:
        return self._pre_processor

    @pre_processor.setter
    def pre_processor(self, pre_processor: PreProcessor):
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

        return (
            self._pre_processor.prepare_dataset(dataset)
            .with_format(
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
        )

    def _process_batch(self, batch: dict[str, list]) -> dict[str, list]:
        input_ids = pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=self._pad_token_id
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
            outputs = self._model.forward(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids.to(self.device),
            )
            output_probabilities: torch.Tensor = outputs.logits.softmax(-1).cpu()
            del outputs

        transition_scores = []
        for seq_probs, target_ids in zip(
            output_probabilities,
            batch["input_ids"],
        ):
            # calculate length of the sequence and get corresponding target probabilities
            seq_len = target_ids.ne(self._pad_token_id).long().sum() - 1
            batch_top_k_probs, batch_top_k_indices = seq_probs[:seq_len].topk(
                self.top_k
            )
            target_probs = seq_probs[:seq_len][
                torch.arange(seq_len), target_ids[1 : seq_len + 1]
            ].flatten()

            seq_results = []
            first_token = target_ids[0].item()
            if first_token not in self._all_special_id_set:
                seq_results.append(TransitionScores(first_token, 0.0, [], []))

            # omit the last token, if it is a special token, e.g. <|endoftext|>
            seq_end = -1 if target_ids[-1] not in self._all_special_id_set else None

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
