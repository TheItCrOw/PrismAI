import inspect
from abc import ABC, abstractmethod
from typing import Iterable

import multiprocess
import torch
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence

from transition_scores.data import (
    FeaturesDict,
    ModelMetadata,
    PreProcessorMetadata,
    ScoresDict,
    TransitionScores,
)
from transition_scores.pre_processor.abc import PreProcessor


class TransitionScorerABC(ABC):
    def __init__(
        self,
        batch_size: int = 128,
        top_k: int = 100,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = torch.device(device)

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

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @abstractmethod
    def get_model_metadata(self) -> ModelMetadata: ...

    def process(
        self,
        dataset: Dataset | Iterable[str],
        pre_processor: PreProcessor,
    ) -> Dataset:
        """
        Calculate transition scores for the given dataset.
        Will use the currently set `pre_processor` to prepare the dataset.

        Note:
            - Use `set_pre_processor` to change the pre-processor.
            - Some pre-processors may require specific fields in the dataset (like `chunks`) &ndash; refer to the pre-processor's documentation.

        Args:
            dataset (Dataset | Iterable[str]): Either a dataset or a sequence of texts to be processed.

        Raises:
            KeyError: If the pre-processor requires a field that is not present in the given dataset.

        Returns:
            Dataset: A new dataset with the calculated transition scores.
                All rows contain the original sequence `_id`, the `transition_scores` and additional information that depends on the pre-processor.
        """
        if not isinstance(dataset, (Dataset, DatasetDict)):
            dataset = Dataset.from_dict({"text": dataset})

        try:
            all_special_ids = set(pre_processor.tokenizer.all_special_ids)
            pad_token_id = (
                pre_processor.tokenizer.pad_token_id
                or pre_processor.tokenizer.eos_token_id
            )
            dataset = pre_processor.prepare_dataset(dataset)
        except KeyError as e:
            raise KeyError(
                f"PreProcessor {type(pre_processor).__name__} requires the field {e.args[0]} in the dataset."
            ) from e

        metadata = {
            "model_metadata": self.get_model_metadata(),
            "pre_processor_metadata": pre_processor.get_metadata(),
        }

        # dataset.set_format(
        #     type="torch",
        #     columns=["input_ids", "attention_mask"],
        #     output_all_columns=True,
        # )
        dataset = dataset.map(
            self._process_batch,
            batched=True,
            batch_size=self.batch_size,
            fn_kwargs={"pad_token_id": pad_token_id},
            input_columns=["input_ids", "attention_mask"],
            remove_columns=["attention_mask"],
            desc="Scorer: Processing Sequences",
        )
        dataset = dataset.map(
            process_probabilities,
            batched=True,
            desc="Scorer: Processing Probabilities",
            fn_kwargs={"all_special_ids": all_special_ids},
            remove_columns=[
                "input_ids",
                "target_probs",
                "top_k_probs",
                "top_k_indices",
            ],
            num_proc=multiprocess.cpu_count() // 2,
        )
        # dataset.set_format(None)

        return dataset.map(
            convert_to_mongo,
            desc="Scorer: Convert to Mongo Format",
            fn_kwargs=metadata,
            remove_columns=dataset.column_names,
            num_proc=multiprocess.cpu_count() // 2,
        )

    def _process_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        *,
        pad_token_id: int = None,
    ) -> dict[str, list[torch.Tensor]]:
        """
        Process the a batch of input sequences and calculate transition scores.
        Runs a forward pass on the model and extracts the top k probabilities.

        Args:
            input_ids (list[list[int]]): A list of input sequences, each represented as a list of token IDs.
            attention_mask (list[list[int]]): A list of attention masks for each input sequence.
            pad_token_id (int, kwarg): The ID of the padding token.

        Returns:
            dict[str, list[list[TransitionScores]]]: A list of transition scores
                for each sequence in the batch.
        """
        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided.")

        outputs = self._forward(input_ids, attention_mask, pad_token_id)

        probs = {
            "target_probs": [],
            "top_k_probs": [],
            "top_k_indices": [],
        }
        for target_ids, seq_probs in zip(input_ids, outputs):
            # Truncate the sequence to the last non-pad token
            seq_len = len(target_ids) - 1
            seq_probs = seq_probs[:seq_len]

            # Get target token and top k probabilities
            target_probs = seq_probs[torch.arange(seq_len), target_ids[1:]].flatten()

            top_k_probs, top_k_indices = seq_probs.topk(self.top_k)

            probs["target_probs"].append(target_probs)
            probs["top_k_probs"].append(top_k_probs)
            probs["top_k_indices"].append(top_k_indices)
        return probs

    def _forward(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        Run a model forward pass for a batch of input sequences and return the output probabilities.

        Args:
            input_ids (list[list[int]]): A list of input sequences, each represented as a list of token IDs.
            attention_mask (list[list[int]]): A list of attention masks for each input sequence.

        Returns:
            dict[str, list[torch.Tensor]]: A dictionary containing `target_probs`, `top_k_probs` and `top_k_indices` for each sequence.
        """
        input_ids = pad_sequence(
            [torch.tensor(seq_ids) for seq_ids in input_ids],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(mask) for mask in attention_mask],
            batch_first=True,
            padding_value=0,
        )

        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            return (
                self._model.forward(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    position_ids=position_ids.to(self.device),
                )
                .logits.softmax(-1)
                .cpu()
            )


def convert_to_mongo(
    row: dict,
    model_metadata: ModelMetadata,
    pre_processor_metadata: PreProcessorMetadata,
) -> FeaturesDict:
    return FeaturesDict.from_scores(
        ScoresDict.new(**row),
        model_metadata=model_metadata,
        pre_processor_metadata=pre_processor_metadata,
    )


def process_probabilities(
    batch: dict[str, list[list[int | float]]],
    *,
    all_special_ids: set[int] = None,
) -> dict[str, list]:
    all_special_ids = all_special_ids or set()

    transition_scores = []
    for target_ids, target_probs, top_k_probs, top_k_indices in zip(
        batch["input_ids"],
        batch["target_probs"],
        batch["top_k_probs"],
        batch["top_k_indices"],
    ):
        seq_scores = []

        # If this model does not use a BOS token, the first token is not predicted,
        # so we add a dummy result with a zero probability
        if (first_token := target_ids[0]) not in all_special_ids:
            seq_scores.append(TransitionScores.new(first_token, 0.0, [], []))

        # Omit the last token if it is a special token, e.g. <|endoftext|>
        seq_end = -1 if target_ids[-1] in all_special_ids else None
        seq_scores.extend(
            map(
                TransitionScores.from_tuple,
                zip(
                    # We skip the first token,
                    # as we will not get predictions for it
                    target_ids[1:seq_end],
                    target_probs,
                    top_k_indices,
                    top_k_probs,
                ),
            )
        )
        transition_scores.append(seq_scores)
    return {"transition_scores": transition_scores}
