import inspect
from abc import ABC, abstractmethod
from functools import partial

import multiprocess
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from transition_scores.data import (
    FeaturesDict,
    ModelMetadata,
    OutputProbabilities,
    PreProcessorMetadata,
)
from transition_scores.utils import PYTORCH_GC_LEVEL, PytorchGcLevel, free_memory


class TransitionScorer(ABC):
    def __init__(
        self,
        batch_size: int = 128,
        top_k: int = 16,
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
    def get_metadata(self) -> ModelMetadata: ...

    def process(
        self,
        dataset: list[dict],
        pad_token_id: int,
    ) -> list[OutputProbabilities]:
        """
        Calculate transition scores for the given pre-processed dataset.

        Args:
            dataset (list[dict]): A sequence of pre-processed documents to be processed.
            pad_token_id (int): The token ID to use for padding.

        Raises:
            KeyError: If the pre-processor requires a field that is not present in the given dataset.

        Returns:
            list[OutputProbabilities]: A list of output probabilities for each input document.
        """
        _collate_fn = partial(collate_fn, pad_token_id=pad_token_id)

        probabilities = []
        for input_ids, attention_mask in tqdm(
            DataLoader(
                dataset,
                shuffle=False,
                collate_fn=_collate_fn,
                batch_size=self.batch_size,
                num_workers=multiprocess.cpu_count() // 2,
            ),
            position=2,
            leave=False,
            desc="Processing Sequences",
        ):
            probabilities.extend(self._process_batch(input_ids, attention_mask))

            if PYTORCH_GC_LEVEL == PytorchGcLevel.BATCH:
                free_memory()

        if PYTORCH_GC_LEVEL == PytorchGcLevel.DATASET:
            free_memory()

        return probabilities

    def _process_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[OutputProbabilities]:
        """
        Process the a batch of input sequences and calculate transition scores.
        Runs a forward pass on the model and extracts the top k probabilities.

        Args:
            input_ids (torch.Tensor): A list of input sequences, each represented as a list of token IDs.
            attention_mask (torch.Tensor): A list of attention masks for each input sequence.

        Returns:
            list[OutputProbabilities]: A list output probability tuples.
        """
        outputs = self._forward(input_ids, attention_mask)

        probabilities = []
        for target_ids, seq_probs in zip(input_ids, outputs):
            # Truncate the sequence to the last non-pad token
            seq_len = len(target_ids) - 1
            seq_probs = seq_probs[:seq_len]

            # Get target token and top k probabilities
            target_probs = seq_probs[torch.arange(seq_len), target_ids[1:]].flatten()

            top_k_probs, top_k_indices = seq_probs.topk(self.top_k)

            probabilities.append(
                OutputProbabilities(
                    target_probs.tolist(), top_k_indices.tolist(), top_k_probs.tolist()
                )
            )
        return probabilities

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a model forward pass for a batch of input sequences and return the output probabilities.

        Args:
            input_ids (torch.Tensor): A list of input sequences, each represented as a list of token IDs.
            attention_mask (torch.Tensor): A list of attention masks for each input sequence.

        Returns:
            dict[str, list[torch.Tensor]]: A dictionary containing `target_probs`, `top_k_probs` and `top_k_indices` for each sequence.
        """
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
                .logits.cpu()
                .softmax(-1)
            )


def convert_to_mongo(
    document: dict,
    model_metadata: ModelMetadata,
    pre_processor_metadata: PreProcessorMetadata,
) -> FeaturesDict:
    document_metadata = document.pop("document")
    return FeaturesDict.new(
        document=document_metadata,
        model=model_metadata,
        pre_processor=pre_processor_metadata,
        transition_scores=document.pop("transition_scores"),
        **document,
    )


def collate_fn(
    batch: list[dict],
    pad_token_id: int,
) -> list[dict]:
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
