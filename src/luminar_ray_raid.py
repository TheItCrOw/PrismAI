import gc
from typing import Iterable

import ray
import torch
from datasets import load_dataset
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import BatchEncoding


class LuminarRaidTokenizer:
    def __init__(
        self,
        feature_dim: int = 256,
        model_name_or_path: str = "gpt2",
        key: str = "generation",
    ):
        self.feature_dim = feature_dim
        self._key = key

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def tokenize(self, batch: list[str], padding: bool = False) -> BatchEncoding:
        return self.tokenizer(
            batch,
            padding=padding,
            truncation=True,
            max_length=self.feature_dim,
            return_length=True,
        )

    def __call__(
        self,
        batch: dict[str, list[str]],
        /,
        key: str | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        encoding = self.tokenize(list(batch[key or self._key]))
        return {
            "id": batch["id"],
            "adv_source_id": batch["adv_source_id"],
            "source_id": batch["source_id"],
            "model": batch["model"],
            "decoding": batch["decoding"],
            "repetition_penalty": batch["repetition_penalty"],
            "attack": batch["attack"],
            "domain": batch["domain"],
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "length": encoding.length,
        }


class LuminarRaidEncoder(LuminarRaidTokenizer):
    def __init__(
        self,
        feature_dim: int = 256,
        model_name_or_path: str = "gpt2",
        device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
        key: str = "generation",
    ):
        super().__init__(
            feature_dim=feature_dim, model_name_or_path=model_name_or_path, key=key
        )

        self.device = torch.device(device)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path
        )
        self.model = self.model.to(self.device)

        if hasattr(self.model, "lm_head"):
            self.model_lm_head: nn.Linear = self.model.lm_head
        elif hasattr(self.model.model, "lm_head"):
            self.model_lm_head: nn.Linear = self.model.model.lm_head
        else:
            raise ValueError("Could not find lm_head in model")

    def __call__(self, batch: dict[str, list[str]]) -> dict[str, list[torch.Tensor]]:
        encoding = {
            "input_ids": batch.pop("input_ids"),
            "attention_mask": batch.pop("attention_mask"),
        }
        return batch | {"features": self.process(encoding)}

    def process(self, batch: dict[str, list[int]]) -> list[torch.Tensor]:
        encoding = self.tokenizer.pad(
            batch,
            max_length=self.feature_dim,
            return_tensors="pt",
        )

        batch_hidden_states = self.forward(encoding.input_ids, encoding.attention_mask)

        intermediate_likelihoods = []
        for input_ids, hidden_states in zip(encoding.input_ids, batch_hidden_states):
            intermediate_likelihoods.append(
                self.compute_intermediate_likelihoods(input_ids, hidden_states)
            )

        return intermediate_likelihoods

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Iterable[tuple[torch.Tensor, ...]]:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                output_hidden_states=True,
            )

            # unpack hidden states to get one list of tensors per input sequence,
            # instead of one hidden state per layer in the model
            hidden_states = zip(*[hs.cpu() for hs in outputs.hidden_states])  # type: ignore

            del outputs
        return hidden_states

    def compute_intermediate_likelihoods(
        self,
        input_ids: torch.Tensor,
        hidden_states: tuple[torch.Tensor],
    ) -> torch.Tensor:
        labels = input_ids[1:].view(-1, 1)

        seq_length = min(labels.ne(self.pad_token_id).sum(), self.feature_dim)
        labels = labels[:seq_length].to(self.device)

        intermediate_likelihoods = []
        with torch.no_grad():
            for hs in hidden_states:
                hs: torch.Tensor = hs[:seq_length].to(self.device)
                il = (
                    # get layer logits
                    self.model_lm_head(hs)
                    # calculate likelihoods
                    .softmax(-1)
                    # get likelihoods of input tokens
                    .gather(-1, labels)
                    .squeeze(-1)
                    .cpu()
                )
                del hs

                # pad with zeros if sequence is shorter than required feature_dim
                if seq_length < self.feature_dim:
                    il = torch.cat([il, torch.zeros(self.feature_dim - seq_length)])

                intermediate_likelihoods.append(il)
        # stack intermediate likelihoods to get tensor of shape (feature_dim, num_layers)
        return torch.stack(intermediate_likelihoods, dim=1)


# ray_ds = ray.data.read_parquet("local:///nvme/projects/PrismAI/PrismAI/data/liamdugan_raid_train")


def load_ray_dataset():
    return ray.data.from_huggingface(load_dataset("liamdugan/raid", split="train"))


ray_ds = load_ray_dataset()
gc.collect()

tiny_ds = ray_ds.limit(2560)
tiny_ds
ds = tiny_ds.map_batches(
    LuminarRaidTokenizer,
    concurrency=20,
    batch_size=1024,
).sort("length")
ds.write_parquet("local:///storage/projects/stoeckel/prismai/raid-gpt2-tokenized")
