from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset, Dataset
from torch import nn
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from luminar.utils import get_matched_datasets


class LuminarEncoder:
    def __init__(
        self,
        max_length: int = 256,
        model_name_or_path: str = "gpt2",
        device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.max_length = max_length
        self.device = torch.device(device)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

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
        return {"features": self.process(batch["text"])}

    def process(self, batch: list[str]) -> list[torch.Tensor]:
        encoding = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_tensors="pt",
        )
        batch_hidden_states = self.forward(
            encoding.input_ids, encoding.attention_mask)

        intermediate_likelihoods = []
        for input_ids, length, hidden_states in zip(encoding.input_ids, encoding.length, batch_hidden_states):
            intermediate_likelihoods.append(
                self.compute_intermediate_likelihoods(
                    input_ids, hidden_states)[:length]
            )

        return intermediate_likelihoods

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Iterable[tuple[torch.Tensor, ...]]:
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )

        # unpack hidden states to get one list of tensors per input sequence,
        # instead of one hidden state per layer in the model
        return zip(*outputs.hidden_states)  # type: ignore

    @torch.inference_mode()
    def compute_intermediate_likelihoods(
        self,
        input_ids: torch.Tensor,
        hidden_states: tuple[torch.Tensor],
    ) -> torch.Tensor:
        labels = input_ids[1:].view(-1, 1)

        seq_length = min(labels.ne(self.pad_token_id).sum(), self.max_length)
        labels = labels[:seq_length].to(self.device)

        intermediate_likelihoods = []
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

            hs.cpu()
            del hs

            # pad with zeros if sequence is shorter than required max_length
            if seq_length < self.max_length:
                il = torch.cat([il, torch.zeros(self.max_length - seq_length)])

            intermediate_likelihoods.append(il)

        # stack intermediate likelihoods to get tensor of shape (max_length, num_layers)
        return torch.stack(intermediate_likelihoods, dim=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    HF_TOKEN = (Path.home() / ".hf_token").read_text().strip()

    agent = "gpt_4o_mini"
    other_agents = "gemma2_9b"
    datasets = {}
    num_proc = 32
    for domain in tqdm(
        [
            "blog_authorship_corpus",
            "student_essays",
            "cnn_news",
            "euro_court_cases",
            "house_of_commons",
            "arxiv_papers",
            "gutenberg_en",
            "bundestag",
            "spiegel_articles",
            # "gutenberg_de",
            "en",
            "de",
        ]
    ):
        datset_config_name = f"{domain}-fulltext"
        dataset_split_name = f"human+{agent}+{other_agents}"
        dataset: Dataset = (
            load_dataset(
                "liberi-luminaris/PrismAI",
                datset_config_name,
                split=dataset_split_name,
                token=HF_TOKEN,
            )  # type: ignore
            .rename_column("label", "labels")
            .filter(
                lambda text: len(text.strip()) > 0,
                input_columns=["text"],
                num_proc=num_proc,
            )
        )
        datasets_matched, dataset_unmatched = get_matched_datasets(
            dataset, agent, num_proc=num_proc
        )
        datasets_matched["unmatched"] = dataset_unmatched
        datasets[domain] = datasets_matched
    del dataset

    model_name = args.model.split("/")[-1].lower().replace("-", "_")

    encoder = LuminarEncoder(
        512, model_name_or_path=args.model, device="cuda:0")
    tq = tqdm(datasets.items(), desc="Encoding datasets")
    agent_str = '_'.join([agent, other_agents])
    base_path = (
        "/storage/projects/stoeckel/prismai/encoded/fulltext/"
        + f"{agent_str}/{model_name}_{encoder.max_length}"
    )
    for config, dataset in tq:
        dataset: Dataset
        tq.set_description_str(f"Encoding {config}")

        path = f"{base_path}/{config}/"
        Path(path).mkdir(parents=True, exist_ok=True)

        dataset.map(encoder, batched=True, batch_size=128).save_to_disk(path)
