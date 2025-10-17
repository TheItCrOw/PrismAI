# Modified from: RAID, Dugan et al. 2024
# > https://github.com/liamdugan/raid/blob/main/detectors/models/binoculars/utils/metrics.py
# > https://github.com/liamdugan/raid/blob/main/detectors/models/binoculars/binoculars.py


from typing import Self

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from luminar.baselines.core import DetectorABC

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(
    encoding: BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0,
):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).masked_fill(
            ~shifted_attention_mask.bool(), float("nan")
        )
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (
            ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
            * shifted_attention_mask
        ).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0,
):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]

    if not temperature:
        p_scores, q_scores = p_logits, q_logits
    else:
        p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (
            ((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()
        )

    return agg_ce


GLOBAL_BINOCULARS_THRESHOLD = (
    0.9015310749276843  # selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
)
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars(DetectorABC):
    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        device_observer: str | torch.device = DEVICE_1,
        device_performer: str | torch.device = DEVICE_2,
    ) -> None:
        super().__init__(AutoTokenizer.from_pretrained(observer_name_or_path))

        self.device_observer = device_observer
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )

        self.device_performer = device_performer
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )

        self.observer_model.eval()
        self.performer_model.eval()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed

        tokenizer = self.tokenizer

        def _tokenize(texts: list[str]) -> BatchEncoding:
            return tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=max_token_observed,
                return_length=True,
                return_token_type_ids=False,
            )

        self.tokenize = _tokenize  # type: ignore

    def tokenize(self):
        pass

    @property
    def device_observer(self) -> torch.device:
        return self._device

    @device_observer.setter
    def device_observer(self, device: str | torch.device):
        self._device = torch.device(device)

    @property
    def device_performer(self) -> torch.device:
        return self._device

    @device_performer.setter
    def device_performer(self, device: str | torch.device):
        self._device = torch.device(device)

    def to(
        self,
        device: str | torch.device | None = None,
        /,
        device_observer: str | torch.device | None = None,
        device_performer: str | torch.device | None = None,
    ) -> Self:
        if not (device or device_observer) or not (device or device_performer):
            raise ValueError(
                "Either `device` or both `device_observer` and `device_performer` have to be specified!"
            )

        self.device_observer = device_observer or device  # type: ignore
        self.device_performer = device_performer or device  # type: ignore

        self.observer_model.to(self.device_observer)  # type: ignore
        self.performer_model.to(self.device_performer)  # type: ignore

        return self

    @torch.inference_mode()
    def _get_logits(
        self, encodings: BatchEncoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(
            **encodings.to(self.observer_model.device)
        ).logits
        performer_logits = self.performer_model(
            **encodings.to(self.performer_model.device)
        ).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    @torch.inference_mode()
    def predict(self, inputs: dict) -> list[float]:
        encodings = self.tokenizer.pad(inputs, return_tensors="pt")
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,  # type: ignore
        )
        binoculars_scores = ppl / x_ppl
        return binoculars_scores.tolist()

    def process(self, inputs: dict) -> dict[str, list[float]]:
        return {
            "y_scores": self.predict(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        }

    @torch.inference_mode()
    def process_texts(self, texts: list[str]) -> list[float]:
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.device)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,  # type: ignore
        )
        binoculars_scores = ppl / x_ppl
        return binoculars_scores.tolist()
