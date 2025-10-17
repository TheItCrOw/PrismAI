import inspect

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from luminar.baselines.log_likelihood import LogLikelihoodABC, PreTrainedModel


class FastDetectGPT(LogLikelihoodABC):
    """Fast-DetectGPT analytic criterion using the **same model** for both reference and scoring.

    Source:
        - Paper: https://arxiv.org/abs/2310.05130
        - GitHub: https://github.com/baoguangsheng/fast-detect-gpt
        - Implementation: https://github.com/baoguangsheng/fast-detect-gpt/blob/b3f689ec886f9d7438ddead7c4f06fef029a7d0c/scripts/fast_detect_gpt.py#L52:L70
    """

    @torch.inference_mode()
    def _calculate_score(
        self,
        probabilities: torch.Tensor,
        log_probabilities: torch.Tensor,
        log_likelihoods: torch.Tensor,
        _target_ranks: torch.Tensor,
        device: torch.device | None = None,
    ) -> float:
        """Fast-DetectGPT analytic criterion implementation.

        Args:
            probabilities (torch.Tensor): Probabilities for each target token.
            log_probabilities (torch.Tensor): Log-probabilities for each target token.
            log_likelihoods (torch.Tensor): Log-likelihoods for each target token.
            device (torch.device | None, optional): Device to use for computation.
                Defaults to None.

        Returns:
            float: The calculated Fast-DetectGPT score.
        """
        # Here, we use the notation from the paper, instead of the implementation (where variables are named `probs_ref`, `mean_ref`, `var_ref`, etc.).
        device = device or self.device

        expectation = (probabilities.to(device) * log_probabilities.to(device)).sum(-1)
        variance = (
            probabilities.to(device) * log_probabilities.to(device).square()
        ).sum(-1) - expectation.square()

        fast_detect_gpt = (
            log_likelihoods.to(device).sum(-1) - expectation.sum(-1)
        ) / variance.sum(-1).sqrt()

        return fast_detect_gpt.cpu().item()


DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class FastDetectGPTwithScoring(FastDetectGPT):
    """Fast-DetectGPT analytic criterion using **different models** for reference and scoring.

    Source:
        - Paper: https://arxiv.org/abs/2310.05130
        - GitHub: https://github.com/baoguangsheng/fast-detect-gpt
        - Implementation:
            - https://github.com/baoguangsheng/fast-detect-gpt/blob/b3f689ec886f9d7438ddead7c4f06fef029a7d0c/scripts/fast_detect_gpt.py#L52:L70
            - https://github.com/baoguangsheng/fast-detect-gpt/blob/b3f689ec886f9d7438ddead7c4f06fef029a7d0c/scripts/fast_detect_gpt.py#L107:L109
    """

    def __init__(
        self,
        reference_model: str,
        scoring_model: str,
        batch_size: int = 128,
        max_length: int = 512,
        device_1: str | torch.device = DEVICE_1,
        device_2: str | torch.device = DEVICE_2,
    ):
        vocab_s = set(AutoTokenizer.from_pretrained(scoring_model).get_vocab().keys())
        vocab_r = set(AutoTokenizer.from_pretrained(reference_model).get_vocab().keys())
        # Check if:
        assert (
            # both vocabularies are equal or
            vocab_r == vocab_s
            # either vocabulary is a superset of the other (-> empty difference)
            or not vocab_r.difference(vocab_s)  # -> reference vocabulary is superset
            or not vocab_s.difference(vocab_r)  # -> scoring vocabulary is superset
        ), (
            "The tokenizer vocabularies of the reference model and scoring model must match!"
        )
        super().__init__(reference_model, batch_size, max_length, device_1)
        self.device_2 = torch.device(device_2 or self.device)
        self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model)
        self.scoring_model.eval()
        self.vocab_size = min(len(vocab_s), len(vocab_r))

    @property
    def scoring_model(self) -> PreTrainedModel:
        return self._scoring_model

    @scoring_model.setter
    def scoring_model(self, model: PreTrainedModel):
        self._scoring_model = model
        self._scoring_requires_position_ids = "position_ids" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )
        self._scoring_model.to(self.device_2)  # type: ignore

    def to(
        self, device: str | torch.device, device_2: str | torch.device | None = None
    ):
        device_2 = device_2 or device
        self.device = torch.device(device)
        self.device_2 = torch.device(device_2)
        self.model.to(self.device)  # type: ignore
        self.scoring_model.to(self.device_2)  # type: ignore
        return self

    @torch.inference_mode()
    def _forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414:L415
        position_ids = None
        if self._requires_position_ids or self._scoring_requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids

        reference_probs: torch.Tensor = (
            self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids is not None and position_ids.to(self.device),
            )
            .logits[:, :, : self.vocab_size]
            .softmax(-1)
        )

        scoring_log_probs: torch.Tensor = (
            self.scoring_model(
                input_ids=input_ids.to(self.device_2),
                attention_mask=attention_mask.to(self.device_2),
                position_ids=(
                    position_ids is not None and position_ids.to(self.device_2)
                ),
            )
            .logits[:, :, : self.vocab_size]
            .log_softmax(-1)
        )

        return (
            reference_probs.to(self.device_2),
            scoring_log_probs.to(self.device_2),
        )
