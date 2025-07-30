from typing import override

import torch

from luminar.baselines.trainable import AutoClassifier


class RoBERTaClassifier(AutoClassifier):
    def __init__(
        self,
        model_name="roberta-base",
        tokenizer_name=None,
        freeze_lm: bool = False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            model_name,
            tokenizer_name=tokenizer_name,
            freeze_lm=freeze_lm,
            device=device,
        )

    @override
    def _freeze_lm(self):
        for name, param in self.model.named_parameters():
            if name.startswith("roberta"):
                param.requires_grad = False
