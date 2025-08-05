# E5 LoRA Detector, indexed in the RAID benchmark leaderboard
# > https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector
# > https://github.com/menglinzhou/microsoft-hackathon-24


import torch

from luminar.baselines.trainable import AutoClassifier


class GPT2Classifier(AutoClassifier):
    def __init__(
        self,
        model_name="gpt2",
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
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _freeze_lm(self):
        for name, param in self.model.named_parameters():
            if name.startswith("transformer"):
                param.requires_grad = False
