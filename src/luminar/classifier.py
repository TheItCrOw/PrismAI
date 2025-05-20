from collections import defaultdict
from typing import Final, Iterable, NamedTuple

import torch
from lightning.pytorch import LightningModule
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from transformers import get_linear_schedule_with_warmup
from transformers.utils.generic import ModelOutput


class ConvolutionalLayerSpec(NamedTuple):
    channels: int
    kernel_size: int | tuple[int, int]
    stride: int = 1

    @property
    def kernel_size_1d(self):
        if isinstance(self.kernel_size, int):
            return self.kernel_size
        return self.kernel_size[0]

    @property
    def kernel_size_2d(self):
        if isinstance(self.kernel_size, int):
            return (self.kernel_size, self.kernel_size)
        return self.kernel_size

    @property
    def padding(self) -> int:
        return (self.kernel_size_1d - 1) // 2

    def __repr__(self):
        return repr(tuple(self))


DEFAULT_CONV_LAYER_SHAPES: Final[tuple[ConvolutionalLayerSpec, ...]] = (
    ConvolutionalLayerSpec(64, 5),
    ConvolutionalLayerSpec(128, 3),
    ConvolutionalLayerSpec(128, 3),
    ConvolutionalLayerSpec(128, 3),
    ConvolutionalLayerSpec(64, 3),
)


class LuminarClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: tuple[int, int],
        conv_layer_shapes: Iterable[ConvolutionalLayerSpec] = DEFAULT_CONV_LAYER_SHAPES,
        projection_dim: int = 128,
        **kwargs,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential()
        for conv in conv_layer_shapes:
            conv = ConvolutionalLayerSpec(*conv)
            self.conv_layers.append(
                nn.LazyConv1d(
                    conv.channels,
                    conv.kernel_size,  # type: ignore
                    conv.stride,
                    conv.padding,
                ),
            )
            self.conv_layers.append(
                nn.LeakyReLU(),
            )
        self.conv_layers.append(nn.Flatten())

        if projection_dim:
            self.projection = nn.Sequential(
                nn.LazyLinear(projection_dim), nn.LeakyReLU()
            )
        else:
            self.projection = nn.Identity()

        self.classifier = nn.LazyLinear(1)

        self.criterion = nn.BCEWithLogitsLoss()

        self.forward(
            torch.randn((kwargs.get("batch_size", 1), *feature_dim)),
            labels=torch.randint(0, 2, (kwargs.get("batch_size", 1),)),
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ModelOutput:
        # We are using 2D features (so `features` is a 3D tensor)
        # but we want to treat the second feature dimension as channels.
        # Thus, we need to transpose the tensor here
        logits = self.classifier(
            self.projection(self.conv_layers(features.transpose(1, 2)).flatten(1))
        )

        if labels is None:
            return ModelOutput(
                logits=logits,
            )

        loss = self.criterion(logits.view(-1), labels.float().view(-1))

        return ModelOutput(
            logits=logits,
            loss=loss,
        )


class LuminarLightningClassifier(LightningModule, LuminarClassifier):
    def __init__(
        self,
        feature_dim: tuple[int, int],
        conv_layer_shapes: Iterable[ConvolutionalLayerSpec] = DEFAULT_CONV_LAYER_SHAPES,
        projection_dim: int = 128,
        learning_rate: float = 0.001,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(
            conv_layer_shapes=conv_layer_shapes,
            projection_dim=projection_dim,
        )

        self.save_hyperparameters(
            {
                "feature_dim": feature_dim,
                "conv_layer_shapes": conv_layer_shapes,
                "projection_dim": projection_dim,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                **kwargs,
            }
        )

        self.example_input_array = torch.randn((train_batch_size, *feature_dim))

        self.outputs = defaultdict(list)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,  # type: ignore
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        # We are using 2D features (so `features` is a 3D tensor)
        # but we want to treat the second feature dimension as channels.
        # Thus, we need to transpose the tensor here
        return self.classifier(
            self.projection(self.conv_layers(inputs.transpose(1, 2)).flatten(1))
        )

    @torch.inference_mode()
    def process(self, inputs: dict[str, torch.Tensor]) -> dict[str, list]:
        scores = self.forward(inputs["features"]).view(-1).sigmoid().cpu()
        return {"scores": scores.tolist(), "preds": scores.ge(0.5).int().tolist()}

    def training_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.criterion(logits.view(-1), batch["label"].float().view(-1))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(**batch)
        loss = self.criterion(logits.view(-1), batch["label"].float().view((-1,)))

        scores = logits.view((-1,))

        labels = batch["label"]

        self.outputs[dataloader_idx].append(
            {"loss": loss, "scores": scores, "label": labels}
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):
        scores, labels, loss = self._process_outputs()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", float(accuracy_score(labels, scores > 0.5)))
        self.log("val/f1", float(f1_score(labels, scores > 0.5)), prog_bar=True)

        if sum(labels) == 0 or sum(labels) == len(labels):
            auroc = -1
        else:
            auroc = float(roc_auc_score(labels, scores))
        self.log("val/auroc", auroc)

        threshold = sorted(scores)[len(labels) - sum(labels) - 1]
        self.log(
            "val/acc_calibrated", float(accuracy_score(labels, scores > threshold))
        )
        self.log("val/f1_calibrated", float(f1_score(labels, scores > threshold)))
        self.log("val/threshold_calibrated", threshold)

        self.outputs.clear()

    @torch.inference_mode()
    def _process_outputs(self) -> tuple[NDArray, NDArray, float]:
        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)

        scores = (
            torch.cat([x["scores"] for x in flat_outputs])
            .detach()
            .sigmoid()
            .flatten()
            .float()
            .cpu()
            .numpy()
        )
        labels = torch.cat([x["label"] for x in flat_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean().item()
        return scores, labels, loss
