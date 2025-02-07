from collections import defaultdict
from typing import NamedTuple

import evaluate
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from sklearn.metrics import auc, roc_curve
from transformers import get_linear_schedule_with_warmup

from luminar.features import OneDimFeatures, ThreeDimFeatures, TwoDimFeatures


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


class DocumentClassficationModel(LightningModule):
    def __init__(
        self,
        feature_size: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures,
        conv_layer_shapes: list[ConvolutionalLayerSpec] = None,
        projection_dim: int = 128,
        learning_rate: float = 0.001,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        conv_layer_shapes = conv_layer_shapes or (
            # Default CNN architecture from SeqXGPT paper (Wang et al., EMNLP 2023)
            # https://github.com/Jihuai-wpy/SeqXGPT/blob/da2f264160c12a95c9573c4820fefaa75a68a0f9/SeqXGPT/SeqXGPT/model.py#L52
            [ConvolutionalLayerSpec(64, 5, 1)]
            + [ConvolutionalLayerSpec(128, 3, 1)] * 3
            + [ConvolutionalLayerSpec(64, 3, 1)]
        )

        self.example_input_array = torch.randn((1, *feature_size))
        self.save_hyperparameters()

        self.conv_layers = nn.ModuleList()
        for conv in conv_layer_shapes:
            match feature_size:
                case OneDimFeatures(_):
                    self.conv_layers.append(
                        nn.Sequential(
                            nn.LazyConv1d(
                                conv.channels,
                                conv.kernel_size_1d,
                                conv.stride,
                                conv.padding,
                            ),
                            nn.LeakyReLU(),
                        )
                    )
                case TwoDimFeatures(_, _):
                    self.conv_layers.append(
                        nn.Sequential(
                            nn.LazyConv2d(
                                conv.channels,
                                conv.kernel_size_2d,
                                conv.stride,
                                conv.padding,
                            ),
                            nn.LeakyReLU(),
                        )
                    )
                case ThreeDimFeatures(_, _, _):
                    self.conv_layers.append(
                        nn.Sequential(
                            nn.LazyConv2d(
                                conv.channels,
                                conv.kernel_size_2d,
                                conv.stride,
                                conv.padding,
                            ),
                            nn.LeakyReLU(),
                        )
                    )

        self.classifier = nn.Sequential(
            nn.LazyLinear(projection_dim), nn.LeakyReLU(), nn.LazyLinear(1)
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self.metrics = {
            metric: evaluate.load(metric) for metric in ("precision", "recall", "f1")
        }
        self.outputs = defaultdict(list)

    def forward(self, features: torch.Tensor, **_):
        if features.ndim == 2:
            features = features.unsqueeze(1)

        for layer in self.conv_layers:
            features = layer(features)

        return self.classifier(features.flatten(1))

    def training_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.criterion(logits.view(-1), batch["labels"].float().view(-1))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(**batch)
        val_loss = self.criterion(logits.view(-1), batch["labels"].float().view(-1))

        preds = logits.squeeze()

        labels = batch["labels"]

        self.outputs[dataloader_idx].append(
            {"loss": val_loss, "preds": preds, "labels": labels}
        )

    def on_validation_epoch_end(self):
        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)

        preds = (
            torch.cat([x["preds"] for x in flat_outputs])
            .detach()
            .sigmoid()
            .flatten()
            .float()
            .cpu()
            .numpy()
        )
        labels = torch.cat([x["labels"] for x in flat_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean()

        self.log("val_loss", loss, prog_bar=True)

        fpr, tpr, _ = roc_curve(labels, preds)
        self.log("roc_auc", auc(fpr, tpr), prog_bar=True)

        preds = (preds > 0.5).astype(float)
        self.log("acc", (preds == labels).sum() / len(preds), prog_bar=True)
        self.outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
