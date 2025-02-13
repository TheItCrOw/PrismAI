from collections import defaultdict
from typing import NamedTuple

import evaluate
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
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

    def __repr__(self):
        return repr(tuple(self))


class DocumentClassficationModel(LightningModule):
    def __init__(
        self,
        feature_dim: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures,
        conv_layer_shapes: list[ConvolutionalLayerSpec] = None,
        projection_dim: int = 128,
        learning_rate: float = 0.001,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        second_dim_as_channels: bool = False,
        **kwargs,
    ):
        super().__init__()
        match feature_dim:
            case (_,):
                feature_dim = OneDimFeatures(*feature_dim)
            case (_, _):
                feature_dim = TwoDimFeatures(*feature_dim)
            case _:
                raise ValueError(f"Invalid feature_dim {feature_dim}")

        self.save_hyperparameters()

        conv_layer_shapes = conv_layer_shapes or (
            # Default CNN architecture from SeqXGPT paper (Wang et al., EMNLP 2023)
            # https://github.com/Jihuai-wpy/SeqXGPT/blob/da2f264160c12a95c9573c4820fefaa75a68a0f9/SeqXGPT/SeqXGPT/model.py#L52
            [ConvolutionalLayerSpec(64, 5, 1)]
            + [ConvolutionalLayerSpec(128, 3, 1)] * 3
            + [ConvolutionalLayerSpec(64, 3, 1)]
        )

        self.example_input_array = torch.randn((train_batch_size, *feature_dim))

        self.conv_layers = nn.Sequential()
        for conv in conv_layer_shapes:
            conv = ConvolutionalLayerSpec(*conv)
            match feature_dim:
                case (_,):
                    self.conv_layers.append(
                        nn.LazyConv1d(
                            conv.channels,
                            conv.kernel_size,
                            conv.stride,
                            conv.padding,
                        ),
                    )
                    self.conv_layers.append(
                        nn.LeakyReLU(),
                    )
                case (_, _) if second_dim_as_channels:
                    self.conv_layers.append(
                        nn.LazyConv1d(
                            conv.channels,
                            conv.kernel_size,
                            conv.stride,
                            conv.padding,
                        ),
                    )
                    self.conv_layers.append(
                        nn.LeakyReLU(),
                    )
                case (_, _) if not second_dim_as_channels:
                    self.conv_layers.append(
                        nn.LazyConv2d(
                            conv.channels,
                            conv.kernel_size,
                            conv.stride,
                            conv.padding,
                        ),
                    )
                    self.conv_layers.append(
                        nn.LeakyReLU(),
                    )
                # case ThreeDimFeatures(_, _, _):
                #     self.conv_layers.append(
                #         nn.LazyConv2d(
                #             conv.channels,
                #             conv.kernel_size_2d,
                #             conv.stride,
                #             conv.padding,
                #         ),
                #     )
                #     self.conv_layers.append(
                #         nn.LeakyReLU(),
                #     )
                case _:
                    raise RuntimeError("Unsupported feature size")
        self.conv_layers.append(nn.Flatten())

        if projection_dim:
            self.projection = nn.Sequential(
                nn.LazyLinear(projection_dim), nn.LeakyReLU()
            )
        else:
            self.projection = nn.Identity()

        self.classifier = nn.LazyLinear(1)

        self.criterion = nn.BCEWithLogitsLoss()

        self.metrics = {
            metric: evaluate.load(metric) for metric in ("precision", "recall", "f1")
        }
        self.best_threshold = 0.5
        self.outputs = defaultdict(list)

    def forward(self, features: torch.Tensor, **_):
        if self.hparams.second_dim_as_channels:
            # We are using 2D features (so `features` is a 3D tensor)
            # but we want to treat the second feature dimension as channels.
            # Thus, we need to transpose the tensor here
            features = features.transpose(1, 2)
        else:
            # Otherwise, we assume single channel input
            # and add the channel dimension here
            features = features.unsqueeze(1)

        for layer in self.conv_layers:
            features = layer(features)

        return self.classifier(self.projection(features.flatten(1)))

    def training_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.criterion(logits.view(-1), batch["labels"].float().view(-1))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(**batch)
        val_loss = self.criterion(logits.view(-1), batch["labels"].float().view((-1,)))

        preds = logits.view((-1,))

        labels = batch["labels"]

        self.outputs[dataloader_idx].append(
            {"loss": val_loss, "preds": preds, "labels": labels}
        )

    def on_validation_epoch_end(self):
        preds, labels, loss = self._process_outputs()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc@0.5", np.mean(((preds > 0.5) == labels)), prog_bar=True)

        thresholds = np.linspace(0, 1, 10001)
        preds_thresholded: np.ndarray = preds > thresholds.reshape(-1, 1)
        acc_thresholded = np.mean((preds_thresholded == labels), axis=1)
        self.best_threshold = thresholds[np.argmax(acc_thresholded)]

        self.log(
            "val_acc@best",
            np.mean((preds > self.best_threshold) == labels),
            prog_bar=True,
        )
        self.log("threshold", self.best_threshold, prog_bar=True)

        fpr, tpr, _ = roc_curve(labels, preds)
        self.log("val_roc_auc", auc(fpr, tpr), prog_bar=True)
        self.outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(**batch)
        test_loss = self.criterion(logits.view(-1), batch["labels"].float().view((-1,)))

        preds = logits.view((-1,))

        labels = batch["labels"]

        self.outputs[dataloader_idx].append(
            {"loss": test_loss, "preds": preds, "labels": labels}
        )

    def on_test_epoch_end(self):
        preds, labels, loss = self._process_outputs()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc@0.5", np.mean((preds > 0.5) == labels), prog_bar=True)

        self.log(
            "test_acc@best",
            np.mean((preds > self.best_threshold) == labels),
            prog_bar=True,
        )

        fpr, tpr, _ = roc_curve(labels, preds)
        self.log("test_roc_auc", auc(fpr, tpr), prog_bar=True)
        self.outputs.clear()

    def _process_outputs(self):
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
        return preds, labels, loss

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
