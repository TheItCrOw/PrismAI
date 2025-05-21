from typing import Final, Iterable, NamedTuple

import torch
from torch import nn
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


class LuminarCNN(nn.Module):
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
