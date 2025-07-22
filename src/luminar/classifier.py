from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from transformers.utils.generic import ModelOutput

from luminar.utils import ConvolutionalLayerSpec, LuminarTrainingConfig


class FeatureRescaler(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Rescale features from range [0, 1] to the range [-1, 1].
        """
        return features.mul(2).sub(1)


class LuminarCNN(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        config = LuminarTrainingConfig(**kwargs)

        feature_len, feature_depth = config.feature_dim

        if config.rescale_features:
            self.rescale = FeatureRescaler()
        else:
            self.rescale = nn.Identity()

        in_channels: int = feature_depth
        if config.projection_dim:
            self.projection = nn.Linear(in_channels, config.projection_dim)
            in_channels = config.projection_dim
        else:
            self.projection = nn.Identity()

        self.cnn = nn.Sequential()
        for conv in config.conv_layer_shapes:
            conv = ConvolutionalLayerSpec(*conv)
            out_channels = conv.channels
            self.cnn.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    conv.kernel_size,  # type: ignore
                    conv.stride,
                    conv.padding,
                ),
            )
            in_channels = out_channels
            self.cnn.append(
                nn.LeakyReLU(),
            )

        ff_input_dim = out_channels
        match config.feed_forward_dim:
            case None | 0:
                ff_output_dim: int = feature_len
                self.feed_forward = nn.Flatten()
            case dim if isinstance(dim, int):
                self.feed_forward = nn.Sequential(
                    nn.Linear(ff_input_dim, dim), nn.SiLU(), nn.Flatten()
                )
                ff_output_dim = dim * feature_len
            case tup if isinstance(tup, (tuple, list)):
                modules = []
                for dim in tup:
                    modules.append(nn.Linear(ff_input_dim, dim))
                    modules.append(nn.SiLU())
                    ff_input_dim = dim
                modules.append(nn.Flatten())
                self.feed_forward = nn.Sequential(*modules)
                ff_output_dim = dim * feature_len  # type: ignore
            case _:
                raise ValueError(
                    f"projection_dim must be None, an int or a tuple of ints, got {config.feed_forward_dim}"
                )

        self.classifier = nn.Linear(ff_output_dim, 1)

        self.criterion = nn.BCEWithLogitsLoss()

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
            self.feed_forward(
                self.cnn(
                    self.projection(self.rescale(features)).transpose(1, 2)
                ).transpose(1, 2)
            )
        )

        if labels is None:
            return ModelOutput(logits=logits)

        loss = self.criterion(logits.view(-1), labels.float().view(-1))

        return ModelOutput(logits=logits, loss=loss)
            )

        loss = self.criterion(logits.view(-1), labels.float().view(-1))

        return ModelOutput(
            logits=logits,
            loss=loss,
        )
