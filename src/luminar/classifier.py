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


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim: int, dropout: float = 0.0, scale: float | None = None
    ):
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout
        self.scale = scale

    def forward(
        self, features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        return F.scaled_dot_product_attention(
            self.query(features),
            self.key(features),
            self.value(features),
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale,
        )


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        return self.linear2(F.silu(self.linear1(x)))


class LuminarAttention(nn.Module):
    """WIP"""
    def __init__(
        self,
        feature_dim: tuple[int, int],
        embed_dim: int = 32,
        ff_dim: int = 32,
        num_layers: int = 1,
        rescale_features: bool = True,
        **kwargs,
    ):
        super().__init__()

        _, feature_depth = feature_dim
        self.projection = nn.Linear(feature_depth, embed_dim)
        self.self_attn = nn.Sequential(
            OrderedDict(
                {
                    f"layer_{i}": nn.Sequential(
                        OrderedDict(
                            {
                                "attn": SelfAttention(embed_dim),
                                "ff": FeedForward(embed_dim, ff_dim),
                            }
                        )
                    )
                    for i in range(num_layers)
                }
            )
        )
        self.classifier = nn.Linear(ff_dim, 1)

        self._rescale_features = rescale_features

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ModelOutput:
        if self._rescale_features:
            features = features.mul(2).sub(1)

        logits = self.classifier(
            self.self_attn(self.projection(features), attention_mask)
        )

        if labels is None:
            return ModelOutput(logits=logits)

        loss = self.criterion(logits.view(-1), labels.float().view(-1))

        return ModelOutput(
            logits=logits,
            loss=loss,
        )
