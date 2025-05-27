import torch
from torch import nn
from transformers.utils.generic import ModelOutput

from luminar.utils import ConvolutionalLayerSpec, LuminarTrainingConfig


class LuminarCNN(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        config = LuminarTrainingConfig(**kwargs)

        feature_len, feature_depth = config.feature_dim

        self.conv_layers = nn.Sequential()
        in_channels = feature_depth
        for conv in config.conv_layer_shapes:
            conv = ConvolutionalLayerSpec(*conv)
            out_channels = conv.channels
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    conv.kernel_size,  # type: ignore
                    conv.stride,
                    conv.padding,
                ),
            )
            in_channels = out_channels
            self.conv_layers.append(
                nn.LeakyReLU(),
            )

        match config.projection_dim:
            case (c, h, p):
                self.projection = nn.Sequential(
                    nn.Linear(out_channels, c),
                    nn.SiLU(),
                    nn.Linear(c, h),
                    nn.SiLU(),
                    nn.Linear(h, p),
                    nn.SiLU(),
                    nn.Flatten(),
                )
            case (h, p):
                self.projection = nn.Sequential(
                    nn.Linear(out_channels, h),
                    nn.SiLU(),
                    nn.Linear(h, p),
                    nn.SiLU(),
                    nn.Flatten(),
                )
            case p if isinstance(p, int):
                self.projection = nn.Sequential(
                    nn.Linear(out_channels, p), nn.SiLU(), nn.Flatten()
                )
            case None:
                p = 1
                self.projection = nn.Flatten()
            case _:
                raise ValueError(
                    f"projection_dim must be an int or a tuple of (projection_dim, hidden_size), got {config.projection_dim}"
                )

        self.classifier = nn.Linear(feature_len * p, 1)

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
            self.projection(self.conv_layers(features.transpose(1, 2)).transpose(1, 2))
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
