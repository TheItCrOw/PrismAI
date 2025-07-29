import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.utils.generic import ModelOutput
from src.luminar.utils import ConvolutionalLayerSpec, LuminarTrainingConfig

class FeatureRescaler(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Rescale features from range [0, 1] to the range [-1, 1].
        """
        return features.mul(2).sub(1)

class LuminarSequence(nn.Module):
    """
    A sequence classifier for AIGT tasks, based on the Luminar architecture.
    General architecture:
    1. Rescale features to [-1, 1] if `rescale_features` is True.
    2. Project features to a lower dimension if `projection_dim` is specified.
    3. Apply a series of convolutional layers defined by `conv_layer_shapes`.
    4. Instead of flattening the output, apply a LSTM layer on the CCN output.
    5. Apply a feed-forward layer to classify the sequences respectively.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        config = LuminarTrainingConfig(**kwargs)
        print(config)
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
        out_channels: int = -1

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

        if out_channels == -1:
            raise ValueError("No convolutional layers defined, out_channels cannot be None.")

        # Instead of flattening the output, we apply a LSTM layer
        self.lstm_hidden_dim = config.lstm_hidden_dim if hasattr(config, "lstm_hidden_dim") else 128
        self.lstm_layers = config.lstm_layers if hasattr(config, "lstm_layers") else 1
        self.lstm = nn.LSTM(
            input_size=out_channels,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Feed Forward Layer (optional, just Identity)
        self.feed_forward = nn.Identity()
        ff_output_dim = self.lstm_hidden_dim

        self.classifier = nn.Linear(ff_output_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, features, sentence_spans, span_labels=None):
        """
        features: (batch, seq_len, feature_dim)
        sentence_spans: List[List[Tuple[start, end]]] per batch item
        span_labels: optional, for loss computation (same shape as sentence_spans)
        """
        batch_sentence_features = []
        batch_lengths = []

        for i, spans in enumerate(sentence_spans):
            for (start, end) in spans:
                span_feat = features[i, start:end, :]
                batch_sentence_features.append(span_feat)
                batch_lengths.append(end - start)

        # Pad sequences to batch shape (pads to the longest span in the batch)
        # (total_spans, max_len, feature_dim)
        padded = nn.utils.rnn.pad_sequence(batch_sentence_features,
                                           batch_first=True)

        x = self.rescale(padded)
        x = self.projection(x)
        x = x.transpose(1, 2)  # CNN expects (batch, channels, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # LSTM expects (batch, seq_len, channels)

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch.nn.utils.rnn.pad_packed_sequence
        # LSTM should ignore the padding, so we pack the padded sequence
        packed = pack_padded_sequence(x, batch_lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)

        logits = self.classifier(self.feed_forward(h_n[-1]))  # One logit per span

        if span_labels is None:
            return ModelOutput(logits=logits)

        # flatten labels to match logits
        flat_labels = torch.cat([torch.tensor(label_seq, dtype=torch.float32) for label_seq in span_labels]).to(
            logits.device)

        loss = self.criterion(logits.view(-1), flat_labels.view(-1))
        return ModelOutput(logits=logits, loss=loss)
