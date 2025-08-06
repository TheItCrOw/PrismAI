import torch
import json
import os

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers.utils.generic import ModelOutput
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from luminar.utils import ConvolutionalLayerSpec, LuminarSequenceTrainingConfig


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
            config : LuminarSequenceTrainingConfig,
    ):
        super().__init__()

        print(config)

        feature_len, feature_depth = config.feature_len, config.num_intermediate_likelihoods

        self.stack_spans = config.stack_spans if hasattr(config, "stack_spans") else 0
        self.apply_delta_augmentation = config.apply_delta_augmentation if hasattr(config, "apply_delta_augmentation") else False
        self.apply_product_augmentation = config.apply_product_augmentation if hasattr(config, "apply_product_augmentation") else False

        # If we apply delta or product augmentation, we double the amount of hidden likelihoods by *2 - 1
        if self.apply_delta_augmentation and self.apply_product_augmentation:
            raise ValueError("apply_delta_augmentation and apply_product_augmentation were both set - can only work with either or None.")
        if config.apply_delta_augmentation or config.apply_product_augmentation:
            feature_depth = feature_depth * 2 - 1

        self.rescale = FeatureRescaler() if config.rescale_features else nn.Identity()

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

        # Feed Forward Layer
        self.feed_forward = nn.Identity()
        ff_output_dim = self.lstm_hidden_dim

        self.classifier = nn.Linear(ff_output_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def _apply_layer_augmentation(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Applies layer-wise delta or product augmentation to a list of [seq_len, num_layers] tensors.

        Returns:
            List of tensors with shape [seq_len, 2*num_layers - 1]
        """
        augmented_features = []

        for feat in features:
            layer_1 = feat[:, 0].unsqueeze(-1)  # [seq_len, 1]
            other_layers = feat[:, 1:]  # [seq_len, num_layers - 1]

            if self.apply_delta_augmentation:
                augmented = other_layers - layer_1  # [seq_len, num_layers - 1]
            elif self.apply_product_augmentation:
                augmented = other_layers * layer_1
            else:
                augmented_features.append(feat)
                continue

            combined = torch.cat([layer_1, other_layers, augmented], dim=-1)
            augmented_features.append(combined)

        return augmented_features

    def forward(
            self, features, sentence_spans, span_labels=None
    ):
        """
        features: (batch, seq_len, feature_dim)
        sentence_spans: List[List[Tuple[start, end]]] per batch item
        span_labels: optional, for loss computation (same shape as sentence_spans)
        """
        batch_sentence_features = []
        batch_lengths = []

        if self.apply_delta_augmentation or self.apply_product_augmentation:
            features = self._apply_layer_augmentation(features)

        for i, spans in enumerate(sentence_spans):
            feature = features[i]  # shape: [seq_len, feat_dim]
            for j, (start, end) in enumerate(spans):
                if self.stack_spans > 0:
                    if j > self.stack_spans - 1:
                        (prev_begin, _) = spans[j - self.stack_spans]
                        start = prev_begin
                    if j < len(spans) - 1 - self.stack_spans - 1:
                        (_, next_end) = spans[j + self.stack_spans]
                        end = next_end
                span_feat = feature[start:end, :]
                batch_sentence_features.append(span_feat)
                batch_lengths.append(end - start)

        # Pad sequences to batch shape (pads to the longest span in the batch)
        # (total_spans, max_len, feature_dim)
        padded = nn.utils.rnn.pad_sequence(batch_sentence_features, batch_first=True)

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

    def save(self, full_path: str):
        """
        Saves the model state_dict and config to the given path.

        Args:
            full_path (str): Directory path to save model weights and config.
        """
        os.makedirs(full_path, exist_ok=True)

        weights_path = os.path.join(full_path, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

        config_dict = self.config.__dict__ if hasattr(self, "config") else {}
        config_path = os.path.join(full_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, full_path: str, device: torch.device = None):
        """
        Loads a LuminarSequence model and its config from the given path.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = os.path.join(full_path, "config.json")
        weights_path = os.path.join(full_path, "pytorch_model.bin")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = LuminarSequenceTrainingConfig(**config_dict)

        model = cls(config)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


class LuminarSequenceAttention(nn.Module):
    """
    WIP!

    A sequence classifier for AIGT tasks, based on the Luminar architecture.
    General architecture:
    1. Rescale features to [-1, 1] if `rescale_features` is True.
    2. Project features to a lower dimension if `projection_dim` is specified.
    3. Apply a series of convolutional layers defined by `conv_layer_shapes`.
    4. Instead of flattening the output, apply Attention on the CCN output.
    5. Apply a feed-forward layer to classify the sequences respectively.
    """

    def __init__(
            self,
            config : LuminarSequenceTrainingConfig,
    ):
        super().__init__()

        feature_len, feature_depth = config.feature_len, config.num_intermediate_likelihoods
        self.stack_spans = getattr(config, "stack_spans", 0)
        self.apply_delta_augmentation = getattr(config, "apply_delta_augmentation", False)
        self.apply_product_augmentation = getattr(config, "apply_product_augmentation", False)

        if self.apply_delta_augmentation and self.apply_product_augmentation:
            raise ValueError("Cannot enable both delta and product augmentation.")

        if self.apply_delta_augmentation or self.apply_product_augmentation:
            feature_depth = feature_depth * 2 - 1

        self.rescale = FeatureRescaler() if config.rescale_features else nn.Identity()

        in_channels = feature_depth
        if config.projection_dim:
            self.projection = nn.Linear(in_channels, config.projection_dim)
            in_channels = config.projection_dim
        else:
            self.projection = nn.Identity()

        # CNN
        self.cnn = nn.Sequential()
        out_channels = -1
        for conv in config.conv_layer_shapes:
            conv = ConvolutionalLayerSpec(*conv)
            out_channels = conv.channels
            self.cnn.append(nn.Conv1d(in_channels, out_channels, conv.kernel_size, conv.stride, conv.padding))
            self.cnn.append(nn.LeakyReLU())
            in_channels = out_channels

        if out_channels == -1:
            raise ValueError("Convolutional layers must be defined.")

        self.attention_dim = out_channels
        self.n_heads = getattr(config, "attention_heads", 4)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.attention_dim))

        # Try: Put an attention head on top for each span classifcation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.n_heads,
            batch_first=True,
        )

        # Self-attention across [CLS] span embeddings
        self.cross_span_attn = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.n_heads,
            batch_first=True,
        )

        self.feed_forward = nn.Identity()
        self.classifier = nn.Linear(self.attention_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def _apply_layer_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """
        Assumes input shape: (batch, seq_len, num_layers)
        Returns: (batch, seq_len, 1 + (num_layers - 1) + (num_layers - 1)) = (batch, seq_len, 2*num_layers - 1)
        """
        layer_1 = features[:, :, 0].unsqueeze(-1)
        other_layers = features[:, :, 1:]

        if self.apply_delta_augmentation:
            augmented = other_layers - layer_1 
        elif self.apply_product_augmentation:
            augmented = other_layers * layer_1
        else:
            return features

        return torch.cat([layer_1, other_layers, augmented], dim=-1)  # (batch, seq_len, 2*num_layers - 1)

    def forward(self, features, sentence_spans, span_labels=None):
            batch_sentence_features = []
            batch_lengths = []

            if self.apply_delta_augmentation or self.apply_product_augmentation:
                features = self._apply_layer_augmentation(features)

            for i, spans in enumerate(sentence_spans):
                feature = features[i]  # shape: [seq_len, feat_dim]
                for j, (start, end) in enumerate(spans):
                    if self.stack_spans > 0:
                        if j > self.stack_spans - 1:
                            (prev_begin, _) = spans[j - self.stack_spans]
                            start = prev_begin
                        if j < len(spans) - 1 - self.stack_spans - 1:
                            (_, next_end) = spans[j + self.stack_spans]
                            end = next_end
                    span_feat = feature[start:end, :]
                    batch_sentence_features.append(span_feat)
                    batch_lengths.append(end - start)

            padded = pad_sequence(batch_sentence_features, batch_first=True)  # (total_spans, max_len, feat_dim)
            attn_mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=padded.device)
            for i, l in enumerate(batch_lengths):
                attn_mask[i, l:] = True

            x = self.rescale(padded)
            x = self.projection(x)
            x = self.cnn(x.transpose(1, 2)).transpose(1, 2)

            # Insert CLS token
            cls_tokens = self.cls_token.expand(x.size(0), 1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            cls_mask = torch.zeros((x.size(0), 1), dtype=torch.bool, device=x.device)
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)

            # Self-attention over tokens in each span
            x, _ = self.self_attention(x, x, x, key_padding_mask=attn_mask)
            # Extract CSL token embeddings
            span_cls_embeddings = x[:, 0, :]

            # Self-attention over [CLS] span embeddings (across all spans)
            span_cls_embeddings_attended, _ = self.cross_span_attn(
                span_cls_embeddings.unsqueeze(0), # q
                span_cls_embeddings.unsqueeze(0), # k
                span_cls_embeddings.unsqueeze(0), #
            )
            span_cls_embeddings_attended = span_cls_embeddings_attended.squeeze(0)

            logits = self.classifier(self.feed_forward(span_cls_embeddings_attended))

            if span_labels is None:
                return ModelOutput(logits=logits)

            flat_labels = torch.cat([torch.tensor(label_seq, dtype=torch.float32) for label_seq in span_labels]).to(
                logits.device)
            loss = self.criterion(logits.view(-1), flat_labels.view(-1))
            return ModelOutput(logits=logits, loss=loss)
    
    def save(self, full_path: str):
        """
        Saves the model state_dict and config to the given path.

        Args:
            full_path (str): Directory path to save model weights and config.
        """
        os.makedirs(full_path, exist_ok=True)

        weights_path = os.path.join(full_path, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

        config_dict = self.config.__dict__ if hasattr(self, "config") else {}
        config_path = os.path.join(full_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, full_path: str, device: torch.device = None):
        """
        Loads a LuminarSequence model and its config from the given path.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = os.path.join(full_path, "config.json")
        weights_path = os.path.join(full_path, "pytorch_model.bin")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = LuminarSequenceTrainingConfig(**config_dict)

        model = cls(config)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model