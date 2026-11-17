"""Compact raw-EEG neural architectures for the Cell 1 experiment.

All models accept raw epochs shaped [batch, channels, samples] and return
unnormalized class logits. ``forward_features`` exposes the learned raw-EEG
representation for future feature-fusion experiments.
"""

import math

import torch
from torch import nn


class _EEGClassifier(nn.Module):
    def __init__(self, in_channels, n_samples, num_classes):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.in_channels = in_channels
        self.n_samples = n_samples
        self.num_classes = num_classes

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Model input must be a torch.Tensor.")
        if x.ndim != 3:
            raise ValueError(
                "Model input must have shape [batch, channels, samples]; "
                f"got {tuple(x.shape)}."
            )
        if x.shape[0] < 1:
            raise ValueError("Model input batch must contain at least one epoch.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel slots, "
                f"got {x.shape[1]}."
            )
        if x.shape[2] != self.n_samples:
            raise ValueError(
                f"Expected {self.n_samples} samples per epoch, got {x.shape[2]}."
            )
        if not torch.is_floating_point(x):
            raise TypeError("Model input must use a floating-point tensor dtype.")

    def forward_features(self, x):
        """Return the learned representation before the classification head."""
        raise NotImplementedError

    def forward(self, x):
        self._validate_input(x)
        return self.classifier(self.forward_features(x))


class EEGNet(_EEGClassifier):
    """Small EEGNet-inspired temporal, depthwise-spatial, separable CNN."""

    def __init__(
        self,
        in_channels=2,
        n_samples=1024,
        num_classes=2,
        temporal_filters=8,
        depth_multiplier=2,
        separable_filters=16,
        temporal_kernel_size=63,
        dropout=0.25,
    ):
        super().__init__(in_channels, n_samples, num_classes)
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer.")
        if min(temporal_filters, depth_multiplier, separable_filters) < 1:
            raise ValueError("EEGNet filter counts must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        spatial_filters = temporal_filters * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, temporal_kernel_size),
                padding=(0, temporal_kernel_size // 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
        )
        self.depthwise_spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(in_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 15),
                padding=(0, 7),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.feature_dim = separable_filters * 8
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        x = self._prepare_input(x)
        x = self.temporal(x)
        x = self.depthwise_spatial(x)
        x = self.separable(x)
        return x.flatten(start_dim=1)

    def _prepare_input(self, x):
        self._validate_input(x)
        return x.unsqueeze(1)


class CNNLSTM(_EEGClassifier):
    """Temporal CNN followed by a compact LSTM over the pooled sequence."""

    def __init__(
        self,
        in_channels=2,
        n_samples=1024,
        num_classes=2,
        conv_channels=(16, 32),
        conv_kernel_sizes=(7, 5),
        hidden_size=24,
        bidirectional=True,
        projection_dim=32,
        dropout=0.25,
    ):
        super().__init__(in_channels, n_samples, num_classes)
        if len(conv_channels) != 2 or len(conv_kernel_sizes) != 2:
            raise ValueError("CNN-LSTM requires exactly two CNN stages.")
        if min(*conv_channels, *conv_kernel_sizes, hidden_size, projection_dim) < 1:
            raise ValueError("CNN-LSTM dimensions must be positive.")
        if any(kernel % 2 == 0 for kernel in conv_kernel_sizes):
            raise ValueError("CNN-LSTM convolution kernels must be odd.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        first_channels, second_channels = conv_channels
        first_kernel, second_kernel = conv_kernel_sizes
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels,
                first_channels,
                kernel_size=first_kernel,
                padding=first_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm1d(first_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(
                first_channels,
                second_channels,
                kernel_size=second_kernel,
                padding=second_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm1d(second_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=second_channels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        recurrent_dim = hidden_size * (2 if bidirectional else 1)
        self.projection = nn.Sequential(
            nn.Linear(recurrent_dim, projection_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.feature_dim = projection_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        self._validate_input(x)
        sequence = self.temporal_cnn(x).transpose(1, 2)
        sequence, _ = self.lstm(sequence)
        pooled = sequence.mean(dim=1)
        return self.projection(pooled)


class _SelfAttentionBlock(nn.Module):
    """Pre-normalized self-attention and feed-forward residual block."""

    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)
        normalized = self.feedforward_norm(tokens)
        return tokens + self.feedforward_dropout(self.feedforward(normalized))


def _conv_output_length(length, kernel_size, stride, padding):
    return math.floor((length + 2 * padding - kernel_size) / stride) + 1


class EEGTransformer(_EEGClassifier):
    """Compact convolutional token embedding with Transformer encoder blocks."""

    def __init__(
        self,
        in_channels=2,
        n_samples=1024,
        num_classes=2,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        token_kernel_size=16,
        token_stride=8,
        dropout=0.2,
    ):
        super().__init__(in_channels, n_samples, num_classes)
        if min(embed_dim, num_heads, num_layers, feedforward_dim) < 1:
            raise ValueError("Transformer dimensions and layer count must be positive.")
        if token_kernel_size < 1 or token_stride < 1:
            raise ValueError("Token kernel size and stride must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")

        token_padding = token_kernel_size // 2
        token_count = _conv_output_length(
            n_samples,
            token_kernel_size,
            token_stride,
            token_padding,
        )
        if token_count < 1:
            raise ValueError("Token embedding produces no temporal tokens.")
        self.token_embedding = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=token_kernel_size,
            stride=token_stride,
            padding=token_padding,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            *(
                _SelfAttentionBlock(
                    embed_dim,
                    num_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(num_layers)
            )
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.feature_dim = embed_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        self._validate_input(x)
        tokens = self.token_embedding(x).transpose(1, 2)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        tokens = self.output_norm(self.encoder(tokens))
        return tokens.max(dim=1).values


class EEGConformer(_EEGClassifier):
    """Compact temporal/spatial convolutional embedding plus attention blocks."""

    def __init__(
        self,
        in_channels=2,
        n_samples=1024,
        num_classes=2,
        temporal_filters=8,
        temporal_kernel_size=25,
        spatial_filters=16,
        pool_size=8,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        feedforward_dim=64,
        dropout=0.2,
    ):
        super().__init__(in_channels, n_samples, num_classes)
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer.")
        if min(
            temporal_filters,
            spatial_filters,
            pool_size,
            embed_dim,
            num_heads,
            num_layers,
            feedforward_dim,
        ) < 1:
            raise ValueError("Conformer dimensions and pool size must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if n_samples < pool_size:
            raise ValueError("pool_size cannot exceed n_samples.")

        token_count = n_samples // pool_size
        self.temporal_spatial_embedding = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, temporal_kernel_size),
                padding=(0, temporal_kernel_size // 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(in_channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_size)),
            nn.Dropout(dropout),
        )
        self.token_projection = nn.Linear(spatial_filters, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            *(
                _SelfAttentionBlock(
                    embed_dim,
                    num_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(num_layers)
            )
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.feature_dim = embed_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        self._validate_input(x)
        tokens = self.temporal_spatial_embedding(x.unsqueeze(1))
        tokens = tokens.squeeze(2).transpose(1, 2)
        tokens = self.token_projection(tokens) + self.position_embedding
        tokens = self.embedding_dropout(tokens)
        tokens = self.output_norm(self.encoder(tokens))
        return tokens.mean(dim=1)


MODEL_REGISTRY = {
    "eegnet": EEGNet,
    "cnn_lstm": CNNLSTM,
    "eeg_transformer": EEGTransformer,
    "eeg_conformer": EEGConformer,
}


def build_model(name, **kwargs):
    """Construct a registered architecture by its configuration key."""
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model {name!r}; available models: {available}."
        ) from error
    return model_class(**kwargs)
