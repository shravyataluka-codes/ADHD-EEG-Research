"""Classification head and integrated Conformer model for EEG signals.

This module implements the model integration specified in Section 5
of `docs/SPRINT_3_TO_8_EXECUTION_PLAN.md` for Sprint 4.

Reference Conformer contract:
- Input: float32 tensor of shape (B, 385, 56).
- Encoder: `EEGConformerEncoder` yielding sequence embeddings of shape (B, 97, 64).
- Temporal Pooling: Concatenation of temporal mean pooling (B, 64) and temporal
  max pooling (B, 64) along the feature dimension, yielding (B, 128).
- Classification Head:
    Linear(128, 64) -> GELU -> Dropout(0.2) -> Linear(64, 3)
- Output: Raw logits of shape (B, 3) from `forward()`.
- Fixed Class Order: `CLASS_ORDER = ("HC", "ADD", "ADHD")`.
    Index 0: HC (Healthy Controls)
    Index 1: ADD (Attention Deficit Disorder)
    Index 2: ADHD (Attention Deficit Hyperactivity Disorder)
- Softmax Scores: Provided by `predict_scores(x)`. These scores represent uncalibrated
  softmax activations and must NOT be interpreted as calibrated clinical probabilities
  or diagnostic certainties.
- Metadata and clinical target labels remain outside the model.

The bounded Sprint 7 registry also exposes one-factor Conformer ablations and a
separate EEGNet baseline. `EEGNetConfig` documents its fixed configuration for
the same input dimensions; feature columns remain anonymous and unordered by
any inferred electrode geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]

from .conformer import (
    DEFAULT_DROPOUT,
    EXPECTED_CHANNELS,
    EXPECTED_TIME_STEPS,
    MODEL_DIM,
    EEGConformerEncoder,
    validate_eeg_input,
)

CLASS_ORDER: Tuple[str, ...] = ("HC", "ADD", "ADHD")
NUM_CLASSES: int = len(CLASS_ORDER)
POOLING_DIM: int = MODEL_DIM * 2  # 64 + 64 = 128
MEAN_POOLING_DIM: int = MODEL_DIM
POOLING_MEAN_MAX = "mean_max"
POOLING_MEAN = "mean"
HEAD_HIDDEN_DIM: int = 64
HEAD_DROPOUT: float = 0.2


_TORCH_MISSING_MSG = (
    "PyTorch is required for EEGConformerClassifier and classifier modules. "
    "Install the optional 'pytorch' extra: pip install .[pytorch]"
)


@dataclass(frozen=True)
class EEGNetConfig:
    """Fixed EEGNet baseline configuration for float32 input shaped (B, 385, 56).

    The architecture uses temporal Conv2D (8 filters, kernel 64), depthwise
    feature-axis filtering (depth multiplier 2), separable temporal Conv2D
    (kernel 16, 16 filters), average-pooling factors 4 and 8, ELU activations,
    batch normalization, and dropout 0.25. Anonymous feature columns are treated
    as a fixed feature axis; this configuration implies no electrode locations.
    """

    expected_time_steps: int = EXPECTED_TIME_STEPS
    expected_channels: int = EXPECTED_CHANNELS
    temporal_filters: int = 8
    depth_multiplier: int = 2
    separable_filters: int = 16
    temporal_kernel_size: int = 64
    separable_kernel_size: int = 16
    first_pool_size: int = 4
    second_pool_size: int = 8
    dropout: float = 0.25

    def validate(self) -> None:
        if (
            isinstance(self.expected_time_steps, bool)
            or not isinstance(self.expected_time_steps, int)
            or self.expected_time_steps != EXPECTED_TIME_STEPS
        ):
            raise ValueError(
                f"EEGNet expected_time_steps must be {EXPECTED_TIME_STEPS}"
            )
        if (
            isinstance(self.expected_channels, bool)
            or not isinstance(self.expected_channels, int)
            or self.expected_channels != EXPECTED_CHANNELS
        ):
            raise ValueError(f"EEGNet expected_channels must be {EXPECTED_CHANNELS}")
        for field in (
            "temporal_filters",
            "depth_multiplier",
            "separable_filters",
            "temporal_kernel_size",
            "separable_kernel_size",
            "first_pool_size",
            "second_pool_size",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be in [0, 1)")
        pooled_steps = (
            self.expected_time_steps
            // self.first_pool_size
            // self.second_pool_size
        )
        if pooled_steps <= 0:
            raise ValueError("pooling factors must leave at least one time step")


class _MissingTorchModule:
    """Fallback placeholder when PyTorch is not installed."""

    def __init__(self, *args, **kwargs) -> None:
        raise ImportError(_TORCH_MISSING_MSG)


if torch is not None:

    class EEGClassificationHead(nn.Module):
        """Classification head mapping pooled representation (B, 128) to logits (B, 3).

        Layers:
        - Linear(128, 64, bias=True)
        - GELU()
        - Dropout(p=0.2)
        - Linear(64, 3, bias=True)
        """

        def __init__(
            self,
            in_features: int = POOLING_DIM,
            hidden_dim: int = HEAD_HIDDEN_DIM,
            num_classes: int = NUM_CLASSES,
            dropout: float = HEAD_DROPOUT,
        ) -> None:
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_dim, bias=True)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(hidden_dim, num_classes, bias=True)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x

    class EEGConformerClassifier(nn.Module):
        """Primary candidate EEG Conformer classifier.

        Integrates the compact EEG Conformer encoder with registered temporal
        pooling and a two-layer classification head. Defaults preserve the
        reference mean-plus-max architecture.

        Contract:
        - Input: Tensor of shape (B, 385, 56), dtype float32.
        - Encoder: Sequence embeddings of shape (B, 97, 64).
        - Temporal Pooling: Concatenated mean and max pooling yielding (B, 128).
        - Classification Head: Linear(128, 64) -> GELU -> Dropout(0.2) -> Linear(64, 3).
        - Output: Raw logits of shape (B, 3) mapped to CLASS_ORDER ("HC", "ADD", "ADHD").
        """

        def __init__(
            self,
            encoder: Optional[EEGConformerEncoder] = None,
            expected_time_steps: int = EXPECTED_TIME_STEPS,
            expected_channels: int = EXPECTED_CHANNELS,
            d_model: int = MODEL_DIM,
            num_classes: int = NUM_CLASSES,
            head_hidden_dim: int = HEAD_HIDDEN_DIM,
            head_dropout: float = HEAD_DROPOUT,
            encoder_dropout: float = DEFAULT_DROPOUT,
            use_self_attention: bool = True,
            use_conv_module: bool = True,
            pooling_mode: str = POOLING_MEAN_MAX,
        ) -> None:
            super().__init__()
            if pooling_mode not in (POOLING_MEAN_MAX, POOLING_MEAN):
                raise ValueError(
                    f"pooling_mode must be {POOLING_MEAN_MAX!r} or {POOLING_MEAN!r}"
                )
            self.expected_time_steps = expected_time_steps
            self.expected_channels = expected_channels
            self.d_model = d_model
            self.num_classes = num_classes
            self.pooling_mode = pooling_mode

            if encoder is not None:
                self.encoder = encoder
            else:
                self.encoder = EEGConformerEncoder(
                    expected_time_steps=expected_time_steps,
                    expected_channels=expected_channels,
                    d_model=d_model,
                    dropout=encoder_dropout,
                    use_self_attention=use_self_attention,
                    use_conv_module=use_conv_module,
                )

            self.head = EEGClassificationHead(
                in_features=d_model * (2 if pooling_mode == POOLING_MEAN_MAX else 1),
                hidden_dim=head_hidden_dim,
                num_classes=num_classes,
                dropout=head_dropout,
            )

        def pool(self, seq: Tensor) -> Tensor:
            """Apply this model's registered temporal pooling operation.

            Args:
                seq: Sequence embeddings of shape (B, T, D) where T=97, D=64.

            Returns:
                Tensor of shape (B, 128) for mean+max or (B, 64) for mean-only.
            """
            mean_pool = seq.mean(dim=1)
            if self.pooling_mode == POOLING_MEAN:
                return mean_pool
            max_pool = torch.amax(seq, dim=1)
            return torch.cat([mean_pool, max_pool], dim=-1)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass mapping raw EEG trial signals to class logits.

            Args:
                x: Tensor of shape (B, 385, 56), dtype float32.

            Returns:
                Tensor of shape (B, 3), dtype float32, representing raw logits.
            """
            validate_eeg_input(x)
            seq = self.encoder(x)
            pooled = self.pool(seq)
            logits = self.head(pooled)
            return logits

        def predict_scores(self, x: Tensor) -> Tensor:
            """Compute softmax class scores from raw model logits.

            Important:
                These scores represent normalized softmax activations over the fixed
                class order ('HC', 'ADD', 'ADHD'). They are uncalibrated model outputs
                and must NOT be interpreted as calibrated clinical probabilities or
                diagnostic certainties.

                This method preserves the current model mode (training or eval) and
                does not silently alter `self.training`. The caller is responsible
                for explicitly setting `model.eval()` and `torch.no_grad()` during
                evaluation if desired.

            Args:
                x: Input tensor of shape (B, 385, 56).

            Returns:
                Tensor of shape (B, 3) containing softmax scores summing to 1.0 along dim=-1.
            """
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    class EEGNetBaseline(nn.Module):
        """Bounded EEGNet comparator compatible with the project's (B, 385, 56) input.

        EEGNet-style stages: temporal convolution, depthwise filtering across the
        56 anonymous columns, temporal separable convolution, and pooled dense
        three-logit output in ``CLASS_ORDER``. With the default configuration,
        pooling yields 12 time positions, flattened to 16 * 12 = 192 features
        for ``Linear(192, 3)``. No metadata is accepted.
        """

        def __init__(
            self,
            config: EEGNetConfig = EEGNetConfig(),
        ) -> None:
            super().__init__()
            config.validate()
            self.config = config
            self.num_classes = NUM_CLASSES
            temporal_before = (config.temporal_kernel_size - 1) // 2
            temporal_after = config.temporal_kernel_size // 2
            separable_before = (config.separable_kernel_size - 1) // 2
            separable_after = config.separable_kernel_size // 2
            depthwise_filters = config.temporal_filters * config.depth_multiplier
            self.features = nn.Sequential(
                nn.ZeroPad2d((0, 0, temporal_before, temporal_after)),
                nn.Conv2d(
                    1,
                    config.temporal_filters,
                    kernel_size=(config.temporal_kernel_size, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(config.temporal_filters),
                nn.Conv2d(
                    config.temporal_filters,
                    depthwise_filters,
                    kernel_size=(1, config.expected_channels),
                    groups=config.temporal_filters,
                    bias=False,
                ),
                nn.BatchNorm2d(depthwise_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(config.first_pool_size, 1)),
                nn.Dropout(config.dropout),
                nn.ZeroPad2d((0, 0, separable_before, separable_after)),
                nn.Conv2d(
                    depthwise_filters,
                    depthwise_filters,
                    kernel_size=(config.separable_kernel_size, 1),
                    groups=depthwise_filters,
                    bias=False,
                ),
                nn.Conv2d(depthwise_filters, config.separable_filters, kernel_size=1),
                nn.BatchNorm2d(config.separable_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(config.second_pool_size, 1)),
                nn.Dropout(config.dropout),
            )
            final_time_steps = (
                config.expected_time_steps
                // config.first_pool_size
                // config.second_pool_size
            )
            self.head = nn.Linear(
                config.separable_filters * final_time_steps, NUM_CLASSES
            )

        def forward(self, x: Tensor) -> Tensor:
            validate_eeg_input(x)
            features = self.features(x.unsqueeze(1)).flatten(start_dim=1)
            return self.head(features)

        def predict_scores(self, x: Tensor) -> Tensor:
            """Return uncalibrated softmax scores in ``CLASS_ORDER``."""
            return torch.softmax(self.forward(x), dim=-1)

else:  # pragma: no cover
    EEGClassificationHead = _MissingTorchModule  # type: ignore[misc,assignment]
    EEGConformerClassifier = _MissingTorchModule  # type: ignore[misc,assignment]
    EEGNetBaseline = _MissingTorchModule  # type: ignore[misc,assignment]



__all__ = [
    "CLASS_ORDER",
    "EEGNetConfig",
    "EEGNetBaseline",
    "HEAD_DROPOUT",
    "HEAD_HIDDEN_DIM",
    "MEAN_POOLING_DIM",
    "NUM_CLASSES",
    "POOLING_DIM",
    "POOLING_MEAN",
    "POOLING_MEAN_MAX",
    "EEGClassificationHead",
    "EEGConformerClassifier",
]
