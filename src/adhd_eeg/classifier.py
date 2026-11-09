"""Classification head and integrated Conformer model for EEG signals.

This module implements the model integration specified in Section 5
of `docs/SPRINT_3_TO_8_EXECUTION_PLAN.md` for Sprint 4.

Architecture & Contract:
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
"""

from __future__ import annotations

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
HEAD_HIDDEN_DIM: int = 64
HEAD_DROPOUT: float = 0.2


_TORCH_MISSING_MSG = (
    "PyTorch is required for EEGConformerClassifier and classifier modules. "
    "Install the optional 'pytorch' extra: pip install .[pytorch]"
)


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

        Integrates the compact EEG Conformer encoder with temporal mean and max
        pooling and a two-layer classification head.

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
        ) -> None:
            super().__init__()
            self.expected_time_steps = expected_time_steps
            self.expected_channels = expected_channels
            self.d_model = d_model
            self.num_classes = num_classes

            if encoder is not None:
                self.encoder = encoder
            else:
                self.encoder = EEGConformerEncoder(
                    expected_time_steps=expected_time_steps,
                    expected_channels=expected_channels,
                    d_model=d_model,
                    dropout=encoder_dropout,
                )

            self.head = EEGClassificationHead(
                in_features=d_model * 2,
                hidden_dim=head_hidden_dim,
                num_classes=num_classes,
                dropout=head_dropout,
            )

        def pool(self, seq: Tensor) -> Tensor:
            """Temporal mean and max pooling concatenated along feature dimension.

            Args:
                seq: Sequence embeddings of shape (B, T, D) where T=97, D=64.

            Returns:
                Tensor of shape (B, 2 * D) = (B, 128).
            """
            mean_pool = seq.mean(dim=1)
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

else:  # pragma: no cover
    EEGClassificationHead = _MissingTorchModule  # type: ignore[misc,assignment]
    EEGConformerClassifier = _MissingTorchModule  # type: ignore[misc,assignment]



__all__ = [
    "CLASS_ORDER",
    "HEAD_DROPOUT",
    "HEAD_HIDDEN_DIM",
    "NUM_CLASSES",
    "POOLING_DIM",
    "EEGClassificationHead",
    "EEGConformerClassifier",
]
