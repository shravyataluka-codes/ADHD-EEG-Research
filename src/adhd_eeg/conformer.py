"""Temporal Conv2D tokenizer and compact Conformer encoder for EEG signals.

This module implements the primary architecture candidate specified in Section 4
of `docs/SPRINT_3_TO_8_EXECUTION_PLAN.md` for Sprint 3.

Contract:
- Input: float32 tensor of shape (B, 385, 56).
- Output: float32 sequence embeddings of shape (B, 97, 64).
- The 56 columns are treated as anonymous, fixed-order channel features. No scalp
  geometry or electrode labels are inferred.
- All 385 input time positions are preserved through explicit padding (7) and stride (4)
  in the temporal Conv2D tokenizer, yielding exactly 97 time tokens.
- Metadata, labels, and clinical classifications remain outside the model.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]


# Architectural Constants (Section 4 Specification)
EXPECTED_TIME_STEPS: int = 385
EXPECTED_CHANNELS: int = 56
TOKEN_TIME_STEPS: int = 97
MODEL_DIM: int = 64
NUM_HEADS: int = 4
FFN_DIM: int = 128
CONV_KERNEL_SIZE: int = 15
NUM_BLOCKS: int = 2
DEFAULT_DROPOUT: float = 0.1
NUM_TEMPORAL_FILTERS: int = 16
TEMPORAL_KERNEL_SIZE: int = 15
TEMPORAL_STRIDE: int = 4
TEMPORAL_PADDING: int = 7


def validate_eeg_input(x: object) -> None:
    """Validate EEG input tensor against the fixed contract: float32 (B, 385, 56).

    Raises:
        TypeError: If x is not a torch.Tensor or has a non-float32 dtype.
        ValueError: If x does not have rank 3, has an empty batch dimension (B <= 0),
            has incorrect sample/channel dimensions, or contains non-finite values (NaN/Inf).
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required for validate_eeg_input. "
            "Install the optional 'pytorch' extra (e.g. pip install .[pytorch])."
        )

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(x).__name__}")

    if x.dtype != torch.float32:
        raise TypeError(f"Expected input dtype torch.float32, got {x.dtype}")

    if x.ndim != 3:
        raise ValueError(
            f"Expected 3D input tensor of shape (B, {EXPECTED_TIME_STEPS}, {EXPECTED_CHANNELS}), "
            f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
        )

    b, t, c = x.shape
    if b <= 0:
        raise ValueError(f"Batch dimension must be positive (B >= 1), got B={b}")

    if t != EXPECTED_TIME_STEPS:
        raise ValueError(
            f"Expected time dimension of {EXPECTED_TIME_STEPS}, got {t}"
        )

    if c != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected channel dimension of {EXPECTED_CHANNELS}, got {c}"
        )

    if not torch.isfinite(x).all():
        raise ValueError("Input tensor contains non-finite values (NaN or Inf)")


def build_sinusoidal_positional_encoding(
    num_positions: int = TOKEN_TIME_STEPS,
    d_model: int = MODEL_DIM,
) -> Tensor:
    """Construct fixed sinusoidal positional encoding table of shape (1, num_positions, d_model).

    Computes standard sinusoidal encoding:
        PE(pos, 2i)   = sin(pos / (10000 ^ (2i / d_model)))
        PE(pos, 2i+1) = cos(pos / (10000 ^ (2i / d_model)))
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required to build positional encoding. "
            "Install the optional 'pytorch' extra (e.g. pip install .[pytorch])."
        )

    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(num_positions, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


if torch is not None:

    class TemporalConv2dTokenizer(nn.Module):
        """Stage 1-4: Temporal Conv2D tokenizer projecting (B, 385, 56) to (B, 97, 64).

        1. Adds a singleton feature-map axis: (B, 1, 385, 56).
        2. Applies Conv2D with 16 filters, kernel (15, 1), stride (4, 1), padding (7, 0), bias enabled,
           yielding (B, 16, 97, 56).
        3. Permutes and flattens feature map/channel values within each time token to (B, 97, 896).
        4. Applies Linear(896, 64), LayerNorm(64), GELU, and Dropout(0.1).
        """

        def __init__(
            self,
            in_channels: int = 1,
            num_filters: int = NUM_TEMPORAL_FILTERS,
            temporal_kernel_size: int = TEMPORAL_KERNEL_SIZE,
            temporal_stride: int = TEMPORAL_STRIDE,
            temporal_padding: int = TEMPORAL_PADDING,
            num_eeg_channels: int = EXPECTED_CHANNELS,
            d_model: int = MODEL_DIM,
            dropout: float = DEFAULT_DROPOUT,
        ) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=(temporal_kernel_size, 1),
                stride=(temporal_stride, 1),
                padding=(temporal_padding, 0),
                bias=True,
            )
            flattened_dim = num_eeg_channels * num_filters  # 56 * 16 = 896
            self.projection = nn.Linear(flattened_dim, d_model, bias=True)
            self.norm = nn.LayerNorm(d_model)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            # Stage 1: (B, 1, 385, 56)
            x = x.unsqueeze(1)
            # Stage 2: (B, 16, 97, 56)
            x = self.conv(x)
            # Stage 3: Permute to (B, 97, 56, 16) and flatten to (B, 97, 896)
            x = x.permute(0, 2, 3, 1).flatten(start_dim=2)
            # Stage 4: Linear projection, LayerNorm, GELU, Dropout
            x = self.projection(x)
            x = self.norm(x)
            x = self.act(x)
            x = self.dropout(x)
            return x

    class SinusoidalPositionalEncoding(nn.Module):
        """Stage 5: Fixed sinusoidal positional encoding over 97 token positions.

        Adds deterministic order information at sample-index token positions.
        Contains no trainable parameters.
        """

        def __init__(
            self,
            num_positions: int = TOKEN_TIME_STEPS,
            d_model: int = MODEL_DIM,
        ) -> None:
            super().__init__()
            pe = build_sinusoidal_positional_encoding(num_positions=num_positions, d_model=d_model)
            self.register_buffer("pe", pe, persistent=False)

        def forward(self, x: Tensor) -> Tensor:
            # x shape: (B, T, d_model)
            return x + self.pe[:, : x.size(1)]

    class ConformerFeedForward(nn.Module):
        """Pre-normalized feed-forward network: 64 -> 128 -> 64, GELU, dropout 0.1."""

        def __init__(
            self,
            d_model: int = MODEL_DIM,
            ffn_dim: int = FFN_DIM,
            dropout: float = DEFAULT_DROPOUT,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.linear1 = nn.Linear(d_model, ffn_dim, bias=True)
            self.act = nn.GELU()
            self.dropout1 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(ffn_dim, d_model, bias=True)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            out = self.norm(x)
            out = self.linear1(out)
            out = self.act(out)
            out = self.dropout1(out)
            out = self.linear2(out)
            out = self.dropout2(out)
            return out

    class ConformerSelfAttention(nn.Module):
        """Pre-normalized multi-head self-attention with 4 heads, no causal mask, output dropout 0.1."""

        def __init__(
            self,
            d_model: int = MODEL_DIM,
            num_heads: int = NUM_HEADS,
            dropout: float = DEFAULT_DROPOUT,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            normed = self.norm(x)
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            return self.dropout(attn_out)

    class ConformerConvModule(nn.Module):
        """Pre-normalized convolution module with depthwise 1D conv and BatchNorm.

        Order:
        - LayerNorm(64)
        - Pointwise Conv1d(64, 128, kernel_size=1) without bias
        - GLU along channel dim to 64 features
        - Depthwise Conv1d(64, 64, kernel_size=15, padding=7, groups=64, bias=True)
        - BatchNorm1d(64)
        - SiLU activation
        - Pointwise Conv1d(64, 64, kernel_size=1, bias=True)
        - Dropout(0.1)
        """

        def __init__(
            self,
            d_model: int = MODEL_DIM,
            kernel_size: int = CONV_KERNEL_SIZE,
            dropout: float = DEFAULT_DROPOUT,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.pw_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, bias=False)
            self.glu = nn.GLU(dim=1)
            padding = (kernel_size - 1) // 2  # (15 - 1) // 2 = 7
            self.dw_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=padding,
                groups=d_model,
                bias=True,
            )
            self.bn = nn.BatchNorm1d(d_model)
            self.act = nn.SiLU()
            self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            # x shape: (B, T, C)
            out = self.norm(x)
            # Transpose to (B, C, T) for 1D convolution
            out = out.transpose(1, 2)
            out = self.pw_conv1(out)
            out = self.glu(out)
            out = self.dw_conv(out)
            out = self.bn(out)
            out = self.act(out)
            out = self.pw_conv2(out)
            out = self.dropout(out)
            # Transpose back to (B, T, C)
            out = out.transpose(1, 2)
            return out

    class ConformerBlock(nn.Module):
        """Conformer-style encoder block preserving token length and feature dimension (B, 97, 64).

        Structure:
        1. Pre-norm FFN 1 with residual scale 0.5.
        2. Optional pre-norm Multi-Head Self-Attention with residual.
        3. Optional pre-norm Convolution Module with residual.
        4. Pre-norm FFN 2 with residual scale 0.5.
        5. Final LayerNorm(64). Optional modules are disabled only by their
           corresponding single-component Sprint 7 ablation.
        """

        def __init__(
            self,
            d_model: int = MODEL_DIM,
            ffn_dim: int = FFN_DIM,
            num_heads: int = NUM_HEADS,
            conv_kernel_size: int = CONV_KERNEL_SIZE,
            dropout: float = DEFAULT_DROPOUT,
            use_self_attention: bool = True,
            use_conv_module: bool = True,
        ) -> None:
            super().__init__()
            if not isinstance(use_self_attention, bool):
                raise TypeError("use_self_attention must be a bool")
            if not isinstance(use_conv_module, bool):
                raise TypeError("use_conv_module must be a bool")
            self.ffn1 = ConformerFeedForward(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)
            self.attn = (
                ConformerSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
                if use_self_attention
                else None
            )
            self.conv = (
                ConformerConvModule(d_model=d_model, kernel_size=conv_kernel_size, dropout=dropout)
                if use_conv_module
                else None
            )
            self.ffn2 = ConformerFeedForward(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)
            self.final_norm = nn.LayerNorm(d_model)

        def forward(self, x: Tensor) -> Tensor:
            # 1. FFN 1 (residual scale 0.5)
            x = x + 0.5 * self.ffn1(x)
            if self.attn is not None:
                x = x + self.attn(x)
            if self.conv is not None:
                x = x + self.conv(x)
            # 4. FFN 2 (residual scale 0.5)
            x = x + 0.5 * self.ffn2(x)
            # 5. Final LayerNorm
            x = self.final_norm(x)
            return x

    class EEGConformerEncoder(nn.Module):
        """Primary candidate EEG Conformer encoder.

        Encodes raw EEG trial signals of shape (B, 385, 56) into sequence
        embeddings of shape (B, 97, 64).

        Stages:
        - Stage 0: Input validation.
        - Stage 1-4: Temporal Conv2D tokenizer.
        - Stage 5: Fixed sinusoidal positional encoding.
        - Stage 6: Two Conformer-style encoder blocks.
        """

        def __init__(
            self,
            expected_time_steps: int = EXPECTED_TIME_STEPS,
            expected_channels: int = EXPECTED_CHANNELS,
            token_time_steps: int = TOKEN_TIME_STEPS,
            d_model: int = MODEL_DIM,
            num_heads: int = NUM_HEADS,
            ffn_dim: int = FFN_DIM,
            conv_kernel_size: int = CONV_KERNEL_SIZE,
            num_blocks: int = NUM_BLOCKS,
            dropout: float = DEFAULT_DROPOUT,
            use_self_attention: bool = True,
            use_conv_module: bool = True,
        ) -> None:
            super().__init__()
            if not isinstance(use_self_attention, bool):
                raise TypeError("use_self_attention must be a bool")
            if not isinstance(use_conv_module, bool):
                raise TypeError("use_conv_module must be a bool")
            self.expected_time_steps = expected_time_steps
            self.expected_channels = expected_channels
            self.token_time_steps = token_time_steps
            self.d_model = d_model

            self.tokenizer = TemporalConv2dTokenizer(
                in_channels=1,
                num_filters=NUM_TEMPORAL_FILTERS,
                temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
                temporal_stride=TEMPORAL_STRIDE,
                temporal_padding=TEMPORAL_PADDING,
                num_eeg_channels=expected_channels,
                d_model=d_model,
                dropout=dropout,
            )
            self.pos_encoder = SinusoidalPositionalEncoding(
                num_positions=token_time_steps,
                d_model=d_model,
            )
            self.blocks = nn.ModuleList(
                [
                    ConformerBlock(
                        d_model=d_model,
                        ffn_dim=ffn_dim,
                        num_heads=num_heads,
                        conv_kernel_size=conv_kernel_size,
                        dropout=dropout,
                        use_self_attention=use_self_attention,
                        use_conv_module=use_conv_module,
                    )
                    for _ in range(num_blocks)
                ]
            )

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass encoding input signals into token sequence embeddings.

            Args:
                x: Tensor of shape (B, 385, 56), dtype float32.

            Returns:
                Tensor of shape (B, 97, 64), dtype float32.
            """
            validate_eeg_input(x)
            x = self.tokenizer(x)
            x = self.pos_encoder(x)
            for block in self.blocks:
                x = block(x)
            return x

        def forward_stages(self, x: Tensor) -> Dict[str, Tensor]:
            """Execute forward pass and return a dictionary of intermediate tensor stages.

            Useful for stage-shape verification and gradient smoke tests.
            """
            validate_eeg_input(x)
            stages: Dict[str, Tensor] = {}
            stages["stage0_input"] = x

            # Tokenizer sub-stages
            s1 = x.unsqueeze(1)
            stages["stage1_unsqueeze"] = s1
            s2 = self.tokenizer.conv(s1)
            stages["stage2_conv2d"] = s2
            s3 = s2.permute(0, 2, 3, 1).flatten(start_dim=2)
            stages["stage3_permute_flatten"] = s3
            s4 = self.tokenizer.dropout(
                self.tokenizer.act(self.tokenizer.norm(self.tokenizer.projection(s3)))
            )
            stages["stage4_projection"] = s4

            # Positional encoding stage
            s5 = self.pos_encoder(s4)
            stages["stage5_pos_encoding"] = s5

            # Encoder blocks
            curr = s5
            for idx, block in enumerate(self.blocks, start=1):
                curr = block(curr)
                stages[f"stage6_block{idx}"] = curr

            stages["output"] = curr
            return stages

else:  # pragma: no cover
    _TORCH_MISSING_MSG = (
        "PyTorch is required for EEGConformerEncoder and conformer modules. "
        "Install the optional 'pytorch' extra: pip install .[pytorch]"
    )

    class _MissingTorchModule:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(_TORCH_MISSING_MSG)

    TemporalConv2dTokenizer = _MissingTorchModule  # type: ignore[misc,assignment]
    SinusoidalPositionalEncoding = _MissingTorchModule  # type: ignore[misc,assignment]
    ConformerFeedForward = _MissingTorchModule  # type: ignore[misc,assignment]
    ConformerSelfAttention = _MissingTorchModule  # type: ignore[misc,assignment]
    ConformerConvModule = _MissingTorchModule  # type: ignore[misc,assignment]
    ConformerBlock = _MissingTorchModule  # type: ignore[misc,assignment]
    EEGConformerEncoder = _MissingTorchModule  # type: ignore[misc,assignment]


__all__ = [
    "CONV_KERNEL_SIZE",
    "ConformerBlock",
    "ConformerConvModule",
    "ConformerFeedForward",
    "ConformerSelfAttention",
    "DEFAULT_DROPOUT",
    "EEGConformerEncoder",
    "EXPECTED_CHANNELS",
    "EXPECTED_TIME_STEPS",
    "FFN_DIM",
    "MODEL_DIM",
    "NUM_BLOCKS",
    "NUM_HEADS",
    "NUM_TEMPORAL_FILTERS",
    "SinusoidalPositionalEncoding",
    "TEMPORAL_KERNEL_SIZE",
    "TEMPORAL_PADDING",
    "TEMPORAL_STRIDE",
    "TOKEN_TIME_STEPS",
    "TemporalConv2dTokenizer",
    "build_sinusoidal_positional_encoding",
    "validate_eeg_input",
]
