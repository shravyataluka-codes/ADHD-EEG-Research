"""Compact fusion model for raw EEG and handcrafted Cell 1 features."""

import torch
from torch import nn

from src.deep_models import EEGTransformer


class EEGTransformerFusion(nn.Module):
    """Fuse Transformer EEG embeddings with a projected 40-feature vector."""

    def __init__(
        self,
        in_channels=2,
        n_samples=1024,
        num_classes=2,
        feature_count=40,
        fusion_dim=16,
        head_hidden_dim=16,
        dropout=0.3,
        freeze_transformer_backbone=False,
        transformer_config=None,
        backbone=None,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("EEGTransformerFusion supports exactly two classes.")
        if feature_count != 40:
            raise ValueError("Cell 1 fusion requires exactly 40 handcrafted features.")
        if min(fusion_dim, head_hidden_dim) < 1:
            raise ValueError("Projection and head dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        transformer_parameters = dict(transformer_config or {})
        self.backbone = (
            backbone
            if backbone is not None
            else EEGTransformer(
                in_channels=in_channels,
                n_samples=n_samples,
                num_classes=num_classes,
                **transformer_parameters,
            )
        )
        if not isinstance(self.backbone, EEGTransformer):
            raise TypeError("backbone must be an EEGTransformer instance.")
        if (
            self.backbone.in_channels != in_channels
            or self.backbone.n_samples != n_samples
        ):
            raise ValueError("Backbone dimensions do not match fusion inputs.")
        self.backbone.classifier = nn.Identity()

        self.in_channels = in_channels
        self.n_samples = n_samples
        self.feature_count = feature_count
        self.fusion_dim = fusion_dim
        self.freeze_transformer_backbone = bool(freeze_transformer_backbone)

        self.raw_projection = nn.Sequential(
            nn.LayerNorm(self.backbone.feature_dim),
            nn.Linear(self.backbone.feature_dim, fusion_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.handcrafted_projection = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, fusion_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, head_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

        if self.freeze_transformer_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _validate_inputs(self, raw_eeg, handcrafted_features):
        if not isinstance(raw_eeg, torch.Tensor):
            raise TypeError("Raw EEG input must be a torch.Tensor.")
        if raw_eeg.ndim != 3:
            raise ValueError(
                "Raw EEG must have shape [batch, channels, samples]; "
                f"got {tuple(raw_eeg.shape)}."
            )
        if raw_eeg.shape[0] < 1:
            raise ValueError("Input batch must contain at least one epoch.")
        if raw_eeg.shape[1:] != (self.in_channels, self.n_samples):
            raise ValueError(
                "Raw EEG dimensions do not match the configured channel slots "
                f"and samples; got {tuple(raw_eeg.shape[1:])}."
            )
        if not torch.is_floating_point(raw_eeg):
            raise TypeError("Raw EEG input must use a floating-point dtype.")
        if not isinstance(handcrafted_features, torch.Tensor):
            raise TypeError("Handcrafted features must be a torch.Tensor.")
        if handcrafted_features.ndim != 2:
            raise ValueError(
                "Handcrafted features must have shape [batch, 40]; "
                f"got {tuple(handcrafted_features.shape)}."
            )
        if handcrafted_features.shape != (raw_eeg.shape[0], self.feature_count):
            raise ValueError(
                "Handcrafted features must have one 40-value row per raw epoch; "
                f"got {tuple(handcrafted_features.shape)}."
            )
        if not torch.is_floating_point(handcrafted_features):
            raise TypeError("Handcrafted features must use a floating-point dtype.")
        if handcrafted_features.device != raw_eeg.device:
            raise ValueError("Raw EEG and handcrafted features must share a device.")
        if not torch.isfinite(raw_eeg).all().item():
            raise ValueError("Raw EEG input contains non-finite values.")
        if not torch.isfinite(handcrafted_features).all().item():
            raise ValueError("Handcrafted features contain non-finite values.")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_transformer_backbone:
            self.backbone.eval()
        return self

    def forward_representations(self, raw_eeg, handcrafted_features):
        """Return separate and fused learned representations for analysis."""
        self._validate_inputs(raw_eeg, handcrafted_features)
        raw_representation = self.backbone.forward_features(raw_eeg)
        raw_projected = self.raw_projection(raw_representation)
        handcrafted_projected = self.handcrafted_projection(handcrafted_features)
        fused = torch.cat((raw_projected, handcrafted_projected), dim=1)
        return {
            "raw_eeg": raw_representation,
            "handcrafted_features": handcrafted_features,
            "raw_projected": raw_projected,
            "handcrafted_projected": handcrafted_projected,
            "fused": fused,
        }

    def forward_features(self, raw_eeg, handcrafted_features):
        return self.forward_representations(raw_eeg, handcrafted_features)["fused"]

    def forward(self, raw_eeg, handcrafted_features):
        representations = self.forward_representations(
            raw_eeg,
            handcrafted_features,
        )
        return self.classifier(representations["fused"])
