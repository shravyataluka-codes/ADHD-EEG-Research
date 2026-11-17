"""Tests for loading and using the self-describing final fusion checkpoint."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.deep_fusion import EEGTransformerFusion
from src.features import _channel_feature_names
from src.inference import CLASS_MAPPING, FinalFusionInference


def make_final_checkpoint(path):
    torch.manual_seed(19)
    transformer = {
        "embed_dim": 8,
        "num_heads": 2,
        "num_layers": 1,
        "feedforward_dim": 16,
        "token_kernel_size": 16,
        "token_stride": 8,
        "dropout": 0.2,
    }
    fusion = {
        "feature_count": 40,
        "projection_dim": 4,
        "classifier_hidden_dim": 4,
        "dropout": 0.3,
        "freeze_transformer_backbone": False,
    }
    model = EEGTransformerFusion(
        in_channels=2,
        n_samples=1024,
        num_classes=2,
        feature_count=40,
        fusion_dim=4,
        head_hidden_dim=4,
        dropout=0.3,
        transformer_config=transformer,
    )
    feature_names = tuple(
        name
        for channel_slot in (1, 2)
        for name in _channel_feature_names(channel_slot)
    )
    torch.save(
        {
            "artifact_type": "final_cell1_transformer_handcrafted_fusion",
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "raw_normalization_mean": [0.1, -0.2],
            "raw_normalization_scale": [1.5, 2.0],
            "feature_normalization_mean": [0.0] * 40,
            "feature_normalization_scale": [1.0] * 40,
            "constant_feature_indices": [],
            "feature_names": list(feature_names),
            "class_mapping": CLASS_MAPPING,
            "model_configuration": {
                "in_channels": 2,
                "n_samples": 1024,
                "num_classes": 2,
                "sampling_rate_hz": 256,
                "transformer": transformer,
            },
            "fusion_configuration": fusion,
            "preprocessing_configuration": {"sampling_rate_hz": 256},
            "random_seed": 19,
            "dataset_metadata": {"cell": 1},
            "training_metadata": {"selected_epochs": 3},
            "metadata": {
                "outer_fold": None,
                "freeze_transformer_backbone": False,
                "group_ids_are_verified_participant_ids": False,
            },
        },
        path,
    )


class FinalFusionInferenceTests(unittest.TestCase):
    def test_checkpoint_loading_and_prediction_are_reproducible(self):
        rng = np.random.default_rng(23)
        epochs = rng.normal(size=(2, 1024, 2)).astype(np.float64)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "final_fusion.pt"
            make_final_checkpoint(checkpoint_path)

            first_predictor = FinalFusionInference.from_checkpoint(
                checkpoint_path,
                device="cpu",
            )
            second_predictor = FinalFusionInference.from_checkpoint(
                checkpoint_path,
                device="cpu",
            )
            first_result = first_predictor.predict(epochs, batch_size=1)
            second_result = second_predictor.predict(epochs, batch_size=1)

        self.assertEqual(first_predictor.class_mapping, {"Control": 0, "ADHD": 1})
        self.assertEqual(first_predictor.random_seed, 19)
        self.assertEqual(first_result.probabilities.shape, (2, 2))
        self.assertEqual(first_result.predicted_labels.shape, (2,))
        self.assertTrue(np.isfinite(first_result.probabilities).all())
        np.testing.assert_allclose(
            first_result.probabilities.sum(axis=1),
            np.ones(2),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            first_result.probabilities,
            second_result.probabilities,
        )
        np.testing.assert_array_equal(
            first_result.predicted_labels,
            second_result.predicted_labels,
        )
        np.testing.assert_array_equal(
            first_result.predicted_labels,
            (first_result.probabilities[:, 1] >= 0.5).astype(np.int64),
        )


if __name__ == "__main__":
    unittest.main()
