"""Tests for recording-level Integrated Gradients explanations."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.deep_fusion import EEGTransformerFusion
from src.epoching import epoch_recording
from src.explainability import explain_recording
from src.features import extract_epoch_features
from src.inference import (
    FinalFusionInference,
    predict_recording,
)
from src.preprocessing import bandpass_filter


def _write_test_checkpoint(path):
    torch.manual_seed(31)
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
    feature_names = [f"feature_{index}" for index in range(40)]
    torch.save(
        {
            "artifact_type": "final_cell1_transformer_handcrafted_fusion",
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "raw_normalization_mean": [0.0, 0.0],
            "raw_normalization_scale": [1.0, 1.0],
            "feature_normalization_mean": [0.0] * 40,
            "feature_normalization_scale": [1.0] * 40,
            "constant_feature_indices": [],
            "feature_names": feature_names,
            "class_mapping": {"Control": 0, "ADHD": 1},
            "model_configuration": {
                "in_channels": 2,
                "n_samples": 1024,
                "num_classes": 2,
                "sampling_rate_hz": 256,
                "transformer": transformer,
            },
            "fusion_configuration": fusion,
            "preprocessing_configuration": {
                "sampling_rate_hz": 256,
                "filter": {
                    "family": "butterworth",
                    "order": 4,
                    "low_cutoff_hz": 0.5,
                    "high_cutoff_hz": 40.0,
                    "zero_phase": True,
                    "implementation": "sosfiltfilt",
                },
                "notch_filter": {"enabled": False},
                "edge_trim_seconds": 1,
                "epoch_duration_seconds": 4,
                "epoch_overlap_seconds": 0,
            },
            "random_seed": 31,
            "dataset_metadata": {"cell": 1},
            "training_metadata": {"selected_epochs": 2},
            "metadata": {
                "outer_fold": None,
                "freeze_transformer_backbone": False,
                "group_ids_are_verified_participant_ids": False,
            },
        },
        path,
    )


class ExplainabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        checkpoint_path = Path(self.temporary_directory.name) / "final.pt"
        _write_test_checkpoint(checkpoint_path)
        self.predictor = FinalFusionInference.from_checkpoint(
            checkpoint_path,
            device="cpu",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _recording(sample_count=1536, seed=73):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(sample_count, 2)).astype(np.float64)

    def test_attribution_shapes_finiteness_completeness_and_determinism(self):
        recording = self._recording()
        first = explain_recording(self.predictor, recording)
        second = explain_recording(self.predictor, recording)

        self.assertEqual(first.status, "ok")
        self.assertEqual(first.accepted_epoch_count, 1)
        explanation = first.epoch_explanations[0]
        self.assertEqual(explanation.raw_attributions.shape, (2, 1024))
        self.assertEqual(explanation.raw_attribution_magnitudes.shape, (2, 1024))
        self.assertEqual(explanation.feature_attributions.shape, (40,))
        self.assertEqual(explanation.feature_attribution_magnitudes.shape, (40,))
        self.assertEqual(explanation.probabilities.shape, (2,))
        self.assertEqual(len(first.feature_names), 40)
        self.assertEqual(first.mean_signed_raw_attribution.shape, (2, 1024))
        self.assertEqual(first.mean_absolute_raw_attribution.shape, (2, 1024))
        self.assertEqual(first.mean_signed_feature_attribution.shape, (40,))
        self.assertEqual(first.mean_absolute_feature_attribution.shape, (40,))
        self.assertTrue(
            np.isfinite(explanation.raw_attributions).all()
            and np.isfinite(explanation.feature_attributions).all()
            and np.isfinite(explanation.probabilities).all()
            and np.isfinite(explanation.completeness_residual)
        )
        self.assertLess(abs(explanation.completeness_residual), 0.03)
        self.assertTrue(np.all(explanation.raw_attribution_magnitudes >= 0))
        self.assertTrue(np.all(explanation.feature_attribution_magnitudes >= 0))
        np.testing.assert_array_equal(
            explanation.raw_attributions,
            second.epoch_explanations[0].raw_attributions,
        )
        np.testing.assert_array_equal(
            explanation.feature_attributions,
            second.epoch_explanations[0].feature_attributions,
        )
        np.testing.assert_array_equal(
            explanation.probabilities,
            second.epoch_explanations[0].probabilities,
        )
        self.assertIn("Zero-valued normalized", first.baseline_description)

    def test_recording_aggregation_and_rejected_epoch_metadata(self):
        recording = self._recording(sample_count=3584, seed=91)
        recording[1280:2304] = 3.0
        with mock.patch(
            "src.inference.bandpass_filter",
            side_effect=lambda data, **kwargs: np.array(data, copy=True),
        ):
            result = explain_recording(
                self.predictor,
                recording,
                integration_steps=16,
            )

        self.assertEqual(result.candidate_epoch_count, 3)
        self.assertEqual(result.accepted_epoch_count, 2)
        self.assertEqual(result.rejected_epoch_count, 1)
        self.assertEqual(result.rejected_epochs[0]["epoch_index"], 2)
        self.assertEqual(result.rejected_epochs[0]["reason"], "flatline_epoch")
        self.assertEqual(result.rejected_epochs[0]["start_sample"], 1280)
        self.assertEqual(result.rejected_epochs[0]["stop_sample"], 2304)

        feature_values = np.stack(
            [item.feature_attributions for item in result.epoch_explanations]
        )
        raw_values = np.stack(
            [item.raw_attributions for item in result.epoch_explanations]
        )
        np.testing.assert_allclose(
            result.mean_signed_feature_attribution,
            feature_values.mean(axis=0),
        )
        np.testing.assert_allclose(
            result.mean_absolute_feature_attribution,
            np.abs(feature_values).mean(axis=0),
        )
        np.testing.assert_allclose(
            result.mean_signed_raw_attribution,
            raw_values.mean(axis=0),
        )
        np.testing.assert_allclose(
            result.mean_absolute_raw_attribution,
            np.abs(raw_values).mean(axis=0),
        )

    def test_ordinary_prediction_matches_pre_refactor_inference_path(self):
        recording = self._recording(sample_count=2560, seed=117)
        filtered = bandpass_filter(
            recording,
            sampling_rate_hz=256,
            low_cutoff_hz=0.5,
            high_cutoff_hz=40.0,
            order=4,
        )
        expected_epochs = epoch_recording(
            filtered,
            sampling_rate_hz=256,
            edge_trim_seconds=1,
            epoch_duration_seconds=4,
        )["epochs"]
        feature_matrix, _ = extract_epoch_features(
            expected_epochs,
            sampling_rate_hz=256,
        )
        normalized_raw = (
            (expected_epochs - self.predictor.raw_mean)
            / self.predictor.raw_scale
        ).astype(np.float32)
        raw_tensor = torch.from_numpy(
            np.transpose(normalized_raw, (0, 2, 1)).copy()
        )
        normalized_features = (
            (feature_matrix - self.predictor.feature_mean)
            / self.predictor.feature_scale
        ).astype(np.float32)
        feature_tensor = torch.from_numpy(normalized_features)
        with torch.no_grad():
            expected_logits = self.predictor.model(raw_tensor, feature_tensor)
            expected_probabilities = torch.softmax(
                expected_logits,
                dim=1,
            ).numpy()
        expected_labels = (expected_probabilities[:, 1] >= 0.5).astype(
            np.int64
        )
        result = predict_recording(
            self.predictor,
            recording,
            sampling_rate_hz=256,
        )
        actual_probabilities = np.asarray(
            [
                [item.probability_control, item.probability_adhd]
                for item in result.epoch_predictions
            ]
        )
        actual_labels = np.asarray(
            [item.predicted_label for item in result.epoch_predictions]
        )
        np.testing.assert_array_equal(
            actual_probabilities,
            expected_probabilities,
        )
        np.testing.assert_array_equal(
            actual_labels,
            expected_labels,
        )

    def test_short_recording_returns_empty_explanation(self):
        result = explain_recording(
            self.predictor,
            self._recording(sample_count=1200),
        )
        self.assertEqual(result.status, "no_valid_epochs")
        self.assertEqual(result.accepted_epoch_count, 0)
        self.assertEqual(result.epoch_explanations, ())
        self.assertIsNone(result.mean_signed_feature_attribution)

    def test_saved_checkpoint_compatibility_and_inference_pipeline(self):
        project_root = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            project_root
            / "results"
            / "deep_learning"
            / "cell_01"
            / "final_fusion"
            / "run_20261116T092136Z"
            / "final_fusion_checkpoint.pt"
        )
        predictor = FinalFusionInference.from_checkpoint(
            checkpoint_path,
            device="cpu",
        )
        recording = self._recording(sample_count=1536, seed=211)
        ordinary_result = predict_recording(
            predictor,
            recording,
            sampling_rate_hz=256,
        )
        explanation = explain_recording(
            predictor,
            recording,
            sampling_rate_hz=256,
            integration_steps=8,
        )

        self.assertEqual(explanation.accepted_epoch_count, 1)
        np.testing.assert_allclose(
            explanation.epoch_explanations[0].probabilities,
            [
                ordinary_result.epoch_predictions[0].probability_control,
                ordinary_result.epoch_predictions[0].probability_adhd,
            ],
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(
            explanation.feature_names,
            tuple(predictor.feature_names),
        )


if __name__ == "__main__":
    unittest.main()
