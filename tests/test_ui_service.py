"""Tests for UI file parsing, validation, and backend orchestration."""

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.deep_fusion import EEGTransformerFusion
from src.inference import FinalFusionInference
from src.ui_service import (
    UIInputError,
    analyze_recording,
    parse_sampling_rate,
    parse_uploaded_recording,
    validate_recording_for_model,
)


def _write_test_checkpoint(path):
    torch.manual_seed(77)
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
            "feature_names": [f"feature_{index}" for index in range(40)],
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
            "random_seed": 77,
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


class UIServiceTests(unittest.TestCase):
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
    def _recording(sample_count=1536, seed=103):
        return np.random.default_rng(seed).normal(
            size=(sample_count, 2)
        ).astype(np.float64)

    def test_parse_csv_with_header_and_without_header(self):
        with_header = parse_uploaded_recording(
            "sample.CSV",
            b"slot_a,slot_b\n1.0,2.0\n3.5,4.5\n",
        )
        without_header = parse_uploaded_recording(
            "sample.csv",
            b"1.0,2.0\n3.5,4.5\n",
        )
        np.testing.assert_array_equal(
            with_header,
            np.asarray([[1.0, 2.0], [3.5, 4.5]]),
        )
        np.testing.assert_array_equal(with_header, without_header)

    def test_parse_csv_rejects_bad_columns_values_and_nonfinite_data(self):
        cases = (
            (b"1,2,3\n", "exactly two columns"),
            (b"slot_a,1\n1,2\n", "two numeric values"),
            (b"1,nan\n2,3\n", "NaN or infinite"),
            (b"", "no numeric recording rows"),
        )
        for content, message in cases:
            with self.subTest(content=content):
                with self.assertRaisesRegex(UIInputError, message):
                    parse_uploaded_recording("recording.csv", content)

    def test_parse_npy_and_reject_wrong_shapes_or_pickled_arrays(self):
        buffer = io.BytesIO()
        np.save(buffer, self._recording(sample_count=12), allow_pickle=False)
        parsed = parse_uploaded_recording("recording.npy", buffer.getvalue())
        self.assertEqual(parsed.shape, (12, 2))
        self.assertTrue(np.isfinite(parsed).all())

        wrong_shape = io.BytesIO()
        np.save(wrong_shape, np.ones((12, 3)), allow_pickle=False)
        with self.assertRaisesRegex(UIInputError, r"\[samples, 2\]"):
            parse_uploaded_recording("wrong.npy", wrong_shape.getvalue())

        object_array = io.BytesIO()
        np.save(object_array, np.asarray([{"not": "numeric"}]), allow_pickle=True)
        with self.assertRaises(UIInputError):
            parse_uploaded_recording("object.npy", object_array.getvalue())

        with self.assertRaisesRegex(UIInputError, r"\.csv or \.npy"):
            parse_uploaded_recording("recording.mat", b"")

    def test_sampling_rate_duration_and_channel_validation(self):
        with self.assertRaisesRegex(UIInputError, "Enter the recording"):
            parse_sampling_rate("")
        with self.assertRaisesRegex(UIInputError, "numeric"):
            parse_sampling_rate("unknown")
        with self.assertRaisesRegex(UIInputError, "positive finite"):
            parse_sampling_rate("inf")

        recording = self._recording()
        accepted = validate_recording_for_model(
            recording,
            256,
            self.predictor,
        )
        self.assertEqual(accepted.shape, recording.shape)
        with self.assertRaisesRegex(UIInputError, "must match"):
            validate_recording_for_model(recording, 128, self.predictor)
        with self.assertRaisesRegex(UIInputError, "too short"):
            validate_recording_for_model(
                self._recording(sample_count=1535),
                256,
                self.predictor,
            )
        flatline = self._recording()
        flatline[:, 1] = 4.0
        with self.assertRaisesRegex(UIInputError, "channel slot\\(s\\): 2"):
            validate_recording_for_model(flatline, 256, self.predictor)

    def test_analyze_recording_returns_aligned_real_backend_outputs(self):
        result = analyze_recording(
            self.predictor,
            self._recording(),
            sampling_rate_hz=256,
            integration_steps=4,
        )
        self.assertEqual(result.sampling_rate_hz, 256.0)
        self.assertEqual(result.inference.summary["valid_epoch_count"], 1)
        self.assertEqual(result.explanation.accepted_epoch_count, 1)
        prediction = result.inference.epoch_predictions[0]
        explanation = result.explanation.epoch_explanations[0]
        self.assertEqual(prediction.epoch_index, explanation.epoch_index)
        np.testing.assert_allclose(
            explanation.probabilities,
            [prediction.probability_control, prediction.probability_adhd],
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(len(result.explanation.feature_names), 40)


if __name__ == "__main__":
    unittest.main()
