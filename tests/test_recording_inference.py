"""Tests for end-to-end inference from an unlabeled continuous recording."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.deep_fusion import EEGTransformerFusion
from src.epoching import epoch_recording
from src.inference import FinalFusionInference, predict_recording
from src.preprocessing import RecordingRejectedError, bandpass_filter


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
    feature_names = [
        f"feature_{index}"
        for index in range(40)
    ]
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


class RecordingInferenceTests(unittest.TestCase):
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
    def _recording(sample_count=4096):
        rng = np.random.default_rng(73)
        return rng.normal(size=(sample_count, 2)).astype(np.float64)

    def test_preprocessing_parity_epoch_shapes_and_prediction_reproducibility(self):
        recording = self._recording()
        filtered = bandpass_filter(
            recording,
            sampling_rate_hz=256,
            low_cutoff_hz=0.5,
            high_cutoff_hz=40.0,
            order=4,
        )
        expected = epoch_recording(
            filtered,
            sampling_rate_hz=256,
            edge_trim_seconds=1,
            epoch_duration_seconds=4,
        )
        with mock.patch.object(
            self.predictor,
            "predict",
            wraps=self.predictor.predict,
        ) as prediction_call:
            result = predict_recording(
                self.predictor,
                recording,
                sampling_rate_hz=256,
            )
        actual_epoch_batch = prediction_call.call_args.args[0]
        np.testing.assert_array_equal(actual_epoch_batch, expected["epochs"])
        self.assertEqual(actual_epoch_batch.shape, (3, 1024, 2))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.candidate_epoch_count, 3)
        self.assertEqual(result.summary["valid_epoch_count"], 3)
        self.assertEqual(len(result.epoch_predictions), 3)
        self.assertEqual(result.epoch_samples, 1024)
        self.assertEqual(result.edge_trim_samples_per_side, 256)
        self.assertEqual(result.discarded_tail_samples, 512)
        for prediction in result.epoch_predictions:
            self.assertIn(prediction.predicted_class, ("Control", "ADHD"))
            self.assertIn(prediction.predicted_label, (0, 1))
            self.assertTrue(
                np.isclose(
                    prediction.probability_control
                    + prediction.probability_adhd,
                    1.0,
                    rtol=0.0,
                    atol=1e-6,
                )
            )
        repeated = predict_recording(
            self.predictor,
            recording,
            sampling_rate_hz=256,
        )
        self.assertEqual(result.epoch_predictions, repeated.epoch_predictions)
        self.assertEqual(result.summary, repeated.summary)

    def test_invalid_input_and_sampling_rate_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_recording(
                self.predictor,
                np.zeros((1024, 3)),
                sampling_rate_hz=256,
            )
        with self.assertRaisesRegex(ValueError, "Sampling rate must match"):
            predict_recording(
                self.predictor,
                self._recording(),
                sampling_rate_hz=128,
            )
        invalid = self._recording()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            predict_recording(
                self.predictor,
                invalid,
                sampling_rate_hz=256,
            )

    def test_short_recording_returns_empty_summary_without_prediction(self):
        with mock.patch.object(self.predictor, "predict") as prediction_call:
            result = predict_recording(
                self.predictor,
                self._recording(sample_count=1200),
                sampling_rate_hz=256,
            )

        self.assertEqual(result.status, "no_valid_epochs")
        self.assertEqual(result.candidate_epoch_count, 0)
        self.assertEqual(result.epoch_predictions, ())
        self.assertEqual(result.summary["valid_epoch_count"], 0)
        prediction_call.assert_not_called()

    def test_flatline_recording_is_rejected_by_existing_preprocessor(self):
        with self.assertRaises(RecordingRejectedError):
            predict_recording(
                self.predictor,
                np.ones((4096, 2), dtype=np.float64),
                sampling_rate_hz=256,
            )


if __name__ == "__main__":
    unittest.main()
