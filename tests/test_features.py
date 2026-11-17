"""Tests for deterministic per-epoch, per-channel feature extraction."""

from dataclasses import replace
import unittest

import numpy as np

from src.features import extract_features
from src.modeling_data import ModelingData


FEATURES_PER_CHANNEL = 20


def make_dataset():
    sampling_rate = 256
    time = np.arange(1024) / sampling_rate
    epochs = np.empty((5, 1024, 2), dtype=np.float64)
    for index in range(5):
        epochs[index, :, 0] = np.sin(2 * np.pi * 10 * time + index * 0.1)
        epochs[index, :, 1] = 0.5 * np.sin(
            2 * np.pi * 20 * time + index * 0.2
        )
    return ModelingData(
        X=epochs,
        y=np.asarray([0, 0, 1, 1, 1], dtype=np.int64),
        groups=np.asarray([f"slot-{index}" for index in range(5)]),
        folds=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        class_mapping={"Control": 0, "ADHD": 1},
    )


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_matrix_dimensions_and_count(self):
        result = extract_features(make_dataset())

        self.assertEqual(result.X.shape, (5, 2 * FEATURES_PER_CHANNEL))
        self.assertEqual(len(result.feature_names), 40)
        self.assertEqual(len(set(result.feature_names)), 40)

    def test_feature_names_identify_channel_slot_and_feature(self):
        result = extract_features(make_dataset())

        self.assertEqual(
            result.feature_names[:5],
            (
                "channel_slot_1_absolute_delta_power",
                "channel_slot_1_absolute_theta_power",
                "channel_slot_1_absolute_alpha_power",
                "channel_slot_1_absolute_beta_power",
                "channel_slot_1_absolute_low_gamma_power",
            ),
        )
        self.assertIn(
            "channel_slot_1_theta_to_beta_power_ratio",
            result.feature_names,
        )
        self.assertIn(
            "channel_slot_2_hjorth_complexity",
            result.feature_names,
        )

    def test_channels_are_extracted_separately(self):
        result = extract_features(make_dataset())
        alpha_1 = result.feature_names.index(
            "channel_slot_1_absolute_alpha_power"
        )
        beta_1 = result.feature_names.index(
            "channel_slot_1_absolute_beta_power"
        )
        alpha_2 = result.feature_names.index(
            "channel_slot_2_absolute_alpha_power"
        )
        beta_2 = result.feature_names.index(
            "channel_slot_2_absolute_beta_power"
        )

        self.assertGreater(result.X[0, alpha_1], result.X[0, beta_1])
        self.assertGreater(result.X[0, beta_2], result.X[0, alpha_2])
        self.assertNotEqual(result.X[0, alpha_1], result.X[0, alpha_2])

    def test_valid_signals_produce_finite_features(self):
        result = extract_features(make_dataset())

        self.assertTrue(np.isfinite(result.X).all())

    def test_rejects_constant_channel_with_clear_error(self):
        dataset = make_dataset()
        dataset.X[0, :, 0] = 2.0

        with self.assertRaisesRegex(ValueError, "channel slot 1 is constant"):
            extract_features(dataset)

    def test_rejects_invalid_epoch_shape(self):
        dataset = replace(make_dataset(), X=np.zeros((5, 1024, 3)))

        with self.assertRaisesRegex(ValueError, "shape \\[epochs, 1024, 2\\]"):
            extract_features(dataset)

    def test_rejects_nonfinite_input(self):
        dataset = make_dataset()
        dataset.X[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            extract_features(dataset)

    def test_preserves_labels_groups_and_folds(self):
        dataset = make_dataset()
        result = extract_features(dataset)

        np.testing.assert_array_equal(result.y, dataset.y)
        np.testing.assert_array_equal(result.groups, dataset.groups)
        np.testing.assert_array_equal(result.folds, dataset.folds)
        self.assertEqual(result.class_mapping, dataset.class_mapping)

    def test_repeated_execution_is_deterministic(self):
        dataset = make_dataset()
        first = extract_features(dataset)
        second = extract_features(dataset)

        np.testing.assert_array_equal(first.X, second.X)
        self.assertEqual(first.feature_names, second.feature_names)


if __name__ == "__main__":
    unittest.main()
