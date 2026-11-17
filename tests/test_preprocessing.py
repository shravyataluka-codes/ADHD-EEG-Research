"""Focused tests for filtering, epoching, rejection, and fold grouping."""

import unittest

import numpy as np

from src.epoching import assign_grouped_folds, epoch_recording
from src.preprocessing import (
    RecordingRejectedError,
    bandpass_filter,
)


class BandpassFilterTests(unittest.TestCase):
    def test_bandpass_preserves_in_band_signal_without_phase_shift(self):
        sampling_rate = 256
        time = np.arange(30 * sampling_rate) / sampling_rate
        expected = np.sin(2 * np.pi * 10 * time)
        input_signal = expected + 0.5 * np.sin(2 * np.pi * 60 * time)
        original = input_signal.copy()

        filtered = bandpass_filter(input_signal[:, None])[:, 0]
        steady = slice(sampling_rate, -sampling_rate)
        correlation = np.corrcoef(filtered[steady], expected[steady])[0, 1]

        self.assertEqual(filtered.shape, input_signal.shape)
        self.assertTrue(np.array_equal(input_signal, original))
        self.assertGreater(correlation, 0.995)

    def test_filter_rejects_nonfinite_values(self):
        signal = np.sin(np.arange(4096) / 10)
        signal[100] = np.nan

        with self.assertRaises(RecordingRejectedError) as error:
            bandpass_filter(signal[:, None])
        self.assertEqual(error.exception.reason, "nonfinite_input")

    def test_filter_rejects_flatline_channel(self):
        time = np.arange(4096) / 256
        recording = np.column_stack(
            (np.sin(2 * np.pi * 10 * time), np.ones_like(time))
        )

        with self.assertRaises(RecordingRejectedError) as error:
            bandpass_filter(recording)
        self.assertEqual(error.exception.reason, "flatline_input_channel")
        self.assertEqual(error.exception.channel_slots, [2])


class EpochingTests(unittest.TestCase):
    def test_edge_trim_and_epoch_dimensions(self):
        random = np.random.default_rng(42)
        recording = random.normal(size=(30 * 256, 2))
        result = epoch_recording(
            recording,
            sampling_rate_hz=256,
            edge_trim_seconds=1,
            epoch_duration_seconds=4,
        )

        self.assertEqual(result["epochs"].shape, (7, 1024, 2))
        self.assertEqual(result["candidate_epoch_count"], 7)
        self.assertEqual(result["accepted_epoch_count"], 7)
        self.assertEqual(result["edge_trim_samples_per_side"], 256)
        self.assertEqual(result["discarded_tail_samples"], 0)

    def test_missing_values_reject_only_affected_epoch(self):
        random = np.random.default_rng(5)
        recording = random.normal(size=(10 * 256, 2))
        recording[256 + 100, 0] = np.nan

        result = epoch_recording(
            recording,
            sampling_rate_hz=256,
            edge_trim_seconds=1,
            epoch_duration_seconds=4,
        )

        self.assertEqual(result["candidate_epoch_count"], 2)
        self.assertEqual(result["accepted_epoch_count"], 1)
        self.assertEqual(len(result["rejected_epochs"]), 1)
        self.assertEqual(result["rejected_epochs"][0]["reason"], "nonfinite_epoch")
        self.assertEqual(result["rejected_epochs"][0]["channel_slots"], [1])

    def test_flatline_epoch_is_rejected(self):
        recording = np.ones((6 * 256, 2))
        result = epoch_recording(
            recording,
            sampling_rate_hz=256,
            edge_trim_seconds=1,
            epoch_duration_seconds=4,
        )

        self.assertEqual(result["accepted_epoch_count"], 0)
        self.assertEqual(len(result["rejected_epochs"]), 1)
        self.assertEqual(result["rejected_epochs"][0]["reason"], "flatline_epoch")
        self.assertEqual(result["rejected_epochs"][0]["channel_slots"], [1, 2])


class GroupedFoldTests(unittest.TestCase):
    def test_grouped_fold_assignment_is_reproducible_and_keeps_groups_together(self):
        groups = [
            f"FC.mat:slot-{slot:03d}"
            for slot in (1, 1, 2, 2, 3, 3, 4, 4)
        ] + [
            f"FADHD.mat:slot-{slot:03d}"
            for slot in (1, 1, 2, 2, 3, 3, 4, 4)
        ]
        labels = ["Control"] * 8 + ["ADHD"] * 8

        first = assign_grouped_folds(groups, labels, n_splits=2)
        second = assign_grouped_folds(groups, labels, n_splits=2)

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), len(groups))
        for group in set(groups):
            assigned_folds = {
                fold for key, fold in zip(groups, first) if key == group
            }
            self.assertEqual(len(assigned_folds), 1)


if __name__ == "__main__":
    unittest.main()
