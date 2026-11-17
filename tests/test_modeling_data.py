"""Tests for the validated Cell 1 modeling-data loader."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.modeling_data import load_cell1_dataset


class Cell1DatasetLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "synthetic_cell1.npz"
        self.arrays = self.valid_arrays()
        self.write_archive(self.arrays)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def valid_arrays():
        rng = np.random.default_rng(42)
        return {
            "epochs": rng.normal(size=(10, 1024, 2)),
            "class_labels": np.asarray(
                ["Control"] * 5 + ["ADHD"] * 5, dtype=str
            ),
            "group_keys": np.asarray(
                [f"file.mat:slot-{slot:03d}" for slot in range(1, 11)],
                dtype=str,
            ),
            "fold_ids": np.asarray([0, 1, 2, 3, 4] * 2, dtype=np.int64),
        }

    def write_archive(self, arrays):
        np.savez(self.path, **arrays)

    def test_loads_valid_data_and_encodes_classes(self):
        loaded = load_cell1_dataset(self.path)

        self.assertEqual(loaded.X.shape, (10, 1024, 2))
        np.testing.assert_array_equal(loaded.y, [0] * 5 + [1] * 5)
        self.assertEqual(loaded.class_mapping, {"Control": 0, "ADHD": 1})
        np.testing.assert_array_equal(loaded.groups, self.arrays["group_keys"])
        np.testing.assert_array_equal(loaded.folds, self.arrays["fold_ids"])

    def test_rejects_incorrect_epoch_dimensions(self):
        self.arrays["epochs"] = np.zeros((10, 512, 2))
        self.write_archive(self.arrays)

        with self.assertRaisesRegex(ValueError, "shape \\[epochs, 1024, 2\\]"):
            load_cell1_dataset(self.path)

    def test_rejects_nan_and_infinite_values(self):
        for invalid_value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=invalid_value):
                arrays = self.valid_arrays()
                arrays["epochs"][0, 0, 0] = invalid_value
                self.write_archive(arrays)

                with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                    load_cell1_dataset(self.path)

    def test_rejects_missing_metadata(self):
        del self.arrays["group_keys"]
        self.write_archive(self.arrays)

        with self.assertRaisesRegex(ValueError, "missing required metadata"):
            load_cell1_dataset(self.path)

    def test_rejects_invalid_labels(self):
        self.arrays["class_labels"][0] = "Unknown"
        self.write_archive(self.arrays)

        with self.assertRaisesRegex(ValueError, "unsupported values"):
            load_cell1_dataset(self.path)

    def test_rejects_mismatched_metadata_lengths(self):
        self.arrays["group_keys"] = self.arrays["group_keys"][:-1]
        self.write_archive(self.arrays)

        with self.assertRaisesRegex(ValueError, "group_keys has 9 entries"):
            load_cell1_dataset(self.path)

    def test_rejects_group_assigned_to_multiple_folds(self):
        self.arrays["group_keys"][1] = self.arrays["group_keys"][0]
        self.write_archive(self.arrays)

        with self.assertRaisesRegex(ValueError, "appears in multiple folds"):
            load_cell1_dataset(self.path)

    def test_rejects_missing_fold_assignment(self):
        self.arrays["fold_ids"] = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        self.write_archive(self.arrays)

        with self.assertRaisesRegex(ValueError, "missing fold IDs: \\[4\\]"):
            load_cell1_dataset(self.path)


if __name__ == "__main__":
    unittest.main()
