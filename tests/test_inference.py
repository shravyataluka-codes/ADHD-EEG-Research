"""Outer-fold inference parity against saved held-out OOF predictions."""

import csv
import unittest
from pathlib import Path

import numpy as np

from src.inference import DEFAULT_CHECKPOINT, FoldFusionInference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL1_DATA = PROJECT_ROOT / "processed" / "cell_01_epochs.npz"
FUSION_ARM_DIRECTORY = DEFAULT_CHECKPOINT.parents[1]
OOF_PREDICTIONS = FUSION_ARM_DIRECTORY / "oof_predictions.csv"
EXPECTED_HELD_OUT_COUNTS = (119, 119, 105, 105, 105)


@unittest.skipUnless(
    CELL1_DATA.is_file() and OOF_PREDICTIONS.is_file(),
    "Full feature-fusion run artifacts are required for inference parity testing.",
)
class AllFoldInferenceTests(unittest.TestCase):
    def test_inference_matches_each_folds_held_out_oof_predictions(self):
        with np.load(CELL1_DATA, allow_pickle=False) as dataset:
            epochs = dataset["epochs"]
            folds = dataset["fold_ids"]

        with OOF_PREDICTIONS.open(
            "r", newline="", encoding="utf-8"
        ) as prediction_file:
            saved_rows = list(csv.DictReader(prediction_file))

        for fold_id, expected_count in enumerate(EXPECTED_HELD_OUT_COUNTS):
            with self.subTest(fold_id=fold_id):
                held_out_indices = np.flatnonzero(folds == fold_id)
                self.assertEqual(held_out_indices.size, expected_count)
                fold_rows = sorted(
                    (
                        row
                        for row in saved_rows
                        if int(row["fold_id"]) == fold_id
                    ),
                    key=lambda row: int(row["epoch_index"]),
                )
                saved_indices = np.asarray(
                    [int(row["epoch_index"]) for row in fold_rows],
                    dtype=np.int64,
                )
                np.testing.assert_array_equal(saved_indices, held_out_indices)

                checkpoint = (
                    FUSION_ARM_DIRECTORY
                    / f"outer_fold_{fold_id}"
                    / "best_checkpoint.pt"
                )
                self.assertTrue(checkpoint.is_file(), f"Missing {checkpoint}.")
                predictor = FoldFusionInference.from_checkpoint(
                    checkpoint,
                    device="cpu",
                    expected_fold=fold_id,
                )
                self.assertEqual(predictor.fold_id, fold_id)

                # Only epochs are supplied; labels remain in the reference CSV
                # and are read below solely for checking saved predictions.
                result = predictor.predict(epochs[held_out_indices])
                expected_probabilities = np.asarray(
                    [
                        [
                            float(row["probability_control"]),
                            float(row["probability_adhd"]),
                        ]
                        for row in fold_rows
                    ],
                    dtype=np.float64,
                )
                expected_labels = np.asarray(
                    [int(row["y_pred"]) for row in fold_rows],
                    dtype=np.int64,
                )

                self.assertEqual(result.probabilities.shape, (expected_count, 2))
                self.assertEqual(result.predicted_labels.shape, (expected_count,))
                self.assertTrue(np.isfinite(result.probabilities).all())
                np.testing.assert_allclose(
                    result.probabilities.sum(axis=1),
                    np.ones(expected_count),
                    rtol=0.0,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    result.probabilities,
                    expected_probabilities,
                    rtol=1e-6,
                    atol=1e-7,
                )
                np.testing.assert_array_equal(
                    result.predicted_labels,
                    expected_labels,
                )

                max_error = float(
                    np.max(np.abs(result.probabilities - expected_probabilities))
                )
                print(
                    f"fold {fold_id}: epochs={expected_count}, "
                    f"max_probability_error={max_error:.12g}, "
                    "labels=exact"
                )


if __name__ == "__main__":
    unittest.main()
