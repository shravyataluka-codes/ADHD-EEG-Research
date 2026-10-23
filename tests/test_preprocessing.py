import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhd_eeg.preprocessing import (
    FilterSettings,
    ICASettings,
    PreprocessingConfig,
    SubjectICAReview,
    preprocess_trials,
    process_dataset,
)
from adhd_eeg.data_access import TrialManifest


class EEGPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir()
        self.output_dir = self.root / "preprocessed"
        self.config = PreprocessingConfig(
            sampling_frequency_hz=256.0,
            notch_frequency_hz=50.0,
            random_seed=17,
            filters=FilterSettings(
                method="fir",
                phase="zero",
                fir_design="firwin",
                filter_length="auto",
                l_trans_bandwidth=0.5,
                h_trans_bandwidth=2.0,
                fir_window="hamming",
                pad="reflect_limited",
                notch_width_hz=1.0,
                notch_transition_bandwidth_hz=1.0,
            ),
            ica=ICASettings(method="infomax", n_components=2, max_iter=500),
        )
        self.review = SubjectICAReview(
            exclude_components=(),
            reviewer="synthetic-test",
            reviewed_at="2026-10-07T00:00:00Z",
            rationale="Synthetic fixture; no artifact components designated.",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _synthetic_trials(count, channels=2):
        time = np.arange(385, dtype=np.float32) / np.float32(256.0)
        trials = np.empty((count, 385, channels), dtype=np.float32)
        for trial_index in range(count):
            for channel_index in range(channels):
                trials[trial_index, :, channel_index] = (
                    np.sin(
                        2 * np.pi * (5 + channel_index % 35) * time
                        + trial_index * 0.01
                    )
                    + 0.15
                    * np.sin(2 * np.pi * 20 * time + channel_index * 0.1)
                )
        return trials

    def _make_dataset(self):
        # HC_01 crosses source-file boundaries to exercise subject mapping.
        y_stim = np.zeros((4, 7), dtype=np.float64)
        y_stim[0] = (1, 1, 1, 1, 1, 1, 2)
        with h5py.File(self.dataset_dir / "y_stim.mat", "w") as metadata:
            metadata.create_dataset("y_stim", data=y_stim)

        all_trials = self._synthetic_trials(7, channels=56)
        self.source_files = []
        for number in range(1, 8):
            path = self.dataset_dir / f"d{number}.mat"
            self.source_files.append(path)
            with h5py.File(path, "w") as source:
                source.create_dataset(
                    f"d{number}",
                    data=all_trials[number - 1:number].transpose(1, 2, 0),
                    dtype=np.float32,
                )
        self.raw_mat_files = sorted(self.dataset_dir.glob("*.mat"))
        return TrialManifest(
            self.dataset_dir,
            file_trial_counts=(1, 1, 1, 1, 1, 1, 1),
            group_ranges=(("HC", 0, 7),),
        )

    def _reviews(self):
        return {
            "HC_01": self.review,
            "HC_02": self.review,
        }

    def test_shape_preservation_and_determinism(self):
        trials = self._synthetic_trials(3)
        first, component_count = preprocess_trials(trials, self.config, self.review)
        second, second_component_count = preprocess_trials(
            trials.copy(),
            self.config,
            self.review,
        )

        self.assertEqual(first.shape, trials.shape)
        self.assertEqual(first.dtype, np.dtype(np.float32))
        self.assertEqual(component_count, 2)
        self.assertEqual(second_component_count, component_count)
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)

    def test_dataset_metadata_shape_counts_and_raw_immutability(self):
        manifest = self._make_dataset()
        hashes_before = {path: self._sha256(path) for path in self.raw_mat_files}

        metadata = process_dataset(
            self.dataset_dir,
            self.output_dir,
            self.config,
            self._reviews(),
            manifest=manifest,
        )

        self.assertEqual(metadata["trial_count"], 7)
        self.assertEqual(metadata["subject_count"], 2)
        self.assertTrue(metadata["validation"]["shape_preserved"])
        self.assertTrue(all(metadata["validation"]["raw_files_unchanged"].values()))
        self.assertIn("y_stim.mat", metadata["validation"]["raw_files_unchanged"])
        self.assertEqual(
            hashes_before,
            {path: self._sha256(path) for path in self.raw_mat_files},
        )
        for number in range(1, 8):
            with h5py.File(self.output_dir / f"d{number}.h5", "r") as output:
                data = output[f"d{number}"]
                self.assertEqual(data.shape, (385, 56, 1))
                self.assertEqual(data.dtype, np.dtype(np.float32))
                self.assertTrue(np.isfinite(data[:]).all())

        with (self.output_dir / "trial_manifest.csv").open(
            newline="",
            encoding="utf-8",
        ) as manifest_file:
            manifest_rows = list(csv.DictReader(manifest_file))
        self.assertEqual(len(manifest_rows), 7)
        self.assertEqual(manifest_rows[2]["global_trial_index"], "2")
        self.assertEqual(manifest_rows[2]["source_file"], "d3.mat")
        self.assertEqual(manifest_rows[2]["subject_id"], "HC_01")
        inspection = json.loads(
            (self.output_dir / "inspection_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(inspection["raw_shape"], [385, 56])
        self.assertEqual(inspection["preprocessed_shape"], [385, 56])
        self.assertEqual(inspection["trial_count"], 7)
        self.assertEqual(inspection["subject_count"], 2)
        with np.load(self.output_dir / "inspection_trial.npz") as representative:
            self.assertEqual(representative["raw"].shape, (385, 56))
            self.assertEqual(representative["preprocessed"].shape, (385, 56))
        with self.assertRaises(FileExistsError):
            process_dataset(
                self.dataset_dir,
                self.output_dir,
                self.config,
                self._reviews(),
                manifest=manifest,
            )

    def test_nan_inf_and_missing_review_decisions_are_rejected(self):
        for invalid_value in (np.nan, np.inf):
            trials = self._synthetic_trials(2)
            trials[0, 0, 0] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    preprocess_trials(trials, self.config, self.review)

        manifest = self._make_dataset()
        with self.assertRaisesRegex(ValueError, "outside the raw dataset"):
            process_dataset(
                self.dataset_dir,
                self.dataset_dir / "preprocessed",
                self.config,
                self._reviews(),
                manifest=manifest,
            )
        with self.assertRaisesRegex(ValueError, "ICA review subject mismatch"):
            process_dataset(
                self.dataset_dir,
                self.output_dir,
                self.config,
                {"HC_01": self.review},
                manifest=manifest,
            )
        self.assertFalse(self.output_dir.exists())

    def test_invalid_component_index_and_notch_setting_are_rejected(self):
        review = SubjectICAReview(
            exclude_components=(2,),
            reviewer="synthetic-test",
            reviewed_at="2026-10-07T00:00:00Z",
            rationale="Out-of-range index test.",
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            preprocess_trials(self._synthetic_trials(3), self.config, review)

        invalid = PreprocessingConfig(
            sampling_frequency_hz=self.config.sampling_frequency_hz,
            notch_frequency_hz=60.0,
            random_seed=self.config.random_seed,
            filters=self.config.filters,
            ica=self.config.ica,
        )
        with self.assertRaisesRegex(ValueError, "approved notch"):
            invalid.validate(channel_count=2)


if __name__ == "__main__":
    unittest.main()
