import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhd_eeg import EEGTrialStore, TrialManifest


class EEGDataAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)
        self.file_trial_counts = (2, 2, 2)
        self.group_ranges = (("HC", 0, 3), ("ADD", 3, 6))

        y_stim = np.zeros((4, 6), dtype=np.float64)
        y_stim[0] = (1, 1, 2, 1, 2, 2)
        with h5py.File(self.dataset_dir / "y_stim.mat", "w") as metadata_file:
            metadata_file.create_dataset("y_stim", data=y_stim)

        self.source_files = []
        for file_number in range(1, 4):
            file_path = self.dataset_dir / f"d{file_number}.mat"
            self.source_files.append(file_path)
            first_global_index = (file_number - 1) * 2
            trials = np.empty((385, 56, 2), dtype=np.float32)
            for within_file_index in range(2):
                trials[:, :, within_file_index] = first_global_index + within_file_index
            with h5py.File(file_path, "w") as eeg_file:
                eeg_file.create_dataset(f"d{file_number}", data=trials, dtype=np.float32)

        self.manifest = TrialManifest(
            self.dataset_dir,
            file_trial_counts=self.file_trial_counts,
            group_ranges=self.group_ranges,
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

    def test_trial_retrieval_shape_and_dtype(self):
        with EEGTrialStore(self.manifest) as store:
            trial, metadata = store.get_trial(4)

        self.assertEqual(trial.shape, (385, 56))
        self.assertEqual(trial.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(trial, np.full((385, 56), 4, dtype=np.float32))
        self.assertEqual(metadata.global_trial_index, 4)

    def test_file_boundary_indices_and_metadata_alignment(self):
        with EEGTrialStore(self.manifest) as store:
            results = [store.get_trial(index) for index in (1, 2, 3, 4)]

        trials, metadata = zip(*results)
        self.assertEqual(
            [(row.source_file, row.within_file_trial_index) for row in metadata],
            [("d1.mat", 1), ("d2.mat", 0), ("d2.mat", 1), ("d3.mat", 0)],
        )
        self.assertEqual(
            [row.candidate_y_stim_column for row in metadata], [1, 2, 3, 4]
        )
        np.testing.assert_array_equal([trial[0, 0] for trial in trials], [1, 2, 3, 4])
        self.assertEqual(metadata[1].candidate_participant_id, "HC_02")
        self.assertEqual(metadata[1].candidate_diagnostic_group, "HC")
        self.assertEqual(metadata[2].candidate_participant_id, "ADD_01")
        self.assertEqual(metadata[2].candidate_diagnostic_group, "ADD")
        self.assertFalse(hasattr(metadata[1], "participant_id"))
        self.assertFalse(hasattr(metadata[1], "diagnostic_group"))

    def test_first_and_last_trial_of_every_file(self):
        expected = (
            (0, "d1.mat", 0, 0),
            (1, "d1.mat", 1, 1),
            (2, "d2.mat", 0, 2),
            (3, "d2.mat", 1, 3),
            (4, "d3.mat", 0, 4),
            (5, "d3.mat", 1, 5),
        )
        with EEGTrialStore(self.manifest) as store:
            results = [store.get_trial(index) for index in (0, 1, 2, 3, 4, 5)]

        for (index, source_file, within_file_index, value), (trial, metadata) in zip(
            expected, results
        ):
            self.assertEqual(metadata.global_trial_index, index)
            self.assertEqual(metadata.source_file, source_file)
            self.assertEqual(metadata.within_file_trial_index, within_file_index)
            np.testing.assert_array_equal(
                trial, np.full((385, 56), value, dtype=np.float32)
            )

    def test_every_file_transition_maps_to_the_next_file(self):
        transitions = ((1, 2), (3, 4))
        with EEGTrialStore(self.manifest) as store:
            results = [
                (store.get_trial(left), store.get_trial(right))
                for left, right in transitions
            ]

        for (left_index, right_index), ((left_trial, left_row), (right_trial, right_row)) in zip(
            transitions, results
        ):
            self.assertEqual(left_row.global_trial_index, left_index)
            self.assertEqual(right_row.global_trial_index, right_index)
            self.assertEqual(left_row.source_file, "d%d.mat" % (left_index // 2 + 1))
            self.assertEqual(right_row.source_file, "d%d.mat" % (right_index // 2 + 1))
            np.testing.assert_array_equal(
                left_trial, np.full((385, 56), left_index, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                right_trial, np.full((385, 56), right_index, dtype=np.float32)
            )

    def test_candidate_participant_and_group_lookup(self):
        subject_rows = tuple(
            self.manifest.iter_candidate_participant("ADD_02")
        )
        group_rows = tuple(self.manifest.iter_candidate_diagnostic_group("HC"))

        self.assertEqual([row.global_trial_index for row in subject_rows], [4, 5])
        self.assertEqual([row.global_trial_index for row in group_rows], [0, 1, 2])
        with self.assertRaises(KeyError):
            tuple(self.manifest.iter_candidate_diagnostic_group("unknown"))

    def test_batches_are_bounded_and_preserve_request_order(self):
        observed_sizes = []
        observed_indices = []
        with EEGTrialStore(self.manifest) as store:
            for batch, rows in store.iter_batches(2, indices=(5, 0, 3, 1, 4)):
                observed_sizes.append(batch.shape[0])
                observed_indices.extend(row.global_trial_index for row in rows)
                self.assertEqual(batch.shape[1:], (385, 56))
                self.assertEqual(batch.dtype, np.dtype(np.float32))

        self.assertEqual(observed_sizes, [2, 2, 1])
        self.assertEqual(observed_indices, [5, 0, 3, 1, 4])

    def test_batches_across_each_transition_preserve_values(self):
        with EEGTrialStore(self.manifest) as store:
            batches = [store.get_batch(indices) for indices in ((1, 2), (3, 4))]

        for indices, (batch, rows) in zip(((1, 2), (3, 4)), batches):
            self.assertEqual(batch.shape, (2, 385, 56))
            self.assertEqual(batch.dtype, np.dtype(np.float32))
            self.assertEqual(
                [row.global_trial_index for row in rows], list(indices)
            )
            expected = np.stack(
                [
                    np.full((385, 56), index, dtype=np.float32)
                    for index in indices
                ]
            )
            np.testing.assert_array_equal(batch, expected)

    def test_manifest_csv_contains_metadata_only(self):
        manifest_path = self.dataset_dir / "manifest.csv"
        self.manifest.write_csv(manifest_path)

        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 7)
        self.assertIn("global_trial_index,source_file", lines[0])
        self.assertIn("HC_01", "\n".join(lines))
        self.assertNotIn("eeg", lines[0].lower())

    def test_source_files_are_unchanged_and_returned_trials_are_detached(self):
        hashes_before = {path: self._sha256(path) for path in self.source_files}
        with EEGTrialStore(self.manifest) as store:
            trial, _ = store.get_trial(0)
            trial[:] = -99
            reread, _ = store.get_trial(0)

        self.assertTrue(np.all(reread == 0))
        self.assertEqual(
            hashes_before,
            {path: self._sha256(path) for path in self.source_files},
        )

    def test_invalid_indices_and_batch_sizes_are_rejected(self):
        with self.assertRaises(IndexError):
            self.manifest.metadata_for(-1)
        with self.assertRaises(IndexError):
            self.manifest.metadata_for(6)
        with EEGTrialStore(self.manifest) as store:
            for index in (-1, self.manifest.total_trials):
                with self.subTest(index=index):
                    with self.assertRaises(IndexError):
                        store.get_trial(index)
            for indices in ((-1,), (self.manifest.total_trials,)):
                with self.subTest(indices=indices):
                    with self.assertRaises(IndexError):
                        store.get_batch(indices)
                    with self.assertRaises(IndexError):
                        tuple(store.iter_batches(1, indices=indices))
            with self.assertRaises(ValueError):
                store.get_batch(())
            with self.assertRaises(ValueError):
                store.get_batch([])
            self.assertEqual(list(store.iter_batches(2, indices=())), [])
            for batch_size in (0, -1, True, 1.5):
                with self.subTest(batch_size=batch_size):
                    with self.assertRaises(ValueError):
                        next(store.iter_batches(batch_size))

    def test_incorrect_eeg_dimensions_are_rejected(self):
        source_file = self.dataset_dir / "d2.mat"
        with h5py.File(source_file, "w") as eeg_file:
            eeg_file.create_dataset(
                "d2", data=np.zeros((384, 56, 2), dtype=np.float32)
            )

        with EEGTrialStore(self.manifest) as store:
            with self.assertRaisesRegex(ValueError, "Unexpected shape"):
                store.get_trial(2)

    def test_incorrect_source_dtype_is_rejected(self):
        source_file = self.dataset_dir / "d2.mat"
        with h5py.File(source_file, "w") as eeg_file:
            eeg_file.create_dataset(
                "d2", data=np.zeros((385, 56, 2), dtype=np.float64)
            )

        with EEGTrialStore(self.manifest) as store:
            with self.assertRaisesRegex(ValueError, "Unexpected dtype"):
                store.get_trial(2)

    def test_non_integral_file_trial_counts_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "file trial count must be an integer"):
            TrialManifest(
                self.dataset_dir,
                file_trial_counts=(2.5, 2, 2),
                group_ranges=self.group_ranges,
            )

    def test_non_integral_group_range_boundaries_are_rejected(self):
        invalid_ranges = (
            (("HC", 0.5, 3), ("ADD", 3, 6)),
            (("HC", 0, 3.5), ("ADD", 3.5, 6)),
        )
        for group_ranges in invalid_ranges:
            with self.subTest(group_ranges=group_ranges):
                with self.assertRaisesRegex(TypeError, "group range"):
                    TrialManifest(
                        self.dataset_dir,
                        file_trial_counts=self.file_trial_counts,
                        group_ranges=group_ranges,
                    )

    def test_duplicate_group_names_after_whitespace_normalization_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate group names"):
            TrialManifest(
                self.dataset_dir,
                file_trial_counts=self.file_trial_counts,
                group_ranges=((" HC ", 0, 3), ("HC", 3, 6)),
            )

    def test_whitespace_only_group_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be named"):
            TrialManifest(
                self.dataset_dir,
                file_trial_counts=self.file_trial_counts,
                group_ranges=((" \t ", 0, 3), ("ADD", 3, 6)),
            )


if __name__ == "__main__":
    unittest.main()