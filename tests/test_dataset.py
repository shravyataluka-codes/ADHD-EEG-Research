import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if torch is not None:
    from adhd_eeg import TrialManifest
    from adhd_eeg.dataset import EEGTrialDataset
    import h5py
    import numpy as np


@unittest.skipIf(torch is None, "Install the optional 'pytorch' extra to run Dataset tests")
class EEGTrialDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)
        y_stim = np.zeros((4, 6), dtype=np.float64)
        y_stim[0] = (1, 1, 2, 1, 2, 2)
        with h5py.File(self.dataset_dir / "y_stim.mat", "w") as metadata_file:
            metadata_file.create_dataset("y_stim", data=y_stim)

        for file_number in range(1, 4):
            first_global_index = (file_number - 1) * 2
            trials = np.empty((385, 56, 2), dtype=np.float32)
            for within_file_index in range(2):
                trials[:, :, within_file_index] = (
                    first_global_index + within_file_index
                )
            with h5py.File(
                self.dataset_dir / f"d{file_number}.mat", "w"
            ) as eeg_file:
                eeg_file.create_dataset(
                    f"d{file_number}", data=trials, dtype=np.float32
                )

        self.manifest = TrialManifest(
            self.dataset_dir,
            file_trial_counts=(2, 2, 2),
            group_ranges=(("HC", 0, 3), ("ADD", 3, 6)),
        )
        self.dataset = EEGTrialDataset(self.manifest)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _assert_file_can_be_renamed(self, source_path):
        renamed_path = source_path.with_name(f"{source_path.stem}.renamed.mat")
        source_path.rename(renamed_path)
        renamed_path.rename(source_path)

    def test_length(self):
        self.assertEqual(len(self.dataset), 6)

    def test_sample_tensor_shape_dtype_and_source_metadata(self):
        signal, metadata = self.dataset[2]

        self.assertIsInstance(signal, torch.Tensor)
        self.assertEqual(tuple(signal.shape), (385, 56))
        self.assertEqual(signal.dtype, torch.float32)
        torch.testing.assert_close(
            signal, torch.full((385, 56), 2.0, dtype=torch.float32)
        )
        self.assertEqual(
            metadata,
            {
                "trial_id": "d2.mat:0",
                "source_filename": "d2.mat",
                "within_file_trial_index": 0,
            },
        )

    def test_unresolved_labels_and_candidate_metadata_are_not_returned(self):
        _, metadata = self.dataset[0]

        self.assertEqual(
            set(metadata),
            {"trial_id", "source_filename", "within_file_trial_index"},
        )
        self.assertNotIn("subject_id", metadata)
        self.assertNotIn("clinical_group", metadata)
        self.assertNotIn("diagnostic_group", metadata)
        self.assertNotIn("target", metadata)

    def test_dataloader_batches_signals_and_source_metadata(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        batches = list(loader)

        self.assertEqual([tuple(signals.shape) for signals, _ in batches], [
            (2, 385, 56),
            (2, 385, 56),
            (2, 385, 56),
        ])
        self.assertTrue(all(signals.dtype == torch.float32 for signals, _ in batches))
        self.assertEqual(
            [trial_id for _, metadata in batches for trial_id in metadata["trial_id"]],
            ["d1.mat:0", "d1.mat:1", "d2.mat:0", "d2.mat:1", "d3.mat:0", "d3.mat:1"],
        )
        self.assertEqual(
            [
                filename
                for _, metadata in batches
                for filename in metadata["source_filename"]
            ],
            ["d1.mat", "d1.mat", "d2.mat", "d2.mat", "d3.mat", "d3.mat"],
        )

    def test_invalid_indices_are_rejected(self):
        for index in (-1, len(self.dataset)):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    self.dataset[index]

    def test_hdf5_file_closes_after_successful_read(self):
        self.dataset[0]
        self._assert_file_can_be_renamed(self.dataset_dir / "d1.mat")

    def test_hdf5_file_closes_when_read_validation_raises(self):
        source_path = self.dataset_dir / "d1.mat"
        with h5py.File(source_path, "w") as eeg_file:
            eeg_file.create_dataset(
                "d1", data=np.zeros((384, 56, 2), dtype=np.float32)
            )

        with self.assertRaisesRegex(ValueError, "Unexpected shape"):
            self.dataset[0]
        self._assert_file_can_be_renamed(source_path)


if __name__ == "__main__":
    unittest.main()
