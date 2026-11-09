"""Unit and integration tests for EEG Conformer classifier and model integration."""

import io
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    Dataset = object  # type: ignore[misc,assignment]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if torch is not None:
    import h5py
    import numpy as np
    from adhd_eeg.classifier import (
        CLASS_ORDER,
        HEAD_DROPOUT,
        HEAD_HIDDEN_DIM,
        NUM_CLASSES,
        POOLING_DIM,
        EEGClassificationHead,
        EEGConformerClassifier,
    )
    from adhd_eeg.conformer import (
        EXPECTED_CHANNELS,
        EXPECTED_TIME_STEPS,
        MODEL_DIM,
        TOKEN_TIME_STEPS,
        EEGConformerEncoder,
    )
    from adhd_eeg.data_access import TrialManifest
    from adhd_eeg.dataset import EEGTrialDataset


@unittest.skipIf(torch is None, "Install the optional 'pytorch' extra to run classifier tests")
class EEGConformerClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = EEGConformerClassifier()

    def test_class_order_contract(self) -> None:
        """Verify the exact class-order contract specified in Sprint 4."""
        self.assertEqual(CLASS_ORDER, ("HC", "ADD", "ADHD"))
        self.assertEqual(NUM_CLASSES, 3)
        self.assertEqual(CLASS_ORDER[0], "HC")
        self.assertEqual(CLASS_ORDER[1], "ADD")
        self.assertEqual(CLASS_ORDER[2], "ADHD")

    def test_forward_batch_size_one(self) -> None:
        """Verify forward pass output shape, dtype, and finiteness for B=1."""
        x = torch.randn(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        self.assertEqual(logits.shape, (1, NUM_CLASSES))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())

    def test_forward_batch_size_greater_than_one(self) -> None:
        """Verify forward pass output shape, dtype, and finiteness for B > 1."""
        for batch_size in (2, 4):
            x = torch.randn(batch_size, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
            self.assertEqual(logits.shape, (batch_size, NUM_CLASSES))
            self.assertEqual(logits.dtype, torch.float32)
            self.assertTrue(torch.isfinite(logits).all())

    def test_pooling_operation(self) -> None:
        """Verify temporal mean and max pooling concatenation produces (B, 128)."""
        seq = torch.randn(2, TOKEN_TIME_STEPS, MODEL_DIM, dtype=torch.float32)
        pooled = self.model.pool(seq)
        self.assertEqual(pooled.shape, (2, POOLING_DIM))
        self.assertEqual(pooled.shape, (2, 128))

        # Check explicit slice values
        expected_mean = seq.mean(dim=1)
        expected_max = torch.amax(seq, dim=1)
        self.assertTrue(torch.allclose(pooled[:, :MODEL_DIM], expected_mean))
        self.assertTrue(torch.allclose(pooled[:, MODEL_DIM:], expected_max))

    def test_cross_entropy_loss_forward_and_backward(self) -> None:
        """Verify CrossEntropyLoss forward and backward passes produce finite gradients."""
        self.model.train()
        criterion = nn.CrossEntropyLoss()

        batch_size = 3
        x = torch.randn(batch_size, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        # Integer synthetic target labels covering each class index {0, 1, 2}
        synthetic_targets = torch.tensor([0, 1, 2], dtype=torch.int64)

        logits = self.model(x)
        self.assertEqual(logits.shape, (batch_size, NUM_CLASSES))

        loss = criterion(logits, synthetic_targets)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient")
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Parameter {name} has non-finite gradient",
                )

    def test_predict_scores_uncalibrated_softmax(self) -> None:
        """Verify predict_scores() computes valid softmax scores summing to 1.0."""
        batch_size = 4
        x = torch.randn(batch_size, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            scores = self.model.predict_scores(x)

        self.assertEqual(scores.shape, (batch_size, NUM_CLASSES))
        self.assertTrue(torch.isfinite(scores).all())
        # Scores must be within [0.0, 1.0]
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 1.0).all())
        # Scores must sum to 1.0 along the class axis
        sums = scores.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_predict_scores_does_not_silently_change_model_mode(self) -> None:
        """Verify predict_scores() preserves the caller's model mode (train vs eval)."""
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)

        # In train mode, calling predict_scores must NOT flip training to False
        self.model.train()
        self.assertTrue(self.model.training)
        _ = self.model.predict_scores(x)
        self.assertTrue(self.model.training)

        # In eval mode, calling predict_scores must NOT flip training to True
        self.model.eval()
        self.assertFalse(self.model.training)
        _ = self.model.predict_scores(x)
        self.assertFalse(self.model.training)

    def test_forward_does_not_apply_softmax(self) -> None:
        """Verify forward() outputs raw logits, not softmax probabilities."""
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        # Logits can be negative and generally do not sum to 1.0
        sums = logits.sum(dim=-1)
        self.assertFalse(torch.allclose(sums, torch.ones_like(sums), atol=1e-3))

    def test_evaluation_mode_determinism(self) -> None:
        """Verify eval() mode produces identical outputs on identical inputs."""
        self.model.eval()
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)

        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)

        self.assertTrue(torch.equal(out1, out2))

    def test_train_mode_dropout_variation(self) -> None:
        """Verify train() mode exhibits stochastic variation due to dropout."""
        self.model.train()
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        out1 = self.model(x)
        out2 = self.model(x)
        self.assertFalse(torch.equal(out1, out2))

    def test_state_dict_save_load_round_trip(self) -> None:
        """Verify state dict can be saved, loaded, and reproduces exact evaluation outputs."""
        self.model.eval()
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)

        with torch.no_grad():
            original_logits = self.model(x)

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)

        new_model = EEGConformerClassifier()
        new_model.load_state_dict(torch.load(buffer, weights_only=True))
        new_model.eval()

        with torch.no_grad():
            reloaded_logits = new_model(x)

        self.assertTrue(torch.equal(original_logits, reloaded_logits))

    def test_exact_parameter_counts(self) -> None:
        """Verify exact parameter counts matching the Sprint 4 architectural accounting:

        - Classification head: 8,451 parameters:
            - Linear(128, 64): 128 * 64 + 64 = 8,256
            - Linear(64, 3): 64 * 3 + 3 = 195
            - Total head = 8,256 + 195 = 8,451
        - Implemented encoder: 185,664 parameters
        - Combined model: 185,664 + 8,451 = 194,115 parameters
        """
        # Head parameter accounting
        head_linear1 = sum(p.numel() for p in self.model.head.linear1.parameters())
        self.assertEqual(head_linear1, 128 * 64 + 64)  # 8,256

        head_linear2 = sum(p.numel() for p in self.model.head.linear2.parameters())
        self.assertEqual(head_linear2, 64 * 3 + 3)  # 195

        total_head_params = sum(p.numel() for p in self.model.head.parameters())
        self.assertEqual(total_head_params, 8451)

        # Encoder parameter accounting
        total_encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        self.assertEqual(total_encoder_params, 185664)

        # Combined model accounting
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.assertEqual(total_trainable_params, 194115)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_params, 194115)

    def test_encoder_injection(self) -> None:
        """Verify EEGConformerClassifier accepts an existing EEGConformerEncoder instance."""
        custom_encoder = EEGConformerEncoder()
        classifier = EEGConformerClassifier(encoder=custom_encoder)
        self.assertIs(classifier.encoder, custom_encoder)

    def test_input_validation_rejection(self) -> None:
        """Verify model rejects invalid shapes, dtypes, non-finite values, and extra args."""
        # Invalid rank
        with self.assertRaises(ValueError):
            self.model(torch.randn(EXPECTED_TIME_STEPS, EXPECTED_CHANNELS))

        # Invalid dimensions
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, EXPECTED_TIME_STEPS - 1, EXPECTED_CHANNELS))

        # Empty batch
        with self.assertRaises(ValueError):
            self.model(torch.empty(0, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS))

        # Invalid dtype
        with self.assertRaises(TypeError):
            self.model(torch.zeros(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float64))

        # Non-finite values
        x_nan = torch.randn(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS)
        x_nan[0, 5, 5] = float("nan")
        with self.assertRaises(ValueError):
            self.model(x_nan)

        # Rejection of metadata or labels inside model forward call
        x_valid = torch.randn(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS)
        with self.assertRaises(TypeError):
            self.model(x_valid, metadata={"trial_id": "test:0"})  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            self.model(x_valid, labels=torch.tensor([0]))  # type: ignore[call-arg]

    def test_dataset_integration_with_synthetic_labeled_fixture(self) -> None:
        """Verify integration with EEGTrialDataset via a synthetic labeled wrapper fixture.

        Confirms that:
        1. Metadata is preserved and never passed into the model forward call.
        2. Signal tensors pass into the model cleanly through a standard DataLoader.
        3. CrossEntropyLoss backward succeeds on synthetic targets.
        """
        # Test-only synthetic labeled wrapper around unlabeled EEGTrialDataset
        class _SyntheticLabeledWrapper(Dataset):
            def __init__(self, base_dataset: EEGTrialDataset) -> None:
                self.base_dataset = base_dataset

            def __len__(self) -> int:
                return len(self.base_dataset)

            def __getitem__(self, idx: int):
                signal, metadata = self.base_dataset[idx]
                # Fabricate an arbitrary synthetic target index in [0, NUM_CLASSES - 1]
                synthetic_target = idx % NUM_CLASSES
                return signal, synthetic_target, metadata

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            y_stim = np.zeros((4, 4), dtype=np.float64)
            y_stim[0] = (1, 1, 1, 1)
            with h5py.File(dataset_dir / "y_stim.mat", "w") as metadata_file:
                metadata_file.create_dataset("y_stim", data=y_stim)

            # 4 synthetic trials of shape (385, 56)
            trials = np.zeros((385, 56, 4), dtype=np.float32)
            with h5py.File(dataset_dir / "d1.mat", "w") as eeg_file:
                eeg_file.create_dataset("d1", data=trials, dtype=np.float32)

            manifest = TrialManifest(
                dataset_dir,
                file_trial_counts=(4,),
                group_ranges=(("HC", 0, 4),),
            )
            base_dataset = EEGTrialDataset(manifest)
            self.assertEqual(len(base_dataset), 4)

            labeled_dataset = _SyntheticLabeledWrapper(base_dataset)
            loader = DataLoader(
                labeled_dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([b[0] for b in batch]),
                    torch.tensor([b[1] for b in batch], dtype=torch.int64),
                    [b[2] for b in batch],
                ),
            )

            self.model.train()
            criterion = nn.CrossEntropyLoss()

            for batch_signals, batch_targets, batch_metadatas in loader:
                # Confirm signals have expected shape (B, 385, 56)
                self.assertEqual(batch_signals.shape, (2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS))
                # Confirm targets are 1D int64 tensors
                self.assertEqual(batch_targets.shape, (2,))
                # Confirm metadata is outside the model and retains source keys
                self.assertEqual(len(batch_metadatas), 2)
                for meta in batch_metadatas:
                    self.assertIn("trial_id", meta)
                    self.assertEqual(meta["source_filename"], "d1.mat")

                # Forward pass receives signals ONLY - metadata is NOT passed
                logits = self.model(batch_signals)
                self.assertEqual(logits.shape, (2, NUM_CLASSES))

                # Loss computation and backward pass
                loss = criterion(logits, batch_targets)
                loss.backward()


class MissingTorchFallbackTests(unittest.TestCase):
    def test_missing_torch_classes_raise_import_error(self) -> None:
        """Verify _MissingTorchModule behavior when PyTorch is unavailable."""
        from adhd_eeg.classifier import _MissingTorchModule

        missing = _MissingTorchModule
        with self.assertRaises(ImportError) as ctx:
            missing()
        self.assertIn("PyTorch is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
