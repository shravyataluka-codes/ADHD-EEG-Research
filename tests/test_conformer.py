"""Unit and integration tests for EEG Conformer encoder and components."""

import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if torch is not None:
    import h5py
    import numpy as np
    from adhd_eeg.conformer import (
        CONV_KERNEL_SIZE,
        ConformerBlock,
        ConformerConvModule,
        ConformerFeedForward,
        ConformerSelfAttention,
        DEFAULT_DROPOUT,
        EEGConformerEncoder,
        EXPECTED_CHANNELS,
        EXPECTED_TIME_STEPS,
        FFN_DIM,
        MODEL_DIM,
        NUM_BLOCKS,
        NUM_HEADS,
        NUM_TEMPORAL_FILTERS,
        SinusoidalPositionalEncoding,
        TEMPORAL_KERNEL_SIZE,
        TEMPORAL_PADDING,
        TEMPORAL_STRIDE,
        TOKEN_TIME_STEPS,
        TemporalConv2dTokenizer,
        build_sinusoidal_positional_encoding,
        validate_eeg_input,
    )
    from adhd_eeg.data_access import TrialManifest
    from adhd_eeg.dataset import EEGTrialDataset


@unittest.skipIf(torch is None, "Install the optional 'pytorch' extra to run Conformer tests")
class EEGConformerEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.encoder = EEGConformerEncoder()

    def test_forward_batch_size_one(self) -> None:
        x = torch.randn(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        self.encoder.eval()
        with torch.no_grad():
            out = self.encoder(x)
        self.assertEqual(out.shape, (1, TOKEN_TIME_STEPS, MODEL_DIM))
        self.assertEqual(out.dtype, torch.float32)

    def test_train_mode_batch_size_one(self) -> None:
        """Verify training-mode forward and backward passes execute cleanly with batch size 1.

        Confirms that 3D BatchNorm1d over (B=1, C=64, T=97) does not raise single-element
        variance errors during training, output is finite, loss backward succeeds, and all
        trainable parameter gradients are finite.
        """
        encoder = EEGConformerEncoder()
        encoder.train()
        x = torch.randn(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)

        # Forward propagation
        out = encoder(x)
        self.assertEqual(out.shape, (1, TOKEN_TIME_STEPS, MODEL_DIM))
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.isfinite(out).all())

        # Scalar loss and backward propagation
        loss = out.sum()
        loss.backward()

        trainable_params = [p for p in encoder.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient")
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Parameter {name} has non-finite gradient",
                )

    def test_forward_batch_size_greater_than_one(self) -> None:
        for batch_size in (2, 4):
            x = torch.randn(batch_size, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
            self.encoder.eval()
            with torch.no_grad():
                out = self.encoder(x)
            self.assertEqual(out.shape, (batch_size, TOKEN_TIME_STEPS, MODEL_DIM))
            self.assertEqual(out.dtype, torch.float32)

    def test_input_validation_invalid_rank(self) -> None:
        invalid_shapes = [
            (EXPECTED_TIME_STEPS,),  # 1D
            (EXPECTED_TIME_STEPS, EXPECTED_CHANNELS),  # 2D
            (1, 1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS),  # 4D
        ]
        for shape in invalid_shapes:
            x = torch.randn(*shape, dtype=torch.float32)
            with self.assertRaises(ValueError):
                self.encoder(x)

    def test_input_validation_invalid_dimensions(self) -> None:
        invalid_shapes = [
            (2, EXPECTED_TIME_STEPS - 1, EXPECTED_CHANNELS),
            (2, EXPECTED_TIME_STEPS + 1, EXPECTED_CHANNELS),
            (2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS - 1),
            (2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS + 1),
        ]
        for shape in invalid_shapes:
            x = torch.randn(*shape, dtype=torch.float32)
            with self.assertRaises(ValueError):
                self.encoder(x)

    def test_input_validation_empty_batch(self) -> None:
        x = torch.empty(0, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.encoder(x)

    def test_input_validation_invalid_dtype(self) -> None:
        invalid_dtypes = [torch.float64, torch.float16, torch.int32, torch.int64]
        for dtype in invalid_dtypes:
            x = torch.zeros((1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS), dtype=dtype)
            with self.assertRaises(TypeError):
                self.encoder(x)

    def test_input_validation_non_tensor(self) -> None:
        x_np = np.zeros((1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS), dtype=np.float32)
        with self.assertRaises(TypeError):
            validate_eeg_input(x_np)
        with self.assertRaises(TypeError):
            self.encoder(x_np)  # type: ignore[arg-type]

    def test_input_validation_nan_and_inf(self) -> None:
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        x[0, 10, 5] = float("nan")
        with self.assertRaises(ValueError):
            self.encoder(x)

        x_inf = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        x_inf[1, 20, 15] = float("inf")
        with self.assertRaises(ValueError):
            self.encoder(x_inf)

        x_neginf = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        x_neginf[1, 20, 15] = float("-inf")
        with self.assertRaises(ValueError):
            self.encoder(x_neginf)

    def test_exact_intermediate_tensor_shapes(self) -> None:
        for b in (1, 3):
            x = torch.randn(b, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
            self.encoder.eval()
            with torch.no_grad():
                stages = self.encoder.forward_stages(x)

            self.assertEqual(stages["stage0_input"].shape, (b, 385, 56))
            self.assertEqual(stages["stage1_unsqueeze"].shape, (b, 1, 385, 56))
            self.assertEqual(stages["stage2_conv2d"].shape, (b, 16, 97, 56))
            self.assertEqual(stages["stage3_permute_flatten"].shape, (b, 97, 896))
            self.assertEqual(stages["stage4_projection"].shape, (b, 97, 64))
            self.assertEqual(stages["stage5_pos_encoding"].shape, (b, 97, 64))
            self.assertEqual(stages["stage6_block1"].shape, (b, 97, 64))
            self.assertEqual(stages["stage6_block2"].shape, (b, 97, 64))
            self.assertEqual(stages["output"].shape, (b, 97, 64))

    def test_deterministic_positional_encoding(self) -> None:
        pe1 = build_sinusoidal_positional_encoding(TOKEN_TIME_STEPS, MODEL_DIM)
        pe2 = build_sinusoidal_positional_encoding(TOKEN_TIME_STEPS, MODEL_DIM)

        self.assertEqual(pe1.shape, (1, TOKEN_TIME_STEPS, MODEL_DIM))
        self.assertTrue(torch.equal(pe1, pe2))
        self.assertTrue(torch.isfinite(pe1).all())

        # Exact values check for pos=0: sin(0)=0 and cos(0)=1
        self.assertAlmostEqual(float(pe1[0, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(pe1[0, 0, 1]), 1.0, places=6)

        pos_module = SinusoidalPositionalEncoding(TOKEN_TIME_STEPS, MODEL_DIM)
        # Verify no trainable parameters are introduced by positional encoding
        self.assertEqual(len(list(pos_module.parameters())), 0)
        self.assertEqual(sum(p.numel() for p in pos_module.parameters()), 0)

        # Forward determinism on module
        toks = torch.zeros(2, TOKEN_TIME_STEPS, MODEL_DIM, dtype=torch.float32)
        out1 = pos_module(toks)
        out2 = pos_module(toks)
        self.assertTrue(torch.equal(out1, out2))
        self.assertTrue(torch.equal(out1[0], pe1[0]))

    def test_finite_outputs(self) -> None:
        x = torch.randn(4, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        self.encoder.eval()
        with torch.no_grad():
            out = self.encoder(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_finite_gradients(self) -> None:
        self.encoder.train()
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        out = self.encoder(x)
        loss = out.sum()
        loss.backward()

        trainable_params = [p for p in self.encoder.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)

        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient")
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Parameter {name} has non-finite gradient",
                )

    def test_evaluation_mode_determinism(self) -> None:
        self.encoder.eval()
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)

        # Record initial BatchNorm running statistics
        bn_stats_before = []
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm1d):
                bn_stats_before.append(
                    (module.running_mean.clone(), module.running_var.clone())
                )

        with torch.no_grad():
            out1 = self.encoder(x)
            out2 = self.encoder(x)

        # Forward passes in eval() mode must produce identical outputs
        self.assertTrue(torch.equal(out1, out2))

        # BatchNorm running statistics must NOT change during eval() mode
        bn_idx = 0
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm1d):
                mean_before, var_before = bn_stats_before[bn_idx]
                self.assertTrue(torch.equal(module.running_mean, mean_before))
                self.assertTrue(torch.equal(module.running_var, var_before))
                bn_idx += 1

    def test_train_mode_dropout_variation(self) -> None:
        # In train mode, dropout causes outputs to vary across stochastic passes
        self.encoder.train()
        x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        out1 = self.encoder(x)
        out2 = self.encoder(x)
        self.assertFalse(torch.equal(out1, out2))

    def test_actual_instantiated_parameter_count(self) -> None:
        # Tokenizer Conv2D: 16 * 1 * 15 * 1 + 16 = 256
        conv_params = sum(p.numel() for p in self.encoder.tokenizer.conv.parameters())
        self.assertEqual(conv_params, 256)

        # Tokenizer Projection Linear: 896 * 64 + 64 = 57,408
        proj_params = sum(p.numel() for p in self.encoder.tokenizer.projection.parameters())
        self.assertEqual(proj_params, 57408)

        # Tokenizer LayerNorm: 2 * 64 = 128
        norm_params = sum(p.numel() for p in self.encoder.tokenizer.norm.parameters())
        self.assertEqual(norm_params, 128)

        # Positional Encoding: 0
        pe_params = sum(p.numel() for p in self.encoder.pos_encoder.parameters())
        self.assertEqual(pe_params, 0)

        # Per Conformer Block:
        # FFN1: LN (128) + Linear (8320) + Linear (8256) = 16,704
        # MHSA: LN (128) + MHA (16640) = 16,768
        # Conv: LN (128) + pw1 (8192) + dw (1024) + bn (128) + pw2 (4160) = 13,632
        # FFN2: LN (128) + Linear (8320) + Linear (8256) = 16,704
        # Final LN: 128
        # Total per block = 63,936
        for block_idx, block in enumerate(self.encoder.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            self.assertEqual(
                block_params,
                63936,
                f"Block {block_idx} parameter count mismatch: expected 63,936 got {block_params}",
            )

        # Two blocks: 63,936 * 2 = 127,872
        blocks_params = sum(p.numel() for p in self.encoder.blocks.parameters())
        self.assertEqual(blocks_params, 127872)

        # Total encoder trainable parameters:
        # 256 + 57408 + 128 + 0 + 127872 = 185,664
        total_trainable_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        self.assertEqual(total_trainable_params, 185664)

        total_params = sum(p.numel() for p in self.encoder.parameters())
        self.assertEqual(total_params, 185664)

    def test_metadata_and_labels_remain_outside_model(self) -> None:
        # Verify model forward rejects unexpected arguments
        x = torch.randn(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        with self.assertRaises(TypeError):
            self.encoder(x, metadata={"trial_id": "test:0"})  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            self.encoder(x, labels=torch.tensor([0]))  # type: ignore[call-arg]

    def test_dataset_sample_integration_with_metadata_discarded(self) -> None:
        """Integration test with EEGTrialDataset without modifying Dataset contracts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            y_stim = np.zeros((4, 2), dtype=np.float64)
            y_stim[0] = (1, 1)
            with h5py.File(dataset_dir / "y_stim.mat", "w") as metadata_file:
                metadata_file.create_dataset("y_stim", data=y_stim)

            trials = np.zeros((385, 56, 2), dtype=np.float32)
            with h5py.File(dataset_dir / "d1.mat", "w") as eeg_file:
                eeg_file.create_dataset("d1", data=trials, dtype=np.float32)

            manifest = TrialManifest(
                dataset_dir,
                file_trial_counts=(2,),
                group_ranges=(("HC", 0, 2),),
            )
            dataset = EEGTrialDataset(manifest)
            self.assertEqual(len(dataset), 2)

            # Retrieve trial from dataset: returns (signal_tensor, metadata_dict)
            signal, metadata = dataset[0]
            self.assertIsInstance(signal, torch.Tensor)
            self.assertIsInstance(metadata, dict)
            self.assertIn("trial_id", metadata)

            # Pass signal to encoder, keeping metadata outside
            batch = signal.unsqueeze(0)  # Shape (1, 385, 56)
            self.encoder.eval()
            with torch.no_grad():
                out = self.encoder(batch)

            self.assertEqual(out.shape, (1, 97, 64))
            # Verify metadata was not altered or passed
            self.assertEqual(metadata["source_filename"], "d1.mat")


if __name__ == "__main__":
    unittest.main()
