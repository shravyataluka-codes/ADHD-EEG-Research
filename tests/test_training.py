"""Unit and integration tests for Sprint 5 training pipeline and validation."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if torch is not None:
    from adhd_eeg.classifier import CLASS_ORDER, NUM_CLASSES, EEGConformerClassifier
    from adhd_eeg.conformer import EXPECTED_CHANNELS, EXPECTED_TIME_STEPS
    from adhd_eeg.training import (
        AUTHORIZED_PROTOCOL_PURPOSE,
        EarlyStopping,
        EpochMetrics,
        NonFiniteLossError,
        TrainingConfig,
        TrainingGateError,
        build_optimizer,
        build_scheduler,
        compute_loss,
        evaluate_epoch,
        get_environment_metadata,
        load_checkpoint,
        resolve_device,
        run_synthetic_training,
        save_checkpoint,
        set_seed,
        train_one_epoch,
    )


@unittest.skipIf(torch is None, "Install the optional 'pytorch' extra to run training tests")
class TrainingPipelineTests(unittest.TestCase):
    """Test suite containing exactly 24 unit and integration tests for Sprint 5."""

    def setUp(self) -> None:
        set_seed(42)

    def test_01_training_config_defaults_and_validation(self) -> None:
        """Test 1: Verify TrainingConfig default values and range validations."""
        cfg = TrainingConfig()
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.max_epochs, 10)
        self.assertEqual(cfg.learning_rate, 5e-4)
        self.assertEqual(cfg.checkpoint_metric, "val_loss")
        self.assertEqual(cfg.device, "cpu")

        # max_epochs must be > 0
        with self.assertRaises(ValueError):
            TrainingConfig(max_epochs=0)
        with self.assertRaises(ValueError):
            TrainingConfig(max_epochs=-1)

        # batch_size must be > 0
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=0)

        # learning_rate must be > 0
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=-0.01)

        # betas range
        with self.assertRaises(ValueError):
            TrainingConfig(betas=(1.5, 0.999))

        # unsupported checkpoint metric
        with self.assertRaises(ValueError):
            TrainingConfig(checkpoint_metric="accuracy")

        # invalid device
        with self.assertRaises(ValueError):
            TrainingConfig(device="tpu")

        # unauthorized protocol_purpose raises TrainingGateError
        with self.assertRaises(TrainingGateError):
            TrainingConfig(protocol_purpose="REAL_COHORT_TRAINING")

        # invalid seed bounds
        with self.assertRaises(ValueError):
            TrainingConfig(seed=-1)
        with self.assertRaises(ValueError):
            TrainingConfig(seed=2**32)

        # set_seed rejects invalid seed bounds
        with self.assertRaises(ValueError):
            set_seed(-1)
        with self.assertRaises(ValueError):
            set_seed(2**32)

    def test_02_training_config_json_roundtrip(self) -> None:
        """Test 2: Verify TrainingConfig dictionary and JSON file serialization round-trip."""
        cfg = TrainingConfig(
            seed=123,
            batch_size=8,
            max_epochs=5,
            learning_rate=1e-3,
            lr_scheduler_type="cosine",
        )
        d = cfg.to_dict()
        cfg_reloaded = TrainingConfig.from_dict(d)
        self.assertEqual(cfg, cfg_reloaded)

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "config.json"
            cfg.to_json_file(json_path)
            cfg_from_file = TrainingConfig.from_json_file(json_path)
            self.assertEqual(cfg, cfg_from_file)

    def test_03_conformer_synthetic_json_loads(self) -> None:
        """Test 3: Verify configs/conformer_synthetic.json exists, loads cleanly, and validates."""
        config_path = Path(__file__).resolve().parents[1] / "configs" / "conformer_synthetic.json"
        self.assertTrue(config_path.is_file(), f"File not found: {config_path}")

        cfg = TrainingConfig.from_json_file(config_path)
        self.assertEqual(cfg.protocol_purpose, "SYNTHETIC_SMOKE_TEST_ONLY")
        self.assertEqual(cfg.max_epochs, 3)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.device, "cpu")

    def test_04_device_resolution_cpu_and_auto(self) -> None:
        """Test 4: Verify resolve_device for 'cpu' and 'auto' on non-CUDA environments."""
        dev_cpu = resolve_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

        with patch("torch.cuda.is_available", return_value=False):
            dev_auto = resolve_device("auto")
            self.assertEqual(dev_auto.type, "cpu")

    def test_05_device_resolution_cuda_unavailable_raises(self) -> None:
        """Test 5: Verify resolve_device('cuda') raises RuntimeError when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_device("cuda")
            self.assertIn("CUDA requested via device='cuda' but CUDA is not available", str(ctx.exception))

    def test_06_set_seed_deterministic_replay(self) -> None:
        """Test 6: Verify deterministic model initialization replay within fixed environment."""
        set_seed(12345)
        model1 = EEGConformerClassifier()
        w1 = next(model1.parameters()).detach().clone()

        set_seed(12345)
        model2 = EEGConformerClassifier()
        w2 = next(model2.parameters()).detach().clone()

        self.assertTrue(torch.equal(w1, w2))

    def test_07_compute_loss_contract(self) -> None:
        """Test 7: Verify compute_loss calculates finite scalar float32 CrossEntropyLoss."""
        logits = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        loss = compute_loss(logits, targets)

        self.assertEqual(loss.dim(), 0)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_08_compute_loss_rejects_invalid_targets(self) -> None:
        """Test 8: Verify compute_loss rejects out-of-range, non-integer, or shape-mismatched targets."""
        logits = torch.randn(2, NUM_CLASSES, dtype=torch.float32)

        # Out-of-range targets (NUM_CLASSES is 3, valid are 0, 1, 2)
        with self.assertRaises(ValueError):
            compute_loss(logits, torch.tensor([0, 3], dtype=torch.long))
        with self.assertRaises(ValueError):
            compute_loss(logits, torch.tensor([-1, 1], dtype=torch.long))

        # Non-integer dtype
        with self.assertRaises(ValueError):
            compute_loss(logits, torch.tensor([0.0, 1.0], dtype=torch.float32))

        # Mismatched shape
        with self.assertRaises(ValueError):
            compute_loss(logits, torch.tensor([[0], [1]], dtype=torch.long))

        # Empty batch rejection
        with self.assertRaises(ValueError):
            compute_loss(torch.empty(0, NUM_CLASSES, dtype=torch.float32), torch.empty(0, dtype=torch.long))
        with self.assertRaises(ValueError):
            compute_loss(logits, torch.empty(0, dtype=torch.long))

    def test_09_train_one_epoch_updates_weights_and_return_types(self) -> None:
        """Test 9: Verify train_one_epoch updates parameters and returns properly typed metrics."""
        model = EEGConformerClassifier()
        signals = torch.randn(4, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        loader = DataLoader(TensorDataset(signals, targets), batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        initial_param = next(model.parameters()).detach().clone()
        metrics = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))

        updated_param = next(model.parameters()).detach().clone()
        self.assertFalse(torch.equal(initial_param, updated_param))

        self.assertIsInstance(metrics["loss"], float)
        self.assertIsInstance(metrics["accuracy"], float)
        self.assertIsInstance(metrics["num_samples"], int)
        self.assertEqual(metrics["num_samples"], 4)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)

    def test_10_train_one_epoch_rejects_non_finite_inputs(self) -> None:
        """Test 10: Verify train_one_epoch rejects NaN/Inf signal inputs via validate_eeg_input."""
        model = EEGConformerClassifier()
        signals = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        signals[0, 10, 10] = float("nan")
        targets = torch.tensor([0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(signals, targets), batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with self.assertRaises(ValueError):
            train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))

    def test_11_train_one_epoch_rejects_non_finite_gradients_clipping(self) -> None:
        """Test 11: Verify non-finite gradients during clipping raise NonFiniteLossError and prevent optimizer.step."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(56, 3)

            def forward(self, x):
                return self.linear(x.mean(dim=1))

        model = DummyModel()
        signals = torch.randn(2, EXPECTED_TIME_STEPS, 56, dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        loader = [(signals, targets)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Inject real NaN into weight gradient during backward to test actual clip_grad_norm_ behavior
        model.linear.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))

        initial_weights = model.linear.weight.detach().clone()
        with patch.object(optimizer, "step") as mock_step:
            with self.assertRaises(NonFiniteLossError) as ctx:
                train_one_epoch(model, loader, optimizer, device=torch.device("cpu"), grad_clip_norm=1.0)
            self.assertIn("Non-finite gradient encountered during clipping", str(ctx.exception))
            mock_step.assert_not_called()

        # Confirm model parameters were not modified
        self.assertTrue(torch.equal(initial_weights, model.linear.weight))

    def test_12_train_one_epoch_rejects_non_finite_gradients_unclipped(self) -> None:
        """Test 12: Verify explicit finite guard catches non-finite gradient when clipping is disabled."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(56, 3)

            def forward(self, x):
                return self.linear(x.mean(dim=1))

        model = DummyModel()
        signals = torch.randn(2, EXPECTED_TIME_STEPS, 56)
        targets = torch.tensor([0, 1], dtype=torch.long)

        class FakeLoader:
            def __iter__(self):
                yield signals, targets

            def __len__(self):
                return 1

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Inject NaN into gradient using PyTorch tensor hook during backward
        model.linear.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))

        with self.assertRaises(NonFiniteLossError) as ctx:
            train_one_epoch(model, FakeLoader(), optimizer, device=torch.device("cpu"), grad_clip_norm=None)
        self.assertIn("Non-finite gradient detected before optimizer step", str(ctx.exception))

    def test_13_gradient_clipping_bounds(self) -> None:
        """Test 13: Verify gradient clipping bounds gradient norm to specified threshold."""
        model = nn.Linear(10, 3)
        x = torch.randn(4, 10) * 100.0
        y = torch.tensor([0, 1, 2, 0])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        total_norm_before = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2
        ).item()
        self.assertGreater(total_norm_before, 1.0)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        total_norm_after = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2
        ).item()
        self.assertAlmostEqual(total_norm_after, 1.0, places=4)

    def test_14_validation_preserves_model_state_and_batchnorm(self) -> None:
        """Test 14: Verify evaluate_epoch preserves model parameters and BatchNorm running statistics bitwise."""
        model = EEGConformerClassifier()
        signals = torch.randn(4, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        loader = DataLoader(TensorDataset(signals, targets), batch_size=2)

        # Record parameter states and BatchNorm buffers
        params_before = [p.detach().clone() for p in model.parameters()]
        buffers_before = [b.detach().clone() for b in model.buffers()]

        metrics = evaluate_epoch(model, loader, device=torch.device("cpu"))

        params_after = [p.detach().clone() for p in model.parameters()]
        buffers_after = [b.detach().clone() for b in model.buffers()]

        for p_b, p_a in zip(params_before, params_after):
            self.assertTrue(torch.equal(p_b, p_a))

        for b_b, b_a in zip(buffers_before, buffers_after):
            self.assertTrue(torch.equal(b_b, b_a))

        self.assertIsInstance(metrics["loss"], float)
        self.assertIsInstance(metrics["accuracy"], float)
        self.assertEqual(metrics["num_samples"], 4)

    def test_15_validation_preserves_gradient_buffers_and_mode(self) -> None:
        """Test 15: Verify evaluate_epoch preserves existing param.grad buffers and restores caller mode."""
        model = EEGConformerClassifier()
        model.train()

        # Create dummy gradients
        dummy_signals = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        dummy_targets = torch.tensor([0, 1], dtype=torch.long)
        loss = compute_loss(model(dummy_signals), dummy_targets)
        loss.backward()

        grad_before = next(p for p in model.parameters() if p.grad is not None).grad.detach().clone()

        # Run evaluate_epoch
        val_signals = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        val_targets = torch.tensor([1, 2], dtype=torch.long)
        val_loader = DataLoader(TensorDataset(val_signals, val_targets), batch_size=2)

        evaluate_epoch(model, val_loader, device=torch.device("cpu"))

        # Check caller's mode is restored
        self.assertTrue(model.training)

        # Check existing gradient was not cleared
        grad_after = next(p for p in model.parameters() if p.grad is not None).grad.detach().clone()
        self.assertTrue(torch.equal(grad_before, grad_after))

    def test_16_early_stopping_behavior(self) -> None:
        """Test 16: Verify EarlyStopping triggers stop=True after exact patience steps without improvement."""
        es = EarlyStopping(patience=3, min_delta=1e-4)
        self.assertFalse(es.step(1.0))
        self.assertTrue(es.is_improved)

        self.assertFalse(es.step(0.99995))  # Improvement less than min_delta
        self.assertFalse(es.is_improved)
        self.assertEqual(es.counter, 1)

        self.assertFalse(es.step(1.0))
        self.assertEqual(es.counter, 2)

        # 3rd non-improving step triggers stop
        self.assertTrue(es.step(1.0))
        self.assertEqual(es.counter, 3)

    def test_17_scheduler_stepping_and_t_max(self) -> None:
        """Test 17: Verify CosineAnnealingLR uses T_max=max_epochs and steps per epoch."""
        model = nn.Linear(10, 3)
        cfg = TrainingConfig(max_epochs=4, learning_rate=1e-3, lr_scheduler_type="cosine", lr_min=1e-5)
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg)

        self.assertIsNotNone(sched)
        self.assertEqual(sched.T_max, 4)
        self.assertEqual(sched.eta_min, 1e-5)

        initial_lr = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        stepped_lr = opt.param_groups[0]["lr"]
        self.assertNotEqual(initial_lr, stepped_lr)

    def test_18_checkpoint_roundtrip_restoration(self) -> None:
        """Test 18: Verify atomic save and strict reload of model, optimizer, scheduler, and metadata."""
        model = EEGConformerClassifier()
        cfg = TrainingConfig(seed=42, max_epochs=4, learning_rate=1e-3, lr_scheduler_type="cosine")
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg)

        # Execute training step to populate optimizer state (momentum, step) and scheduler state
        dummy_x = torch.randn(2, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        dummy_y = torch.tensor([0, 1], dtype=torch.long)
        opt.zero_grad()
        loss = compute_loss(model(dummy_x), dummy_y)
        loss.backward()
        opt.step()
        sched.step()

        saved_lr = opt.param_groups[0]["lr"]
        saved_sched_epoch = sched.last_epoch
        saved_step_keys = list(opt.state.keys())
        self.assertGreater(len(saved_step_keys), 0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_ckpt.pt"
            save_checkpoint(
                filepath=ckpt_path,
                model=model,
                config=cfg,
                epoch=1,
                step=10,
                optimizer=opt,
                scheduler=sched,
                val_loss=0.45,
                val_accuracy=0.85,
                is_best=True,
            )

            new_model = EEGConformerClassifier()
            new_opt = build_optimizer(new_model, cfg)
            new_sched = build_scheduler(new_opt, cfg)
            payload = load_checkpoint(ckpt_path, new_model, optimizer=new_opt, scheduler=new_sched)

            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["step"], 10)
            self.assertEqual(payload["val_loss"], 0.45)
            self.assertEqual(tuple(payload["class_order"]), CLASS_ORDER)

            # Verify optimizer and scheduler restoration
            self.assertEqual(len(new_opt.state), len(opt.state))
            self.assertEqual(new_opt.param_groups[0]["lr"], saved_lr)
            self.assertEqual(new_sched.last_epoch, saved_sched_epoch)

            # Compare forward passes
            x = torch.randn(1, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS)
            model.eval()
            new_model.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), new_model(x)))

    def test_19_checkpoint_disabled_when_dir_none(self) -> None:
        """Test 19: Verify checkpoint_dir=None explicitly disables checkpoint creation."""
        cfg = TrainingConfig(max_epochs=1, batch_size=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            orig_cwd = Path.cwd()
            res = run_synthetic_training(
                config=cfg,
                checkpoint_dir=None,
                num_train_samples=4,
                num_val_samples=2,
            )
            self.assertEqual(len(res["history"]), 1)
            # Assert no checkpoint files were created in cwd or temp dir
            cwd_checkpoints = list(orig_cwd.glob("checkpoint_*.pt"))
            self.assertEqual(cwd_checkpoints, [])
            tmp_checkpoints = list(Path(tmp_dir).glob("**/*"))
            self.assertEqual(tmp_checkpoints, [])

    def test_20_checkpoint_without_validation(self) -> None:
        """Test 20: Verify training without validation saves last checkpoint only, not best."""
        cfg = TrainingConfig(max_epochs=1, batch_size=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_synthetic_training(
                config=cfg,
                checkpoint_dir=tmp_dir,
                num_train_samples=4,
                num_val_samples=None,
            )
            self.assertIsNone(res["best_epoch"])
            self.assertIsNone(res["best_val_loss"])

            ckpt_dir = Path(tmp_dir)
            self.assertTrue((ckpt_dir / "checkpoint_last.pt").is_file())
            self.assertFalse((ckpt_dir / "checkpoint_best.pt").is_file())

    def test_21_checkpoint_rejects_class_order_mismatch(self) -> None:
        """Test 21: Verify load_checkpoint raises ValueError if saved class order differs from CLASS_ORDER."""
        model = EEGConformerClassifier()
        cfg = TrainingConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "bad_order.pt"
            save_checkpoint(ckpt_path, model, cfg, epoch=1, step=1)

            # Tamper with class order
            payload = torch.load(ckpt_path, weights_only=False)
            payload["class_order"] = ["ADHD", "HC", "ADD"]
            torch.save(payload, ckpt_path)

            new_model = EEGConformerClassifier()
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(ckpt_path, new_model)
            self.assertIn("Checkpoint class order mismatch", str(ctx.exception))

    def test_22_run_synthetic_training_sample_count_validation(self) -> None:
        """Test 22: Verify sample count validation rejects non-positive, non-integer, and boolean values."""
        cfg = TrainingConfig()

        # Reject non-positive train samples
        with self.assertRaises(ValueError):
            run_synthetic_training(cfg, num_train_samples=0)
        with self.assertRaises(ValueError):
            run_synthetic_training(cfg, num_train_samples=-4)

        # Reject booleans (bool is subclass of int)
        with self.assertRaises(ValueError):
            run_synthetic_training(cfg, num_train_samples=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            run_synthetic_training(cfg, num_val_samples=True)  # type: ignore[arg-type]

        # Reject non-integer train/val samples
        with self.assertRaises(ValueError):
            run_synthetic_training(cfg, num_train_samples=4.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            run_synthetic_training(cfg, num_val_samples=0)
        with self.assertRaises(ValueError):
            run_synthetic_training(cfg, num_val_samples=-2)

    def test_23_run_synthetic_training_full_conformer_pipeline(self) -> None:
        """Test 23: Verify run_synthetic_training executes full EEGConformerClassifier end-to-end and restores checkpoints."""
        cfg = TrainingConfig(
            max_epochs=2,
            batch_size=4,
            learning_rate=5e-4,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_synthetic_training(
                config=cfg,
                checkpoint_dir=tmp_dir,
                num_train_samples=8,
                num_val_samples=4,
            )

            self.assertEqual(len(res["history"]), 2)
            self.assertIn("train_loss", res["history"][0])
            self.assertIn("val_loss", res["history"][0])
            self.assertIsNotNone(res["best_epoch"])
            self.assertIsNotNone(res["best_val_loss"])

            ckpt_dir = Path(tmp_dir)
            last_path = ckpt_dir / "checkpoint_last.pt"
            best_path = ckpt_dir / "checkpoint_best.pt"
            self.assertTrue(last_path.is_file())
            self.assertTrue(best_path.is_file())

            # Load and verify restoration of both checkpoints
            restored_model_last = EEGConformerClassifier()
            restored_opt_last = build_optimizer(restored_model_last, cfg)
            payload_last = load_checkpoint(last_path, restored_model_last, optimizer=restored_opt_last)
            self.assertEqual(payload_last["epoch"], 2)
            self.assertIn("model_state_dict", payload_last)
            self.assertIn("optimizer_state_dict", payload_last)

            restored_model_best = EEGConformerClassifier()
            payload_best = load_checkpoint(best_path, restored_model_best)
            self.assertEqual(payload_best["epoch"], res["best_epoch"])
            self.assertEqual(payload_best["val_loss"], res["best_val_loss"])
            self.assertTrue(payload_best["is_best"])

    def test_24_deterministic_synthetic_convergence(self) -> None:
        """Test 24: Engineering smoke test verifying >= 50% loss reduction using EEGConformerClassifier and train_one_epoch."""
        set_seed(42)
        model = EEGConformerClassifier()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Fixed deterministic synthetic input and target
        signals = torch.randn(4, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        loader = DataLoader(TensorDataset(signals, targets), batch_size=4, shuffle=False)

        # First epoch metrics via train_one_epoch
        metrics_e1 = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))
        initial_loss = metrics_e1["loss"]

        # Train for 9 more epochs using train_one_epoch
        latest_metrics = metrics_e1
        for _ in range(9):
            latest_metrics = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))

        final_loss = latest_metrics["loss"]

        # Assert at least 50% loss drop achieved using train_one_epoch
        self.assertLess(final_loss, 0.5 * initial_loss)


if __name__ == "__main__":
    unittest.main()
