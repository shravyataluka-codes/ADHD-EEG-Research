"""Training pipeline, validation engine, and reproducibility controls for EEG Conformer.

This module implements the Sprint 5 specification of `docs/SPRINT_3_TO_8_EXECUTION_PLAN.md`.
All training procedures operate strictly within a synthetic execution boundary.
No real ADHD diagnostic labels, participant mappings, or unapproved preprocessing
are used or authorized by this module.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset, TensorDataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

from .classifier import CLASS_ORDER, NUM_CLASSES, EEGConformerClassifier
from .conformer import EXPECTED_CHANNELS, EXPECTED_TIME_STEPS, validate_eeg_input

_TORCH_MISSING_MSG = (
    "PyTorch is required for training modules. "
    "Install the optional 'pytorch' extra: pip install .[pytorch]"
)


AUTHORIZED_PROTOCOL_PURPOSE: str = "SYNTHETIC_SMOKE_TEST_ONLY"


class NonFiniteLossError(FloatingPointError):
    """Raised when loss or gradients evaluate to NaN or Inf."""


class TrainingGateError(RuntimeError):
    """Raised when attempting to execute real cohort training prior to Sprint 6 gates."""


class EpochMetrics(TypedDict):
    loss: float
    accuracy: float
    num_samples: int


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration container for training and validation runs."""

    seed: int = 42
    batch_size: int = 16
    max_epochs: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0
    early_stopping_patience: Optional[int] = 5
    early_stopping_min_delta: float = 1e-4
    checkpoint_metric: str = "val_loss"
    lr_scheduler_type: Optional[str] = None
    lr_step_size: int = 5
    lr_gamma: float = 0.5
    lr_min: float = 1e-6
    device: str = "cpu"
    protocol_purpose: str = AUTHORIZED_PROTOCOL_PURPOSE

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate parameter ranges and types, failing closed on invalid configurations."""
        if self.protocol_purpose != AUTHORIZED_PROTOCOL_PURPOSE:
            raise TrainingGateError(
                f"Unauthorized protocol_purpose: {self.protocol_purpose!r}. "
                f"Real cohort training is blocked until Sprint 6 gates. "
                f"Only {AUTHORIZED_PROTOCOL_PURPOSE!r} is permitted."
            )

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"seed must be an integer, got {type(self.seed).__name__}")
        if not (0 <= self.seed < 2**32):
            raise ValueError(f"seed must be in [0, 2**32 - 1], got {self.seed}")

        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}")

        if isinstance(self.max_epochs, bool) or not isinstance(self.max_epochs, int) or self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be a positive integer (> 0), got {self.max_epochs!r}")

        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be a positive float, got {self.learning_rate!r}")

        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay!r}")

        if not isinstance(self.betas, (tuple, list)) or len(self.betas) != 2:
            raise ValueError(f"betas must be a tuple of 2 floats, got {self.betas!r}")
        b1, b2 = self.betas
        if isinstance(b1, bool) or not isinstance(b1, (int, float)) or not (0.0 <= b1 < 1.0):
            raise ValueError(f"betas[0] must be in [0, 1), got {b1!r}")
        if isinstance(b2, bool) or not isinstance(b2, (int, float)) or not (0.0 <= b2 < 1.0):
            raise ValueError(f"betas[1] must be in [0, 1), got {b2!r}")

        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)) or self.eps <= 0:
            raise ValueError(f"eps must be a positive float, got {self.eps!r}")

        if self.grad_clip_norm is not None:
            if isinstance(self.grad_clip_norm, bool) or not isinstance(self.grad_clip_norm, (int, float)) or self.grad_clip_norm <= 0:
                raise ValueError(f"grad_clip_norm must be positive or None, got {self.grad_clip_norm!r}")

        if self.early_stopping_patience is not None:
            if isinstance(self.early_stopping_patience, bool) or not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience <= 0:
                raise ValueError(f"early_stopping_patience must be a positive integer or None, got {self.early_stopping_patience!r}")

        if isinstance(self.early_stopping_min_delta, bool) or not isinstance(self.early_stopping_min_delta, (int, float)) or self.early_stopping_min_delta < 0:
            raise ValueError(f"early_stopping_min_delta must be non-negative, got {self.early_stopping_min_delta!r}")

        if self.checkpoint_metric != "val_loss":
            raise ValueError(f"Only checkpoint_metric='val_loss' is supported, got {self.checkpoint_metric!r}")

        if self.lr_scheduler_type is not None and self.lr_scheduler_type not in ("step", "cosine"):
            raise ValueError(f"lr_scheduler_type must be None, 'step', or 'cosine', got {self.lr_scheduler_type!r}")

        if isinstance(self.lr_step_size, bool) or not isinstance(self.lr_step_size, int) or self.lr_step_size <= 0:
            raise ValueError(f"lr_step_size must be a positive integer, got {self.lr_step_size!r}")

        if isinstance(self.lr_gamma, bool) or not isinstance(self.lr_gamma, (int, float)) or self.lr_gamma <= 0:
            raise ValueError(f"lr_gamma must be a positive float, got {self.lr_gamma!r}")

        if isinstance(self.lr_min, bool) or not isinstance(self.lr_min, (int, float)) or self.lr_min < 0:
            raise ValueError(f"lr_min must be non-negative, got {self.lr_min!r}")

        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {self.device!r}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary suitable for JSON serialization."""
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingConfig:
        """Create TrainingConfig from dictionary."""
        d = dict(data)
        if "betas" in d and isinstance(d["betas"], list):
            d["betas"] = tuple(d["betas"])
        d.pop("_comment", None)
        return cls(**d)

    def to_json_file(self, filepath: Path | str) -> None:
        """Serialize configuration to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json_file(cls, filepath: Path | str) -> TrainingConfig:
        """Deserialize configuration from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def resolve_device(device_str: str) -> Any:
    """Resolve device string to torch.device.

    Raises:
        RuntimeError: If device_str is 'cuda' but CUDA is not available.
        ValueError: If device_str is not 'cpu', 'cuda', or 'auto'.
    """
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)

    if device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via device='cuda' but CUDA is not available")
        return torch.device("cuda")
    elif device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Unknown device: {device_str!r}. Must be 'cpu', 'cuda', or 'auto'.")


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Set random seeds across random, numpy, and torch (CPU/CUDA)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed < 2**32):
        raise ValueError(f"seed must be an integer in [0, 2**32 - 1], got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initialization function deriving deterministic seed."""
    if torch is not None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def get_environment_metadata() -> Dict[str, Any]:
    """Capture environment and library audit metadata."""
    cuda_available = torch.cuda.is_available() if torch is not None else False
    device_name = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "Unknown CUDA Device"

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__ if torch is not None else None,
        "cuda_available": cuda_available,
        "cuda_device_name": device_name,
    }


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute CrossEntropyLoss from raw logits (B, 3) and integer targets (B,).

    Args:
        logits: Tensor of shape (B, 3), dtype float32.
        targets: Tensor of shape (B,), integer dtype, values in [0, 2].

    Returns:
        Scalar float32 loss tensor.
    """
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"targets must be a torch.Tensor, got {type(targets).__name__}")

    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must have shape (B, {NUM_CLASSES}), got {tuple(logits.shape)}")
    if logits.shape[0] == 0:
        raise ValueError("logits batch size must be positive, got 0")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must have shape (B,) matching logits batch size, got {tuple(targets.shape)}")
    if targets.shape[0] == 0:
        raise ValueError("targets batch size must be positive, got 0")

    if targets.dtype not in (torch.int64, torch.long, torch.int32):
        raise ValueError(f"targets must have integer dtype (torch.long), got {targets.dtype}")

    if targets.dtype != torch.long:
        targets = targets.long()

    if not torch.isfinite(logits).all():
        raise NonFiniteLossError("logits contain non-finite values (NaN or Inf)")

    if (targets < 0).any() or (targets >= NUM_CLASSES).any():
        raise ValueError(
            f"targets values must be in [0, {NUM_CLASSES - 1}], "
            f"got min={targets.min().item()}, max={targets.max().item()}"
        )

    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, targets)

    if not torch.isfinite(loss):
        raise NonFiniteLossError("Loss evaluated to non-finite value (NaN or Inf)")

    return loss


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Any:
    """Instantiate AdamW optimizer with configured parameters."""
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )


def build_scheduler(
    optimizer: Any,
    config: TrainingConfig,
) -> Optional[Any]:
    """Instantiate optional learning rate scheduler configured to step once per epoch."""
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)

    if config.lr_scheduler_type is None:
        return None
    elif config.lr_scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        )
    elif config.lr_scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.lr_min,
        )
    else:
        raise ValueError(f"Unknown lr_scheduler_type: {config.lr_scheduler_type!r}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Any,
    device: Optional[Any] = None,
    grad_clip_norm: Optional[float] = None,
) -> EpochMetrics:
    """Execute one training epoch over (signals, targets) batches."""
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)
    if device is None:
        device = torch.device("cpu")

    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(
                f"Batch must be a tuple/list of at least (signals, targets), got {type(batch).__name__}"
            )
        signals = batch[0]
        targets = batch[1]

        validate_eeg_input(signals)
        signals = signals.to(device)
        targets = targets.to(device)

        batch_size = signals.shape[0]

        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = compute_loss(logits, targets)

        loss.backward()

        # Gradient clipping with error_if_nonfinite=True
        if grad_clip_norm is not None:
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as err:
                raise NonFiniteLossError(f"Non-finite gradient encountered during clipping: {err}") from err

        # Explicit finite gradient guard before optimizer.step()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise NonFiniteLossError("Non-finite gradient detected before optimizer step")

        optimizer.step()

        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=-1)
        running_correct += (preds == targets).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Dataloader yielded 0 samples for training epoch")

    return {
        "loss": running_loss / total_samples,
        "accuracy": float(running_correct) / total_samples,
        "num_samples": total_samples,
    }


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[Any] = None,
) -> EpochMetrics:
    """Execute evaluation over (signals, targets) batches without modifying model state."""
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)
    if device is None:
        device = torch.device("cpu")

    prior_mode = model.training
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError(
                        f"Batch must be a tuple/list of at least (signals, targets), got {type(batch).__name__}"
                    )
                signals = batch[0]
                targets = batch[1]

                validate_eeg_input(signals)
                signals = signals.to(device)
                targets = targets.to(device)

                batch_size = signals.shape[0]
                logits = model(signals)
                loss = compute_loss(logits, targets)

                running_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=-1)
                running_correct += (preds == targets).sum().item()
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Dataloader yielded 0 samples for evaluation epoch")

        return {
            "loss": running_loss / total_samples,
            "accuracy": float(running_correct) / total_samples,
            "num_samples": total_samples,
        }
    finally:
        model.train(prior_mode)


class EarlyStopping:
    """Validation loss early stopping tracker."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter: int = 0
        self._is_improved: bool = False

    def step(self, current_loss: float) -> bool:
        """Update tracker with current validation loss. Return True if stopped."""
        if not math.isfinite(current_loss):
            raise NonFiniteLossError(f"EarlyStopping received non-finite loss: {current_loss}")

        if self.best_loss is None:
            self.best_loss = current_loss
            self._is_improved = True
            self.counter = 0
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self._is_improved = True
            self.counter = 0
            return False
        else:
            self._is_improved = False
            self.counter += 1
            return self.counter >= self.patience

    @property
    def is_improved(self) -> bool:
        """True if the most recent step represented an improvement."""
        return self._is_improved


def save_checkpoint(
    filepath: Path | str,
    model: nn.Module,
    config: TrainingConfig,
    epoch: int,
    step: int,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    val_loss: Optional[float] = None,
    val_accuracy: Optional[float] = None,
    is_best: bool = False,
) -> None:
    """Atomically write training checkpoint to disk."""
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)

    dest_path = Path(filepath)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".pt.tmp")

    payload = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config.to_dict(),
        "class_order": list(CLASS_ORDER),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "is_best": is_best,
        "environment": get_environment_metadata(),
    }

    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(dest_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def load_checkpoint(
    filepath: Path | str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Strictly load model state and restore optimizer/scheduler state dicts."""
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)
    if device is None:
        device = torch.device("cpu")

    src_path = Path(filepath)
    if not src_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {src_path}")

    checkpoint = torch.load(src_path, map_location=device, weights_only=False)

    saved_class_order = tuple(checkpoint.get("class_order", ()))
    if saved_class_order != CLASS_ORDER:
        raise ValueError(
            f"Checkpoint class order mismatch: saved {saved_class_order}, expected {CLASS_ORDER}"
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def run_synthetic_training(
    config: TrainingConfig,
    checkpoint_dir: Optional[Path | str] = None,
    num_train_samples: int = 32,
    num_val_samples: Optional[int] = 16,
) -> Dict[str, Any]:
    """Execute synthetic training using internally generated synthetic signals and targets.

    This function generates its own synthetic dataset internally. It does NOT accept external
    DataLoaders, preventing accidental ingestion of unverified real or candidate cohort data.

    Args:
        config: Validated TrainingConfig instance.
        checkpoint_dir: Directory to save checkpoints. If None, checkpoint writing is
            EXPLICITLY DISABLED; no files are written.
        num_train_samples: Number of synthetic training samples (must be > 0).
        num_val_samples: Number of synthetic validation samples (must be None or > 0).

    Returns:
        Dictionary containing history, best metrics, early stopping status, and audit metadata.
    """
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)

    config.validate()

    # Validate sample counts rejecting non-positive values, non-integers, and booleans
    if isinstance(num_train_samples, bool) or not isinstance(num_train_samples, int) or num_train_samples <= 0:
        raise ValueError(f"num_train_samples must be a positive integer, got {num_train_samples!r}")

    if num_val_samples is not None:
        if isinstance(num_val_samples, bool) or not isinstance(num_val_samples, int) or num_val_samples <= 0:
            raise ValueError(f"num_val_samples must be None or a positive integer, got {num_val_samples!r}")

    set_seed(config.seed)
    device = resolve_device(config.device)

    # Internally generate synthetic training data: (B, 385, 56) float32, targets in [0, 2]
    train_signals = torch.randn(num_train_samples, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
    train_targets = torch.randint(0, NUM_CLASSES, (num_train_samples,), dtype=torch.long)
    train_dataset = TensorDataset(train_signals, train_targets)

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, num_train_samples),
        shuffle=True,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )

    val_loader: Optional[DataLoader] = None
    if num_val_samples is not None:
        val_signals = torch.randn(num_val_samples, EXPECTED_TIME_STEPS, EXPECTED_CHANNELS, dtype=torch.float32)
        val_targets = torch.randint(0, NUM_CLASSES, (num_val_samples,), dtype=torch.long)
        val_dataset = TensorDataset(val_signals, val_targets)
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(config.batch_size, num_val_samples),
            shuffle=False,
        )

    model = EEGConformerClassifier()
    model.to(device)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    early_stopping: Optional[EarlyStopping] = None
    if val_loader is not None and config.early_stopping_patience is not None:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

    ckpt_dir_path: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir_path is not None:
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    stopped_early = False
    cumulative_step = 0

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        cumulative_step += len(train_loader)

        val_metrics: Optional[EpochMetrics] = None
        if val_loader is not None:
            val_metrics = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
            )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"] if val_metrics is not None else None,
            "val_accuracy": val_metrics["accuracy"] if val_metrics is not None else None,
            "lr": current_lr,
        }
        history.append(epoch_record)

        # Scheduler steps once per completed epoch
        if scheduler is not None:
            scheduler.step()

        # Checkpoint management & early stopping
        is_best = False
        if val_metrics is not None:
            val_loss = val_metrics["loss"]
            if best_val_loss is None or val_loss < best_val_loss - config.early_stopping_min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                is_best = True

            if ckpt_dir_path is not None:
                save_checkpoint(
                    filepath=ckpt_dir_path / "checkpoint_last.pt",
                    model=model,
                    config=config,
                    epoch=epoch,
                    step=cumulative_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_loss=val_loss,
                    val_accuracy=val_metrics["accuracy"],
                    is_best=False,
                )
                if is_best:
                    save_checkpoint(
                        filepath=ckpt_dir_path / "checkpoint_best.pt",
                        model=model,
                        config=config,
                        epoch=epoch,
                        step=cumulative_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        val_loss=val_loss,
                        val_accuracy=val_metrics["accuracy"],
                        is_best=True,
                    )

            if early_stopping is not None and early_stopping.step(val_loss):
                stopped_early = True
                break
        else:
            # When validation is absent, only save checkpoint_last.pt if checkpoint_dir is provided
            if ckpt_dir_path is not None:
                save_checkpoint(
                    filepath=ckpt_dir_path / "checkpoint_last.pt",
                    model=model,
                    config=config,
                    epoch=epoch,
                    step=cumulative_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_loss=None,
                    val_accuracy=None,
                    is_best=False,
                )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped_early,
        "environment": get_environment_metadata(),
        "config": config.to_dict(),
    }
