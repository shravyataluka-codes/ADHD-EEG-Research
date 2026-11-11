"""Registered model variants and matched synthetic engineering experiments.

All experiment data and labels are generated locally from the configured seed.
No EEG files, participant mappings, candidate metadata, or real labels are read.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import DataLoader, Subset, TensorDataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Subset = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

from .analysis import PredictionArtifact, PredictionRecord
from .classifier import (
    CLASS_ORDER,
    EEGConformerClassifier,
    EEGNetBaseline,
    EEGNetConfig,
    POOLING_MEAN,
)
from .conformer import EXPECTED_CHANNELS, EXPECTED_TIME_STEPS
from .evaluation import GroupedSplit, grouped_train_validation_test_split
from .training import (
    AUTHORIZED_PROTOCOL_PURPOSE,
    EarlyStopping,
    TrainingConfig,
    build_optimizer,
    build_scheduler,
    evaluate_epoch,
    resolve_device,
    seed_worker,
    set_seed,
    train_one_epoch,
)

_TORCH_MISSING_MSG = (
    "PyTorch is required for synthetic ablation experiments. "
    "Install the optional 'pytorch' extra: pip install .[pytorch]"
)


class ModelVariant(str, Enum):
    """The bounded Sprint 7 model registry."""

    CONFORMER_REFERENCE = "conformer_reference"
    CONFORMER_NO_ATTENTION = "conformer_no_attention"
    CONFORMER_NO_CONV = "conformer_no_conv"
    CONFORMER_MEAN_POOL = "conformer_mean_pool"
    EEGNET = "eegnet"


MODEL_VARIANT_REGISTRY: Mapping[str, str] = MappingProxyType(
    {
        ModelVariant.CONFORMER_REFERENCE.value: "Reference Conformer",
        ModelVariant.CONFORMER_NO_ATTENTION.value: "Conformer without self-attention",
        ModelVariant.CONFORMER_NO_CONV.value: "Conformer without local convolution",
        ModelVariant.CONFORMER_MEAN_POOL.value: "Conformer with mean-only pooling",
        ModelVariant.EEGNET.value: "EEGNet baseline",
    }
)


def build_model(variant: ModelVariant | str) -> nn.Module:
    """Construct exactly one registered model while preserving the reference default."""
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)
    try:
        selected = ModelVariant(variant)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unregistered model variant: {variant!r}") from error

    if selected is ModelVariant.CONFORMER_REFERENCE:
        return EEGConformerClassifier()
    if selected is ModelVariant.CONFORMER_NO_ATTENTION:
        return EEGConformerClassifier(use_self_attention=False)
    if selected is ModelVariant.CONFORMER_NO_CONV:
        return EEGConformerClassifier(use_conv_module=False)
    if selected is ModelVariant.CONFORMER_MEAN_POOL:
        return EEGConformerClassifier(pooling_mode=POOLING_MEAN)
    if selected is ModelVariant.EEGNET:
        return EEGNetBaseline(config=EEGNetConfig())
    raise AssertionError(f"registry entry has no model factory: {selected.value}")


@dataclass(frozen=True)
class MatchedSyntheticConfig:
    """Frozen synthetic fixture and common training configuration."""

    training: TrainingConfig = TrainingConfig(max_epochs=3, batch_size=4)
    variants: tuple[str, ...] = tuple(MODEL_VARIANT_REGISTRY)
    participants_per_class: int = 5
    trials_per_participant: int = 2
    protocol_purpose: str = AUTHORIZED_PROTOCOL_PURPOSE

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.training, TrainingConfig):
            raise TypeError("training must be a TrainingConfig")
        self.training.validate()
        if self.protocol_purpose != AUTHORIZED_PROTOCOL_PURPOSE:
            raise ValueError(
                f"protocol_purpose must be {AUTHORIZED_PROTOCOL_PURPOSE!r}"
            )
        if not isinstance(self.variants, (tuple, list)) or not self.variants:
            raise ValueError("variants must be a non-empty sequence")
        normalized: list[str] = []
        for variant in self.variants:
            try:
                normalized.append(ModelVariant(variant).value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"unregistered model variant: {variant!r}") from error
        if len(set(normalized)) != len(normalized):
            raise ValueError("variants must not contain duplicates")
        object.__setattr__(self, "variants", tuple(normalized))
        if (
            isinstance(self.participants_per_class, bool)
            or not isinstance(self.participants_per_class, int)
            or self.participants_per_class < 3
        ):
            raise ValueError("participants_per_class must be an integer >= 3")
        if (
            isinstance(self.trials_per_participant, bool)
            or not isinstance(self.trials_per_participant, int)
            or self.trials_per_participant <= 0
        ):
            raise ValueError("trials_per_participant must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "participants_per_class": self.participants_per_class,
            "protocol_purpose": self.protocol_purpose,
            "training": self.training.to_dict(),
            "trials_per_participant": self.trials_per_participant,
            "variants": list(self.variants),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MatchedSyntheticConfig:
        required = {
            "participants_per_class",
            "protocol_purpose",
            "training",
            "trials_per_participant",
            "variants",
        }
        if not isinstance(value, Mapping):
            raise TypeError("matched synthetic configuration must be a mapping")
        normalized = dict(value)
        normalized.pop("_comment", None)
        if set(normalized) != required:
            raise ValueError(
                f"configuration fields must be exactly {sorted(required)!r}"
            )
        training = normalized["training"]
        if not isinstance(training, TrainingConfig):
            if not isinstance(training, Mapping):
                raise TypeError("training must be a TrainingConfig or mapping")
            training = TrainingConfig.from_dict(dict(training))
        variants = normalized["variants"]
        if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
            raise TypeError("variants must be a sequence")
        return cls(
            training=training,
            variants=tuple(variants),
            participants_per_class=normalized["participants_per_class"],
            trials_per_participant=normalized["trials_per_participant"],
            protocol_purpose=normalized["protocol_purpose"],
        )

    @classmethod
    def from_json_file(cls, filepath: Path | str) -> MatchedSyntheticConfig:
        with open(filepath, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


@dataclass(frozen=True)
class MatchedSyntheticFixture:
    """Generated tensors plus participant-disjoint synthetic split identifiers."""

    signals: Tensor
    targets: Tensor
    sample_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]
    split: GroupedSplit
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class MatchedVariantResult:
    """One model's validation-selected checkpoint and held-out predictions."""

    variant: str
    parameter_count: int
    trainable_parameter_count: int
    compute_cost_measurement: str
    selected_epoch: int
    selected_validation_loss: float
    training_history: tuple[Mapping[str, Any], ...]
    selection_partition: str
    predictions: PredictionArtifact


@dataclass(frozen=True)
class MatchedSyntheticExperimentResult:
    """Synthetic engineering result with a deterministic configuration manifest."""

    manifest: Mapping[str, Any]
    variants: tuple[MatchedVariantResult, ...]

    def manifest_json(self) -> str:
        return json.dumps(self.manifest, sort_keys=True, indent=2, allow_nan=False)


def _build_fixture(config: MatchedSyntheticConfig) -> MatchedSyntheticFixture:
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)
    config.validate()

    sample_ids: list[str] = []
    participant_ids: list[str] = []
    targets: list[int] = []
    partition_samples: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    partition_participants: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for class_index, class_name in enumerate(CLASS_ORDER):
        class_participants = [
            f"synthetic-fixture:participant-{class_name}-{index:03d}"
            for index in range(config.participants_per_class)
        ]
        class_samples: list[str] = []
        class_groups: list[str] = []
        class_targets: list[int] = []
        for participant_id in class_participants:
            for trial_index in range(config.trials_per_participant):
                sample_id = (
                    f"{participant_id}:trial-{trial_index:03d}"
                )
                sample_ids.append(sample_id)
                participant_ids.append(participant_id)
                targets.append(class_index)
                class_samples.append(sample_id)
                class_groups.append(participant_id)
                class_targets.append(class_index)
        class_split = grouped_train_validation_test_split(
            class_samples,
            class_groups,
            group_counts={
                "train": config.participants_per_class - 2,
                "validation": 1,
                "test": 1,
            },
            random_seed=config.training.seed,
            labels=class_targets,
        )
        for partition, split_samples, split_participants in (
            ("train", class_split.train_sample_ids, class_split.train_group_ids),
            (
                "validation",
                class_split.validation_sample_ids,
                class_split.validation_group_ids,
            ),
            ("test", class_split.test_sample_ids, class_split.test_group_ids),
        ):
            partition_samples[partition].extend(split_samples)
            partition_participants[partition].extend(split_participants)

    split = GroupedSplit(
        train_sample_ids=tuple(partition_samples["train"]),
        validation_sample_ids=tuple(partition_samples["validation"]),
        test_sample_ids=tuple(partition_samples["test"]),
        train_group_ids=tuple(partition_participants["train"]),
        validation_group_ids=tuple(partition_participants["validation"]),
        test_group_ids=tuple(partition_participants["test"]),
        class_counts=None,
    )
    train_groups = set(split.train_group_ids)
    validation_groups = set(split.validation_group_ids)
    test_groups = set(split.test_group_ids)
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise RuntimeError("synthetic participant split unexpectedly overlaps")
    if set(split.train_sample_ids) | set(split.validation_sample_ids) | set(split.test_sample_ids) != set(sample_ids):
        raise RuntimeError("synthetic sample split does not provide exact coverage")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.training.seed)
    signals = torch.randn(
        len(sample_ids),
        EXPECTED_TIME_STEPS,
        EXPECTED_CHANNELS,
        dtype=torch.float32,
        generator=generator,
    )
    target_tensor = torch.tensor(targets, dtype=torch.long)
    manifest = {
        "class_order": list(CLASS_ORDER),
        "configuration": config.to_dict(),
        "data_inclusion": "same synthetic sample IDs for every registered variant",
        "input_contract": [EXPECTED_TIME_STEPS, EXPECTED_CHANNELS],
        "model_input_fields": ["signal_tensor"],
        "model_selection": {
            "criterion": "minimum validation cross-entropy loss",
            "eligible_partition": "validation",
            "test_partition_used_for_selection": False,
        },
        "participant_ids_by_partition": {
            partition: list(partition_participants[partition])
            for partition in ("train", "validation", "test")
        },
        "protocol_purpose": config.protocol_purpose,
        "result_kind": "synthetic engineering validation",
        "sample_ids_by_partition": {
            partition: list(partition_samples[partition])
            for partition in ("train", "validation", "test")
        },
        "splitter": "grouped_train_validation_test_split, independently per synthetic class",
        "trial_inclusion": "all generated trials are assigned by synthetic participant",
    }
    return MatchedSyntheticFixture(
        signals=signals,
        targets=target_tensor,
        sample_ids=tuple(sample_ids),
        participant_ids=tuple(participant_ids),
        split=split,
        manifest=manifest,
    )


def build_matched_synthetic_manifest(
    config: MatchedSyntheticConfig,
) -> dict[str, Any]:
    """Build deterministic split/config metadata without exposing tensor values."""
    return dict(_build_fixture(config).manifest)


def _indices_for_samples(
    fixture: MatchedSyntheticFixture, sample_ids: Sequence[str]
) -> list[int]:
    index_by_id = {sample_id: index for index, sample_id in enumerate(fixture.sample_ids)}
    return [index_by_id[sample_id] for sample_id in sample_ids]


def _predict_test_partition(
    model: nn.Module,
    fixture: MatchedSyntheticFixture,
    config: MatchedSyntheticConfig,
    device: Any,
    variant: str,
) -> PredictionArtifact:
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)
    indices = _indices_for_samples(fixture, fixture.split.test_sample_ids)
    dataset = TensorDataset(fixture.signals, fixture.targets)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=min(config.training.batch_size, len(indices)),
        shuffle=False,
    )
    model.eval()
    predictions: list[PredictionRecord] = []
    test_samples = set(fixture.split.test_sample_ids)
    with torch.no_grad():
        for signals, targets in loader:
            logits = model(signals.to(device))
            probabilities = torch.softmax(logits, dim=-1).cpu()
            for batch_index, probabilities_row in enumerate(probabilities):
                # The Subset is ordered by its original sample ID list.
                source_index = indices[len(predictions)]
                sample_id = fixture.sample_ids[source_index]
                if sample_id not in test_samples:
                    raise RuntimeError("test prediction escaped the held-out partition")
                predictions.append(
                    PredictionRecord(
                        sample_id=sample_id,
                        participant_id=fixture.participant_ids[source_index],
                        fold_id="synthetic-fixture:test",
                        seed=config.training.seed,
                        target=int(targets[batch_index].item()),
                        probabilities=tuple(
                            float(value) for value in probabilities_row.tolist()
                        ),
                    )
                )
    return PredictionArtifact(
        model_variant=variant,
        records=tuple(predictions),
        expected_sample_ids=tuple(fixture.split.test_sample_ids),
    )


def run_matched_synthetic_experiment(
    config: MatchedSyntheticConfig = MatchedSyntheticConfig(),
) -> MatchedSyntheticExperimentResult:
    """Train registered variants on one shared synthetic split and predict test once.

    Validation cross-entropy is the sole checkpoint-selection measure. The test
    subset is not constructed for evaluation until the selected validation
    checkpoint has been restored.
    """
    if torch is None:
        raise ImportError(_TORCH_MISSING_MSG)
    if not isinstance(config, MatchedSyntheticConfig):
        raise TypeError("config must be a MatchedSyntheticConfig")
    config.validate()
    fixture = _build_fixture(config)
    manifest = dict(fixture.manifest)
    results: list[MatchedVariantResult] = []
    train_indices = _indices_for_samples(fixture, fixture.split.train_sample_ids)
    validation_indices = _indices_for_samples(
        fixture, fixture.split.validation_sample_ids
    )
    training_dataset = TensorDataset(
        fixture.signals[train_indices], fixture.targets[train_indices]
    )
    validation_dataset = TensorDataset(
        fixture.signals[validation_indices], fixture.targets[validation_indices]
    )
    device = resolve_device(config.training.device)

    for variant in config.variants:
        set_seed(config.training.seed)
        model = build_model(variant).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        optimizer = build_optimizer(model, config.training)
        scheduler = build_scheduler(optimizer, config.training)
        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(config.training.seed)
        train_loader = DataLoader(
            training_dataset,
            batch_size=min(config.training.batch_size, len(train_indices)),
            shuffle=True,
            generator=train_generator,
            worker_init_fn=seed_worker,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=min(config.training.batch_size, len(validation_indices)),
            shuffle=False,
        )
        stopper = (
            EarlyStopping(
                config.training.early_stopping_patience,
                config.training.early_stopping_min_delta,
            )
            if config.training.early_stopping_patience is not None
            else None
        )
        best_state: dict[str, Tensor] | None = None
        best_loss = float("inf")
        best_epoch = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, config.training.max_epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                grad_clip_norm=config.training.grad_clip_norm,
            )
            validation_metrics = evaluate_epoch(
                model, validation_loader, device=device
            )
            validation_loss = validation_metrics["loss"]
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if scheduler is not None:
                scheduler.step()
            if stopper is not None and stopper.step(validation_loss):
                break

        if best_state is None:
            raise RuntimeError("synthetic model selection produced no checkpoint")
        model.load_state_dict(best_state, strict=True)
        # Test data is first passed to a loader here, after model selection.
        prediction_artifact = _predict_test_partition(
            model, fixture, config, device, variant
        )
        results.append(
            MatchedVariantResult(
                variant=variant,
                parameter_count=parameter_count,
                trainable_parameter_count=trainable_parameter_count,
                compute_cost_measurement="not_measured",
                selected_epoch=best_epoch,
                selected_validation_loss=best_loss,
                training_history=tuple(history),
                selection_partition="validation",
                predictions=prediction_artifact,
            )
        )
    return MatchedSyntheticExperimentResult(
        manifest=manifest, variants=tuple(results)
    )


__all__ = [
    "MODEL_VARIANT_REGISTRY",
    "MatchedSyntheticConfig",
    "MatchedSyntheticExperimentResult",
    "MatchedSyntheticFixture",
    "MatchedVariantResult",
    "ModelVariant",
    "build_matched_synthetic_manifest",
    "build_model",
    "run_matched_synthetic_experiment",
]
