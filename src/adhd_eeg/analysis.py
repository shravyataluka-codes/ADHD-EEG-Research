"""Synthetic engineering analysis for participant-level prediction artifacts.

The evaluation unit is one synthetic participant within one (seed, fold) group.
When a participant has multiple trials, their class-probability vectors are
averaged first and the participant receives one argmax prediction. Accuracy
uncertainty is estimated separately within each seed/fold using a participant-
level bootstrap stratified by true class, preserving the observed class counts.
Repeated predictions for one participant across seeds contribute to run
variability, not to the participant-level bootstrap sample size.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .classifier import CLASS_ORDER
from .training import AUTHORIZED_PROTOCOL_PURPOSE

_ARTIFACT_VERSION = 1
_ROW_FIELDS = frozenset(
    ("sample_id", "participant_id", "fold_id", "seed", "target", "probabilities")
)


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_class_index(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer class index")
    if not 0 <= value < len(CLASS_ORDER):
        raise ValueError(f"{field} must be in [0, {len(CLASS_ORDER) - 1}]")
    return value


@dataclass(frozen=True)
class PredictionRecord:
    """One synthetic trial's target and predicted class probabilities."""

    sample_id: str
    participant_id: str
    fold_id: str
    seed: int
    target: int
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        _require_text(self.participant_id, "participant_id")
        _require_text(self.fold_id, "fold_id")
        if not self.sample_id.startswith("synthetic-fixture:"):
            raise ValueError("sample_id must be explicitly tagged synthetic")
        if not self.participant_id.startswith("synthetic-fixture:"):
            raise ValueError("participant_id must be explicitly tagged synthetic")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32 - 1]")
        _require_class_index(self.target, "target")
        if not isinstance(self.probabilities, (tuple, list)):
            raise TypeError("probabilities must be a sequence")
        if len(self.probabilities) != len(CLASS_ORDER):
            raise ValueError(
                f"probabilities must follow {CLASS_ORDER!r} with "
                f"{len(CLASS_ORDER)} values"
            )
        normalized: list[float] = []
        for index, probability in enumerate(self.probabilities):
            if isinstance(probability, bool) or not isinstance(
                probability, (int, float)
            ):
                raise TypeError(f"probabilities[{index}] must be numeric")
            value = float(probability)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"probabilities[{index}] must be finite and in [0, 1]")
            normalized.append(value)
        if not math.isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1 in CLASS_ORDER")
        object.__setattr__(self, "probabilities", tuple(normalized))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PredictionRecord:
        if not isinstance(value, Mapping) or set(value) != _ROW_FIELDS:
            raise ValueError(f"prediction row fields must be exactly {sorted(_ROW_FIELDS)!r}")
        return cls(
            sample_id=value["sample_id"],
            participant_id=value["participant_id"],
            fold_id=value["fold_id"],
            seed=value["seed"],
            target=value["target"],
            probabilities=tuple(value["probabilities"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "participant_id": self.participant_id,
            "fold_id": self.fold_id,
            "seed": self.seed,
            "target": self.target,
            "probabilities": list(self.probabilities),
        }


@dataclass(frozen=True)
class PredictionArtifact:
    """Strict synthetic-only predictions sufficient for deterministic reanalysis."""

    model_variant: str
    records: tuple[PredictionRecord, ...]
    expected_sample_ids: tuple[str, ...]
    class_order: tuple[str, ...] = CLASS_ORDER
    protocol_purpose: str = AUTHORIZED_PROTOCOL_PURPOSE
    artifact_version: int = _ARTIFACT_VERSION

    def __post_init__(self) -> None:
        _require_text(self.model_variant, "model_variant")
        if not isinstance(self.records, (tuple, list)):
            raise TypeError("records must be a sequence")
        immutable_records = tuple(self.records)
        object.__setattr__(self, "records", immutable_records)
        if not isinstance(self.expected_sample_ids, (tuple, list)):
            raise TypeError("expected_sample_ids must be a sequence")
        immutable_sample_ids = tuple(self.expected_sample_ids)
        object.__setattr__(self, "expected_sample_ids", immutable_sample_ids)
        if not isinstance(self.class_order, (tuple, list)):
            raise TypeError("class_order must be a sequence")
        object.__setattr__(self, "class_order", tuple(self.class_order))
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be exactly {CLASS_ORDER!r}")
        if self.protocol_purpose != AUTHORIZED_PROTOCOL_PURPOSE:
            raise ValueError(
                f"protocol_purpose must be {AUTHORIZED_PROTOCOL_PURPOSE!r}"
            )
        if (
            isinstance(self.artifact_version, bool)
            or not isinstance(self.artifact_version, int)
            or self.artifact_version != _ARTIFACT_VERSION
        ):
            raise ValueError(f"unsupported artifact_version {self.artifact_version!r}")
        if not self.records:
            raise ValueError("records must contain at least one prediction")
        if not self.expected_sample_ids:
            raise ValueError("expected_sample_ids must define complete prediction coverage")
        if len(set(self.expected_sample_ids)) != len(self.expected_sample_ids):
            raise ValueError("expected_sample_ids must be unique")
        for sample_id in self.expected_sample_ids:
            if not isinstance(sample_id, str) or not sample_id.startswith(
                "synthetic-fixture:"
            ):
                raise ValueError("expected_sample_ids must be explicitly tagged synthetic")
        if any(not isinstance(record, PredictionRecord) for record in self.records):
            raise TypeError("records must contain PredictionRecord values")

        seen_rows: set[tuple[int, str, str]] = set()
        samples_by_seed: dict[int, set[str]] = defaultdict(set)
        fold_by_participant_seed: dict[tuple[int, str], str] = {}
        target_by_participant: dict[str, int] = {}
        participant_by_sample: dict[str, str] = {}
        for record in self.records:
            if record.sample_id not in self.expected_sample_ids:
                raise ValueError(
                    f"unexpected prediction sample {record.sample_id!r}"
                )
            previous_participant = participant_by_sample.setdefault(
                record.sample_id, record.participant_id
            )
            if previous_participant != record.participant_id:
                raise ValueError("a sample must map to one participant across seeds")
            row_key = (record.seed, record.fold_id, record.sample_id)
            if row_key in seen_rows:
                raise ValueError(f"duplicate prediction row: {row_key!r}")
            seen_rows.add(row_key)
            samples_by_seed[record.seed].add(record.sample_id)
            participant_seed = (record.seed, record.participant_id)
            previous_fold = fold_by_participant_seed.setdefault(
                participant_seed, record.fold_id
            )
            if previous_fold != record.fold_id:
                raise ValueError(
                    "a participant must appear in only one fold for each seed"
                )
            previous_target = target_by_participant.setdefault(
                record.participant_id, record.target
            )
            if previous_target != record.target:
                raise ValueError(
                    "a participant must have one target across seeds"
                )
        expected = set(self.expected_sample_ids)
        for seed, observed in samples_by_seed.items():
            if observed != expected:
                missing = sorted(expected - observed)
                raise ValueError(
                    f"prediction coverage is incomplete for seed {seed}: "
                    f"missing {missing!r}"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PredictionArtifact:
        required = {
            "artifact_version",
            "class_order",
            "expected_sample_ids",
            "model_variant",
            "protocol_purpose",
            "records",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"artifact fields must be exactly {sorted(required)!r}")
        rows = value["records"]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError("records must be a sequence")
        return cls(
            model_variant=value["model_variant"],
            records=tuple(PredictionRecord.from_mapping(row) for row in rows),
            expected_sample_ids=tuple(value["expected_sample_ids"]),
            class_order=tuple(value["class_order"]),
            protocol_purpose=value["protocol_purpose"],
            artifact_version=value["artifact_version"],
        )

    @classmethod
    def from_json_file(cls, filepath: Path | str) -> PredictionArtifact:
        with open(filepath, "r", encoding="utf-8") as file:
            value = json.load(file)
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "class_order": list(self.class_order),
            "expected_sample_ids": list(self.expected_sample_ids),
            "model_variant": self.model_variant,
            "protocol_purpose": self.protocol_purpose,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False)

    def to_json_file(self, filepath: Path | str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")


def _participant_predictions(
    artifact: PredictionArtifact,
) -> dict[tuple[int, str, str], tuple[int, int]]:
    grouped: dict[tuple[int, str, str], list[PredictionRecord]] = defaultdict(list)
    for record in artifact.records:
        grouped[(record.seed, record.fold_id, record.participant_id)].append(record)

    aggregated: dict[tuple[int, str, str], tuple[int, int]] = {}
    for key, records in grouped.items():
        target = records[0].target
        mean_probabilities = [
            sum(record.probabilities[class_index] for record in records)
            / len(records)
            for class_index in range(len(CLASS_ORDER))
        ]
        aggregated[key] = (target, max(range(len(CLASS_ORDER)), key=mean_probabilities.__getitem__))
    return aggregated


def _metrics(observations: Sequence[tuple[int, int]]) -> dict[str, Any]:
    confusion = [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER]
    for target, predicted in observations:
        confusion[target][predicted] += 1
    total = len(observations)
    accuracy = sum(confusion[index][index] for index in range(len(CLASS_ORDER))) / total
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(CLASS_ORDER):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[name] = {
            "false_negatives": support - true_positive,
            "false_positives": predicted_count - true_positive,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": support,
        }
        f1_values.append(f1)
    return {
        "accuracy": accuracy,
        "class_order": list(CLASS_ORDER),
        "confusion_matrix": confusion,
        "macro_f1": sum(f1_values) / len(CLASS_ORDER),
        "num_participants": total,
        "per_class": per_class,
    }


def _validate_bootstrap_config(
    num_replicates: int,
    random_seed: int,
    confidence_level: float,
) -> float:
    if (
        isinstance(num_replicates, bool)
        or not isinstance(num_replicates, int)
        or num_replicates <= 0
    ):
        raise ValueError("num_replicates must be a positive integer")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or not 0 <= random_seed < 2**32
    ):
        raise ValueError("random_seed must be an integer in [0, 2**32 - 1]")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be finite and in (0, 1)")
    return float(confidence_level)


def _stratified_bootstrap_replicates(
    observations: Sequence[tuple[int, int]],
    *,
    num_replicates: int,
    random_seed: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Resample participant outcomes with replacement within each true class."""
    _validate_bootstrap_config(num_replicates, random_seed, 0.95)
    if not isinstance(observations, Sequence) or isinstance(
        observations, (str, bytes)
    ):
        raise TypeError("observations must be a sequence of (target, prediction) pairs")
    if not observations:
        raise ValueError("observations must not be empty")

    by_class: dict[int, list[tuple[int, int]]] = {
        class_index: [] for class_index in range(len(CLASS_ORDER))
    }
    for index, observation in enumerate(observations):
        if not isinstance(observation, (tuple, list)) or len(observation) != 2:
            raise ValueError(f"observations[{index}] must be a (target, prediction) pair")
        target, prediction = observation
        for value, name in ((target, "target"), (prediction, "prediction")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"observations[{index}] {name} must be an integer class index"
                )
            if not 0 <= value < len(CLASS_ORDER):
                raise ValueError(
                    f"observations[{index}] {name} must be a class index in "
                    f"[0, {len(CLASS_ORDER) - 1}]"
                )
        by_class[target].append((target, prediction))

    insufficient = [
        CLASS_ORDER[class_index]
        for class_index, class_observations in by_class.items()
        if len(class_observations) < 2
    ]
    if insufficient:
        raise ValueError(
            "at least two participants per true class are required for "
            f"stratified bootstrap; insufficient classes: {insufficient!r}"
        )

    generator = random.Random(random_seed)
    replicates: list[tuple[tuple[int, int], ...]] = []
    for _ in range(num_replicates):
        replicate = tuple(
            sampled
            for class_observations in by_class.values()
            for sampled in generator.choices(
                class_observations, k=len(class_observations)
            )
        )
        replicates.append(replicate)
    return tuple(replicates)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated empirical percentile."""
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _stratified_bootstrap_interval(
    observations: Sequence[tuple[int, int]],
    *,
    num_replicates: int = 2000,
    random_seed: int = 42,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Compute a percentile interval for overall accuracy from stratified replicates."""
    confidence_level = _validate_bootstrap_config(
        num_replicates, random_seed, confidence_level
    )
    replicates = _stratified_bootstrap_replicates(
        observations,
        num_replicates=num_replicates,
        random_seed=random_seed,
    )
    accuracies = sorted(
        sum(target == prediction for target, prediction in replicate)
        / len(replicate)
        for replicate in replicates
    )
    tail_probability = (1.0 - confidence_level) / 2.0
    return (
        _percentile(accuracies, tail_probability),
        _percentile(accuracies, 1.0 - tail_probability),
    )


def analyze_prediction_artifact(
    artifact: PredictionArtifact,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Recalculate participant-level metrics, errors, variability, and uncertainty."""
    if not isinstance(artifact, PredictionArtifact):
        raise TypeError("artifact must be a PredictionArtifact")
    confidence_level = _validate_bootstrap_config(
        bootstrap_replicates, bootstrap_seed, confidence_level
    )
    participant_predictions = _participant_predictions(artifact)
    by_seed_fold: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for (seed, fold_id, _participant_id), observation in participant_predictions.items():
        by_seed_fold[(seed, fold_id)].append(observation)

    groups: list[dict[str, Any]] = []
    for group_index, ((seed, fold_id), observations) in enumerate(
        sorted(by_seed_fold.items())
    ):
        metrics = _metrics(observations)
        class_counts = {
            CLASS_ORDER[class_index]: sum(
                target == class_index for target, _ in observations
            )
            for class_index in range(len(CLASS_ORDER))
        }
        group_bootstrap_seed = (bootstrap_seed + group_index) % 2**32
        insufficient_classes = [
            class_name for class_name, count in class_counts.items() if count < 2
        ]
        if insufficient_classes:
            accuracy_interval = None
            bootstrap_status = "insufficient_class_sample_size"
        else:
            accuracy_interval = list(
                _stratified_bootstrap_interval(
                    observations,
                    num_replicates=bootstrap_replicates,
                    random_seed=group_bootstrap_seed,
                    confidence_level=confidence_level,
                )
            )
            bootstrap_status = "computed"
        groups.append(
            {
                "fold_id": fold_id,
                "metrics": metrics,
                "seed": seed,
                "uncertainty": {
                    "accuracy_interval": accuracy_interval,
                    "class_counts": class_counts,
                    "confidence_level": confidence_level,
                    "method": (
                        "participant-level class-stratified percentile bootstrap "
                        "with linearly interpolated quantiles"
                    ),
                    "num_replicates": bootstrap_replicates,
                    "random_seed": group_bootstrap_seed,
                    "resampling_unit": "one participant within this seed/fold group",
                    "status": bootstrap_status,
                    "unavailable_reason": (
                        f"fewer than two participants in true classes: "
                        f"{insufficient_classes!r}"
                        if insufficient_classes
                        else None
                    ),
                },
            }
        )

    metrics_by_name = {
        metric_name: [group["metrics"][metric_name] for group in groups]
        for metric_name in ("accuracy", "macro_f1")
    }
    variability: dict[str, dict[str, float | int | None]] = {}
    for metric_name, values in metrics_by_name.items():
        variability[metric_name] = {
            "group_count": len(values),
            "mean": statistics.mean(values),
            "sample_standard_deviation": (
                statistics.stdev(values) if len(values) > 1 else None
            ),
            "minimum": min(values),
            "maximum": max(values),
        }

    all_observations = [
        observation for observations in by_seed_fold.values() for observation in observations
    ]
    return {
        "aggregation": (
            "mean trial probabilities per participant within each seed/fold, "
            "then one argmax prediction per participant"
        ),
        "class_order": list(CLASS_ORDER),
        "class_wise_errors": _metrics(all_observations)["per_class"],
        "class_wise_errors_unit": (
            "participant-seed-fold run outcomes; a participant may recur across "
            "seeds, so these pooled counts are not unique-participant counts"
        ),
        "fold_seed_groups": groups,
        "model_variant": artifact.model_variant,
        "protocol_purpose": AUTHORIZED_PROTOCOL_PURPOSE,
        "result_kind": "synthetic engineering validation",
        "uncertainty_assumptions": (
            "Each percentile bootstrap resamples participants with replacement "
            "within their true class, preserving observed class counts, and is "
            "conditional on one seed/fold group's predictions. It requires at "
            "least two participants in every class. It assumes participants are "
            "independent and representative within each class; predictions are "
            "held fixed. Repeated predictions for the same participant across "
            "seeds are not additional participants; seed/fold variability is "
            "reported separately."
        ),
        "uncertainty_method": (
            "two-sided class-stratified participant bootstrap percentile "
            "interval for overall accuracy with linearly interpolated quantiles"
        ),
        "variability": variability,
        "variability_unit": (
            "seed/fold group metrics; repeated predictions for a participant "
            "contribute to run variability, not participant-level bootstrap sample size"
        ),
    }


__all__ = [
    "PredictionArtifact",
    "PredictionRecord",
    "analyze_prediction_artifact",
]
