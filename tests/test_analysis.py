"""Synthetic known-outcome tests for Sprint 7 prediction analysis."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhd_eeg.analysis import (
    PredictionArtifact,
    PredictionRecord,
    _stratified_bootstrap_interval,
    _stratified_bootstrap_replicates,
    analyze_prediction_artifact,
)
from adhd_eeg.classifier import CLASS_ORDER


def _record(
    sample: str,
    participant: str,
    target: int,
    probabilities: tuple[float, float, float],
    *,
    seed: int = 7,
    fold: str = "synthetic-fixture:test-fold-0",
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=f"synthetic-fixture:{sample}",
        participant_id=f"synthetic-fixture:{participant}",
        fold_id=fold,
        seed=seed,
        target=target,
        probabilities=probabilities,
    )


def _known_artifact() -> PredictionArtifact:
    records = (
        _record("hc-1a", "hc-1", 0, (0.8, 0.1, 0.1)),
        _record("hc-1b", "hc-1", 0, (0.2, 0.7, 0.1)),
        _record("add-1a", "add-1", 1, (0.6, 0.3, 0.1)),
        _record("add-1b", "add-1", 1, (0.6, 0.3, 0.1)),
        _record("adhd-1", "adhd-1", 2, (0.1, 0.2, 0.7)),
    )
    return PredictionArtifact(
        model_variant="conformer_reference",
        records=records,
        expected_sample_ids=tuple(record.sample_id for record in records),
    )


class PredictionAnalysisTests(unittest.TestCase):
    def test_participant_aggregation_and_classwise_errors(self) -> None:
        report = analyze_prediction_artifact(_known_artifact())
        group = report["fold_seed_groups"][0]
        metrics = group["metrics"]
        self.assertEqual(report["class_order"], list(CLASS_ORDER))
        self.assertEqual(metrics["num_participants"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
        )
        self.assertEqual(metrics["per_class"]["ADD"]["false_negatives"], 1)
        self.assertEqual(metrics["per_class"]["ADD"]["support"], 1)
        self.assertAlmostEqual(metrics["per_class"]["HC"]["precision"], 0.5)
        self.assertEqual(
            report["aggregation"],
            "mean trial probabilities per participant within each seed/fold, "
            "then one argmax prediction per participant",
        )
        self.assertEqual(report["result_kind"], "synthetic engineering validation")

    def test_insufficient_class_size_is_reported_without_interval(self) -> None:
        first = list(_known_artifact().records)
        second = [
            _record(
                sample,
                participant,
                target,
                probabilities,
                seed=8,
            )
            for sample, participant, target, probabilities in (
                ("hc-1a", "hc-1", 0, (0.8, 0.1, 0.1)),
                ("hc-1b", "hc-1", 0, (0.8, 0.1, 0.1)),
                ("add-1a", "add-1", 1, (0.1, 0.8, 0.1)),
                ("add-1b", "add-1", 1, (0.1, 0.8, 0.1)),
                ("adhd-1", "adhd-1", 2, (0.1, 0.2, 0.7)),
            )
        ]
        artifact = PredictionArtifact(
            "conformer_reference",
            tuple(first + second),
            expected_sample_ids=tuple(record.sample_id for record in first),
        )
        report = analyze_prediction_artifact(artifact)
        self.assertEqual(len(report["fold_seed_groups"]), 2)
        self.assertAlmostEqual(report["variability"]["accuracy"]["mean"], 5 / 6)
        self.assertAlmostEqual(
            report["variability"]["accuracy"]["sample_standard_deviation"],
            (1 / 18) ** 0.5,
        )
        uncertainty = report["fold_seed_groups"][0]["uncertainty"]
        self.assertIsNone(uncertainty["accuracy_interval"])
        self.assertEqual(uncertainty["status"], "insufficient_class_sample_size")
        self.assertEqual(uncertainty["class_counts"], {"HC": 1, "ADD": 1, "ADHD": 1})
        self.assertEqual(
            uncertainty["method"],
            "participant-level class-stratified percentile bootstrap "
            "with linearly interpolated quantiles",
        )
        self.assertIn("Repeated predictions", report["uncertainty_assumptions"])

    def test_bootstrap_replicates_preserve_true_class_counts(self) -> None:
        observations = (
            (0, 0), (0, 1), (0, 0),
            (1, 1), (1, 2),
            (2, 2), (2, 0), (2, 2), (2, 1),
        )
        expected_counts = Counter(target for target, _ in observations)
        replicates = _stratified_bootstrap_replicates(
            observations, num_replicates=25, random_seed=321
        )
        self.assertEqual(len(replicates), 25)
        for replicate in replicates:
            self.assertEqual(Counter(target for target, _ in replicate), expected_counts)
            self.assertEqual(len(replicate), len(observations))

    def test_bootstrap_interval_is_deterministic_and_has_expected_bounds(self) -> None:
        observations = (
            (0, 0), (0, 0),
            (1, 1), (1, 0),
            (2, 1), (2, 0),
        )
        first = _stratified_bootstrap_interval(
            observations,
            num_replicates=1000,
            random_seed=11,
            confidence_level=0.95,
        )
        second = _stratified_bootstrap_interval(
            observations,
            num_replicates=1000,
            random_seed=11,
            confidence_level=0.95,
        )
        self.assertEqual(first, second)
        lower, upper = first
        self.assertEqual(lower, 1 / 3)
        self.assertEqual(upper, 2 / 3)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLessEqual(lower, upper)

    def test_bootstrap_rejects_invalid_and_insufficient_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _stratified_bootstrap_interval(())
        with self.assertRaisesRegex(ValueError, "pair"):
            _stratified_bootstrap_interval(((0,),))
        with self.assertRaisesRegex(TypeError, "integer class index"):
            _stratified_bootstrap_interval(((True, 0),))
        with self.assertRaisesRegex(ValueError, "class index"):
            _stratified_bootstrap_interval(((3, 0),))
        with self.assertRaisesRegex(ValueError, "at least two participants"):
            _stratified_bootstrap_interval(((0, 0), (1, 1), (2, 2), (2, 0)))
        with self.assertRaisesRegex(ValueError, "num_replicates"):
            _stratified_bootstrap_interval(
                ((0, 0), (0, 0), (1, 1), (1, 1), (2, 2), (2, 2)),
                num_replicates=0,
            )
        with self.assertRaisesRegex(ValueError, "random_seed"):
            _stratified_bootstrap_interval(
                ((0, 0), (0, 0), (1, 1), (1, 1), (2, 2), (2, 2)),
                random_seed=-1,
            )
        with self.assertRaisesRegex(ValueError, "confidence_level"):
            _stratified_bootstrap_interval(
                ((0, 0), (0, 0), (1, 1), (1, 1), (2, 2), (2, 2)),
                confidence_level=1.0,
            )
        with self.assertRaisesRegex(ValueError, "num_replicates"):
            analyze_prediction_artifact(_known_artifact(), bootstrap_replicates=True)

    def test_bootstrap_uncertainty_is_separate_from_seed_variability(self) -> None:
        participant_classes = (
            ("hc-a", "hc-a", 0), ("hc-b", "hc-b", 0),
            ("add-a", "add-a", 1), ("add-b", "add-b", 1),
            ("adhd-a", "adhd-a", 2), ("adhd-b", "adhd-b", 2),
        )
        first_seed = [
            _record(sample, participant, target, tuple(
                1.0 if class_index == target else 0.0
                for class_index in range(3)
            ), seed=7)
            for sample, participant, target in participant_classes
        ]
        second_seed = [
            _record(
                sample,
                participant,
                target,
                tuple(
                    1.0 if class_index == prediction else 0.0
                    for class_index in range(3)
                ),
                seed=8,
            )
            for (sample, participant, target), prediction in zip(
                participant_classes, (0, 0, 1, 0, 0, 1)
            )
        ]
        artifact = PredictionArtifact(
            "conformer_reference",
            tuple(first_seed + second_seed),
            expected_sample_ids=tuple(
                f"synthetic-fixture:{sample}"
                for sample, _participant, _target in participant_classes
            ),
        )
        report = analyze_prediction_artifact(
            artifact, bootstrap_replicates=500, bootstrap_seed=99
        )
        self.assertEqual(len(report["fold_seed_groups"]), 2)
        self.assertEqual(
            [group["metrics"]["num_participants"] for group in report["fold_seed_groups"]],
            [6, 6],
        )
        self.assertEqual(
            [group["uncertainty"]["class_counts"] for group in report["fold_seed_groups"]],
            [{"HC": 2, "ADD": 2, "ADHD": 2}] * 2,
        )
        self.assertEqual(report["variability"]["accuracy"]["mean"], 0.75)
        self.assertIn("not additional participants", report["uncertainty_assumptions"])
        self.assertIn("participant may recur", report["class_wise_errors_unit"])

    def test_serialization_is_deterministic_and_metrics_recompute(self) -> None:
        artifact = _known_artifact()
        self.assertEqual(artifact.to_json(), artifact.to_json())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.json"
            artifact.to_json_file(path)
            restored = PredictionArtifact.from_json_file(path)
        self.assertEqual(restored, artifact)
        self.assertEqual(
            analyze_prediction_artifact(restored),
            analyze_prediction_artifact(artifact),
        )

    def test_rejects_invalid_or_inconsistent_artifacts(self) -> None:
        with self.assertRaises(ValueError):
            PredictionRecord(
                "synthetic-fixture:s",
                "synthetic-fixture:p",
                "synthetic-fixture:f",
                1,
                3,
                (0.2, 0.3, 0.5),
            )
        with self.assertRaises(ValueError):
            PredictionRecord(
                "synthetic-fixture:s",
                "synthetic-fixture:p",
                "synthetic-fixture:f",
                1,
                0,
                (0.2, 0.3, 0.4),
            )
        with self.assertRaises(ValueError):
            PredictionRecord(
                "real-trial",
                "synthetic-fixture:p",
                "synthetic-fixture:f",
                1,
                0,
                (1.0, 0.0, 0.0),
            )
        valid = _known_artifact()
        with self.assertRaises(ValueError):
            PredictionArtifact(
                valid.model_variant,
                valid.records + (valid.records[0],),
                expected_sample_ids=valid.expected_sample_ids,
            )
        with self.assertRaises(ValueError):
            PredictionArtifact(
                valid.model_variant,
                valid.records,
                expected_sample_ids=valid.expected_sample_ids,
                class_order=("ADD", "HC", "ADHD"),
            )
        with self.assertRaises(ValueError):
            PredictionArtifact(
                valid.model_variant,
                valid.records,
                expected_sample_ids=valid.expected_sample_ids,
                protocol_purpose="REAL_COHORT_TRAINING",
            )
        changed_target = _record("hc-other", "hc-1", 1, (0.1, 0.8, 0.1))
        with self.assertRaisesRegex(ValueError, "one target"):
            PredictionArtifact(
                valid.model_variant,
                valid.records + (changed_target,),
                expected_sample_ids=valid.expected_sample_ids
                + (changed_target.sample_id,),
            )
        changed_fold = _record(
            "hc-other-fold",
            "hc-1",
            0,
            (0.8, 0.1, 0.1),
            fold="synthetic-fixture:test-fold-1",
        )
        with self.assertRaisesRegex(ValueError, "only one fold"):
            PredictionArtifact(
                valid.model_variant,
                valid.records + (changed_fold,),
                expected_sample_ids=valid.expected_sample_ids
                + (changed_fold.sample_id,),
            )
        with self.assertRaises(ValueError):
            PredictionArtifact(
                valid.model_variant,
                (),
                expected_sample_ids=valid.expected_sample_ids,
            )
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            PredictionArtifact(
                valid.model_variant,
                valid.records[:-1],
                expected_sample_ids=valid.expected_sample_ids,
            )

    def test_analysis_is_deterministic(self) -> None:
        report = analyze_prediction_artifact(_known_artifact())
        self.assertEqual(report, analyze_prediction_artifact(_known_artifact()))


if __name__ == "__main__":
    unittest.main()
