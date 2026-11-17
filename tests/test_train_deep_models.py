"""Tests for grouped training helpers, metrics, and smoke execution."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.deep_models import EEGNet
from src.modeling_data import ModelingData
from src.train_deep_models import (
    _mean_loss,
    calculate_metrics,
    fit_channel_normalizer,
    group_aware_inner_split,
    load_best_checkpoint,
    run_training_pipeline,
    save_best_checkpoint,
    validate_prediction_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_grouped_data():
    rng = np.random.default_rng(42)
    epochs = []
    labels = []
    groups = []
    folds = []
    for fold in range(5):
        for label in (0, 1):
            epochs.append(rng.normal(loc=label * 0.1, size=(1024, 2)))
            labels.append(label)
            groups.append(f"class-{label}-fold-{fold}")
            folds.append(fold)
    return ModelingData(
        X=np.asarray(epochs, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        folds=np.asarray(folds, dtype=np.int64),
        class_mapping={"Control": 0, "ADHD": 1},
    )


def load_project_configs():
    with (ROOT / "configs" / "training_cell1.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        training_config = yaml.safe_load(file)
    with (ROOT / "configs" / "modeling_cell1.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        modeling_config = yaml.safe_load(file)
    with (ROOT / "configs" / "deep_models_cell1.yaml").open(
        "r", encoding="utf-8"
    ) as file:
        model_config = yaml.safe_load(file)
    return training_config, modeling_config, model_config


class GroupSplitTests(unittest.TestCase):
    def test_group_aware_inner_split_has_no_group_overlap(self):
        groups = np.asarray(
            [f"control-{index}" for index in range(5)]
            + [f"adhd-{index}" for index in range(5)]
        )
        labels = np.asarray([0] * 5 + [1] * 5)

        training_indices, validation_indices = group_aware_inner_split(
            groups,
            labels,
            validation_fraction=0.2,
            seed=42,
        )

        self.assertFalse(
            set(groups[training_indices]) & set(groups[validation_indices])
        )
        self.assertEqual(set(labels[training_indices]), {0, 1})
        self.assertEqual(set(labels[validation_indices]), {0, 1})


class NormalizationTests(unittest.TestCase):
    def test_normalization_statistics_come_only_from_training_epochs(self):
        training = np.arange(2 * 4 * 2, dtype=np.float64).reshape(2, 4, 2)
        validation = np.full((1, 4, 2), 1e6)
        test = np.full((1, 4, 2), -1e6)

        normalizer = fit_channel_normalizer(training)
        transformed_validation = normalizer.transform(validation)
        transformed_test = normalizer.transform(test)

        np.testing.assert_allclose(normalizer.mean, training.mean(axis=(0, 1)))
        np.testing.assert_allclose(normalizer.scale, training.std(axis=(0, 1)))
        self.assertGreater(float(np.abs(transformed_validation).mean()), 1e3)
        self.assertGreater(float(np.abs(transformed_test).mean()), 1e3)


class LossAggregationTests(unittest.TestCase):
    class LogitModel(torch.nn.Module):
        def forward(self, logits):
            return logits

    def test_weighted_loss_matches_full_dataset_with_uneven_batch_classes(self):
        logits = torch.tensor(
            [
                [2.0, -0.5],
                [-0.2, 0.8],
                [1.1, 0.2],
                [0.1, 1.0],
                [1.0, 0.4],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        weights = torch.tensor([0.6, 2.4], dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(logits, labels),
            batch_size=3,
            shuffle=False,
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        aggregated_loss = _mean_loss(
            self.LogitModel(),
            loader,
            criterion,
            torch.device("cpu"),
        )
        expected_loss = criterion(logits, labels).item()

        self.assertAlmostEqual(aggregated_loss, expected_loss, places=7)

    def test_unweighted_loss_matches_full_dataset_with_uneven_batch_sizes(self):
        logits = torch.tensor(
            [
                [2.0, -0.5],
                [-0.2, 0.8],
                [1.1, 0.2],
                [0.1, 1.0],
                [1.0, 0.4],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        loader = DataLoader(
            TensorDataset(logits, labels),
            batch_size=3,
            shuffle=False,
        )
        criterion = torch.nn.CrossEntropyLoss()

        aggregated_loss = _mean_loss(
            self.LogitModel(),
            loader,
            criterion,
            torch.device("cpu"),
        )
        expected_loss = criterion(logits, labels).item()

        self.assertAlmostEqual(aggregated_loss, expected_loss, places=7)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_model(self):
        model = EEGNet(temporal_filters=2, depth_multiplier=1, separable_filters=4)
        normalizer = fit_channel_normalizer(
            np.random.default_rng(1).normal(size=(3, 1024, 2))
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            save_best_checkpoint(
                path,
                model,
                normalizer,
                {"outer_fold": 0, "best_epoch": 1},
            )
            restored = EEGNet(
                temporal_filters=2,
                depth_multiplier=1,
                separable_filters=4,
            )
            checkpoint = load_best_checkpoint(path, restored, torch.device("cpu"))

        self.assertEqual(checkpoint["metadata"]["best_epoch"], 1)
        np.testing.assert_array_equal(
            checkpoint["normalization_mean"], normalizer.mean
        )
        for first, second in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(first, second))


class PredictionAndMetricTests(unittest.TestCase):
    def test_prediction_coverage_requires_exactly_one_prediction(self):
        validate_prediction_coverage(np.ones(5, dtype=np.int64))

        with self.assertRaisesRegex(ValueError, "missing indices"):
            validate_prediction_coverage(np.asarray([1, 0, 1]))
        with self.assertRaisesRegex(ValueError, "duplicated indices"):
            validate_prediction_coverage(np.asarray([1, 2, 1]))

    def test_metrics_and_confusion_matrix_use_adhd_as_positive(self):
        result = calculate_metrics(
            np.asarray([0, 1, 1, 0]),
            np.asarray([0.1, 0.8, 0.4, 0.7]),
        )

        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall_sensitivity"], 0.5)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["f1_score"], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertNotEqual(result["roc_curve"]["false_positive_rate"], [])

    def test_undefined_metrics_are_explicit(self):
        result = calculate_metrics(np.asarray([0, 0]), np.asarray([0.1, 0.2]))

        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["recall_sensitivity"])
        self.assertIn("roc_auc", result["undefined_metrics"])
        self.assertIn("recall_sensitivity", result["undefined_metrics"])


class SmokeRunTests(unittest.TestCase):
    def test_smoke_run_trains_all_models_and_writes_artifacts(self):
        training_config, modeling_config, model_config = load_project_configs()
        training_config = copy.deepcopy(training_config)
        training_config["inner_validation"]["fraction"] = 0.25
        training_config["optimization"]["maximum_epochs"] = 1
        training_config["optimization"]["early_stopping_patience"] = 1
        training_config["optimization"]["batch_size"] = 4
        training_config["smoke_test"]["groups_per_class_per_fold"] = 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "smoke"
            report_path = output_directory / "deep_learning_training_report.md"
            results = run_training_pipeline(
                data=synthetic_grouped_data(),
                training_config=training_config,
                modeling_config=modeling_config,
                model_config=model_config,
                output_directory=output_directory,
                report_path=report_path,
                smoke_test=True,
                device=torch.device("cpu"),
            )

            self.assertEqual(results["run_mode"], "smoke_test")
            self.assertEqual(set(results["models"]), set(model_config["models"]))
            self.assertTrue(report_path.exists())
            loaded_json = json.loads(
                (output_directory / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded_json["group_ids_are_verified_participant_ids"], False)
            for model_name, model_result in results["models"].items():
                self.assertEqual(model_result["oof_epoch_count"], 10)
                self.assertEqual(len(model_result["per_fold_metrics"]), 5)
                for fold in range(5):
                    fold_directory = (
                        output_directory
                        / model_name
                        / f"outer_fold_{fold}"
                    )
                    self.assertTrue((fold_directory / "best_checkpoint.pt").exists())
                    self.assertTrue((fold_directory / "training_history.json").exists())
                    self.assertTrue((fold_directory / "group_split.json").exists())
                self.assertTrue(
                    (output_directory / model_name / "oof_predictions.csv").exists()
                )
                self.assertTrue(
                    (output_directory / model_name / "roc_curves.json").exists()
                )
                with (
                    output_directory / model_name / "oof_predictions.csv"
                ).open("r", newline="", encoding="utf-8") as file:
                    prediction_rows = list(csv.DictReader(file))
                epoch_indices = [int(row["epoch_index"]) for row in prediction_rows]
                self.assertEqual(sorted(epoch_indices), list(range(10)))
                self.assertEqual(len(set(epoch_indices)), 10)
                split = json.loads(
                    (
                        output_directory
                        / model_name
                        / "outer_fold_0"
                        / "group_split.json"
                    ).read_text(encoding="utf-8")
                )
                training_groups = set(split["training_group_ids"])
                validation_groups = set(split["validation_group_ids"])
                test_groups = set(split["test_group_ids"])
                self.assertFalse(training_groups & validation_groups)
                self.assertFalse(training_groups & test_groups)
                self.assertFalse(validation_groups & test_groups)


if __name__ == "__main__":
    unittest.main()
