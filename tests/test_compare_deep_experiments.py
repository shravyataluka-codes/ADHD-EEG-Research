import unittest

import numpy as np

from src.compare_deep_experiments import (
    calculate_metrics,
    parse_markdown_tables,
    render_report,
    summarize_folds,
)


class CompareDeepExperimentsTests(unittest.TestCase):
    def test_metrics_use_adhd_as_positive_and_return_confusion_matrix(self):
        metrics = calculate_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 0]),
            np.asarray([0.1, 0.8, 0.9, 0.2]),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["recall_sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_fold_summary_uses_sample_standard_deviation(self):
        folds = [
            {"accuracy": value, "balanced_accuracy": value,
             "recall_sensitivity": value, "specificity": value,
             "precision": value, "f1_score": value, "roc_auc": value}
            for value in (0.5, 0.7, 0.9)
        ]

        summary = summarize_folds(folds)

        self.assertAlmostEqual(summary["accuracy"]["mean"], 0.7)
        self.assertAlmostEqual(
            summary["accuracy"]["sample_standard_deviation"], 0.2
        )

    def test_markdown_table_parser_keeps_headers_and_rows(self):
        tables = parse_markdown_tables(
            "| Model | Balanced accuracy |\n"
            "| --- | --- |\n"
            "| EEGNet | 0.82 |\n"
        )

        self.assertEqual(tables[0]["header"], ["model", "balancedaccuracy"])
        self.assertEqual(
            tables[0]["rows"][0],
            {"model": "EEGNet", "balancedaccuracy": "0.82"},
        )

    def test_report_separates_pooled_and_fold_summary_sections(self):
        metrics = {
            "accuracy": 0.8,
            "balanced_accuracy": 0.8,
            "recall_sensitivity": 0.8,
            "specificity": 0.8,
            "precision": 0.8,
            "f1_score": 0.8,
            "roc_auc": 0.8,
            "confusion_matrix": [[2, 0], [0, 2]],
        }
        summary = {
            metric: {
                "mean": 0.8,
                "sample_standard_deviation": 0.1,
                "fold_values": [0.7, 0.9],
            }
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "recall_sensitivity",
                "specificity",
                "precision",
                "f1_score",
                "roc_auc",
            )
        }
        experiments = []
        keys_and_names = [
            ("eegnet", "EEGNet-inspired"),
            ("cnn_lstm", "CNN-LSTM"),
            ("eeg_transformer", "EEG-Transformer-inspired"),
            ("eeg_conformer", "EEG Conformer"),
            ("handcrafted_logistic", "Handcrafted features only (Logistic Regression)"),
            ("raw_transformer_baseline", "Raw EEG Transformer baseline (fusion Arm A)"),
            ("transformer_feature_fusion", "Transformer + handcrafted feature fusion"),
        ]
        for key, name in keys_and_names:
            experiments.append(
                {
                    "key": key,
                    "name": name,
                    "family": "deep_learning",
                    "source_run": "source",
                    "artifact_directory": "source/artifacts",
                    "prediction_count": 4,
                    "pooled_metrics": metrics,
                    "fold_metrics": [
                        {"fold_id": 0, "epoch_count": 2, **metrics},
                        {"fold_id": 1, "epoch_count": 2, **metrics},
                    ],
                    "fold_summary": summary,
                    "roc_curves": {"overall": {}, "per_fold": []},
                    "learning_curve_analysis": None,
                    "pooled_balanced_accuracy_rank": 1,
                }
            )
        comparison = {
            "evaluation": {
                "epoch_count": 4,
                "class_counts": {"Control": 2, "ADHD": 2},
            },
            "source_runs": {"deep_learning": "base", "feature_fusion": "fusion"},
            "experiments": experiments,
            "validation": {
                "transformer_baseline_duplicate_check": {
                    "exact_prediction_rows_match": True
                },
                "existing_report_numeric_crosschecks": [],
            },
        }
        report = render_report(comparison)

        self.assertIn("## Pooled out-of-fold metrics", report)
        self.assertIn("## Five-fold mean ± sample standard deviation", report)
        self.assertIn("## All five outer-fold results", report)
        self.assertIn("not independent evidence", report)


if __name__ == "__main__":
    unittest.main()
