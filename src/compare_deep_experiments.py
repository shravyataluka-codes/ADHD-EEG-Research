"""Compare saved Cell 1 experiments without retraining any models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.modeling_data import load_cell1_dataset


BASELINE_RUN = Path(
    "results/deep_learning/cell_01/full/run_20261115T090201955591Z"
)
FUSION_RUN = Path(
    "results/deep_learning/cell_01/feature_fusion/full/"
    "run_20261115T110540461128Z"
)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "recall_sensitivity",
    "specificity",
    "precision",
    "f1_score",
    "roc_auc",
)
PREDICTION_COLUMNS = (
    "epoch_index",
    "y_true",
    "y_pred",
    "probability_control",
    "probability_adhd",
    "group_id",
    "fold_id",
)
SPECS = (
    {
        "key": "eegnet",
        "name": "EEGNet-inspired",
        "source_run": BASELINE_RUN,
        "subdirectory": "eegnet",
        "family": "deep_learning",
    },
    {
        "key": "cnn_lstm",
        "name": "CNN-LSTM",
        "source_run": BASELINE_RUN,
        "subdirectory": "cnn_lstm",
        "family": "deep_learning",
    },
    {
        "key": "eeg_transformer",
        "name": "EEG-Transformer-inspired",
        "source_run": BASELINE_RUN,
        "subdirectory": "eeg_transformer",
        "family": "deep_learning",
    },
    {
        "key": "eeg_conformer",
        "name": "EEG Conformer",
        "source_run": BASELINE_RUN,
        "subdirectory": "eeg_conformer",
        "family": "deep_learning",
    },
    {
        "key": "handcrafted_logistic",
        "name": "Handcrafted features only (Logistic Regression)",
        "source_run": FUSION_RUN,
        "subdirectory": "B_handcrafted_logistic",
        "family": "handcrafted_only",
    },
    {
        "key": "raw_transformer_baseline",
        "name": "Raw EEG Transformer baseline (fusion Arm A)",
        "source_run": FUSION_RUN,
        "subdirectory": "A_raw_eeg_transformer",
        "family": "raw_baseline",
    },
    {
        "key": "transformer_feature_fusion",
        "name": "Transformer + handcrafted feature fusion",
        "source_run": FUSION_RUN,
        "subdirectory": "C_transformer_plus_handcrafted",
        "family": "feature_fusion",
    },
)
REPORT_METRIC_COLUMNS = {
    "accuracy": "accuracy",
    "balancedaccuracy": "balanced_accuracy",
    "sensitivity": "recall_sensitivity",
    "specificity": "specificity",
    "precision": "precision",
    "f1": "f1_score",
    "rocauc": "roc_auc",
}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved artifact is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(PREDICTION_COLUMNS).issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Unexpected OOF prediction columns in {path}.")
        rows = []
        for row in reader:
            rows.append(
                {
                    "epoch_index": int(row["epoch_index"]),
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                    "probability_control": float(row["probability_control"]),
                    "probability_adhd": float(row["probability_adhd"]),
                    "group_id": row["group_id"],
                    "fold_id": int(row["fold_id"]),
                }
            )
    return rows


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probability_adhd: np.ndarray
) -> dict[str, Any]:
    """Calculate binary metrics independently from saved OOF prediction rows."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_sensitivity": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability_adhd)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def summarize_folds(
    fold_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return unweighted mean and sample SD across folds for every metric."""
    summary = {}
    for metric in METRICS:
        values = np.asarray([fold[metric] for fold in fold_metrics], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)),
            "fold_values": [float(value) for value in values],
        }
    return summary


def _assert_number_matches(
    saved: Any, calculated: float, description: str, tolerance: float = 1e-10
) -> None:
    if saved is None or not math.isfinite(float(saved)):
        raise ValueError(f"{description} is not a finite saved number.")
    if not math.isclose(
        float(saved), calculated, rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ValueError(
            f"{description} mismatch: saved={saved!r}, recalculated={calculated!r}."
        )


def _validate_saved_metrics(
    metric_file: dict[str, Any], calculated: dict[str, Any], description: str
) -> None:
    for metric in METRICS:
        _assert_number_matches(
            metric_file.get(metric), calculated[metric], f"{description}.{metric}"
        )
    if metric_file.get("confusion_matrix") != calculated["confusion_matrix"]:
        raise ValueError(f"{description}.confusion_matrix does not match OOF rows.")


def _validate_roc_data(
    roc_data: dict[str, Any],
    y_true: np.ndarray,
    probability_adhd: np.ndarray,
    expected_auc: float,
    description: str,
) -> None:
    if "false_positive_rate" not in roc_data or "true_positive_rate" not in roc_data:
        raise ValueError(f"{description} is missing FPR/TPR coordinates.")
    fpr = np.asarray(roc_data["false_positive_rate"], dtype=float)
    tpr = np.asarray(roc_data["true_positive_rate"], dtype=float)
    if fpr.ndim != 1 or tpr.ndim != 1 or fpr.size != tpr.size or fpr.size < 2:
        raise ValueError(f"{description} has malformed ROC coordinates.")
    if not np.isfinite(fpr).all() or not np.isfinite(tpr).all():
        raise ValueError(f"{description} contains non-finite ROC coordinates.")
    if (
        fpr[0] != 0.0
        or tpr[0] != 0.0
        or fpr[-1] != 1.0
        or tpr[-1] != 1.0
        or np.any(np.diff(fpr) < 0)
        or np.any(np.diff(tpr) < 0)
    ):
        raise ValueError(f"{description} is not a monotone ROC curve from (0,0) to (1,1).")
    saved_auc = float(np.trapz(tpr, fpr))
    _assert_number_matches(saved_auc, expected_auc, f"{description}.area")
    recalculated_auc = float(roc_auc_score(y_true, probability_adhd))
    _assert_number_matches(saved_auc, recalculated_auc, f"{description}.recalculated_area")


def _check_reported_metrics(
    report_path: Path,
    lookup: dict[str, dict[str, Any]],
    expected_type: str,
) -> dict[str, Any]:
    tables = parse_markdown_tables(report_path.read_text(encoding="utf-8"))
    if expected_type == "long_fold_summary":
        table = next(
            (
                candidate
                for candidate in tables
                if {"arm", "metric", "mean"}.issubset(candidate["header"])
                and any(header.startswith("samplesd") for header in candidate["header"])
            ),
            None,
        )
    elif expected_type == "fold_summary":
        expected_headers = {
            "model",
            "accuracy",
            "balancedaccuracy",
            "sensitivity",
            "specificity",
            "f1",
            "rocauc",
        }
        table = next(
            (
                candidate
                for candidate in tables
                if expected_headers.issubset(set(candidate["header"]))
                and any(
                    re.search(r"\+/-|±", cell)
                    for row in candidate["rows"]
                    for cell in row.values()
                )
            ),
            None,
        )
    else:
        expected_headers = {
            "accuracy",
            "balancedaccuracy",
            "sensitivity",
            "specificity",
            "f1",
            "rocauc",
        }
        table = next(
            (
                candidate
                for candidate in tables
                if expected_headers.issubset(set(candidate["header"]))
                and (
                    expected_type != "training_pooled"
                    or "precision" in candidate["header"]
                )
            ),
            None,
        )
    if table is None:
        raise ValueError(f"Could not find the expected metrics table in {report_path}.")

    checks = 0
    for row in table["rows"]:
        label = row.get("model", row.get("arm", ""))
        normalized_label = normalize_header(label)
        if normalized_label not in lookup:
            continue
        expected = lookup[normalized_label]
        if expected_type == "long_fold_summary":
            raw_metric = normalize_header(row.get("metric", ""))
            metric = {
                "recallsensitivity": "recall_sensitivity",
                "balancedaccuracy": "balanced_accuracy",
                "rocauc": "roc_auc",
                "f1score": "f1_score",
            }.get(raw_metric, raw_metric)
            if metric not in expected:
                continue
            mean_values = re.findall(
                r"[-+]?\d*\.?\d+", row.get("mean", "")
            )
            sd_column = next(
                header for header in table["header"] if header.startswith("samplesd")
            )
            sd_values = re.findall(
                r"[-+]?\d*\.?\d+", row.get(sd_column, "")
            )
            if not mean_values or not sd_values:
                raise ValueError(
                    f"Missing fold summary values for {label}.{metric} in {report_path}."
                )
            _assert_number_matches(
                float(mean_values[0]),
                expected[metric]["mean"],
                f"{report_path.name}:{label}.{metric}.mean",
                tolerance=5.1e-5,
            )
            _assert_number_matches(
                float(sd_values[0]),
                expected[metric]["sample_standard_deviation"],
                f"{report_path.name}:{label}.{metric}.sample_sd",
                tolerance=5.1e-5,
            )
            checks += 2
            continue
        for column, metric in REPORT_METRIC_COLUMNS.items():
            if column not in row:
                continue
            values = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", row[column])]
            if not values:
                raise ValueError(f"No numeric value for {label}.{column} in {report_path}.")
            if expected_type == "fold_summary":
                if len(values) < 2:
                    raise ValueError(
                        f"Expected mean and sample SD for {label}.{column} in {report_path}."
                    )
                _assert_number_matches(
                    values[0],
                    expected[metric]["mean"],
                    f"{report_path.name}:{label}.{metric}.mean",
                    tolerance=5.1e-5,
                )
                _assert_number_matches(
                    values[1],
                    expected[metric]["sample_standard_deviation"],
                    f"{report_path.name}:{label}.{metric}.sample_sd",
                    tolerance=5.1e-5,
                )
            else:
                _assert_number_matches(
                    values[0],
                    expected[metric],
                    f"{report_path.name}:{label}.{metric}",
                    tolerance=5.1e-5,
                )
            checks += 1
    if checks == 0:
        raise ValueError(f"No experiment metrics were validated in {report_path}.")
    return {
        "file": report_path.as_posix(),
        "status": "PASS",
        "numeric_values_checked": checks,
    }


def _check_reported_confusion_matrices(
    report_path: Path, experiments: list[dict[str, Any]]
) -> dict[str, Any]:
    markdown = report_path.read_text(encoding="utf-8")
    checked = 0
    for experiment in experiments:
        key = re.escape(experiment["key"])
        match = re.search(
            rf"(?ms)^###\s+{key}\s*$([\s\S]*?)(?=^###\s|^##\s|\Z)",
            markdown,
        )
        if match is None:
            raise ValueError(
                f"Could not find {experiment['key']} confusion matrix in {report_path}."
            )
        section = match.group(1)
        control = re.search(r"\|\s*Control\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", section)
        adhd = re.search(r"\|\s*ADHD\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", section)
        if control is None or adhd is None:
            raise ValueError(
                f"Malformed {experiment['key']} confusion matrix in {report_path}."
            )
        reported = [
            [int(control.group(1)), int(control.group(2))],
            [int(adhd.group(1)), int(adhd.group(2))],
        ]
        if reported != experiment["pooled_metrics"]["confusion_matrix"]:
            raise ValueError(
                f"{report_path.name}:{experiment['key']} confusion matrix "
                "does not match saved OOF predictions."
            )
        checked += 4
    return {
        "file": report_path.as_posix(),
        "status": "PASS",
        "confusion_matrix_cells_checked": checked,
    }


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_markdown_tables(markdown: str) -> list[dict[str, Any]]:
    lines = markdown.splitlines()
    tables = []
    index = 0
    while index + 1 < len(lines):
        if not lines[index].strip().startswith("|") or not re.search(
            r"\|?\s*:?-{3,}", lines[index + 1]
        ):
            index += 1
            continue
        header = [normalize_header(cell) for cell in _split_markdown_row(lines[index])]
        index += 2
        rows = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            cells = _split_markdown_row(lines[index])
            if len(cells) == len(header):
                rows.append(dict(zip(header, cells)))
            index += 1
        tables.append({"header": header, "rows": rows})
    return tables


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _history_summary(history_path: Path) -> dict[str, Any]:
    history = _read_json(history_path)
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is empty or malformed: {history_path}")
    validation = np.asarray([entry["validation_loss"] for entry in history], dtype=float)
    training = np.asarray([entry["training_loss"] for entry in history], dtype=float)
    if not np.isfinite(validation).all() or not np.isfinite(training).all():
        raise ValueError(f"Training history contains non-finite losses: {history_path}")
    best_index = int(np.argmin(validation))
    return {
        "epochs_recorded": len(history),
        "best_epoch": int(history[best_index]["epoch"]),
        "best_validation_loss": float(validation[best_index]),
        "final_validation_loss": float(validation[-1]),
        "final_training_loss": float(training[-1]),
        "training_loss_at_best_epoch": float(training[best_index]),
        "final_validation_above_best": bool(validation[-1] > validation[best_index]),
        "training_fell_after_best": bool(training[-1] < training[best_index]),
    }


def _load_experiment(
    root: Path, spec: dict[str, Any], data: Any
) -> dict[str, Any]:
    source_run = root / spec["source_run"]
    directory = source_run / spec["subdirectory"]
    predictions_path = directory / "oof_predictions.csv"
    overall_path = directory / "overall_metrics.json"
    per_fold_path = directory / "per_fold_metrics.json"
    roc_path = directory / "roc_curves.json"
    rows = _read_predictions(predictions_path)
    if len(rows) != len(data.y):
        raise ValueError(
            f"{spec['name']} has {len(rows)} OOF rows; expected {len(data.y)}."
        )

    epoch_indices = np.asarray([row["epoch_index"] for row in rows], dtype=np.int64)
    if not np.array_equal(np.sort(epoch_indices), np.arange(len(data.y))):
        raise ValueError(f"{spec['name']} does not have exactly one OOF row per epoch.")
    ordered = sorted(rows, key=lambda row: row["epoch_index"])
    y_true = np.asarray([row["y_true"] for row in ordered], dtype=np.int64)
    y_pred = np.asarray([row["y_pred"] for row in ordered], dtype=np.int64)
    probability_control = np.asarray(
        [row["probability_control"] for row in ordered], dtype=float
    )
    probability_adhd = np.asarray(
        [row["probability_adhd"] for row in ordered], dtype=float
    )
    groups = np.asarray([row["group_id"] for row in ordered], dtype=str)
    fold_ids = np.asarray([row["fold_id"] for row in ordered], dtype=np.int64)
    if not np.array_equal(y_true, data.y):
        raise ValueError(f"{spec['name']} saved labels differ from Cell 1 source labels.")
    if not np.array_equal(groups, data.groups):
        raise ValueError(f"{spec['name']} saved groups differ from Cell 1 source groups.")
    if not np.array_equal(fold_ids, data.folds):
        raise ValueError(f"{spec['name']} saved folds differ from Cell 1 fold assignments.")
    if not np.isfinite(probability_control).all() or not np.isfinite(probability_adhd).all():
        raise ValueError(f"{spec['name']} has non-finite probabilities.")
    if not np.allclose(probability_control + probability_adhd, 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"{spec['name']} probabilities do not sum to one.")
    if not np.array_equal(y_pred, (probability_adhd >= 0.5).astype(np.int64)):
        raise ValueError(f"{spec['name']} predictions do not match the saved 0.5 threshold.")

    pooled_metrics = calculate_metrics(y_true, y_pred, probability_adhd)
    saved_overall = _read_json(overall_path)
    _validate_saved_metrics(saved_overall, pooled_metrics, f"{spec['key']}.overall")
    confusion_path = directory / "confusion_matrix.json"
    standalone_confusion_matrix = confusion_path.is_file()
    if standalone_confusion_matrix:
        saved_confusion = _read_json(confusion_path)
        if isinstance(saved_confusion, dict):
            saved_confusion = saved_confusion.get("confusion_matrix")
        if saved_confusion != pooled_metrics["confusion_matrix"]:
            raise ValueError(
                f"{spec['name']} standalone confusion matrix differs from OOF rows."
            )

    saved_folds = _read_json(per_fold_path)
    if not isinstance(saved_folds, list) or len(saved_folds) != 5:
        raise ValueError(f"{spec['name']} must have five saved per-fold metric records.")
    fold_metrics = []
    fold_roc_curves = []
    roc_data = _read_json(roc_path)
    per_fold_roc = roc_data.get("per_fold")
    if not isinstance(per_fold_roc, list) or len(per_fold_roc) != 5:
        raise ValueError(f"{spec['name']} does not have five saved ROC fold curves.")
    for fold_id in range(5):
        mask = fold_ids == fold_id
        calculated_fold = calculate_metrics(y_true[mask], y_pred[mask], probability_adhd[mask])
        saved_fold = saved_folds[fold_id]
        _validate_saved_metrics(
            saved_fold, calculated_fold, f"{spec['key']}.fold_{fold_id}"
        )
        fold_file = directory / f"outer_fold_{fold_id}" / "fold_metrics.json"
        if fold_file.is_file():
            _validate_saved_metrics(
                _read_json(fold_file),
                calculated_fold,
                f"{spec['key']}.outer_fold_{fold_id}",
            )
        _validate_roc_data(
            per_fold_roc[fold_id],
            y_true[mask],
            probability_adhd[mask],
            calculated_fold["roc_auc"],
            f"{spec['key']}.fold_{fold_id}.roc_curve",
        )
        fold_metrics.append(
            {
                "fold_id": fold_id,
                "epoch_count": int(mask.sum()),
                **calculated_fold,
            }
        )
        fold_roc_curves.append(per_fold_roc[fold_id])

    overall_roc = roc_data.get("overall")
    if not isinstance(overall_roc, dict):
        raise ValueError(f"{spec['name']} is missing its pooled ROC curve.")
    _validate_roc_data(
        overall_roc,
        y_true,
        probability_adhd,
        pooled_metrics["roc_auc"],
        f"{spec['key']}.overall.roc_curve",
    )

    history = None
    history_path = directory / "outer_fold_0" / "training_history.json"
    if history_path.is_file():
        fold_histories = []
        for fold_id in range(5):
            fold_histories.append(
                {
                    "fold_id": fold_id,
                    **_history_summary(
                        directory
                        / f"outer_fold_{fold_id}"
                        / "training_history.json"
                    ),
                }
            )
        history = {
            "folds": fold_histories,
            "folds_final_validation_above_best": sum(
                item["final_validation_above_best"] for item in fold_histories
            ),
            "folds_training_fell_after_best": sum(
                item["training_fell_after_best"] for item in fold_histories
            ),
            "folds_showing_both_signals": sum(
                item["final_validation_above_best"]
                and item["training_fell_after_best"]
                for item in fold_histories
            ),
        }

    return {
        "key": spec["key"],
        "name": spec["name"],
        "family": spec["family"],
        "source_run": spec["source_run"].as_posix(),
        "artifact_directory": (
            spec["source_run"] / spec["subdirectory"]
        ).as_posix(),
        "prediction_count": len(rows),
        "has_standalone_confusion_matrix": standalone_confusion_matrix,
        "pooled_metrics": pooled_metrics,
        "fold_metrics": fold_metrics,
        "fold_summary": summarize_folds(fold_metrics),
        "roc_curves": {
            "overall": overall_roc,
            "per_fold": fold_roc_curves,
        },
        "learning_curve_analysis": history,
        "_ordered_predictions": ordered,
    }


def _remove_private_values(experiments: list[dict[str, Any]]) -> None:
    for experiment in experiments:
        experiment.pop("_ordered_predictions", None)


def _compare_transformer_rows(
    baseline: dict[str, Any], fusion_reference: dict[str, Any]
) -> dict[str, Any]:
    baseline_rows = baseline["_ordered_predictions"]
    fusion_rows = fusion_reference["_ordered_predictions"]
    fields = (
        "epoch_index",
        "y_true",
        "y_pred",
        "probability_control",
        "probability_adhd",
        "group_id",
        "fold_id",
    )
    identical = all(
        left[field] == right[field]
        for left, right in zip(baseline_rows, fusion_rows)
        for field in fields
    )
    return {
        "status": "PASS",
        "exact_prediction_rows_match": bool(identical),
        "compared_rows": len(baseline_rows),
        "interpretation": (
            "The two named Transformer baseline entries are the same saved OOF "
            "predictions and are duplicate evidence."
            if identical
            else "The two Transformer baseline entries have distinct saved OOF predictions."
        ),
    }


def build_comparison(root: Path) -> dict[str, Any]:
    root = Path(root)
    data = load_cell1_dataset(root / "processed" / "cell_01_epochs.npz")
    experiments = [_load_experiment(root, spec, data) for spec in SPECS]
    experiment_by_key = {item["key"]: item for item in experiments}

    duplicate_check = _compare_transformer_rows(
        experiment_by_key["eeg_transformer"],
        experiment_by_key["raw_transformer_baseline"],
    )
    rank_order = sorted(
        experiments,
        key=lambda item: item["pooled_metrics"]["balanced_accuracy"],
        reverse=True,
    )
    previous_score = None
    previous_rank = 0
    for position, item in enumerate(rank_order, start=1):
        score = item["pooled_metrics"]["balanced_accuracy"]
        if previous_score is None or not math.isclose(
            score, previous_score, rel_tol=0, abs_tol=1e-15
        ):
            previous_rank = position
            previous_score = score
        item["pooled_balanced_accuracy_rank"] = previous_rank

    pooled_lookup = {
        normalize_header(item["key"]): item["pooled_metrics"] for item in experiments
    }
    pooled_lookup.update(
        {
            normalize_header(item["name"]): item["pooled_metrics"]
            for item in experiments
        }
    )
    fold_lookup = {
        normalize_header(item["key"]): item["fold_summary"] for item in experiments
    }
    fold_lookup.update(
        {
            normalize_header(item["name"]): item["fold_summary"]
            for item in experiments
        }
    )

    baseline_keys = {
        item["key"]: item for item in experiments if item["family"] == "deep_learning"
    }
    baseline_aliases = {
        "eegnet": baseline_keys["eegnet"]["pooled_metrics"],
        "cnnlstm": baseline_keys["cnn_lstm"]["pooled_metrics"],
        "eegtransformer": baseline_keys["eeg_transformer"]["pooled_metrics"],
        "eegconformer": baseline_keys["eeg_conformer"]["pooled_metrics"],
    }
    baseline_fold_aliases = {
        "eegnet": baseline_keys["eegnet"]["fold_summary"],
        "cnnlstm": baseline_keys["cnn_lstm"]["fold_summary"],
        "eegtransformer": baseline_keys["eeg_transformer"]["fold_summary"],
        "eegconformer": baseline_keys["eeg_conformer"]["fold_summary"],
    }
    fusion_items = {
        item["key"]: item
        for item in experiments
        if item["family"] in {"raw_baseline", "handcrafted_only", "feature_fusion"}
    }
    fusion_aliases = {
        "araw eegtransformer": fusion_items["raw_transformer_baseline"]["pooled_metrics"],
        "bhandcraftedlogistic": fusion_items["handcrafted_logistic"]["pooled_metrics"],
        "ctransformerplushandcrafted": fusion_items["transformer_feature_fusion"]["pooled_metrics"],
    }
    fusion_fold_aliases = {
        "araw eegtransformer": fusion_items["raw_transformer_baseline"]["fold_summary"],
        "bhandcraftedlogistic": fusion_items["handcrafted_logistic"]["fold_summary"],
        "ctransformerplushandcrafted": fusion_items["transformer_feature_fusion"]["fold_summary"],
    }
    report_checks = [
        _check_reported_metrics(
            root / "reports" / "deep_learning_training_report.md",
            baseline_aliases,
            "training_pooled",
        ),
        _check_reported_metrics(
            root / "reports" / "deep_learning_error_analysis.md",
            baseline_fold_aliases,
            "fold_summary",
        ),
        _check_reported_metrics(
            root / "reports" / "deep_feature_fusion_report.md",
            fusion_aliases,
            "pooled",
        ),
        _check_reported_metrics(
            root / "reports" / "deep_feature_fusion_report.md",
            fusion_fold_aliases,
            "long_fold_summary",
        ),
        _check_reported_confusion_matrices(
            root / "reports" / "deep_learning_error_analysis.md",
            list(baseline_keys.values()),
        ),
    ]
    standalone_confusion_count = sum(
        item["has_standalone_confusion_matrix"] for item in experiments
    )
    if standalone_confusion_count != 3:
        raise ValueError(
            "Expected standalone confusion-matrix artifacts for all three feature-fusion arms; "
            f"found {standalone_confusion_count}."
        )

    _remove_private_values(experiments)
    return {
        "title": "Comparative Cell 1 ADHD vs Healthy Control experiments",
        "class_mapping": {"Control": 0, "ADHD": 1},
        "positive_class": "ADHD",
        "evaluation": {
            "cell": 1,
            "epoch_count": int(len(data.y)),
            "class_counts": {
                "Control": int(np.count_nonzero(data.y == 0)),
                "ADHD": int(np.count_nonzero(data.y == 1)),
            },
            "provisional_group_count": int(np.unique(data.groups).size),
            "outer_fold_ids": sorted(int(value) for value in np.unique(data.folds)),
            "decision_threshold": 0.5,
            "fold_summary": "Unweighted mean and sample standard deviation (ddof=1) over five outer folds.",
            "pooled_summary": "Metrics recalculated once from all 553 saved OOF predictions.",
        },
        "source_runs": {
            "deep_learning": BASELINE_RUN.as_posix(),
            "feature_fusion": FUSION_RUN.as_posix(),
            "cell1_metadata": "processed/cell_01_epochs.npz",
        },
        "validation": {
            "status": "PASS",
            "oof_labels_groups_folds_probabilities_metrics_and_confusion_matrices": "PASS",
            "saved_per_fold_metrics_and_outer_fold_metrics": "PASS",
            "saved_pooled_and_per_fold_roc_curves": "PASS",
            "saved_standalone_confusion_matrices": "PASS",
            "existing_report_numeric_crosschecks": report_checks,
            "transformer_baseline_duplicate_check": duplicate_check,
        },
        "experiments": experiments,
    }


def _metric_text(metrics: dict[str, Any], metric: str) -> str:
    return f"{metrics[metric]:.4f}"


def _fold_metric_text(summary: dict[str, Any], metric: str) -> str:
    item = summary[metric]
    return f"{item['mean']:.4f} ± {item['sample_standard_deviation']:.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def render_report(comparison: dict[str, Any]) -> str:
    experiments = comparison["experiments"]
    pooled_rows = [
        [
            str(item["pooled_balanced_accuracy_rank"]),
            item["name"],
            *[_metric_text(item["pooled_metrics"], metric) for metric in (
                "accuracy",
                "balanced_accuracy",
                "recall_sensitivity",
                "specificity",
                "precision",
                "f1_score",
                "roc_auc",
            )],
        ]
        for item in sorted(
            experiments,
            key=lambda entry: (
                entry["pooled_balanced_accuracy_rank"],
                -entry["pooled_metrics"]["roc_auc"],
            ),
        )
    ]
    fold_rows = [
        [
            item["name"],
            *[_fold_metric_text(item["fold_summary"], metric) for metric in (
                "accuracy",
                "balanced_accuracy",
                "recall_sensitivity",
                "specificity",
                "precision",
                "f1_score",
                "roc_auc",
            )],
        ]
        for item in experiments
    ]
    per_fold_rows = []
    for item in experiments:
        for fold in item["fold_metrics"]:
            per_fold_rows.append(
                [
                    item["name"],
                    str(fold["fold_id"]),
                    str(fold["epoch_count"]),
                    *[f"{fold[metric]:.4f}" for metric in (
                        "accuracy",
                        "balanced_accuracy",
                        "recall_sensitivity",
                        "specificity",
                        "precision",
                        "f1_score",
                        "roc_auc",
                    )],
                ]
            )
    confusion_rows = []
    for item in experiments:
        tn, fp = item["pooled_metrics"]["confusion_matrix"][0]
        fn, tp = item["pooled_metrics"]["confusion_matrix"][1]
        confusion_rows.append([item["name"], str(tn), str(fp), str(fn), str(tp)])

    by_key = {item["key"]: item for item in experiments}
    handcrafted = by_key["handcrafted_logistic"]["pooled_metrics"]
    raw = by_key["raw_transformer_baseline"]["pooled_metrics"]
    fusion = by_key["transformer_feature_fusion"]["pooled_metrics"]
    delta_handcrafted = {
        metric: handcrafted[metric] - raw[metric] for metric in METRICS
    }
    delta_fusion = {metric: fusion[metric] - raw[metric] for metric in METRICS}
    balanced_ranked = sorted(
        experiments,
        key=lambda item: item["pooled_metrics"]["balanced_accuracy"],
        reverse=True,
    )
    auc_ranked = sorted(
        experiments,
        key=lambda item: item["pooled_metrics"]["roc_auc"],
        reverse=True,
    )
    leading_ba = balanced_ranked[0]
    leading_auc = auc_ranked[0]
    leading_sensitivity = max(
        experiments,
        key=lambda item: item["pooled_metrics"]["recall_sensitivity"],
    )
    leading_specificity = max(
        experiments,
        key=lambda item: item["pooled_metrics"]["specificity"],
    )
    fp_dominant = []
    fn_dominant = []
    for item in experiments:
        fp = item["pooled_metrics"]["confusion_matrix"][0][1]
        fn = item["pooled_metrics"]["confusion_matrix"][1][0]
        if fp > fn:
            fp_dominant.append(item["name"])
        elif fn > fp:
            fn_dominant.append(item["name"])
    duplicate_note = comparison["validation"]["transformer_baseline_duplicate_check"]
    if duplicate_note["exact_prediction_rows_match"]:
        transformer_note = (
            "The EEG-Transformer-inspired model and the fusion-run Raw EEG Transformer "
            "baseline have identical saved OOF rows, including probabilities. They are "
            "shown under both requested names but are one baseline result, not independent evidence."
        )
    else:
        transformer_note = (
            "The two Transformer baseline entries have distinct saved OOF rows and are "
            "reported separately."
        )

    history_rows = []
    for item in experiments:
        history = item["learning_curve_analysis"]
        if history is None:
            detail = "Not applicable: classical fit; no epoch-wise training history."
            history_rows.append([item["name"], "—", "—", detail])
            continue
        detail = (
            f"{history['folds_showing_both_signals']}/5 folds show both indicators"
        )
        history_rows.append(
            [
                item["name"],
                f"{history['folds_final_validation_above_best']}/5",
                f"{history['folds_training_fell_after_best']}/5",
                detail,
            ]
        )

    validation_rows = [
        [
            Path(item["file"]).name,
            str(
                item.get(
                    "numeric_values_checked",
                    item.get("confusion_matrix_cells_checked", 0),
                )
            ),
            item["status"],
        ]
        for item in comparison["validation"]["existing_report_numeric_crosschecks"]
    ]
    plot_prefix = (
        "../results/deep_learning/cell_01/final_comparison/plots/"
    )

    lines = [
        "# Comparative Model Evaluation: ADHD vs Healthy Control",
        "",
        "## Scope and evaluation",
        "",
        f"- **Cell:** 1 only; {comparison['evaluation']['epoch_count']} epochs "
        f"({comparison['evaluation']['class_counts']['Control']} Control, "
        f"{comparison['evaluation']['class_counts']['ADHD']} ADHD).",
        "- **Experiments:** four raw-EEG deep architectures, handcrafted-feature-only "
        "logistic regression, raw EEG Transformer baseline, and Transformer plus handcrafted fusion.",
        "- **Source runs:**",
        f"  - Deep learning: `{comparison['source_runs']['deep_learning']}`",
        f"  - Feature fusion: `{comparison['source_runs']['feature_fusion']}`",
        "- The existing saved five outer-fold assignments and OOF predictions were used; "
        "no model was retrained and no threshold was tuned on test folds.",
        "- Class mapping is Control = 0 and ADHD = 1. Sensitivity is recall for ADHD. "
        "All reported metrics were recalculated from saved predictions.",
        "- Pooled OOF metrics and unweighted five-fold mean ± sample SD are distinct summaries. "
        "The fold SD is descriptive (ddof=1), not a confidence interval.",
        "",
        "## Pooled out-of-fold metrics",
        "",
        "Ranking is by pooled balanced accuracy. A duplicate Transformer baseline is retained "
        "under its requested label; see the note below.",
        "",
        _markdown_table(
            [
                "Rank",
                "Experiment",
                "Accuracy",
                "Balanced accuracy",
                "ADHD sensitivity",
                "Specificity",
                "Precision",
                "F1",
                "ROC-AUC",
            ],
            pooled_rows,
        ),
        "",
        "## Five-fold mean ± sample standard deviation",
        "",
        "Each cell below is the unweighted mean ± sample SD of the five outer-fold metrics; "
        "these are not pooled OOF values.",
        "",
        _markdown_table(
            [
                "Experiment",
                "Accuracy",
                "Balanced accuracy",
                "ADHD sensitivity",
                "Specificity",
                "Precision",
                "F1",
                "ROC-AUC",
            ],
            fold_rows,
        ),
        "",
        "## All five outer-fold results",
        "",
        _markdown_table(
            [
                "Experiment",
                "Fold",
                "Epochs",
                "Accuracy",
                "Balanced accuracy",
                "ADHD sensitivity",
                "Specificity",
                "Precision",
                "F1",
                "ROC-AUC",
            ],
            per_fold_rows,
        ),
        "",
        "## Confusion matrices and ROC-AUC",
        "",
        "Confusion matrix counts are pooled OOF values with rows = true [Control, ADHD] "
        "and columns = predicted [Control, ADHD]. TN/FP/FN/TP correspond to that ordering.",
        "",
        _markdown_table(
            ["Experiment", "TN", "FP (Control → ADHD)", "FN (ADHD → Control)", "TP"],
            confusion_rows,
        ),
        "",
        f"By pooled OOF ROC-AUC, the leader is **{leading_auc['name']}** "
        f"({leading_auc['pooled_metrics']['roc_auc']:.4f}); the ROC plot compares saved "
        "OOF curves. The pooled AUC ranking is separate from the balanced-accuracy ranking.",
        "",
        f"![Pooled OOF ROC curves]({plot_prefix}pooled_roc_curves.png)",
        "",
        f"![Pooled OOF confusion matrices]({plot_prefix}pooled_confusion_matrices.png)",
        "",
        "## Model ranking and interpretation",
        "",
        f"- **Highest pooled balanced accuracy:** {leading_ba['name']} "
        f"({leading_ba['pooled_metrics']['balanced_accuracy']:.4f}). "
        f"**Highest pooled ROC-AUC:** {leading_auc['name']} "
        f"({leading_auc['pooled_metrics']['roc_auc']:.4f}). These criteria do not identify "
        "one universally superior model.",
        f"- Among the four raw-EEG deep architectures, "
        f"{by_key['eeg_transformer']['name']} has the highest pooled balanced accuracy "
        f"({by_key['eeg_transformer']['pooled_metrics']['balanced_accuracy']:.4f}); "
        f"handcrafted-only Logistic Regression ({handcrafted['balanced_accuracy']:.4f}) "
        f"and fusion ({fusion['balanced_accuracy']:.4f}) are higher on this comparison. "
        "This is descriptive evidence for these saved splits only.",
        f"- Handcrafted-only Logistic Regression versus the raw Transformer reference: "
        f"pooled accuracy {delta_handcrafted['accuracy']:+.4f}, balanced accuracy "
        f"{delta_handcrafted['balanced_accuracy']:+.4f}, sensitivity "
        f"{delta_handcrafted['recall_sensitivity']:+.4f}, specificity "
        f"{delta_handcrafted['specificity']:+.4f}, precision "
        f"{delta_handcrafted['precision']:+.4f}, F1 {delta_handcrafted['f1_score']:+.4f}, "
        f"and ROC-AUC {delta_handcrafted['roc_auc']:+.4f}. All seven pooled point metrics "
        "and all seven fold means are higher for this handcrafted-only arm than for the "
        "raw Transformer reference in these saved splits; this does not establish a general "
        "advantage for handcrafted features.",
        f"- Transformer + handcrafted fusion versus the raw Transformer reference: pooled "
        f"accuracy {delta_fusion['accuracy']:+.4f}, balanced accuracy "
        f"{delta_fusion['balanced_accuracy']:+.4f}, sensitivity "
        f"{delta_fusion['recall_sensitivity']:+.4f}, specificity "
        f"{delta_fusion['specificity']:+.4f}, precision {delta_fusion['precision']:+.4f}, "
        f"F1 {delta_fusion['f1_score']:+.4f}, and ROC-AUC {delta_fusion['roc_auc']:+.4f}. "
        "All seven pooled point metrics and all seven fold means are higher for fusion than "
        "the raw Transformer reference in this run; with five folds, this is descriptive "
        "rather than inferential.",
        f"- Against handcrafted-only Logistic Regression, fusion changes pooled balanced "
        f"accuracy by {fusion['balanced_accuracy'] - handcrafted['balanced_accuracy']:+.4f}, "
        f"sensitivity by {fusion['recall_sensitivity'] - handcrafted['recall_sensitivity']:+.4f}, "
        f"specificity by {fusion['specificity'] - handcrafted['specificity']:+.4f}, and "
        f"ROC-AUC by {fusion['roc_auc'] - handcrafted['roc_auc']:+.4f}. Thus fusion's "
        "relative advantage depends on metric: it is not uniformly better than the handcrafted-only arm.",
        f"- **Sensitivity/specificity trade-off:** pooled ADHD sensitivity is highest for "
        f"{leading_sensitivity['name']} ({leading_sensitivity['pooled_metrics']['recall_sensitivity']:.4f}); "
        f"specificity is highest for {leading_specificity['name']} "
        f"({leading_specificity['pooled_metrics']['specificity']:.4f}). At the fixed 0.5 "
        "threshold, false-positive errors exceed false negatives for "
        f"{', '.join(fp_dominant)}, while false negatives exceed false positives for "
        f"{', '.join(fn_dominant)}. These are descriptive classification trade-offs, "
        "not diagnostic recommendations.",
        f"- {transformer_note}",
        "",
        f"![Fold-mean sensitivity and specificity]({plot_prefix}sensitivity_specificity_fold_means.png)",
        "",
        f"![Pooled metric comparison]({plot_prefix}pooled_metric_comparison.png)",
        "",
        f"![Fold balanced accuracy and ROC-AUC]({plot_prefix}fold_metric_means.png)",
        "",
        "## Training-history and variability evidence",
        "",
        "For neural models, a final validation loss above the best validation loss together "
        "with training loss below its best-validation-epoch value is a descriptive pattern "
        "consistent with possible late-epoch overfitting. It is not proof of overfitting. "
        "The logistic-regression estimator has no epoch-wise history.",
        "",
        _markdown_table(
            [
                "Experiment",
                "Folds final validation > best",
                "Folds training loss fell after best",
                "Interpretation",
            ],
            history_rows,
        ),
        "",
        "Variation across only five folds is substantial for some metrics, especially "
        "sensitivity/specificity for several neural models. Fold SDs are descriptive and "
        "do not quantify uncertainty for new participants. The scatter plot shows fold means "
        "with sample-SD error bars.",
        "",
        "## Existing-report checks",
        "",
        "The established training, error-analysis, and fusion reports were numerically checked "
        "against metrics independently recomputed from saved OOF predictions.",
        "",
        _markdown_table(
            ["Report", "Values/cells checked", "Result"],
            validation_rows,
        ),
        "",
        "## Limitations",
        "",
        "- Only Cell 1 was used in this comparison.",
        "- Participant identities were not independently verified. Group IDs are provisional "
        "filename/storage-slot identifiers, not verified participant IDs.",
        "- Consequently, the saved group folds do not establish participant-independent "
        "generalization.",
        "- Results do not establish clinical diagnostic performance or utility.",
        "- ADHD subtype classification is not part of this objective.",
        "- Five-fold summaries are descriptive; no significance testing or universal ranking "
        "is supported by these saved experiments.",
        f"- {transformer_note}",
        "",
        "## Reproducibility and generated artifacts",
        "",
        "All source prediction, metric, history, and report artifacts were left unchanged. "
        "This comparison creates a separate JSON, CSV tables, and plots under "
        "`results/deep_learning/cell_01/final_comparison/`.",
        "",
    ]
    return "\n".join(lines)


def _experiment_label(experiment: dict[str, Any]) -> str:
    labels = {
        "eegnet": "EEGNet-inspired",
        "cnn_lstm": "CNN-LSTM",
        "eeg_transformer": "EEG-Transformer-inspired",
        "eeg_conformer": "EEG Conformer",
        "handcrafted_logistic": "Handcrafted logistic",
        "raw_transformer_baseline": "Raw Transformer baseline",
        "transformer_feature_fusion": "Transformer + features",
    }
    return labels[experiment["key"]]


def write_comparison_csvs(comparison: dict[str, Any], output_dir: Path) -> None:
    pooled_path = output_dir / "pooled_comparison.csv"
    headers = [
        "experiment",
        "balanced_accuracy_rank",
        *METRICS,
    ]
    with pooled_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for item in comparison["experiments"]:
            writer.writerow(
                {
                    "experiment": item["name"],
                    "balanced_accuracy_rank": item["pooled_balanced_accuracy_rank"],
                    **{
                        metric: item["pooled_metrics"][metric]
                        for metric in METRICS
                    },
                }
            )

    fold_path = output_dir / "per_fold_comparison.csv"
    fold_headers = ["experiment", "fold_id", "epoch_count", *METRICS]
    with fold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_headers)
        writer.writeheader()
        for item in comparison["experiments"]:
            for fold in item["fold_metrics"]:
                writer.writerow(
                    {
                        "experiment": item["name"],
                        "fold_id": fold["fold_id"],
                        "epoch_count": fold["epoch_count"],
                        **{metric: fold[metric] for metric in METRICS},
                    }
                )


def create_plots(comparison: dict[str, Any], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    experiments = comparison["experiments"]
    labels = [_experiment_label(item) for item in experiments]

    metric_names = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("recall_sensitivity", "ADHD sensitivity"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("f1_score", "F1"),
        ("roc_auc", "ROC-AUC"),
    ]
    x = np.arange(len(experiments))
    width = 0.1
    fig, axis = plt.subplots(figsize=(13, 6))
    for index, (metric, label) in enumerate(metric_names):
        values = [item["pooled_metrics"][metric] for item in experiments]
        axis.bar(x + (index - 3) * width, values, width, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Pooled OOF score")
    axis.set_title("Pooled out-of-fold metric comparison")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "pooled_metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for axis, item in zip(axes.flat, experiments):
        matrix = np.asarray(item["pooled_metrics"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(_experiment_label(item))
        axis.set_xticks([0, 1], labels=["Control", "ADHD"])
        axis.set_yticks([0, 1], labels=["Control", "ADHD"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
    for axis in axes.flat[len(experiments) :]:
        axis.axis("off")
    fig.suptitle("Pooled OOF confusion matrices")
    fig.tight_layout()
    fig.savefig(plot_dir / "pooled_confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7))
    for item in experiments:
        curve = item["roc_curves"]["overall"]
        axis.plot(
            curve["false_positive_rate"],
            curve["true_positive_rate"],
            label=f"{_experiment_label(item)} (AUC {item['pooled_metrics']['roc_auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Pooled OOF ROC curves")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "pooled_roc_curves.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7))
    for item in experiments:
        sensitivity = item["fold_summary"]["recall_sensitivity"]
        specificity = item["fold_summary"]["specificity"]
        axis.errorbar(
            sensitivity["mean"],
            specificity["mean"],
            xerr=sensitivity["sample_standard_deviation"],
            yerr=specificity["sample_standard_deviation"],
            marker="o",
            capsize=3,
            label=_experiment_label(item),
        )
    axis.set_xlim(0.55, 1.05)
    axis.set_ylim(0.55, 1.05)
    axis.set_xlabel("Fold-mean ADHD sensitivity")
    axis.set_ylabel("Fold-mean specificity")
    axis.set_title("Fold-mean sensitivity/specificity (sample SD error bars)")
    axis.legend(fontsize=8, loc="lower left")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "sensitivity_specificity_fold_means.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, metric, title in (
        (axes[0], "balanced_accuracy", "Balanced accuracy"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ):
        means = [item["fold_summary"][metric]["mean"] for item in experiments]
        deviations = [
            item["fold_summary"][metric]["sample_standard_deviation"]
            for item in experiments
        ]
        axis.errorbar(
            np.arange(len(experiments)),
            means,
            yerr=deviations,
            marker="o",
            capsize=3,
            linestyle="none",
        )
        axis.set_xticks(np.arange(len(experiments)))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_ylim(0.65, 1.03)
        axis.set_title(f"Fold mean {title} ± sample SD")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "fold_metric_means.png", dpi=180)
    plt.close(fig)


def generate_outputs(root: Path) -> dict[str, Any]:
    root = Path(root)
    output_dir = root / "results" / "deep_learning" / "cell_01" / "final_comparison"
    report_path = root / "reports" / "final_model_comparison.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"

    comparison = build_comparison(root)
    report_text = render_report(comparison)
    create_plots(comparison, plot_dir)
    write_comparison_csvs(comparison, output_dir)
    json_path = output_dir / "model_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, allow_nan=False)
        handle.write("\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return {
        "report": report_path,
        "json": json_path,
        "output_directory": output_dir,
        "experiment_count": len(comparison["experiments"]),
        "validation": comparison["validation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare existing saved Cell 1 experiments without retraining."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing processed/, reports/, and results/.",
    )
    args = parser.parse_args()
    result = generate_outputs(args.project_root)
    print(f"Generated report: {result['report']}")
    print(f"Generated JSON: {result['json']}")
    print(f"Generated tables and plots: {result['output_directory']}")
    print(f"Experiments compared: {result['experiment_count']}")
    print(f"Validation: {result['validation']['status']}")


if __name__ == "__main__":
    main()
