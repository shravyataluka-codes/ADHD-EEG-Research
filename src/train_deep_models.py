"""Leakage-aware grouped cross-validation for raw Cell 1 EEG models."""

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.deep_models import MODEL_REGISTRY, build_model
from src.modeling_data import ModelingData, load_cell1_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ChannelNormalizer:
    """Per-channel training-set normalization parameters."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, epochs):
        data = np.asarray(epochs, dtype=np.float64)
        if data.ndim != 3 or data.shape[2] != self.mean.size:
            raise ValueError(
                "Epochs must have shape [epochs, samples, channels] matching "
                "the fitted normalizer."
            )
        if not np.isfinite(data).all():
            raise ValueError("Cannot normalize epochs containing non-finite values.")
        return ((data - self.mean) / self.scale).astype(np.float32)


def fit_channel_normalizer(training_epochs):
    """Fit channel-wise mean and standard deviation on training epochs only."""
    data = np.asarray(training_epochs, dtype=np.float64)
    if data.ndim != 3 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(
            "Training epochs must have non-empty "
            "[epochs, samples, channels] dimensions."
        )
    if not np.isfinite(data).all():
        raise ValueError("Training epochs contain non-finite values.")
    mean = np.mean(data, axis=(0, 1))
    scale = np.std(data, axis=(0, 1))
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Training normalization statistics are non-finite.")
    if np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("A training channel has zero or negligible variance.")
    return ChannelNormalizer(mean=mean, scale=scale)


def group_aware_inner_split(groups, labels, validation_fraction, seed):
    """Return epoch indices for a deterministic, class-stratified group split."""
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    if groups.ndim != 1 or labels.ndim != 1 or len(groups) != len(labels):
        raise ValueError("groups and labels must be equal-length one-dimensional arrays.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")

    group_labels = {}
    for group, label in zip(groups.tolist(), labels.tolist()):
        if group in group_labels and group_labels[group] != label:
            raise ValueError(f"Group {group!r} has inconsistent class labels.")
        group_labels[group] = label

    class_groups = {}
    for group, label in group_labels.items():
        class_groups.setdefault(label, []).append(group)
    if len(class_groups) != 2:
        raise ValueError("Inner split requires both binary classes.")

    validation_groups = set()
    rng = np.random.default_rng(seed)
    for label in sorted(class_groups):
        ordered = sorted(class_groups[label])
        if len(ordered) < 2:
            raise ValueError(
                f"Class {label!r} needs at least two groups for an inner split."
            )
        shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
        validation_count = max(1, math.ceil(len(ordered) * validation_fraction))
        validation_count = min(validation_count, len(ordered) - 1)
        validation_groups.update(shuffled[:validation_count])

    validation_indices = np.asarray(
        [index for index, group in enumerate(groups) if group in validation_groups],
        dtype=np.int64,
    )
    training_indices = np.asarray(
        [index for index, group in enumerate(groups) if group not in validation_groups],
        dtype=np.int64,
    )
    if not training_indices.size or not validation_indices.size:
        raise ValueError("Inner split produced an empty training or validation set.")
    if set(groups[training_indices]) & set(groups[validation_indices]):
        raise RuntimeError("Group leakage detected in inner split.")
    if set(labels[training_indices].tolist()) != {0, 1}:
        raise ValueError("Inner training split must contain both classes.")
    if set(labels[validation_indices].tolist()) != {0, 1}:
        raise ValueError("Inner validation split must contain both classes.")
    return training_indices, validation_indices


def class_weights_from_training_labels(labels, enabled, imbalance_ratio_threshold):
    """Compute balanced weights from training labels only, when warranted."""
    labels = np.asarray(labels)
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training data must contain both classes.")
    ratio = float(counts.max() / counts.min())
    if not enabled or ratio < imbalance_ratio_threshold:
        return None, {"counts": counts.tolist(), "ratio": ratio, "applied": False}
    weights = labels.size / (2.0 * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32), {
        "counts": counts.tolist(),
        "ratio": ratio,
        "applied": True,
        "weights": weights.tolist(),
    }


def calculate_metrics(y_true, positive_probabilities, threshold=0.5):
    """Calculate binary metrics with ADHD (1) as the positive class.

    Undefined metrics are returned as ``None`` and explained in
    ``undefined_metrics`` rather than silently assigned a numeric fallback.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    if y_true.ndim != 1 or probabilities.ndim != 1 or y_true.size == 0:
        raise ValueError("Metric inputs must be non-empty one-dimensional arrays.")
    if y_true.size != probabilities.size:
        raise ValueError("Labels and probabilities must have equal lengths.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("Labels must contain only 0 (Control) and 1 (ADHD).")
    if not np.isfinite(probabilities).all():
        raise ValueError("Positive-class probabilities must be finite.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Positive-class probabilities must be in [0, 1].")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    y_pred = (probabilities >= threshold).astype(np.int64)
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    predicted_positive_count = true_positive + false_positive
    notes = {}

    precision = (
        true_positive / predicted_positive_count
        if predicted_positive_count
        else None
    )
    if precision is None:
        notes["precision"] = "No epochs were predicted as ADHD."
    recall = (
        true_positive / positive_count
        if positive_count
        else None
    )
    if recall is None:
        notes["recall_sensitivity"] = "No ADHD epochs are present in y_true."
    specificity = (
        true_negative / negative_count
        if negative_count
        else None
    )
    if specificity is None:
        notes["specificity"] = "No Control epochs are present in y_true."
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = (
        2 * true_positive / f1_denominator
        if f1_denominator
        else None
    )
    if f1 is None:
        notes["f1_score"] = "F1 is undefined when there are no positive cases or predictions."
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    if balanced_accuracy is None:
        notes["balanced_accuracy"] = "Both classes are required."

    roc_curve = {"false_positive_rate": [], "true_positive_rate": [], "thresholds": []}
    roc_auc = None
    if positive_count and negative_count:
        ranks = rankdata(probabilities, method="average")
        roc_auc = float(
            (ranks[y_true == 1].sum() - positive_count * (positive_count + 1) / 2)
            / (positive_count * negative_count)
        )
        sort_indices = np.argsort(-probabilities, kind="mergesort")
        sorted_probabilities = probabilities[sort_indices]
        sorted_labels = y_true[sort_indices]
        tp = 0
        fp = 0
        fpr_values = [0.0]
        tpr_values = [0.0]
        threshold_values = [None]
        index = 0
        while index < y_true.size:
            score = sorted_probabilities[index]
            next_index = index
            while (
                next_index < y_true.size
                and sorted_probabilities[next_index] == score
            ):
                if sorted_labels[next_index] == 1:
                    tp += 1
                else:
                    fp += 1
                next_index += 1
            fpr_values.append(fp / negative_count)
            tpr_values.append(tp / positive_count)
            threshold_values.append(float(score))
            index = next_index
        roc_curve = {
            "false_positive_rate": fpr_values,
            "true_positive_rate": tpr_values,
            "thresholds": threshold_values,
        }
    else:
        notes["roc_auc"] = "ROC-AUC requires both classes in y_true."

    return {
        "accuracy": float((true_positive + true_negative) / y_true.size),
        "balanced_accuracy": (
            float(balanced_accuracy) if balanced_accuracy is not None else None
        ),
        "precision": float(precision) if precision is not None else None,
        "recall_sensitivity": float(recall) if recall is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "f1_score": float(f1) if f1 is not None else None,
        "roc_auc": roc_auc,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "undefined_metrics": notes,
        "roc_curve": roc_curve,
        "threshold": float(threshold),
        "positive_class": 1,
        "class_mapping": {"Control": 0, "ADHD": 1},
    }


def validate_prediction_coverage(prediction_counts):
    """Require one and only one out-of-fold prediction for each epoch."""
    counts = np.asarray(prediction_counts)
    if counts.ndim != 1:
        raise ValueError("Prediction coverage counts must be one-dimensional.")
    missing = np.flatnonzero(counts == 0).tolist()
    duplicated = np.flatnonzero(counts > 1).tolist()
    invalid = np.flatnonzero((counts < 0) | (counts > 1)).tolist()
    if missing or duplicated or invalid:
        raise ValueError(
            "Out-of-fold prediction coverage failed: "
            f"missing indices={missing}, duplicated indices={duplicated}."
        )


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(requested)


def save_best_checkpoint(path, model, normalizer, metadata):
    """Save model weights and training-only normalization statistics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization_mean": normalizer.mean.tolist(),
            "normalization_scale": normalizer.scale.tolist(),
            "metadata": metadata,
        },
        path,
    )


def load_best_checkpoint(path, model, device):
    """Load a trusted project checkpoint without unrestricted pickle loading."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} has no model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _validate_group_disjointness(train_groups, validation_groups, test_groups):
    train_groups = set(train_groups.tolist())
    validation_groups = set(validation_groups.tolist())
    test_groups = set(test_groups.tolist())
    if train_groups & validation_groups:
        raise RuntimeError("A group occurs in both training and validation.")
    if train_groups & test_groups:
        raise RuntimeError("An outer-test group occurs in training.")
    if validation_groups & test_groups:
        raise RuntimeError("An outer-test group occurs in validation.")


def _smoke_subset(data, groups_per_class_per_fold, seed):
    group_info = {}
    for index, (group, label, fold) in enumerate(
        zip(data.groups.tolist(), data.y.tolist(), data.folds.tolist())
    ):
        existing = group_info.setdefault(
            group, {"label": label, "fold": fold, "indices": []}
        )
        if existing["label"] != label or existing["fold"] != fold:
            raise ValueError(f"Group {group!r} has inconsistent metadata.")
        existing["indices"].append(index)

    rng = np.random.default_rng(seed)
    selected_groups = set()
    for fold in sorted(set(data.folds.tolist())):
        for label in (0, 1):
            candidates = sorted(
                group
                for group, info in group_info.items()
                if info["fold"] == fold and info["label"] == label
            )
            if len(candidates) < groups_per_class_per_fold:
                raise ValueError(
                    f"Smoke test needs {groups_per_class_per_fold} groups for "
                    f"class {label} in fold {fold}; found {len(candidates)}."
                )
            shuffled = [candidates[i] for i in rng.permutation(len(candidates))]
            selected_groups.update(shuffled[:groups_per_class_per_fold])

    selected_indices = np.asarray(
        [
            index
            for index, group in enumerate(data.groups.tolist())
            if group in selected_groups
        ],
        dtype=np.int64,
    )
    return ModelingData(
        X=data.X[selected_indices],
        y=data.y[selected_indices],
        groups=data.groups[selected_indices],
        folds=data.folds[selected_indices],
        class_mapping=dict(data.class_mapping),
    )


def _tensor_epochs(epochs, normalizer):
    normalized = normalizer.transform(epochs)
    return torch.from_numpy(np.transpose(normalized, (0, 2, 1)).copy())


def _batch_loss_total_and_denominator(loss, targets, criterion):
    """Recover a batch loss sum and its correct CrossEntropyLoss denominator."""
    if criterion.weight is None:
        denominator = float(targets.numel())
    else:
        denominator = float(criterion.weight[targets].sum().item())
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("Loss aggregation denominator must be positive and finite.")
    return float(loss.item()) * denominator, denominator


def _mean_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_denominator = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = criterion(model(batch_x), batch_y)
            batch_loss_sum, batch_denominator = _batch_loss_total_and_denominator(
                loss, batch_y, criterion
            )
            total_loss += batch_loss_sum
            total_denominator += batch_denominator
    if total_denominator == 0:
        raise ValueError("Cannot calculate validation loss for an empty split.")
    return total_loss / total_denominator


def _predict_probabilities_once(model, epochs, device):
    """Run exactly one model forward pass over an outer-test fold."""
    model.eval()
    with torch.no_grad():
        logits = model(epochs.to(device))
        probabilities = torch.softmax(logits, dim=1)[:, 1]
    return probabilities.detach().cpu().numpy()


def _json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def _train_outer_fold(
    model_name,
    model_parameters,
    data,
    outer_fold,
    training_indices,
    validation_indices,
    test_indices,
    training_config,
    model_config,
    output_directory,
    device,
):
    optimizer_config = training_config["optimization"]
    seed = int(training_config["seed"]) + outer_fold
    set_global_seed(seed)

    _validate_group_disjointness(
        data.groups[training_indices],
        data.groups[validation_indices],
        data.groups[test_indices],
    )
    normalizer = fit_channel_normalizer(data.X[training_indices])
    train_x = _tensor_epochs(data.X[training_indices], normalizer)
    validation_x = _tensor_epochs(data.X[validation_indices], normalizer)
    test_x = _tensor_epochs(data.X[test_indices], normalizer)
    train_y = torch.from_numpy(data.y[training_indices].astype(np.int64, copy=True))
    validation_y = torch.from_numpy(
        data.y[validation_indices].astype(np.int64, copy=True)
    )

    class_weight_config = training_config["class_weighting"]
    class_weights, class_weight_report = class_weights_from_training_labels(
        data.y[training_indices],
        enabled=class_weight_config["enabled"],
        imbalance_ratio_threshold=class_weight_config[
            "imbalance_ratio_threshold"
        ],
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(
        model_name,
        in_channels=model_config["data"]["input_channels"],
        n_samples=model_config["data"]["input_samples"],
        num_classes=len(model_config["data"]["classes"]),
        **model_parameters,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )
    scheduler_config = optimizer_config["learning_rate_scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        min_lr=scheduler_config["minimum_learning_rate"],
    )

    dataset = TensorDataset(train_x, train_y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=optimizer_config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_x, validation_y),
        batch_size=optimizer_config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model_fold_directory = (
        Path(output_directory) / model_name / f"outer_fold_{outer_fold}"
    )
    checkpoint_path = model_fold_directory / "best_checkpoint.pt"
    history = []
    best_validation_loss = math.inf
    best_epoch = None
    epochs_without_improvement = 0
    maximum_epochs = optimizer_config["maximum_epochs"]
    patience = optimizer_config["early_stopping_patience"]
    minimum_improvement = optimizer_config["minimum_validation_improvement"]

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_training_loss = 0.0
        training_loss_denominator = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{model_name} fold {outer_fold} produced non-finite loss."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                optimizer_config["gradient_clip_norm"],
            )
            optimizer.step()
            batch_loss_sum, batch_denominator = _batch_loss_total_and_denominator(
                loss, batch_y, criterion
            )
            total_training_loss += batch_loss_sum
            training_loss_denominator += batch_denominator

        validation_loss = _mean_loss(
            model, validation_loader, criterion, device
        )
        if not math.isfinite(validation_loss):
            raise FloatingPointError(
                f"{model_name} fold {outer_fold} produced non-finite "
                "validation loss."
            )
        scheduler.step(validation_loss)
        record = {
            "epoch": epoch,
            "training_loss": total_training_loss / training_loss_denominator,
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        if validation_loss < best_validation_loss - minimum_improvement:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_best_checkpoint(
                checkpoint_path,
                model,
                normalizer,
                {
                    "model_name": model_name,
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "group_ids_are_verified_participant_ids": False,
                },
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_epoch is None:
        raise RuntimeError(
            f"No best checkpoint was selected for {model_name} fold {outer_fold}."
        )

    _json_write(model_fold_directory / "training_history.json", history)
    reloaded_checkpoint = load_best_checkpoint(checkpoint_path, model, device)
    checkpoint_normalizer = ChannelNormalizer(
        mean=np.asarray(reloaded_checkpoint["normalization_mean"], dtype=np.float64),
        scale=np.asarray(
            reloaded_checkpoint["normalization_scale"], dtype=np.float64
        ),
    )
    if not (
        np.array_equal(checkpoint_normalizer.mean, normalizer.mean)
        and np.array_equal(checkpoint_normalizer.scale, normalizer.scale)
    ):
        raise RuntimeError(
            "Loaded checkpoint normalization statistics do not match."
        )

    test_x_from_checkpoint = _tensor_epochs(data.X[test_indices], checkpoint_normalizer)
    test_probabilities = _predict_probabilities_once(
        model,
        test_x_from_checkpoint,
        device,
    )
    fold_metrics = calculate_metrics(data.y[test_indices], test_probabilities)
    fold_metrics.update(
        {
            "outer_fold": int(outer_fold),
            "test_epoch_count": int(test_indices.size),
            "test_group_count": int(len(set(data.groups[test_indices].tolist()))),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "class_weighting": class_weight_report,
        }
    )
    _json_write(model_fold_directory / "fold_metrics.json", fold_metrics)

    split_record = {
        "outer_fold": int(outer_fold),
        "training_group_ids": sorted(set(data.groups[training_indices].tolist())),
        "validation_group_ids": sorted(set(data.groups[validation_indices].tolist())),
        "test_group_ids": sorted(set(data.groups[test_indices].tolist())),
        "group_ids_are_verified_participant_ids": False,
    }
    _json_write(model_fold_directory / "group_split.json", split_record)
    return test_probabilities, fold_metrics


def _prediction_rows(data, probabilities, folds):
    predictions = (probabilities >= 0.5).astype(np.int64)
    return [
        {
            "epoch_index": int(index),
            "y_true": int(data.y[index]),
            "y_pred": int(predictions[index]),
            "probability_control": float(1.0 - probabilities[index]),
            "probability_adhd": float(probabilities[index]),
            "group_id": str(data.groups[index]),
            "fold_id": int(folds[index]),
        }
        for index in range(data.y.size)
    ]


def _write_prediction_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "epoch_index",
        "y_true",
        "y_pred",
        "probability_control",
        "probability_adhd",
        "group_id",
        "fold_id",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path, results):
    lines = [
        "# Deep-Learning Training and Evaluation Report",
        "",
        f"- Run mode: **{results['run_mode']}**",
        f"- Cell: **{results['cell']} only**",
        f"- Input epochs: **{results['epoch_count']}**",
        f"- Raw epoch shape: **{results['input_shape']}**",
        f"- Class mapping: **Control = 0, ADHD = 1**",
        f"- Device: **{results['device']}**",
        f"- Seed: **{results['seed']}**",
        "",
        "## Evaluation procedure",
        "",
        "Five existing Cell 1 fold assignments were used as outer test folds. "
        "For each outer fold, the other four folds were split into training "
        "and inner validation groups using deterministic class-stratified "
        "group sampling. No group crossed training, validation, and test "
        "splits. Training-only per-channel mean and standard deviation were "
        "fitted on the inner-training epochs and then applied unchanged to "
        "validation and test epochs. Early stopping and checkpoint selection "
        "used inner validation loss only. The selected checkpoint was loaded "
        "before one prediction pass on the outer test fold.",
        "",
        "Out-of-fold predictions were checked so each included epoch had "
        "exactly one prediction. ADHD is the positive class. Undefined metrics "
        "are recorded as `null` with an explanation rather than replaced by "
        "a misleading number.",
        "",
        "## Data and identity limitations",
        "",
        "The experiment uses raw Cell 1 epochs only; it does not use the "
        "40-column handcrafted feature matrix or combine recording cells. "
        "The two channel slots remain unnamed because channel order is not "
        "independently verified. Group IDs are provisional "
        "source-file/storage-slot identifiers, not verified participant IDs. "
        "Consequently, this procedure provides provisional group separation "
        "only and must not be described as verified participant-independent "
        "evaluation. No clinical diagnostic performance is established.",
        "",
        "## Models and results",
        "",
        "| Model | Parameters | OOF epochs | Accuracy | Balanced accuracy | "
        "Precision | Sensitivity | Specificity | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model_result in results["models"].items():
        metrics = model_result["overall_metrics"]
        values = [
            metrics.get("accuracy"),
            metrics.get("balanced_accuracy"),
            metrics.get("precision"),
            metrics.get("recall_sensitivity"),
            metrics.get("specificity"),
            metrics.get("f1_score"),
            metrics.get("roc_auc"),
        ]
        cells = ["N/A" if value is None else f"{value:.4f}" for value in values]
        lines.append(
            f"| {name} | {model_result['trainable_parameters']} | "
            f"{model_result['oof_epoch_count']} | " + " | ".join(cells) + " |"
        )
    lines.extend(["", "## Confusion matrices", ""])
    for name, model_result in results["models"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "Rows are true labels `[Control, ADHD]`; columns are predicted "
                "labels `[Control, ADHD]`.",
                "",
                "```json",
                json.dumps(
                    model_result["overall_metrics"]["confusion_matrix"],
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Hyperparameters",
            "",
            f"- Inner validation fraction: "
            f"`{results['inner_validation_fraction']}` (group-stratified).",
            f"- Optimization settings: "
            f"`{json.dumps(results['optimization'], sort_keys=True)}`",
            f"- Class weighting policy: "
            f"`{json.dumps(results['class_weighting_config'], sort_keys=True)}`",
            "",
            "Model-specific architecture settings:",
            "",
        ]
    )
    for name, parameters in results["model_configurations"].items():
        lines.extend(
            [
                f"- **{name}:** `{json.dumps(parameters, sort_keys=True)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Configuration and artifacts",
            "",
            f"- Model configuration: `{results['model_config_path']}`",
            f"- Training configuration: `{results['training_config_path']}`",
            f"- Output directory: `{results['output_directory']}`",
            "- Per-fold checkpoints, training histories, validation losses, "
            "group split records, metrics, out-of-fold predictions, and ROC "
            "curve data are saved under the output directory.",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _create_run_directory(output_root, smoke_test):
    mode = "smoke_test" if smoke_test else "full"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_directory = Path(output_root) / mode / f"run_{run_id}"
    output_directory.mkdir(parents=True, exist_ok=False)
    return output_directory


def _validate_configs(training_config, modeling_config, model_config):
    if modeling_config.get("dataset", {}).get("cell") != 1:
        raise ValueError("Training is restricted to Cell 1.")
    if modeling_config["dataset"].get("combine_other_cells") is not False:
        raise ValueError("Other recording cells must not be combined.")
    if model_config.get("data", {}).get("cell") != 1:
        raise ValueError("Deep-model configuration must select Cell 1.")
    if model_config["data"].get("input_representation") != "raw_epochs":
        raise ValueError("Deep models must use raw EEG epochs.")
    if model_config["data"].get("use_handcrafted_features_as_model_input") is not False:
        raise ValueError("Handcrafted features cannot be model inputs in this run.")
    if model_config["data"].get("input_channels") != 2:
        raise ValueError("Cell 1 must use two generic input channel slots.")
    if model_config["data"].get("input_samples") != 1024:
        raise ValueError("Cell 1 epochs must contain 1024 samples.")
    if model_config["data"].get("classes") != {"Control": 0, "ADHD": 1}:
        raise ValueError("Expected class mapping Control=0 and ADHD=1.")
    if set(model_config.get("models", {})) != set(MODEL_REGISTRY):
        raise ValueError(
            "Model configuration must provide exactly these models: "
            f"{', '.join(sorted(MODEL_REGISTRY))}."
        )
    if training_config.get("seed") != model_config.get("seed"):
        raise ValueError("Training and model configuration seeds must match.")


def run_training_pipeline(
    data,
    training_config,
    modeling_config,
    model_config,
    output_directory,
    report_path=None,
    smoke_test=False,
    device="cpu",
):
    """Run all four architectures over the existing five outer folds."""
    _validate_configs(training_config, modeling_config, model_config)
    if smoke_test:
        data = _smoke_subset(
            data,
            training_config["smoke_test"]["groups_per_class_per_fold"],
            training_config["seed"],
        )
    output_directory = Path(output_directory)
    if output_directory.exists():
        if any(output_directory.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_directory}"
            )
    else:
        output_directory.mkdir(parents=True)

    active_training_config = json.loads(json.dumps(training_config))
    if smoke_test:
        active_training_config["optimization"]["maximum_epochs"] = (
            training_config["smoke_test"]["maximum_epochs"]
        )
        active_training_config["optimization"]["early_stopping_patience"] = (
            training_config["smoke_test"]["early_stopping_patience"]
        )
        active_training_config["optimization"]["batch_size"] = (
            training_config["smoke_test"]["batch_size"]
        )

    unique_folds = sorted(np.unique(data.folds).tolist())
    if unique_folds != [0, 1, 2, 3, 4]:
        raise ValueError(
            f"Expected existing outer fold IDs 0-4, got {unique_folds}."
        )
    if set(np.unique(data.y).tolist()) != {0, 1}:
        raise ValueError("Cell 1 data must contain Control and ADHD labels.")

    results = {
        "run_mode": "smoke_test" if smoke_test else "full",
        "cell": 1,
        "epoch_count": int(data.y.size),
        "input_shape": [int(data.X.shape[1]), int(data.X.shape[2])],
        "class_counts": {
            "Control": int(np.sum(data.y == 0)),
            "ADHD": int(np.sum(data.y == 1)),
        },
        "group_count": int(len(set(data.groups.tolist()))),
        "group_ids_are_verified_participant_ids": False,
        "class_mapping": {"Control": 0, "ADHD": 1},
        "seed": int(training_config["seed"]),
        "device": str(device),
        "optimization": active_training_config["optimization"],
        "inner_validation_fraction": training_config["inner_validation"][
            "fraction"
        ],
        "class_weighting_config": training_config["class_weighting"],
        "training_config_path": str(training_config.get("_path", "")),
        "model_config_path": str(model_config.get("_path", "")),
        "output_directory": str(output_directory.resolve()),
        "model_configurations": model_config["models"],
        "models": {},
    }
    for model_name, model_parameters in model_config["models"].items():
        model_result = {
            "model_name": model_name,
            "model_parameters": model_parameters,
            "trainable_parameters": None,
            "per_fold_metrics": [],
            "overall_metrics": None,
            "oof_epoch_count": 0,
        }
        oof_probabilities = np.full(data.y.size, np.nan, dtype=np.float64)
        prediction_counts = np.zeros(data.y.size, dtype=np.int64)
        folds_for_prediction = np.full(data.y.size, -1, dtype=np.int64)

        for outer_fold in unique_folds:
            test_indices = np.flatnonzero(data.folds == outer_fold)
            outer_training_indices = np.flatnonzero(data.folds != outer_fold)
            relative_training, relative_validation = group_aware_inner_split(
                data.groups[outer_training_indices],
                data.y[outer_training_indices],
                validation_fraction=training_config["inner_validation"]["fraction"],
                seed=training_config["seed"] + outer_fold,
            )
            training_indices = outer_training_indices[relative_training]
            validation_indices = outer_training_indices[relative_validation]
            _validate_group_disjointness(
                data.groups[training_indices],
                data.groups[validation_indices],
                data.groups[test_indices],
            )

            model = build_model(
                model_name,
                in_channels=model_config["data"]["input_channels"],
                n_samples=model_config["data"]["input_samples"],
                num_classes=len(model_config["data"]["classes"]),
                **model_parameters,
            )
            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            if model_result["trainable_parameters"] is None:
                model_result["trainable_parameters"] = int(parameter_count)
            elif model_result["trainable_parameters"] != parameter_count:
                raise RuntimeError(
                    f"{model_name} parameter count changed between folds."
                )

            del model
            test_probabilities, fold_metrics = _train_outer_fold(
                model_name=model_name,
                model_parameters=model_parameters,
                data=data,
                outer_fold=outer_fold,
                training_indices=training_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                training_config=active_training_config,
                model_config=model_config,
                output_directory=output_directory,
                device=device,
            )
            if np.any(prediction_counts[test_indices] != 0):
                raise RuntimeError(
                    f"Duplicate out-of-fold prediction in outer fold {outer_fold}."
                )
            oof_probabilities[test_indices] = test_probabilities
            prediction_counts[test_indices] += 1
            folds_for_prediction[test_indices] = outer_fold
            model_result["per_fold_metrics"].append(fold_metrics)

        validate_prediction_coverage(prediction_counts)
        if not np.isfinite(oof_probabilities).all():
            raise RuntimeError(f"{model_name} has non-finite OOF probabilities.")
        model_result["overall_metrics"] = calculate_metrics(
            data.y, oof_probabilities
        )
        model_result["oof_epoch_count"] = int(prediction_counts.size)
        rows = _prediction_rows(data, oof_probabilities, folds_for_prediction)
        _write_prediction_csv(
            output_directory / model_name / "oof_predictions.csv",
            rows,
        )
        _json_write(
            output_directory / model_name / "per_fold_metrics.json",
            model_result["per_fold_metrics"],
        )
        _json_write(
            output_directory / model_name / "overall_metrics.json",
            model_result["overall_metrics"],
        )
        _json_write(
            output_directory / model_name / "roc_curves.json",
            {
                "overall": model_result["overall_metrics"]["roc_curve"],
                "per_fold": [
                    {
                        "outer_fold": metric["outer_fold"],
                        **metric["roc_curve"],
                    }
                    for metric in model_result["per_fold_metrics"]
                ],
            },
        )
        results["models"][model_name] = model_result

    _json_write(output_directory / "results.json", results)
    if report_path is None:
        report_path = output_directory / "deep_learning_training_report.md"
    _write_report(report_path, results)
    if Path(report_path).resolve() != (
        output_directory / "deep_learning_training_report.md"
    ).resolve():
        _write_report(
            output_directory / "deep_learning_training_report.md",
            results,
        )
    results["report_path"] = str(Path(report_path).resolve())
    _json_write(output_directory / "results.json", results)
    return results


def _load_yaml(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must contain a mapping.")
    config["_path"] = str(path)
    return config


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and evaluate Cell 1 deep EEG models with grouped CV."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "training_cell1.yaml"),
        help="Path to training_cell1.yaml.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a reduced one-epoch experiment in a separate smoke-test folder.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device (defaults to CUDA when available, otherwise CPU).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional new, empty output directory; it must not already exist.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    training_config_path = Path(args.config).resolve()
    training_config = _load_yaml(training_config_path)
    config_directory = training_config_path.parent
    modeling_config_path = (
        config_directory / training_config["configs"]["modeling_data"]
    ).resolve()
    model_config_path = (
        config_directory / training_config["configs"]["deep_models"]
    ).resolve()
    modeling_config = _load_yaml(modeling_config_path)
    model_config = _load_yaml(model_config_path)
    _validate_configs(training_config, modeling_config, model_config)

    input_path = (
        modeling_config_path.parent / modeling_config["dataset"]["input"]
    ).resolve()
    deep_config_input = (
        model_config_path.parent / model_config["data"]["input"]
    ).resolve()
    if input_path != deep_config_input:
        raise ValueError(
            "Modeling and deep-model configurations must point to the same "
            "Cell 1 input file."
        )
    data = load_cell1_dataset(input_path)
    expected_shape = (
        model_config["data"]["input_samples"],
        model_config["data"]["input_channels"],
    )
    if data.X.shape[1:] != expected_shape:
        raise ValueError(
            f"Loaded Cell 1 epochs have shape {data.X.shape[1:]}, "
            f"expected {expected_shape}."
        )

    device = choose_device(args.device)
    output_root = (config_directory / training_config["paths"]["output_root"]).resolve()
    output_directory = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else _create_run_directory(output_root, args.smoke_test)
    )
    if args.output_dir:
        output_directory.mkdir(parents=True, exist_ok=False)
    report_path = (
        output_directory / "deep_learning_training_report.md"
        if args.smoke_test
        else (config_directory / training_config["paths"]["full_report"]).resolve()
    )
    training_config["_path"] = str(training_config_path)
    model_config["_path"] = str(model_config_path)

    results = run_training_pipeline(
        data=data,
        training_config=training_config,
        modeling_config=modeling_config,
        model_config=model_config,
        output_directory=output_directory,
        report_path=report_path,
        smoke_test=args.smoke_test,
        device=device,
    )
    print(f"Completed {results['run_mode']} run.")
    print(f"Output directory: {results['output_directory']}")
    print(f"Report: {results['report_path']}")
    for model_name, model_result in results["models"].items():
        print(
            f"{model_name}: parameters={model_result['trainable_parameters']}, "
            f"epochs={model_result['oof_epoch_count']}, "
            f"accuracy={model_result['overall_metrics']['accuracy']:.4f}"
        )
    return results


if __name__ == "__main__":
    main()
