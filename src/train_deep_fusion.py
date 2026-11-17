"""Leakage-aware Cell 1 ablation of EEG, handcrafted, and fused inputs."""

import argparse
import csv
import json
import math
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.deep_fusion import EEGTransformerFusion
from src.deep_models import EEGTransformer
from src.features import extract_features
from src.modeling_data import load_cell1_dataset
from src.train_deep_models import (
    _batch_loss_total_and_denominator,
    _json_write,
    _tensor_epochs,
    calculate_metrics,
    choose_device,
    class_weights_from_training_labels,
    fit_channel_normalizer,
    group_aware_inner_split,
    load_best_checkpoint,
    set_global_seed,
    validate_prediction_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1_score",
    "roc_auc",
)


@dataclass(frozen=True)
class FeatureNormalizer:
    """Standardize handcrafted features using training epochs only."""

    mean: np.ndarray
    scale: np.ndarray
    constant_feature_indices: tuple

    def transform(self, features):
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError(
                "Handcrafted features must have shape [epochs, fitted_features]."
            )
        if not np.isfinite(values).all():
            raise ValueError("Cannot normalize non-finite handcrafted features.")
        return ((values - self.mean) / self.scale).astype(np.float32)


def fit_feature_normalizer(training_features):
    """Fit feature-wise mean and scale from the inner-training rows only."""
    values = np.asarray(training_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 40:
        raise ValueError(
            "Training features must be a non-empty [epochs, 40] matrix."
        )
    if not np.isfinite(values).all():
        raise ValueError("Training features contain NaN or infinite values.")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Feature normalization statistics are non-finite.")
    constant = np.flatnonzero(scale <= np.finfo(np.float64).eps)
    safe_scale = scale.copy()
    safe_scale[constant] = 1.0
    return FeatureNormalizer(
        mean=mean,
        scale=safe_scale,
        constant_feature_indices=tuple(int(index) for index in constant),
    )


def validate_feature_alignment(data, feature_dataset, expected_feature_count=40):
    """Require exact epoch-order and metadata alignment across modalities."""
    raw_epochs = np.asarray(data.X)
    features = np.asarray(feature_dataset.X)
    if raw_epochs.ndim != 3 or raw_epochs.shape[1:] != (1024, 2):
        raise ValueError(
            f"Raw Cell 1 epochs must have shape [N, 1024, 2]; got {raw_epochs.shape}."
        )
    if features.ndim != 2 or features.shape != (
        raw_epochs.shape[0],
        expected_feature_count,
    ):
        raise ValueError(
            "Handcrafted feature matrix must have exactly one 40-value row "
            f"per raw epoch; got {features.shape}."
        )
    if len(feature_dataset.feature_names) != expected_feature_count:
        raise ValueError("Feature name count does not match the feature matrix.")
    for field in ("y", "groups", "folds"):
        raw_metadata = np.asarray(getattr(data, field))
        feature_metadata = np.asarray(getattr(feature_dataset, field))
        if raw_metadata.shape != feature_metadata.shape or not np.array_equal(
            raw_metadata, feature_metadata
        ):
            raise ValueError(
                f"Raw EEG and handcrafted feature {field} are not exactly aligned."
            )
    if dict(data.class_mapping) != dict(feature_dataset.class_mapping):
        raise ValueError("Class mappings differ between input modalities.")
    if not np.isfinite(features).all():
        raise ValueError("Handcrafted feature matrix contains non-finite values.")


def validate_group_partition(data, training_indices, validation_indices, test_indices):
    """Validate complete outer coverage and disjoint provisional groups."""
    training_indices = np.asarray(training_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    training_groups = set(data.groups[training_indices].tolist())
    validation_groups = set(data.groups[validation_indices].tolist())
    test_groups = set(data.groups[test_indices].tolist())
    if training_groups & validation_groups:
        raise RuntimeError("A provisional group occurs in training and validation.")
    if training_groups & test_groups:
        raise RuntimeError("An outer-test group occurs in training.")
    if validation_groups & test_groups:
        raise RuntimeError("An outer-test group occurs in validation.")
    if not training_indices.size or not validation_indices.size or not test_indices.size:
        raise ValueError("Training, validation, and test splits must be non-empty.")
    if set(data.y[training_indices].tolist()) != {0, 1}:
        raise ValueError("Training split must contain Control and ADHD.")
    if set(data.y[validation_indices].tolist()) != {0, 1}:
        raise ValueError("Validation split must contain Control and ADHD.")
    if set(data.y[test_indices].tolist()) != {0, 1}:
        raise ValueError("Outer-test split must contain Control and ADHD.")
    return {
        "training_group_ids": sorted(training_groups),
        "validation_group_ids": sorted(validation_groups),
        "test_group_ids": sorted(test_groups),
        "group_ids_are_verified_participant_ids": False,
    }


def _validate_baseline_metadata(results, data, model_config, fusion_config):
    expected_count = data.y.size
    if results.get("run_mode") != "full" or results.get("cell") != 1:
        raise ValueError("Baseline reference must be the completed full Cell 1 run.")
    if results.get("epoch_count") != expected_count:
        raise ValueError("Baseline epoch count does not match the Cell 1 loader.")
    if results.get("group_ids_are_verified_participant_ids") is not False:
        raise ValueError("Baseline must retain provisional group identity metadata.")
    if results.get("seed") != fusion_config["seed"]:
        raise ValueError("Fusion and baseline random seeds must match.")
    if results.get("class_mapping") != {"Control": 0, "ADHD": 1}:
        raise ValueError("Baseline class mapping must be Control=0 and ADHD=1.")
    baseline_name = fusion_config["baseline"]["model_name"]
    baseline_parameters = model_config["models"]["eeg_transformer"]
    if results.get("model_configurations", {}).get(baseline_name) != baseline_parameters:
        raise ValueError("Baseline Transformer architecture config has changed.")
    if baseline_name not in results.get("models", {}):
        raise ValueError(f"Baseline run is missing model {baseline_name!r}.")


def _read_baseline_predictions(path, data):
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    indices = [int(row["epoch_index"]) for row in rows]
    if sorted(indices) != list(range(data.y.size)) or len(set(indices)) != data.y.size:
        raise ValueError("Baseline OOF predictions do not cover each epoch exactly once.")
    ordered = [None] * data.y.size
    for row in rows:
        ordered[int(row["epoch_index"])] = row
    probabilities = np.empty(data.y.size, dtype=np.float64)
    for index, row in enumerate(ordered):
        if int(row["y_true"]) != int(data.y[index]):
            raise ValueError(f"Baseline label misalignment at epoch {index}.")
        if row["group_id"] != data.groups[index]:
            raise ValueError(f"Baseline group misalignment at epoch {index}.")
        if int(row["fold_id"]) != int(data.folds[index]):
            raise ValueError(f"Baseline fold misalignment at epoch {index}.")
        probabilities[index] = float(row["probability_adhd"])
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Baseline OOF probabilities must be finite and in [0, 1].")
    return probabilities


def _validate_recomputed_metrics(actual, expected, description):
    for metric_name in METRIC_NAMES:
        actual_value = actual.get(metric_name)
        expected_value = expected.get(metric_name)
        if actual_value is None or expected_value is None:
            if actual_value is not None or expected_value is not None:
                raise ValueError(
                    f"{description} has inconsistent undefined metric {metric_name}."
                )
        elif not math.isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{description} {metric_name} does not match saved baseline."
            )
    if actual["confusion_matrix"] != expected["confusion_matrix"]:
        raise ValueError(f"{description} confusion matrix does not match baseline.")


def _load_baseline_splits(
    baseline_directory,
    data,
    model_parameters,
    validation_fraction,
    seed,
):
    """Verify baseline checkpoints and reuse their exact group split records."""
    splits = {}
    for outer_fold in range(5):
        fold_directory = (
            Path(baseline_directory)
            / "eeg_transformer"
            / f"outer_fold_{outer_fold}"
        )
        split_path = fold_directory / "group_split.json"
        with split_path.open("r", encoding="utf-8") as file:
            saved_split = json.load(file)
        checkpoint_path = fold_directory / "best_checkpoint.pt"
        backbone = EEGTransformer(
            in_channels=2,
            n_samples=1024,
            num_classes=2,
            **model_parameters,
        )
        checkpoint = load_best_checkpoint(
            checkpoint_path,
            backbone,
            torch.device("cpu"),
        )
        if len(checkpoint.get("normalization_mean", [])) != 2 or len(
            checkpoint.get("normalization_scale", [])
        ) != 2:
            raise ValueError(
                f"Baseline fold {outer_fold} has invalid EEG normalization metadata."
            )

        test_indices = np.flatnonzero(data.folds == outer_fold)
        outer_training_indices = np.flatnonzero(data.folds != outer_fold)
        relative_training, relative_validation = group_aware_inner_split(
            data.groups[outer_training_indices],
            data.y[outer_training_indices],
            validation_fraction=validation_fraction,
            seed=seed + outer_fold,
        )
        expected_training_groups = set(
            data.groups[outer_training_indices[relative_training]].tolist()
        )
        expected_validation_groups = set(
            data.groups[outer_training_indices[relative_validation]].tolist()
        )
        saved_training_groups = set(saved_split["training_group_ids"])
        saved_validation_groups = set(saved_split["validation_group_ids"])
        saved_test_groups = set(saved_split["test_group_ids"])
        actual_test_groups = set(data.groups[test_indices].tolist())
        if (
            saved_training_groups != expected_training_groups
            or saved_validation_groups != expected_validation_groups
            or saved_test_groups != actual_test_groups
        ):
            raise ValueError(
                f"Baseline fold {outer_fold} group splits do not match "
                "the configured deterministic split."
            )
        outer_training_groups = data.groups[outer_training_indices]
        training_indices = outer_training_indices[
            np.isin(outer_training_groups, list(saved_training_groups))
        ]
        validation_indices = outer_training_indices[
            np.isin(outer_training_groups, list(saved_validation_groups))
        ]
        expected_normalizer = fit_channel_normalizer(data.X[training_indices])
        if not (
            np.allclose(
                checkpoint["normalization_mean"],
                expected_normalizer.mean,
                rtol=1e-12,
                atol=1e-12,
            )
            and np.allclose(
                checkpoint["normalization_scale"],
                expected_normalizer.scale,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"Baseline fold {outer_fold} normalization was not fit "
                "on its saved training epochs."
            )
        record = validate_group_partition(
            data,
            training_indices,
            validation_indices,
            test_indices,
        )
        splits[outer_fold] = {
            "training_indices": training_indices,
            "validation_indices": validation_indices,
            "test_indices": test_indices,
            "record": record,
            "source": "completed_baseline_group_split",
        }
    return splits


def _smoke_indices(data, groups_per_class_per_fold, seed):
    group_info = {}
    for index, (group, label, fold) in enumerate(
        zip(data.groups.tolist(), data.y.tolist(), data.folds.tolist())
    ):
        info = group_info.setdefault(
            group,
            {"label": int(label), "fold": int(fold), "indices": []},
        )
        if info["label"] != label or info["fold"] != fold:
            raise ValueError(f"Group {group!r} has inconsistent metadata.")
        info["indices"].append(index)
    rng = np.random.default_rng(seed)
    selected_groups = set()
    for fold in range(5):
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
    return np.asarray(
        [
            index
            for index, group in enumerate(data.groups.tolist())
            if group in selected_groups
        ],
        dtype=np.int64,
    )


def _make_smoke_splits(data, included_indices, validation_fraction, seed):
    splits = {}
    for outer_fold in range(5):
        test_indices = included_indices[data.folds[included_indices] == outer_fold]
        outer_indices = included_indices[data.folds[included_indices] != outer_fold]
        relative_training, relative_validation = group_aware_inner_split(
            data.groups[outer_indices],
            data.y[outer_indices],
            validation_fraction=validation_fraction,
            seed=seed + outer_fold,
        )
        training_indices = outer_indices[relative_training]
        validation_indices = outer_indices[relative_validation]
        record = validate_group_partition(
            data,
            training_indices,
            validation_indices,
            test_indices,
        )
        splits[outer_fold] = {
            "training_indices": training_indices,
            "validation_indices": validation_indices,
            "test_indices": test_indices,
            "record": record,
            "source": "smoke_test_group_aware_split",
        }
    return splits


def _write_prediction_csv(path, data, included_indices, probabilities):
    if probabilities.shape != (len(included_indices),) or not np.isfinite(
        probabilities
    ).all():
        raise ValueError("Prediction probabilities do not cover included epochs.")
    rows = []
    for index, probability in zip(included_indices, probabilities):
        rows.append(
            {
                "epoch_index": int(index),
                "y_true": int(data.y[index]),
                "y_pred": int(probability >= 0.5),
                "probability_control": float(1.0 - probability),
                "probability_adhd": float(probability),
                "group_id": str(data.groups[index]),
                "fold_id": int(data.folds[index]),
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _predict_fusion(model, raw_epochs, features, device, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, raw_epochs.shape[0], batch_size):
            logits = model(
                raw_epochs[start : start + batch_size].to(device),
                features[start : start + batch_size].to(device),
            )
            predictions.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(predictions, dtype=np.float64)


def _fusion_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_denominator = 0.0
    with torch.no_grad():
        for batch_raw, batch_features, batch_labels in loader:
            batch_raw = batch_raw.to(device)
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            loss = criterion(model(batch_raw, batch_features), batch_labels)
            batch_total, batch_denominator = _batch_loss_total_and_denominator(
                loss,
                batch_labels,
                criterion,
            )
            total_loss += batch_total
            total_denominator += batch_denominator
    if total_denominator <= 0:
        raise ValueError("Cannot calculate validation loss for an empty split.")
    return total_loss / total_denominator


def _save_fusion_checkpoint(path, model, raw_normalizer, feature_normalizer, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "raw_normalization_mean": raw_normalizer.mean.tolist(),
            "raw_normalization_scale": raw_normalizer.scale.tolist(),
            "feature_normalization_mean": feature_normalizer.mean.tolist(),
            "feature_normalization_scale": feature_normalizer.scale.tolist(),
            "constant_feature_indices": list(
                feature_normalizer.constant_feature_indices
            ),
            "metadata": metadata,
        },
        path,
    )


def _load_fusion_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Fusion checkpoint at {path} has no model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _build_fusion_model(model_config, fusion_config, initial_checkpoint, device):
    backbone = EEGTransformer(
        in_channels=2,
        n_samples=1024,
        num_classes=2,
        **model_config["models"]["eeg_transformer"],
    )
    if initial_checkpoint is not None:
        checkpoint = torch.load(
            initial_checkpoint,
            map_location=device,
            weights_only=True,
        )
        backbone.load_state_dict(checkpoint["model_state_dict"])
    model = EEGTransformerFusion(
        in_channels=2,
        n_samples=1024,
        num_classes=2,
        feature_count=fusion_config["feature_count"],
        fusion_dim=fusion_config["projection_dim"],
        head_hidden_dim=fusion_config["classifier_hidden_dim"],
        dropout=fusion_config["dropout"],
        freeze_transformer_backbone=fusion_config[
            "freeze_transformer_backbone"
        ],
        backbone=backbone,
    )
    return model.to(device)


def _train_fusion_fold(
    data,
    feature_dataset,
    split,
    outer_fold,
    model_config,
    fusion_config,
    training_config,
    initial_checkpoint,
    fold_directory,
    device,
):
    training_indices = split["training_indices"]
    validation_indices = split["validation_indices"]
    test_indices = split["test_indices"]
    optimizer_config = training_config["optimization"]
    seed = int(training_config["seed"]) + outer_fold
    set_global_seed(seed)

    raw_normalizer = fit_channel_normalizer(data.X[training_indices])
    feature_normalizer = fit_feature_normalizer(
        feature_dataset.X[training_indices]
    )
    train_raw = _tensor_epochs(data.X[training_indices], raw_normalizer)
    validation_raw = _tensor_epochs(data.X[validation_indices], raw_normalizer)
    test_raw = _tensor_epochs(data.X[test_indices], raw_normalizer)
    train_features = torch.from_numpy(
        feature_normalizer.transform(feature_dataset.X[training_indices])
    )
    validation_features = torch.from_numpy(
        feature_normalizer.transform(feature_dataset.X[validation_indices])
    )
    test_features = torch.from_numpy(
        feature_normalizer.transform(feature_dataset.X[test_indices])
    )
    train_labels = torch.from_numpy(data.y[training_indices].astype(np.int64))
    validation_labels = torch.from_numpy(
        data.y[validation_indices].astype(np.int64)
    )

    class_weights, class_weight_report = class_weights_from_training_labels(
        data.y[training_indices],
        enabled=training_config["class_weighting"]["enabled"],
        imbalance_ratio_threshold=training_config["class_weighting"][
            "imbalance_ratio_threshold"
        ],
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = _build_fusion_model(
        model_config,
        fusion_config,
        initial_checkpoint,
        device,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_raw, train_features, train_labels),
        batch_size=optimizer_config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_raw, validation_features, validation_labels),
        batch_size=optimizer_config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    fold_directory = Path(fold_directory)
    checkpoint_path = fold_directory / "best_checkpoint.pt"
    history = []
    best_validation_loss = math.inf
    best_epoch = None
    epochs_without_improvement = 0
    for epoch in range(1, optimizer_config["maximum_epochs"] + 1):
        model.train()
        training_loss_total = 0.0
        training_loss_denominator = 0.0
        for batch_raw, batch_features, batch_labels in train_loader:
            batch_raw = batch_raw.to(device)
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_raw, batch_features), batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Fusion fold {outer_fold} produced a non-finite loss."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                optimizer_config["gradient_clip_norm"],
            )
            optimizer.step()
            batch_total, batch_denominator = _batch_loss_total_and_denominator(
                loss,
                batch_labels,
                criterion,
            )
            training_loss_total += batch_total
            training_loss_denominator += batch_denominator

        validation_loss = _fusion_loss(
            model,
            validation_loader,
            criterion,
            device,
        )
        if not math.isfinite(validation_loss):
            raise FloatingPointError(
                f"Fusion fold {outer_fold} produced a non-finite validation loss."
            )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss_total / training_loss_denominator,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        minimum_improvement = optimizer_config["minimum_validation_improvement"]
        if validation_loss < best_validation_loss - minimum_improvement:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_fusion_checkpoint(
                checkpoint_path,
                model,
                raw_normalizer,
                feature_normalizer,
                {
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "class_weighting": class_weight_report,
                    "group_ids_are_verified_participant_ids": False,
                    "initial_transformer_checkpoint": (
                        str(Path(initial_checkpoint).resolve())
                        if initial_checkpoint is not None
                        else None
                    ),
                    "freeze_transformer_backbone": fusion_config[
                        "freeze_transformer_backbone"
                    ],
                },
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= optimizer_config["early_stopping_patience"]:
            break
    if best_epoch is None:
        raise RuntimeError(f"No fusion checkpoint selected for fold {outer_fold}.")

    fold_directory.mkdir(parents=True, exist_ok=True)
    _json_write(fold_directory / "training_history.json", history)
    checkpoint = _load_fusion_checkpoint(checkpoint_path, model, device)
    loaded_raw_normalizer = np.asarray(
        checkpoint["raw_normalization_mean"], dtype=np.float64
    )
    loaded_feature_mean = np.asarray(
        checkpoint["feature_normalization_mean"], dtype=np.float64
    )
    if not (
        np.array_equal(loaded_raw_normalizer, raw_normalizer.mean)
        and np.array_equal(
            checkpoint["raw_normalization_scale"], raw_normalizer.scale
        )
        and np.array_equal(loaded_feature_mean, feature_normalizer.mean)
        and np.array_equal(
            checkpoint["feature_normalization_scale"], feature_normalizer.scale
        )
    ):
        raise RuntimeError("Loaded checkpoint normalization statistics differ.")

    validation_probabilities = _predict_fusion(
        model,
        validation_raw,
        validation_features,
        device,
        optimizer_config["batch_size"],
    )
    test_probabilities = _predict_fusion(
        model,
        test_raw,
        test_features,
        device,
        optimizer_config["batch_size"],
    )
    validation_metrics = calculate_metrics(
        data.y[validation_indices],
        validation_probabilities,
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
            "constant_feature_indices": list(
                feature_normalizer.constant_feature_indices
            ),
        }
    )
    _json_write(fold_directory / "fold_metrics.json", fold_metrics)
    _json_write(fold_directory / "validation_metrics.json", validation_metrics)
    np.savez_compressed(
        fold_directory / "feature_normalization.npz",
        mean=feature_normalizer.mean,
        scale=feature_normalizer.scale,
        constant_feature_indices=np.asarray(
            feature_normalizer.constant_feature_indices,
            dtype=np.int64,
        ),
    )
    return test_probabilities, fold_metrics


def _train_logistic_fold(
    data,
    feature_dataset,
    split,
    outer_fold,
    configuration,
    training_config,
    fold_directory,
):
    training_indices = split["training_indices"]
    validation_indices = split["validation_indices"]
    test_indices = split["test_indices"]
    normalizer = fit_feature_normalizer(
        feature_dataset.X[training_indices]
    )
    train_x = normalizer.transform(feature_dataset.X[training_indices])
    validation_x = normalizer.transform(feature_dataset.X[validation_indices])
    test_x = normalizer.transform(feature_dataset.X[test_indices])
    train_y = data.y[training_indices]
    counts = np.bincount(train_y.astype(np.int64), minlength=2)
    ratio = float(counts.max() / counts.min())
    use_class_weight = bool(
        configuration["class_weighting_uses_training_fold"]
        and training_config["class_weighting"]["enabled"]
        and ratio
        >= training_config["class_weighting"]["imbalance_ratio_threshold"]
    )
    class_weight = "balanced" if use_class_weight else None
    classifier = LogisticRegression(
        C=configuration["C"],
        max_iter=configuration["maximum_iterations"],
        class_weight=class_weight,
        random_state=configuration["random_state"] + outer_fold,
        solver="lbfgs",
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(train_x, train_y)
    convergence_warnings = [
        str(item.message)
        for item in captured
        if issubclass(item.category, ConvergenceWarning)
    ]

    def positive_probability(values):
        positive_index = np.flatnonzero(classifier.classes_ == 1)
        if positive_index.size != 1:
            raise RuntimeError("Logistic regression did not fit the ADHD class.")
        return classifier.predict_proba(values)[:, int(positive_index[0])]

    validation_probabilities = positive_probability(validation_x)
    test_probabilities = positive_probability(test_x)
    validation_metrics = calculate_metrics(
        data.y[validation_indices],
        validation_probabilities,
    )
    fold_metrics = calculate_metrics(data.y[test_indices], test_probabilities)
    fold_metrics.update(
        {
            "outer_fold": int(outer_fold),
            "test_epoch_count": int(test_indices.size),
            "test_group_count": int(len(set(data.groups[test_indices].tolist()))),
            "training_class_counts": counts.tolist(),
            "class_weight": class_weight,
            "constant_feature_indices": list(normalizer.constant_feature_indices),
            "convergence_warnings": convergence_warnings,
        }
    )
    fold_directory = Path(fold_directory)
    fold_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": classifier,
            "normalization_mean": normalizer.mean,
            "normalization_scale": normalizer.scale,
            "feature_names": list(feature_dataset.feature_names),
            "constant_feature_indices": list(normalizer.constant_feature_indices),
        },
        fold_directory / "classical_checkpoint.joblib",
    )
    restored = joblib.load(fold_directory / "classical_checkpoint.joblib")
    if not (
        np.array_equal(
            restored["estimator"].coef_,
            classifier.coef_,
        )
        and np.array_equal(restored["estimator"].intercept_, classifier.intercept_)
        and np.array_equal(restored["normalization_mean"], normalizer.mean)
        and np.array_equal(restored["normalization_scale"], normalizer.scale)
    ):
        raise RuntimeError("Classical checkpoint reload changed fitted parameters.")
    _json_write(
        fold_directory / "fit_summary.json",
        {
            "outer_fold": int(outer_fold),
            "model": "LogisticRegression",
            "C": float(configuration["C"]),
            "maximum_iterations": int(configuration["maximum_iterations"]),
            "class_weight": class_weight,
            "training_class_counts": counts.tolist(),
            "iterations": classifier.n_iter_.tolist(),
            "convergence_warnings": convergence_warnings,
            "normalization_mean": normalizer.mean.tolist(),
            "normalization_scale": normalizer.scale.tolist(),
            "constant_feature_indices": list(normalizer.constant_feature_indices),
            "group_ids_are_verified_participant_ids": False,
        },
    )
    _json_write(fold_directory / "validation_metrics.json", validation_metrics)
    _json_write(fold_directory / "fold_metrics.json", fold_metrics)
    return test_probabilities, fold_metrics


def _write_group_splits(output_directory, splits):
    for outer_fold, split in splits.items():
        _json_write(
            Path(output_directory)
            / "group_splits"
            / f"outer_fold_{outer_fold}.json",
            {
                "outer_fold": int(outer_fold),
                "split_source": split["source"],
                **split["record"],
            },
        )


def _write_arm_results(
    output_directory,
    arm_name,
    data,
    included_indices,
    probabilities,
    per_fold_metrics,
    model_metadata,
):
    arm_directory = Path(output_directory) / arm_name
    arm_directory.mkdir(parents=True, exist_ok=True)
    if probabilities.shape != (included_indices.size,):
        raise ValueError(f"{arm_name} OOF probability count is incorrect.")
    overall_metrics = calculate_metrics(
        data.y[included_indices],
        probabilities,
    )
    _write_prediction_csv(
        arm_directory / "oof_predictions.csv",
        data,
        included_indices,
        probabilities,
    )
    _json_write(arm_directory / "per_fold_metrics.json", per_fold_metrics)
    _json_write(arm_directory / "overall_metrics.json", overall_metrics)
    _json_write(
        arm_directory / "confusion_matrix.json",
        overall_metrics["confusion_matrix"],
    )
    _json_write(
        arm_directory / "roc_curves.json",
        {
            "overall": overall_metrics["roc_curve"],
            "per_fold": [
                {
                    "outer_fold": metric["outer_fold"],
                    **metric["roc_curve"],
                }
                for metric in per_fold_metrics
            ],
        },
    )
    _json_write(
        arm_directory / "arm_metadata.json",
        {
            **model_metadata,
            "included_epoch_count": int(included_indices.size),
            "group_ids_are_verified_participant_ids": False,
            "threshold": 0.5,
            "positive_class": "ADHD (1)",
        },
    )
    return overall_metrics


def _metric_summary(per_fold_metrics):
    summary = {}
    for name in METRIC_NAMES:
        values = [
            metric[name]
            for metric in per_fold_metrics
            if metric.get(name) is not None
        ]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "sample_standard_deviation": (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            ),
            "defined_folds": len(values),
        }
    return summary


def _write_comparison_tables(output_directory, arms):
    comparison_directory = Path(output_directory) / "comparison"
    comparison_directory.mkdir(parents=True, exist_ok=True)
    with (comparison_directory / "fold_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = ("arm", "outer_fold", "test_epoch_count", *METRIC_NAMES)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for arm_name, arm in arms.items():
            for metrics in arm["per_fold_metrics"]:
                writer.writerow(
                    {
                        "arm": arm_name,
                        "outer_fold": metrics["outer_fold"],
                        "test_epoch_count": metrics["test_epoch_count"],
                        **{name: metrics.get(name) for name in METRIC_NAMES},
                    }
                )

    summary_rows = []
    for arm_name, arm in arms.items():
        arm_summary = _metric_summary(arm["per_fold_metrics"])
        for metric_name, summary in arm_summary.items():
            summary_rows.append(
                {
                    "arm": arm_name,
                    "metric": metric_name,
                    "fold_mean": summary["mean"],
                    "fold_sample_standard_deviation": summary[
                        "sample_standard_deviation"
                    ],
                    "pooled_oof": arm["overall_metrics"].get(metric_name),
                }
            )
    with (comparison_directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = (
            "arm",
            "metric",
            "fold_mean",
            "fold_sample_standard_deviation",
            "pooled_oof",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def _write_report(path, results):
    path = Path(path)
    if results["fusion_configuration"]["warm_start_from_fold_baseline"]:
        initialization_note = (
            "Full runs initialize from the matching fold's baseline Transformer "
            "checkpoint; smoke runs start from a seeded initialization to "
            "avoid using baseline training data in their newly formed validation "
            "splits."
        )
    else:
        initialization_note = (
            "The fusion Transformer starts from a seeded initialization in both "
            "smoke and full runs."
        )
    lines = [
        "# Deep Feature-Fusion Experiment",
        "",
        f"- Run mode: **{results['run_mode']}**",
        f"- Cell: **Cell {results['cell']} only**",
        f"- Epochs in source Cell 1 data: **{results['raw_epoch_count']}**",
        f"- Epochs evaluated: **{results['included_epoch_count']}** "
        f"(Control={results['class_counts']['Control']}, "
        f"ADHD={results['class_counts']['ADHD']})",
        f"- Raw EEG shape: **[2, 1024]**; handcrafted input: **40 features**",
        "- Class mapping: **Control = 0, ADHD = 1**; ADHD is positive.",
        "- Threshold: **0.5**, fixed; no threshold tuning on test folds.",
        f"- Group IDs are provisional: **{results['group_ids_are_verified_participant_ids'] is False}**",
        "",
        "## Methods and leakage controls",
        "",
        "The existing completed EEG-Transformer full-run OOF predictions and "
        "fold checkpoints provide Arm A without modifying or retraining the "
        "baseline. Its saved group splits were checked against the Cell 1 "
        "metadata and deterministic group-aware split procedure. Arms B and C "
        "use those exact same inner and outer splits in a full run. The smoke "
        "run uses a small group subset and creates new group-aware inner splits, "
        "while retaining the same five outer-fold IDs.",
        "",
        "Handcrafted features were generated independently per epoch using the "
        "existing feature extractor, preserving its 40 names and ordering. "
        "Raw EEG and feature rows, labels, groups, and folds were checked for "
        "exact epoch-order alignment. Both raw-channel normalization and "
        "feature-wise standardization were fitted on inner-training epochs "
        "only. The logistic-regression estimator is fit on inner-training rows "
        "only; the fusion model uses validation loss for scheduling, early "
        "stopping, and checkpoint selection. Each included epoch receives "
        "exactly one OOF prediction.",
        "",
        "Group IDs are provisional filename/storage-slot identifiers, not "
        "verified participant IDs. Results are not verified participant-"
        "independent evaluation and do not establish clinical utility. The two "
        "channel slots remain unnamed.",
        "",
        "## Ablation arms",
        "",
        "- **A - Raw EEG Transformer:** existing full-run OOF reference. "
        "Checkpoint and history copies are stored in the fusion output; the "
        "original baseline files remain unchanged.",
        "- **B - Handcrafted-only logistic regression:** 40 existing features, "
        "training-only feature normalization, fixed L2 logistic regression.",
        "- **C - EEG Transformer + handcrafted fusion:** "
        f"{initialization_note} The model uses a compact "
        f"{results['fusion_configuration']['projection_dim']}-dimensional "
        "projection per modality and a 2-class head. Backbone freezing is "
        "configurable.",
        "",
        "## Results",
        "",
    ]
    if results["run_mode"] == "smoke_test":
        lines.extend(
            [
                "**Smoke-test only. This reduced run is an implementation "
                "check, not the full ablation experiment. It cannot establish "
                "whether fusion improves performance. Run the full command "
                "below for the configured five-fold comparison.**",
                "",
            ]
        )
    lines.extend(
        [
            "| Arm | Accuracy | Balanced accuracy | Sensitivity | Specificity | "
            "F1 | ROC-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm_name, arm in results["arms"].items():
        metrics = arm["overall_metrics"]
        values = [
            metrics.get("accuracy"),
            metrics.get("balanced_accuracy"),
            metrics.get("recall_sensitivity"),
            metrics.get("specificity"),
            metrics.get("f1_score"),
            metrics.get("roc_auc"),
        ]
        cells = ["N/A" if value is None else f"{value:.4f}" for value in values]
        lines.append(f"| {arm_name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### Fold mean and sample standard deviation",
            "",
            "| Arm | Metric | Mean | Sample SD (ddof=1) |",
            "|---|---|---:|---:|",
        ]
    )
    for arm_name, arm in results["arms"].items():
        for metric_name, summary in _metric_summary(
            arm["per_fold_metrics"]
        ).items():
            mean = summary["mean"]
            deviation = summary["sample_standard_deviation"]
            lines.append(
                f"| {arm_name} | {metric_name} | "
                f"{'N/A' if mean is None else f'{mean:.4f}'} | "
                f"{'N/A' if deviation is None else f'{deviation:.4f}'} |"
            )
    lines.extend(
        [
            "",
            "### Per-fold sensitivity and specificity",
            "",
            "| Arm | Fold | Sensitivity | Specificity | Balanced accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm_name, arm in results["arms"].items():
        for metric in arm["per_fold_metrics"]:
            lines.append(
                f"| {arm_name} | {metric['outer_fold']} | "
                f"{metric['recall_sensitivity']:.4f} | "
                f"{metric['specificity']:.4f} | "
                f"{metric['balanced_accuracy']:.4f} |"
            )
    lines.extend(
        [
            "",
            "### Fusion comparison",
            "",
            results["fusion_comparison_text"],
            "",
            "Confusion matrices and ROC data are saved per arm. Metric "
            "undefinedness is preserved as `null` with the explanations from "
            "the existing metrics implementation. Fold means and standard "
            "deviations are descriptive only; no significance testing or "
            "threshold selection was performed.",
            "",
            "## Configuration and artifacts",
            "",
            f"- Baseline source (read-only): `{results['baseline_directory']}`",
            f"- Run output: `{results['output_directory']}`",
            f"- Feature order preserved: `{json.dumps(results['feature_names'])}`",
            f"- Transformer configuration: `{json.dumps(results['transformer_configuration'], sort_keys=True)}`",
            f"- Handcrafted baseline configuration: `{json.dumps(results['handcrafted_baseline_configuration'], sort_keys=True)}`",
            f"- Optimization configuration: `{json.dumps(results['training_configuration'], sort_keys=True)}`",
            f"- Fusion configuration: `{json.dumps(results['fusion_configuration'], sort_keys=True)}`",
            "- Per-fold checkpoints, normalization parameters, training "
            "histories, group-split records, OOF predictions, metrics, "
            "confusion matrices, ROC data, and comparison tables are under "
            "the run output directory.",
            "",
            "## Commands",
            "",
            "```powershell",
            "python -m src.train_deep_fusion --smoke-test --device cpu",
            "python -m src.train_deep_fusion --device cpu",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fusion_comparison_text(results):
    if results["run_mode"] == "smoke_test":
        return (
            "No performance conclusion is drawn from the smoke subset. The "
            "full run will compare fusion with the baseline across the same "
            "five outer folds."
        )
    baseline = results["arms"]["A_raw_eeg_transformer"]
    fusion = results["arms"]["C_transformer_plus_handcrafted"]
    comparison_metrics = (
        "accuracy",
        "balanced_accuracy",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
    )
    lines = []
    for name in comparison_metrics:
        differences = [
            fusion_fold[name] - baseline_fold[name]
            for baseline_fold, fusion_fold in zip(
                baseline["per_fold_metrics"],
                fusion["per_fold_metrics"],
            )
            if baseline_fold.get(name) is not None
            and fusion_fold.get(name) is not None
        ]
        if not differences:
            lines.append(f"- {name}: unavailable for one or more folds.")
            continue
        mean_difference = float(np.mean(differences))
        direction = "higher" if mean_difference > 0 else (
            "lower" if mean_difference < 0 else "unchanged"
        )
        lines.append(
            f"- {name}: fusion was {direction} by {abs(mean_difference):.4f} "
            "on average across paired outer folds "
            f"(fold differences: {', '.join(f'{value:+.4f}' for value in differences)})."
        )
    lines.append(
        "This is a descriptive five-fold comparison. Improvement is not "
        "claimed unless the paired fold results and pooled OOF metrics support "
        "it; fold variability and provisional group identities limit "
        "generalization."
    )
    return "\n".join(lines)


def _validate_configs(fusion_config, modeling_config, model_config):
    if fusion_config.get("seed") != model_config.get("seed"):
        raise ValueError("Fusion and deep-model seeds must match.")
    if fusion_config.get("data", {}).get("cell") != 1:
        raise ValueError("Feature fusion is restricted to Cell 1.")
    if fusion_config["data"].get("handcrafted_feature_count") != 40:
        raise ValueError("Cell 1 must use the existing 40 handcrafted features.")
    if modeling_config.get("dataset", {}).get("cell") != 1:
        raise ValueError("Modeling configuration must select Cell 1.")
    if modeling_config["dataset"].get("combine_other_cells") is not False:
        raise ValueError("Other recording cells must not be combined.")
    if model_config["data"].get("input_representation") != "raw_epochs":
        raise ValueError("The Transformer arm must use raw EEG epochs.")
    if model_config["data"].get("input_channels") != 2:
        raise ValueError("Cell 1 must use two generic channel slots.")
    if model_config["data"].get("input_samples") != 1024:
        raise ValueError("Cell 1 epochs must contain 1024 samples.")
    if model_config["data"].get("classes") != {"Control": 0, "ADHD": 1}:
        raise ValueError("Expected Control=0 and ADHD=1.")
    if set(model_config.get("models", {})) != {
        "eegnet",
        "cnn_lstm",
        "eeg_transformer",
        "eeg_conformer",
    }:
        raise ValueError("Existing deep-model configuration is incomplete.")


def _prepare_full_splits(
    data,
    included_indices,
    baseline_splits,
    training_config,
    smoke_test,
):
    if smoke_test:
        return _make_smoke_splits(
            data,
            included_indices,
            training_config["inner_validation"]["fraction"],
            training_config["seed"],
        )
    return baseline_splits


def run_feature_fusion_pipeline(
    data,
    feature_dataset,
    fusion_config,
    modeling_config,
    model_config,
    training_config,
    baseline_directory,
    output_directory,
    report_path,
    smoke_test=False,
    device=torch.device("cpu"),
):
    """Run three Cell 1 ablation arms over the existing five outer folds."""
    validate_feature_alignment(
        data,
        feature_dataset,
        fusion_config["data"]["handcrafted_feature_count"],
    )
    _validate_configs(fusion_config, modeling_config, model_config)
    baseline_directory = Path(baseline_directory).resolve()
    baseline_results_path = baseline_directory / "results.json"
    with baseline_results_path.open("r", encoding="utf-8") as file:
        baseline_results = json.load(file)
    _validate_baseline_metadata(
        baseline_results,
        data,
        model_config,
        fusion_config,
    )
    baseline_prediction = _read_baseline_predictions(
        baseline_directory
        / fusion_config["baseline"]["model_name"]
        / "oof_predictions.csv",
        data,
    )
    baseline_model_result = baseline_results["models"][
        fusion_config["baseline"]["model_name"]
    ]
    _validate_recomputed_metrics(
        calculate_metrics(data.y, baseline_prediction),
        baseline_model_result["overall_metrics"],
        "Baseline overall OOF metrics",
    )
    for outer_fold in range(5):
        test_indices = np.flatnonzero(data.folds == outer_fold)
        expected_fold = next(
            item
            for item in baseline_model_result["per_fold_metrics"]
            if item["outer_fold"] == outer_fold
        )
        _validate_recomputed_metrics(
            calculate_metrics(
                data.y[test_indices],
                baseline_prediction[test_indices],
            ),
            expected_fold,
            f"Baseline fold {outer_fold} metrics",
        )
    baseline_splits = _load_baseline_splits(
        baseline_directory,
        data,
        model_config["models"]["eeg_transformer"],
        fusion_config["inner_validation"]["fraction"],
        fusion_config["seed"],
    )

    if smoke_test:
        included_indices = _smoke_indices(
            data,
            fusion_config["smoke_test"]["groups_per_class_per_fold"],
            fusion_config["seed"],
        )
    else:
        included_indices = np.arange(data.y.size, dtype=np.int64)
    if len(np.unique(included_indices)) != included_indices.size:
        raise RuntimeError("Included epoch indices must be unique.")
    splits = _prepare_full_splits(
        data,
        included_indices,
        baseline_splits,
        fusion_config,
        smoke_test,
    )
    output_directory = Path(output_directory)
    if output_directory.exists():
        if any(output_directory.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_directory}")
    else:
        output_directory.mkdir(parents=True)
    _write_group_splits(output_directory, splits)

    training_config = json.loads(json.dumps(training_config))
    if smoke_test:
        training_config["optimization"]["batch_size"] = fusion_config[
            "smoke_test"
        ]["batch_size"]
        training_config["optimization"]["maximum_epochs"] = fusion_config[
            "smoke_test"
        ]["maximum_epochs"]
        training_config["optimization"]["early_stopping_patience"] = (
            fusion_config["smoke_test"]["early_stopping_patience"]
        )

    arms = {}
    for arm_name, configuration in (
        ("A_raw_eeg_transformer", {"kind": "baseline_reference"}),
        ("B_handcrafted_logistic", fusion_config["handcrafted_baseline"]),
        ("C_transformer_plus_handcrafted", fusion_config["fusion"]),
    ):
        probabilities = np.full(included_indices.size, np.nan, dtype=np.float64)
        prediction_counts = np.zeros(included_indices.size, dtype=np.int64)
        local_index_by_global = {
            int(global_index): local_index
            for local_index, global_index in enumerate(included_indices)
        }
        per_fold_metrics = []
        arm_directory = Path(output_directory) / arm_name
        for outer_fold in range(5):
            split = splits[outer_fold]
            test_indices = split["test_indices"]
            fold_directory = arm_directory / f"outer_fold_{outer_fold}"
            if arm_name == "A_raw_eeg_transformer":
                test_probabilities = baseline_prediction[test_indices]
                metric = calculate_metrics(
                    data.y[test_indices],
                    test_probabilities,
                )
                metric.update(
                    {
                        "outer_fold": outer_fold,
                        "test_epoch_count": int(test_indices.size),
                        "test_group_count": int(
                            len(set(data.groups[test_indices].tolist()))
                        ),
                        "baseline_best_epoch": int(
                            next(
                                item["best_epoch"]
                                for item in baseline_results["models"][
                                    fusion_config["baseline"]["model_name"]
                                ]["per_fold_metrics"]
                                if item["outer_fold"] == outer_fold
                            )
                        ),
                    }
                )
                baseline_fold_directory = (
                    baseline_directory
                    / "eeg_transformer"
                    / f"outer_fold_{outer_fold}"
                )
                fold_directory.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    baseline_fold_directory / "best_checkpoint.pt",
                    fold_directory / "baseline_checkpoint_reference.pt",
                )
                shutil.copy2(
                    baseline_fold_directory / "training_history.json",
                    fold_directory / "training_history.json",
                )
                _json_write(
                    fold_directory / "baseline_reference.json",
                    {
                        "source_checkpoint": str(
                            (
                                baseline_fold_directory / "best_checkpoint.pt"
                            ).resolve()
                        ),
                        "source_history": str(
                            (
                                baseline_fold_directory / "training_history.json"
                            ).resolve()
                        ),
                        "copied_without_modifying_source": True,
                    },
                )
            elif arm_name == "B_handcrafted_logistic":
                test_probabilities, metric = _train_logistic_fold(
                    data,
                    feature_dataset,
                    split,
                    outer_fold,
                    configuration,
                    training_config,
                    fold_directory,
                )
            else:
                initial_checkpoint = None
                if (
                    not smoke_test
                    and configuration["warm_start_from_fold_baseline"]
                ):
                    initial_checkpoint = (
                        baseline_directory
                        / "eeg_transformer"
                        / f"outer_fold_{outer_fold}"
                        / "best_checkpoint.pt"
                    )
                test_probabilities, metric = _train_fusion_fold(
                    data,
                    feature_dataset,
                    split,
                    outer_fold,
                    model_config,
                    configuration,
                    training_config,
                    initial_checkpoint,
                    fold_directory,
                    device,
                )
            if arm_name == "A_raw_eeg_transformer":
                _json_write(fold_directory / "fold_metrics.json", metric)
                _json_write(
                    fold_directory / "group_split_reference.json",
                    splits[outer_fold]["record"],
                )
            if test_probabilities.shape != (test_indices.size,):
                raise RuntimeError(
                    f"{arm_name} fold {outer_fold} returned an invalid prediction shape."
                )
            for index, probability in zip(test_indices, test_probabilities):
                local_index = local_index_by_global[int(index)]
                if prediction_counts[local_index] != 0:
                    raise RuntimeError(
                        f"Duplicate OOF prediction for epoch {int(index)}."
                    )
                probabilities[local_index] = probability
                prediction_counts[local_index] += 1
            per_fold_metrics.append(metric)

        validate_prediction_coverage(prediction_counts)
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"{arm_name} produced non-finite OOF probabilities.")
        metadata = {
            "kind": configuration["kind"]
            if arm_name == "A_raw_eeg_transformer"
            else arm_name,
            "baseline_reference": str(baseline_directory),
            "model_parameters": (
                model_config["models"]["eeg_transformer"]
                if arm_name == "A_raw_eeg_transformer"
                else configuration
            ),
            "training_configuration": training_config["optimization"],
            "inner_validation_fraction": fusion_config["inner_validation"][
                "fraction"
            ],
            "out_of_fold_coverage_verified": True,
            "included_epoch_indices": included_indices.tolist(),
        }
        overall_metrics = _write_arm_results(
            output_directory,
            arm_name,
            data,
            included_indices,
            probabilities,
            per_fold_metrics,
            metadata,
        )
        arms[arm_name] = {
            "per_fold_metrics": per_fold_metrics,
            "overall_metrics": overall_metrics,
        }

    comparison_summary = _write_comparison_tables(output_directory, arms)
    results = {
        "run_mode": "smoke_test" if smoke_test else "full",
        "cell": 1,
        "raw_epoch_count": int(data.y.size),
        "included_epoch_count": int(included_indices.size),
        "included_epoch_indices": included_indices.tolist(),
        "class_counts": {
            "Control": int(np.sum(data.y[included_indices] == 0)),
            "ADHD": int(np.sum(data.y[included_indices] == 1)),
        },
        "class_mapping": {"Control": 0, "ADHD": 1},
        "positive_class": "ADHD",
        "group_ids_are_verified_participant_ids": False,
        "baseline_directory": str(baseline_directory),
        "output_directory": str(output_directory.resolve()),
        "report_path": str(Path(report_path).resolve()),
        "device": str(device),
        "seed": int(fusion_config["seed"]),
        "feature_names": list(feature_dataset.feature_names),
        "transformer_configuration": model_config["models"]["eeg_transformer"],
        "handcrafted_baseline_configuration": fusion_config[
            "handcrafted_baseline"
        ],
        "fusion_configuration": fusion_config["fusion"],
        "training_configuration": training_config["optimization"],
        "fold_split_source": (
            "smoke_subset_group_stratified"
            if smoke_test
            else "existing_baseline_group_splits"
        ),
        "out_of_fold_coverage_verified": True,
        "arms": arms,
        "comparison_summary": comparison_summary,
    }
    results["fusion_comparison_text"] = _fusion_comparison_text(results)
    _json_write(output_directory / "results.json", results)
    _write_report(report_path, results)
    run_report_path = output_directory / "deep_feature_fusion_report.md"
    if Path(report_path).resolve() != run_report_path.resolve():
        _write_report(run_report_path, results)
    results["run_report_path"] = str(run_report_path.resolve())
    _json_write(output_directory / "results.json", results)
    return results


def _load_yaml(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration at {path} must contain a mapping.")
    value["_path"] = str(path)
    return value


def _create_run_directory(output_root, smoke_test):
    mode = "smoke_test" if smoke_test else "full"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_directory = Path(output_root) / mode / f"run_{run_id}"
    output_directory.mkdir(parents=True, exist_ok=False)
    return output_directory


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run leakage-aware Cell 1 deep feature-fusion ablations."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "deep_fusion_cell1.yaml"),
        help="Path to deep_fusion_cell1.yaml.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run reduced, one-epoch checks in a separate smoke-test directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional new output directory; it must not already exist.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    fusion_config_path = Path(args.config).resolve()
    fusion_config = _load_yaml(fusion_config_path)
    config_directory = fusion_config_path.parent
    modeling_config_path = (
        config_directory / fusion_config["configs"]["modeling_data"]
    ).resolve()
    model_config_path = (
        config_directory / fusion_config["configs"]["deep_models"]
    ).resolve()
    modeling_config = _load_yaml(modeling_config_path)
    model_config = _load_yaml(model_config_path)
    _validate_configs(fusion_config, modeling_config, model_config)

    input_path = (
        modeling_config_path.parent / modeling_config["dataset"]["input"]
    ).resolve()
    deep_input_path = (
        model_config_path.parent / model_config["data"]["input"]
    ).resolve()
    fusion_input_path = (
        fusion_config_path.parent / fusion_config["data"]["input"]
    ).resolve()
    if input_path != deep_input_path or input_path != fusion_input_path:
        raise ValueError("Fusion, modeling, and deep-model configs must use the same Cell 1 data.")
    data = load_cell1_dataset(input_path)
    feature_dataset = extract_features(
        data,
        sampling_rate_hz=model_config["data"]["sampling_rate_hz"],
    )
    validate_feature_alignment(data, feature_dataset)

    baseline_directory = (
        fusion_config_path.parent / fusion_config["baseline"]["run_directory"]
    ).resolve()
    device = choose_device(args.device)
    output_root = (
        fusion_config_path.parent / fusion_config["paths"]["output_root"]
    ).resolve()
    output_directory = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else _create_run_directory(output_root, args.smoke_test)
    )
    if args.output_dir:
        output_directory.mkdir(parents=True, exist_ok=False)
    report_path = (
        fusion_config_path.parent / fusion_config["paths"]["report"]
    ).resolve()
    results = run_feature_fusion_pipeline(
        data=data,
        feature_dataset=feature_dataset,
        fusion_config=fusion_config,
        modeling_config=modeling_config,
        model_config=model_config,
        training_config=fusion_config,
        baseline_directory=baseline_directory,
        output_directory=output_directory,
        report_path=report_path,
        smoke_test=args.smoke_test,
        device=device,
    )
    print(f"Completed {results['run_mode']} feature-fusion run.")
    print(f"Output directory: {results['output_directory']}")
    print(f"Report: {results['report_path']}")
    for arm_name, arm in results["arms"].items():
        metrics = arm["overall_metrics"]
        print(
            f"{arm_name}: epochs={results['included_epoch_count']}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"sensitivity={metrics['recall_sensitivity']:.4f}, "
            f"specificity={metrics['specificity']:.4f}"
        )
    return results


if __name__ == "__main__":
    main()
