"""Select a training duration group-safely, then fit fusion on all Cell 1 data."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import extract_features
from src.inference import CLASS_MAPPING
from src.modeling_data import load_cell1_dataset
from src.train_deep_fusion import (
    _build_fusion_model,
    _fusion_loss,
    fit_feature_normalizer,
    validate_feature_alignment,
)
from src.train_deep_models import (
    _batch_loss_total_and_denominator,
    _tensor_epochs,
    class_weights_from_training_labels,
    choose_device,
    fit_channel_normalizer,
    group_aware_inner_split,
    set_global_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "final_fusion_cell1.yaml"


def _read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)
    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration at {path} must contain a mapping.")
    return configuration


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def _source_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_selection_split(data, validation_fraction, seed):
    training_relative, validation_relative = group_aware_inner_split(
        data.groups,
        data.y,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    training_indices = np.asarray(training_relative, dtype=np.int64)
    validation_indices = np.asarray(validation_relative, dtype=np.int64)
    if np.intersect1d(training_indices, validation_indices).size:
        raise RuntimeError("Training and validation epoch indices overlap.")
    if not np.array_equal(
        np.sort(np.concatenate((training_indices, validation_indices))),
        np.arange(data.y.size),
    ):
        raise RuntimeError("The selection split does not cover every Cell 1 epoch.")
    training_groups = set(data.groups[training_indices].tolist())
    validation_groups = set(data.groups[validation_indices].tolist())
    if training_groups & validation_groups:
        raise RuntimeError("A provisional group crosses the selection split.")
    if set(data.y[training_indices].tolist()) != {0, 1}:
        raise ValueError("Selection training split must contain both classes.")
    if set(data.y[validation_indices].tolist()) != {0, 1}:
        raise ValueError("Selection validation split must contain both classes.")
    return {
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "training_group_count": len(training_groups),
        "validation_group_count": len(validation_groups),
        "training_groups_are_validation_disjoint": True,
    }


def _make_loader(raw, features, labels, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(raw, features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def _train_one_epoch(model, loader, criterion, optimizer, device, clip_norm):
    model.train()
    loss_total = 0.0
    loss_denominator = 0.0
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    for raw, features, labels in loader:
        raw = raw.to(device)
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(raw, features), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Final fusion training produced non-finite loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, clip_norm)
        optimizer.step()
        batch_total, batch_denominator = _batch_loss_total_and_denominator(
            loss,
            labels,
            criterion,
        )
        loss_total += batch_total
        loss_denominator += batch_denominator
    if loss_denominator <= 0:
        raise ValueError("Cannot calculate training loss for an empty split.")
    return loss_total / loss_denominator


def _select_training_duration(
    data,
    feature_dataset,
    selection_split,
    model_config,
    fusion_training_config,
    device,
    seed,
):
    training_indices = selection_split["training_indices"]
    validation_indices = selection_split["validation_indices"]
    optimization = fusion_training_config["optimization"]
    fusion_configuration = fusion_training_config["fusion"]

    raw_normalizer = fit_channel_normalizer(data.X[training_indices])
    feature_normalizer = fit_feature_normalizer(
        feature_dataset.X[training_indices]
    )
    training_raw = _tensor_epochs(data.X[training_indices], raw_normalizer)
    validation_raw = _tensor_epochs(data.X[validation_indices], raw_normalizer)
    training_features = torch.from_numpy(
        feature_normalizer.transform(feature_dataset.X[training_indices])
    )
    validation_features = torch.from_numpy(
        feature_normalizer.transform(feature_dataset.X[validation_indices])
    )
    training_labels = torch.from_numpy(data.y[training_indices].astype(np.int64))
    validation_labels = torch.from_numpy(data.y[validation_indices].astype(np.int64))

    class_weights, class_weight_report = class_weights_from_training_labels(
        data.y[training_indices],
        enabled=fusion_training_config["class_weighting"]["enabled"],
        imbalance_ratio_threshold=fusion_training_config["class_weighting"][
            "imbalance_ratio_threshold"
        ],
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    model = _build_fusion_model(
        model_config,
        fusion_configuration,
        initial_checkpoint=None,
        device=device,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
    )
    scheduler_config = optimization["learning_rate_scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        min_lr=scheduler_config["minimum_learning_rate"],
    )
    training_loader = _make_loader(
        training_raw,
        training_features,
        training_labels,
        optimization["batch_size"],
        shuffle=True,
        seed=seed,
    )
    validation_loader = _make_loader(
        validation_raw,
        validation_features,
        validation_labels,
        optimization["batch_size"],
        shuffle=False,
        seed=seed,
    )

    history = []
    best_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    for epoch in range(1, optimization["maximum_epochs"] + 1):
        training_loss = _train_one_epoch(
            model,
            training_loader,
            criterion,
            optimizer,
            device,
            optimization["gradient_clip_norm"],
        )
        validation_loss = _fusion_loss(model, validation_loader, criterion, device)
        if not np.isfinite(validation_loss):
            raise FloatingPointError(
                "Duration-selection validation loss is non-finite."
            )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if validation_loss < best_loss - optimization["minimum_validation_improvement"]:
            best_loss = float(validation_loss)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= optimization["early_stopping_patience"]:
            break
    if best_epoch is None:
        raise RuntimeError("No training duration was selected.")
    return {
        "selected_epochs": int(best_epoch),
        "best_validation_loss": best_loss,
        "history": history,
        "class_weighting": class_weight_report,
    }


def _fit_all_epochs(
    data,
    feature_dataset,
    model_config,
    fusion_training_config,
    device,
    seed,
    epochs,
    learning_rate_schedule,
):
    raw_normalizer = fit_channel_normalizer(data.X)
    feature_normalizer = fit_feature_normalizer(feature_dataset.X)
    raw = _tensor_epochs(data.X, raw_normalizer)
    features = torch.from_numpy(
        feature_normalizer.transform(feature_dataset.X)
    )
    labels = torch.from_numpy(data.y.astype(np.int64))
    optimization = fusion_training_config["optimization"]
    class_weights, class_weight_report = class_weights_from_training_labels(
        data.y,
        enabled=fusion_training_config["class_weighting"]["enabled"],
        imbalance_ratio_threshold=fusion_training_config["class_weighting"][
            "imbalance_ratio_threshold"
        ],
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    set_global_seed(seed)
    model = _build_fusion_model(
        model_config,
        fusion_training_config["fusion"],
        initial_checkpoint=None,
        device=device,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
    )
    loader = _make_loader(
        raw,
        features,
        labels,
        optimization["batch_size"],
        shuffle=True,
        seed=seed,
    )
    history = []
    for epoch in range(1, epochs + 1):
        learning_rate = learning_rate_schedule[epoch - 1]
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        history.append(
            {
                "epoch": epoch,
                "training_loss": _train_one_epoch(
                    model,
                    loader,
                    criterion,
                    optimizer,
                    device,
                    optimization["gradient_clip_norm"],
                ),
                "learning_rate": float(learning_rate),
            }
        )
    model.eval()
    return model, raw_normalizer, feature_normalizer, class_weight_report, history


def _create_output_directory(root, supplied_path=None):
    if supplied_path is not None:
        directory = Path(supplied_path).resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = Path(root).resolve() / f"run_{timestamp}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    return directory


def run_final_fusion(config_path=DEFAULT_CONFIG, output_dir=None, device="cpu"):
    config_path = Path(config_path).resolve()
    configuration = _read_yaml(config_path)
    config_directory = config_path.parent
    modeling_configuration = _read_yaml(
        config_directory / configuration["configs"]["modeling_data"]
    )
    model_configuration = _read_yaml(
        config_directory / configuration["configs"]["deep_models"]
    )
    fusion_training_configuration = _read_yaml(
        config_directory / configuration["configs"]["fusion_training"]
    )
    preprocessing_configuration = _read_yaml(
        config_directory / configuration["configs"]["preprocessing"]
    )
    seed = int(configuration["seed"])
    if seed != int(fusion_training_configuration["seed"]):
        raise ValueError("Final and fusion configuration seeds must match.")
    if configuration["validation"]["seed"] != seed:
        raise ValueError("Validation split seed must match the configured seed.")
    if configuration["validation"]["strategy"] != "stratified_group_holdout":
        raise ValueError("Unsupported validation strategy.")

    input_path = (
        config_directory / modeling_configuration["dataset"]["input"]
    ).resolve()
    if input_path != (
        config_directory
        / fusion_training_configuration["data"]["input"]
    ).resolve():
        raise ValueError("Final fusion and modeling configs must use the same Cell 1 data.")
    data = load_cell1_dataset(input_path)
    if data.X.shape[0] != configuration["expected_epoch_count"]:
        raise ValueError(
            "Expected "
            f"{configuration['expected_epoch_count']} eligible Cell 1 epochs, "
            f"found {data.X.shape[0]}."
        )
    if data.class_mapping != CLASS_MAPPING:
        raise ValueError("Cell 1 class mapping must be Control=0 and ADHD=1.")

    feature_dataset = extract_features(
        data,
        sampling_rate_hz=model_configuration["data"]["sampling_rate_hz"],
    )
    validate_feature_alignment(data, feature_dataset)
    if len(feature_dataset.feature_names) != 40:
        raise ValueError("Expected exactly 40 handcrafted feature names.")

    selection_split = _prepare_selection_split(
        data,
        validation_fraction=configuration["validation"]["fraction"],
        seed=seed,
    )
    device = choose_device(device)
    set_global_seed(seed)
    duration_selection = _select_training_duration(
        data,
        feature_dataset,
        selection_split,
        model_configuration,
        fusion_training_configuration,
        device,
        seed,
    )
    learning_rate_schedule = [
        fusion_training_configuration["optimization"]["learning_rate"]
    ]
    learning_rate_schedule.extend(
        item["learning_rate"]
        for item in duration_selection["history"][: duration_selection["selected_epochs"] - 1]
    )
    model, raw_normalizer, feature_normalizer, full_class_weighting, final_history = (
        _fit_all_epochs(
            data,
            feature_dataset,
            model_configuration,
            fusion_training_configuration,
            device,
            seed,
            duration_selection["selected_epochs"],
            learning_rate_schedule,
        )
    )

    run_directory = _create_output_directory(
        config_directory / configuration["paths"]["output_root"],
        output_dir,
    )
    checkpoint_path = run_directory / "final_fusion_checkpoint.pt"
    training_metadata = {
        "duration_selection_method": "stratified group holdout from Cell 1",
        "validation_fraction": float(configuration["validation"]["fraction"]),
        "validation_seed": seed,
        "selection_training_group_count": selection_split["training_group_count"],
        "selection_validation_group_count": selection_split["validation_group_count"],
        "selection_groups_disjoint": True,
        "selected_epochs": duration_selection["selected_epochs"],
        "best_validation_loss": duration_selection["best_validation_loss"],
        "validation_groups_used_for_duration_selection": True,
        "validation_groups_included_in_final_fit_after_duration_selection": True,
        "final_fit_uses_all_eligible_epochs": True,
        "final_fit_epoch_count": int(data.y.size),
        "independent_test_performance_evaluated": False,
        "duration_selection_class_weighting": duration_selection[
            "class_weighting"
        ],
        "final_fit_class_weighting": full_class_weighting,
        "optimization": fusion_training_configuration["optimization"],
    }
    dataset_metadata = {
        "cell": 1,
        "input_path": str(input_path),
        "input_sha256": _source_digest(input_path),
        "epoch_count": int(data.y.size),
        "epoch_shape": list(data.X.shape[1:]),
        "class_counts": {
            "Control": int(np.count_nonzero(data.y == 0)),
            "ADHD": int(np.count_nonzero(data.y == 1)),
        },
        "provisional_group_count": int(np.unique(data.groups).size),
        "group_ids_are_verified_participant_ids": False,
        "fold_ids_present": sorted(int(value) for value in np.unique(data.folds)),
        "eligible_epoch_indices": list(range(int(data.y.size))),
    }
    model_configuration_for_checkpoint = {
        "in_channels": 2,
        "n_samples": 1024,
        "num_classes": 2,
        "sampling_rate_hz": int(model_configuration["data"]["sampling_rate_hz"]),
        "transformer": model_configuration["models"]["eeg_transformer"],
    }
    checkpoint = {
        "artifact_type": "final_cell1_transformer_handcrafted_fusion",
        "format_version": 1,
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "raw_normalization_mean": raw_normalizer.mean.tolist(),
        "raw_normalization_scale": raw_normalizer.scale.tolist(),
        "feature_normalization_mean": feature_normalizer.mean.tolist(),
        "feature_normalization_scale": feature_normalizer.scale.tolist(),
        "constant_feature_indices": list(
            feature_normalizer.constant_feature_indices
        ),
        "feature_names": list(feature_dataset.feature_names),
        "class_mapping": dict(data.class_mapping),
        "model_configuration": model_configuration_for_checkpoint,
        "fusion_configuration": dict(fusion_training_configuration["fusion"]),
        "preprocessing_configuration": preprocessing_configuration,
        "random_seed": seed,
        "dataset_metadata": dataset_metadata,
        "training_metadata": training_metadata,
        "metadata": {
            "outer_fold": None,
            "freeze_transformer_backbone": bool(
                fusion_training_configuration["fusion"][
                    "freeze_transformer_backbone"
                ]
            ),
            "group_ids_are_verified_participant_ids": False,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    _write_json(
        run_directory / "training_metadata.json",
        {
            **training_metadata,
            "dataset_metadata": dataset_metadata,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "preprocessing_configuration": preprocessing_configuration,
            "model_configuration": model_configuration_for_checkpoint,
            "fusion_configuration": fusion_training_configuration["fusion"],
            "feature_names": list(feature_dataset.feature_names),
            "class_mapping": dict(data.class_mapping),
            "random_seed": seed,
        },
    )
    _write_json(
        run_directory / "duration_selection_history.json",
        duration_selection["history"],
    )
    _write_json(run_directory / "final_training_history.json", final_history)
    print(f"Final fusion checkpoint: {checkpoint_path.resolve()}")
    print(
        f"Selected epochs: {duration_selection['selected_epochs']}; "
        f"final fit epochs: {data.y.size}; validation loss: "
        f"{duration_selection['best_validation_loss']:.8f}"
    )
    print("Independent test performance was not evaluated.")
    return {
        "checkpoint_path": checkpoint_path,
        "run_directory": run_directory,
        "selected_epochs": duration_selection["selected_epochs"],
        "epoch_count": int(data.y.size),
        "validation_loss": duration_selection["best_validation_loss"],
        "training_metadata": training_metadata,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Select duration group-safely and fit fusion on all Cell 1 epochs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory; an existing directory is never overwritten.",
    )
    args = parser.parse_args(argv)
    run_final_fusion(args.config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
