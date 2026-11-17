"""Load and validate the existing Cell 1 epochs for downstream modeling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


CLASS_MAPPING = {"Control": 0, "ADHD": 1}
EXPECTED_EPOCH_SHAPE = (1024, 2)
EXPECTED_FOLD_IDS = frozenset(range(5))
REQUIRED_KEYS = ("epochs", "class_labels", "group_keys", "fold_ids")


@dataclass(frozen=True)
class ModelingData:
    """Validated Cell 1 arrays and their existing grouping metadata."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    folds: np.ndarray
    class_mapping: Dict[str, int]


def load_cell1_dataset(path):
    """Load Cell 1 epochs and validate labels, groups, and saved fold IDs.

    The loader does not infer participant IDs or create replacement groups or
    folds. Groups are provisional filename/storage-slot identifiers.
    """
    path = Path(path)
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not safely load dataset at {path}: {error}") from error

    with archive:
        missing_keys = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing_keys:
            raise ValueError(
                f"Dataset is missing required metadata: {', '.join(missing_keys)}."
            )

        X = np.array(archive["epochs"], copy=True)
        raw_labels = np.array(archive["class_labels"], copy=True)
        groups = np.array(archive["group_keys"], copy=True)
        folds = np.array(archive["fold_ids"], copy=True)

    if X.ndim != 3 or X.shape[1:] != EXPECTED_EPOCH_SHAPE:
        raise ValueError(
            "Epoch data must have shape [epochs, 1024, 2]; "
            f"got {X.shape}."
        )
    if X.shape[0] == 0:
        raise ValueError("Epoch data must contain at least one epoch.")
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError("Epoch data must contain real numeric EEG values.")
    if not np.isfinite(X).all():
        raise ValueError("Epoch data contains NaN or infinite values.")

    expected_length = X.shape[0]
    for name, metadata in (
        ("class_labels", raw_labels),
        ("group_keys", groups),
        ("fold_ids", folds),
    ):
        if metadata.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        if metadata.shape[0] != expected_length:
            raise ValueError(
                f"{name} has {metadata.shape[0]} entries, but epochs has "
                f"{expected_length}."
            )

    if raw_labels.dtype.kind not in ("U", "S"):
        raise ValueError("class_labels must contain the strings 'Control' and 'ADHD'.")
    labels = raw_labels.astype(str)
    invalid_labels = sorted(set(labels.tolist()) - set(CLASS_MAPPING))
    if invalid_labels:
        raise ValueError(
            "class_labels contains unsupported values: "
            f"{invalid_labels}. Expected only 'Control' and 'ADHD'."
        )
    present_labels = set(labels.tolist())
    missing_classes = sorted(set(CLASS_MAPPING) - present_labels)
    if missing_classes:
        raise ValueError(
            f"Both classes must be represented; missing: {', '.join(missing_classes)}."
        )
    y = np.asarray([CLASS_MAPPING[label] for label in labels], dtype=np.int64)

    if groups.dtype.kind not in ("U", "S"):
        raise ValueError("group_keys must contain non-empty strings.")
    groups = groups.astype(str)
    if any(not group.strip() for group in groups.tolist()):
        raise ValueError("group_keys must not contain empty values.")

    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("fold_ids must contain integer fold IDs.")
    folds = folds.astype(np.int64, copy=False)
    present_folds = set(folds.tolist())
    unexpected_folds = sorted(present_folds - EXPECTED_FOLD_IDS)
    if unexpected_folds:
        raise ValueError(
            f"fold_ids contains unexpected fold IDs: {unexpected_folds}; "
            f"expected {sorted(EXPECTED_FOLD_IDS)}."
        )
    missing_folds = sorted(EXPECTED_FOLD_IDS - present_folds)
    if missing_folds:
        raise ValueError(
            f"All five existing folds must be represented; "
            f"missing fold IDs: {missing_folds}."
        )

    group_to_fold = {}
    group_to_label = {}
    for group, label, fold in zip(groups, labels, folds):
        if group in group_to_fold and group_to_fold[group] != fold:
            raise ValueError(
                f"Group {group!r} appears in multiple folds "
                f"({group_to_fold[group]} and {fold})."
            )
        if group in group_to_label and group_to_label[group] != label:
            raise ValueError(f"Group {group!r} has inconsistent class labels.")
        group_to_fold[group] = int(fold)
        group_to_label[group] = label

    return ModelingData(
        X=X,
        y=y,
        groups=groups,
        folds=folds,
        class_mapping=CLASS_MAPPING.copy(),
    )
