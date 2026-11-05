"""Generic participant-grouped split utilities.

This module does not read ``TrialManifest`` values or assign provenance status.
Callers must supply the grouping key and decide whether it is authoritative
for their intended evaluation.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral, Real
from typing import Hashable, Mapping, Sequence

import numpy as np


_PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class GroupedSplit:
    """Sample and group assignments plus optional observed class counts."""

    train_sample_ids: tuple[Hashable, ...]
    validation_sample_ids: tuple[Hashable, ...]
    test_sample_ids: tuple[Hashable, ...]
    train_group_ids: tuple[Hashable, ...]
    validation_group_ids: tuple[Hashable, ...]
    test_group_ids: tuple[Hashable, ...]
    class_counts: Mapping[str, Mapping[Hashable, int]] | None


def _validate_identifier(value: Hashable, *, field: str, index: int) -> None:
    """Accept hashable identifiers except null, blank strings, NaN, and NaT."""
    if value is None:
        raise ValueError(f"{field}[{index}] must not be missing")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field}[{index}] must not be empty")
    if isinstance(value, (np.datetime64, np.timedelta64)) and np.isnat(value):
        raise ValueError(f"{field}[{index}] must not be missing")
    if isinstance(value, Decimal) and value.is_nan():
        raise ValueError(f"{field}[{index}] must not be missing")
    if isinstance(value, Real) and math.isnan(value):
        raise ValueError(f"{field}[{index}] must not be missing")
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{field}[{index}] must be hashable") from error


def grouped_train_validation_test_split(
    sample_ids: Sequence[Hashable],
    group_ids: Sequence[Hashable],
    *,
    group_counts: Mapping[str, int],
    random_seed: int,
    labels: Sequence[Hashable] | None = None,
) -> GroupedSplit:
    """Randomly assign whole supplied groups to train, validation, and test.

    ``group_counts`` specifies the exact number of distinct groups assigned to
    each partition; it avoids imposing an unapproved ratio on participant
    counts. Group sizes may differ, so resulting sample counts need not match
    these proportions. Labels, when supplied, are summarized per partition;
    they are not used for stratification or interpreted as clinical diagnoses.
    """
    if len(sample_ids) == 0:
        raise ValueError("sample_ids must not be empty")
    if len(sample_ids) != len(group_ids):
        raise ValueError("sample_ids and group_ids must have equal lengths")
    if labels is not None and len(labels) != len(sample_ids):
        raise ValueError("labels must have the same length as sample_ids")

    seen_samples: set[Hashable] = set()
    for index, sample_id in enumerate(sample_ids):
        _validate_identifier(sample_id, field="sample_ids", index=index)
        if sample_id in seen_samples:
            raise ValueError(f"duplicate sample identifier: {sample_id!r}")
        seen_samples.add(sample_id)

    for index, group_id in enumerate(group_ids):
        _validate_identifier(group_id, field="group_ids", index=index)

    if labels is not None:
        for index, label in enumerate(labels):
            _validate_identifier(label, field="labels", index=index)

    if set(group_counts) != set(_PARTITIONS):
        raise ValueError(
            "group_counts must define exactly train, validation, and test"
        )
    normalized_counts: dict[str, int] = {}
    for partition in _PARTITIONS:
        count = group_counts[partition]
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise ValueError(
                f"group_counts[{partition!r}] must be a positive integer"
            )
        normalized_counts[partition] = int(count)

    unique_groups = list(dict.fromkeys(group_ids))
    if sum(normalized_counts.values()) != len(unique_groups):
        raise ValueError(
            "group_counts must sum to the number of unique group identifiers"
        )
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be an integer")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")

    shuffled_groups = unique_groups.copy()
    random.Random(int(random_seed)).shuffle(shuffled_groups)

    groups_by_partition: dict[str, tuple[Hashable, ...]] = {}
    cursor = 0
    for partition in _PARTITIONS:
        count = normalized_counts[partition]
        groups_by_partition[partition] = tuple(shuffled_groups[cursor : cursor + count])
        cursor += count

    partition_by_group = {
        group_id: partition
        for partition, partition_groups in groups_by_partition.items()
        for group_id in partition_groups
    }
    samples_by_partition: dict[str, list[Hashable]] = {
        partition: [] for partition in _PARTITIONS
    }
    labels_by_partition: dict[str, Counter[Hashable]] = {
        partition: Counter() for partition in _PARTITIONS
    }
    for index, (sample_id, group_id) in enumerate(zip(sample_ids, group_ids)):
        partition = partition_by_group[group_id]
        samples_by_partition[partition].append(sample_id)
        if labels is not None:
            labels_by_partition[partition][labels[index]] += 1

    classes = list(dict.fromkeys(labels)) if labels is not None else []
    class_counts = (
        {
            partition: {
                label: labels_by_partition[partition].get(label, 0)
                for label in classes
            }
            for partition in _PARTITIONS
        }
        if labels is not None
        else None
    )
    return GroupedSplit(
        train_sample_ids=tuple(samples_by_partition["train"]),
        validation_sample_ids=tuple(samples_by_partition["validation"]),
        test_sample_ids=tuple(samples_by_partition["test"]),
        train_group_ids=groups_by_partition["train"],
        validation_group_ids=groups_by_partition["validation"],
        test_group_ids=groups_by_partition["test"],
        class_counts=class_counts,
    )
