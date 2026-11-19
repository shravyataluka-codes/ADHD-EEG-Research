"""Canonicalize Phase 2 recordings without applying signal preprocessing."""

import numpy as np

from .compatibility import validate_recording_compatibility


def canonical_trials(recording) -> list[np.ndarray]:
    errors = validate_recording_compatibility(recording)
    if errors:
        raise ValueError(" ".join(errors))
    if recording.array_3d is None:
        return [np.asarray(recording.array)]
    # Independent trials remain independent; never join them across boundaries.
    return [np.asarray(recording.array_3d[index]) for index in range(recording.array_3d.shape[0])]
