"""Trim recordings and divide them into non-overlapping EEG epochs."""

import numpy as np


def _duration_to_samples(duration_seconds, sampling_rate_hz, name, allow_zero=False):
    """Convert a duration to an exact sample count."""
    if sampling_rate_hz <= 0:
        raise ValueError("Sampling rate must be positive.")
    if duration_seconds < 0 or (duration_seconds == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive.")

    sample_count = duration_seconds * sampling_rate_hz
    rounded_count = round(sample_count)
    if not np.isclose(sample_count, rounded_count, rtol=0, atol=1e-9):
        raise ValueError(f"{name} does not correspond to a whole number of samples.")
    return int(rounded_count)


def candidate_epoch_count(
    sample_count,
    sampling_rate_hz,
    edge_trim_seconds,
    epoch_duration_seconds,
):
    """Return the number of complete epochs left after trimming both edges."""
    trim_samples = _duration_to_samples(
        edge_trim_seconds, sampling_rate_hz, "Edge trim", allow_zero=True
    )
    epoch_samples = _duration_to_samples(
        epoch_duration_seconds, sampling_rate_hz, "Epoch duration"
    )
    usable_samples = max(0, sample_count - 2 * trim_samples)
    return usable_samples // epoch_samples


def epoch_recording(
    recording,
    sampling_rate_hz,
    edge_trim_seconds,
    epoch_duration_seconds,
):
    """Trim the recording edges and return complete, clean epochs.

    The input is samples x channels. Epoch boundaries and rejection locations
    use zero-based sample offsets from the original recording.
    """
    data = np.asarray(recording)
    if data.ndim != 2:
        raise ValueError("A recording must have samples x channels dimensions.")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("A recording must contain samples and channels.")

    trim_samples = _duration_to_samples(
        edge_trim_seconds, sampling_rate_hz, "Edge trim", allow_zero=True
    )
    epoch_samples = _duration_to_samples(
        epoch_duration_seconds, sampling_rate_hz, "Epoch duration"
    )
    if data.shape[0] <= 2 * trim_samples:
        raise ValueError("Edge trimming would leave no usable samples.")

    trimmed = data[trim_samples:-trim_samples] if trim_samples else data
    candidate_count = trimmed.shape[0] // epoch_samples
    discarded_tail_samples = trimmed.shape[0] % epoch_samples
    accepted_epochs = []
    accepted_epoch_indices = []
    rejected_epochs = []

    for epoch_index in range(candidate_count):
        start = trim_samples + epoch_index * epoch_samples
        stop = start + epoch_samples
        epoch = data[start:stop]

        if not np.isfinite(epoch).all():
            invalid_channels = (
                np.flatnonzero(~np.isfinite(epoch).all(axis=0)) + 1
            ).tolist()
            rejected_epochs.append(
                {
                    "epoch_index": epoch_index + 1,
                    "start_sample": start,
                    "stop_sample": stop,
                    "reason": "nonfinite_epoch",
                    "channel_slots": invalid_channels,
                }
            )
            continue

        flatline_channels = (np.ptp(epoch, axis=0) == 0)
        if flatline_channels.any():
            rejected_epochs.append(
                {
                    "epoch_index": epoch_index + 1,
                    "start_sample": start,
                    "stop_sample": stop,
                    "reason": "flatline_epoch",
                    "channel_slots": (
                        np.flatnonzero(flatline_channels) + 1
                    ).tolist(),
                }
            )
            continue
        accepted_epochs.append(epoch)
        accepted_epoch_indices.append(epoch_index + 1)

    if accepted_epochs:
        epochs = np.stack(accepted_epochs, axis=0)
    else:
        epochs = np.empty((0, epoch_samples, data.shape[1]), dtype=data.dtype)

    return {
        "epochs": epochs,
        "candidate_epoch_count": candidate_count,
        "accepted_epoch_count": len(accepted_epochs),
        "accepted_epoch_indices": accepted_epoch_indices,
        "rejected_epochs": rejected_epochs,
        "edge_trim_samples_per_side": trim_samples,
        "epoch_samples": epoch_samples,
        "discarded_tail_samples": discarded_tail_samples,
    }


def assign_grouped_folds(group_keys, class_labels, n_splits=5):
    """Assign stable folds without splitting any group key across folds.

    Groups are assigned round-robin within each class. Group keys should be
    provisional source-file/storage-slot keys, not invented participant IDs.
    """
    if len(group_keys) != len(class_labels):
        raise ValueError("Group keys and class labels must have equal lengths.")
    if n_splits < 2:
        raise ValueError("At least two folds are required.")
    if not group_keys:
        return np.asarray([], dtype=np.int64)

    group_classes = {}
    class_groups = {}
    for group_key, class_label in zip(group_keys, class_labels):
        previous_label = group_classes.setdefault(group_key, class_label)
        if previous_label != class_label:
            raise ValueError("A group key cannot belong to multiple classes.")
        class_groups.setdefault(class_label, set()).add(group_key)

    group_to_fold = {}
    for class_label in sorted(class_groups):
        ordered_groups = sorted(class_groups[class_label])
        if len(ordered_groups) < n_splits:
            raise ValueError(
                f"Class {class_label!r} has fewer groups than requested folds."
            )
        for group_index, group_key in enumerate(ordered_groups):
            group_to_fold[group_key] = group_index % n_splits

    return np.asarray(
        [group_to_fold[group_key] for group_key in group_keys],
        dtype=np.int64,
    )
