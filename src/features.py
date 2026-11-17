"""Deterministic per-epoch, per-channel EEG feature extraction."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


DEFAULT_SAMPLING_RATE_HZ = 256
WELCH_SEGMENT_SAMPLES = 512
WELCH_OVERLAP_SAMPLES = 256
BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 40.0),
)


def _channel_feature_names(channel_slot):
    names = [
        f"channel_slot_{channel_slot}_absolute_{band}_power"
        for band, _, _ in BANDS
    ]
    names.extend(
        f"channel_slot_{channel_slot}_relative_{band}_power"
        for band, _, _ in BANDS
    )
    names.extend(
        (
            f"channel_slot_{channel_slot}_theta_to_beta_power_ratio",
            f"channel_slot_{channel_slot}_mean",
            f"channel_slot_{channel_slot}_standard_deviation",
            f"channel_slot_{channel_slot}_variance",
            f"channel_slot_{channel_slot}_root_mean_square",
            f"channel_slot_{channel_slot}_skewness",
            f"channel_slot_{channel_slot}_kurtosis_excess",
            f"channel_slot_{channel_slot}_hjorth_activity",
            f"channel_slot_{channel_slot}_hjorth_mobility",
            f"channel_slot_{channel_slot}_hjorth_complexity",
        )
    )
    return names


def _integrated_power(frequencies, psd, low_hz, high_hz, description):
    selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    if np.count_nonzero(selected) < 2:
        raise ValueError(
            f"Welch frequency grid has insufficient bins for {description}."
        )
    power = float(trapezoid(psd[selected], frequencies[selected]))
    if not np.isfinite(power) or power < 0:
        raise FloatingPointError(f"Invalid integrated power for {description}.")
    return power


def _extract_channel_features(signal, sampling_rate_hz, epoch_index, channel_slot):
    signal = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError(
            f"Epoch {epoch_index}, channel slot {channel_slot} "
            "contains NaN or infinite samples."
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = float(np.mean(signal))
        centered = signal - mean
        variance = float(np.mean(centered * centered))
        standard_deviation = float(np.sqrt(variance))

    if not np.isfinite(variance) or not np.isfinite(standard_deviation):
        raise FloatingPointError(
            f"Epoch {epoch_index}, channel slot {channel_slot} "
            "has non-finite time-domain variance."
        )
    if variance == 0:
        raise ValueError(
            f"Epoch {epoch_index}, channel slot {channel_slot} is constant; "
            "normalized moments and power ratios are undefined."
        )

    frequencies, psd = welch(
        signal,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=WELCH_SEGMENT_SAMPLES,
        noverlap=WELCH_OVERLAP_SAMPLES,
        detrend="constant",
        scaling="density",
        average="mean",
    )
    absolute_powers = {
        band: _integrated_power(
            frequencies,
            psd,
            low_hz,
            high_hz,
            f"{band} band power",
        )
        for band, low_hz, high_hz in BANDS
    }
    broadband_power = _integrated_power(
        frequencies, psd, 0.5, 40.0, "0.5-40 Hz broadband power"
    )
    if broadband_power == 0:
        raise FloatingPointError(
            f"Epoch {epoch_index}, channel slot {channel_slot} has zero "
            "0.5-40 Hz power; relative powers are undefined."
        )
    if absolute_powers["beta"] == 0:
        raise FloatingPointError(
            f"Epoch {epoch_index}, channel slot {channel_slot} has zero beta "
            "power; theta-to-beta ratio is undefined."
        )

    first_difference = np.diff(signal)
    second_difference = np.diff(first_difference)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        first_difference_variance = float(np.var(first_difference))
        second_difference_variance = float(np.var(second_difference))
        mobility = float(
            np.sqrt(first_difference_variance / variance)
        )
        # A linear discrete signal has a constant first difference; its
        # complexity is defined as zero rather than evaluating an undefined 0/0.
        complexity = (
            0.0
            if first_difference_variance == 0
            else float(
                np.sqrt(second_difference_variance / first_difference_variance)
                / mobility
            )
        )
        rms = float(np.sqrt(np.mean(signal * signal)))
        standardized = centered / standard_deviation
        skewness = float(np.mean(standardized**3))
        kurtosis_excess = float(np.mean(standardized**4) - 3.0)

    values = [
        *(absolute_powers[band] for band, _, _ in BANDS),
        *(absolute_powers[band] / broadband_power for band, _, _ in BANDS),
        absolute_powers["theta"] / absolute_powers["beta"],
        mean,
        standard_deviation,
        variance,
        rms,
        skewness,
        kurtosis_excess,
        variance,
        mobility,
        complexity,
    ]
    if not np.isfinite(values).all():
        raise FloatingPointError(
            f"Epoch {epoch_index}, channel slot {channel_slot} produced "
            "non-finite features."
        )
    return values


@dataclass(frozen=True)
class FeatureDataset:
    """Features and unchanged epoch-level metadata."""

    X: np.ndarray
    feature_names: Tuple[str, ...]
    y: np.ndarray
    groups: np.ndarray
    folds: np.ndarray
    class_mapping: Dict[str, int]


def extract_epoch_features(epochs, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ):
    """Extract a 40-feature matrix from unlabeled EEG epochs.

    Welch PSD uses two-second Hann segments with 50% overlap. Absolute band
    power is PSD integrated over each inclusive band; relative power divides
    each band by total integrated 0.5-40 Hz power. Skewness is the standardized
    third central moment and kurtosis is excess kurtosis. Hjorth activity is
    population variance; mobility and complexity use discrete sample
    differences. Constant channels and mathematically undefined power ratios
    are rejected rather than filled with arbitrary values.
    """
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 80:
        raise ValueError(
            "sampling_rate_hz must be finite and greater than 80 Hz "
            "to cover the configured bands below Nyquist."
        )

    epochs = np.asarray(epochs)
    if (
        epochs.ndim != 3
        or epochs.shape[1:] != (1024, 2)
        or epochs.shape[0] == 0
    ):
        raise ValueError(
            "Input epochs must have shape [epochs, 1024, 2]; "
            f"got {epochs.shape}."
        )
    if not np.issubdtype(epochs.dtype, np.number) or np.iscomplexobj(epochs):
        raise ValueError("Input epochs must contain real numeric EEG samples.")
    if not np.isfinite(epochs).all():
        raise ValueError("Input epochs contain NaN or infinite samples.")

    epoch_count = epochs.shape[0]
    names = tuple(
        _channel_feature_names(channel_slot)
        for channel_slot in (1, 2)
    )
    feature_names = tuple(name for channel_names in names for name in channel_names)
    features = np.empty((epoch_count, len(feature_names)), dtype=np.float64)

    for epoch_index, epoch in enumerate(epochs):
        row = []
        for channel_index in range(2):
            row.extend(
                _extract_channel_features(
                    epoch[:, channel_index],
                    sampling_rate_hz,
                    epoch_index,
                    channel_index + 1,
                )
            )
        features[epoch_index] = row

    return features, feature_names


def extract_features(data, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ):
    """Extract 40 features per epoch and preserve aligned dataset metadata."""
    features, feature_names = extract_epoch_features(
        data.X,
        sampling_rate_hz=sampling_rate_hz,
    )
    epoch_count = features.shape[0]
    metadata = {}
    for field in ("y", "groups", "folds"):
        values = np.asarray(getattr(data, field))
        if values.ndim != 1 or values.shape[0] != epoch_count:
            raise ValueError(
                f"{field} must have one entry per epoch ({epoch_count}); "
                f"got shape {values.shape}."
            )
        metadata[field] = np.array(values, copy=True)

    class_mapping = dict(getattr(data, "class_mapping", {}))
    return FeatureDataset(
        X=features,
        feature_names=feature_names,
        y=metadata["y"],
        groups=metadata["groups"],
        folds=metadata["folds"],
        class_mapping=class_mapping,
    )
