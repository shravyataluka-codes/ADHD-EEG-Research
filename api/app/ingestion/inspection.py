"""EEG signal quality inspection for Drushti Detect Phase 2.

This module computes descriptive statistics and quality indicators for an
uploaded EEG recording.  No filtering, resampling, normalisation, or artifact
removal is applied — the original signal is preserved exactly.

All computations are performed on the actual uploaded samples.  For 3D
(epoched) recordings the inspection is performed on the first trial only;
the full recording is not modified.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .recording import EEGRecording

LOGGER = logging.getLogger(__name__)

# A channel is considered flatline when its peak-to-peak range is below this
# fraction of the global inter-channel range.  This is a display heuristic,
# not a clinical rejection criterion.
_FLATLINE_RANGE_THRESHOLD = 1e-10


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChannelStats:
    """Per-channel signal statistics."""

    channel_index: int
    channel_name: str | None
    n_samples: int
    n_nonfinite: int
    signal_min: float | None      # None when all samples are non-finite
    signal_max: float | None
    signal_range: float | None
    signal_mean: float | None
    signal_std: float | None
    is_flatline: bool
    has_nonfinite: bool


@dataclass
class SignalQualityReport:
    """Signal quality information for one recording (or one trial of a 3D recording).

    Attributes
    ----------
    recording_id:
        Matches the parent EEGRecording.
    trial_index:
        0-based trial index inspected.  ``None`` for continuous recordings.
    n_channels:
        Number of channels inspected.
    n_samples:
        Number of time samples per channel.
    sampling_rate_hz:
        Sampling rate used for duration calculation, or ``None``.
    duration_seconds:
        ``n_samples / sampling_rate_hz``, or ``None`` when rate is unknown.
    total_nonfinite_count:
        Total count of NaN / ±Inf values across all channels.
    flatline_channel_indices:
        List of channel indices identified as flatline.
    channel_stats:
        Per-channel statistics list (length = n_channels).
    warnings:
        Signal quality warnings.
    """

    recording_id: str
    trial_index: int | None
    n_channels: int
    n_samples: int
    sampling_rate_hz: float | None
    duration_seconds: float | None
    total_nonfinite_count: int
    flatline_channel_indices: list[int]
    channel_stats: list[ChannelStats]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inspection logic
# ---------------------------------------------------------------------------


def _channel_stats(
    channel_index: int,
    channel_data: np.ndarray,
    channel_name: str | None,
) -> ChannelStats:
    """Compute statistics for a single channel vector."""
    n = int(channel_data.size)
    finite_mask = np.isfinite(channel_data)
    n_nonfinite = int((~finite_mask).sum())
    has_nonfinite = n_nonfinite > 0

    if finite_mask.any():
        finite_vals = channel_data[finite_mask]
        sig_min = float(finite_vals.min())
        sig_max = float(finite_vals.max())
        sig_range = sig_max - sig_min
        sig_mean = float(finite_vals.mean())
        sig_std = float(finite_vals.std(ddof=0))
    else:
        sig_min = sig_max = sig_range = sig_mean = sig_std = None

    is_flatline = (
        sig_range is not None and sig_range <= _FLATLINE_RANGE_THRESHOLD
    )

    return ChannelStats(
        channel_index=channel_index,
        channel_name=channel_name,
        n_samples=n,
        n_nonfinite=n_nonfinite,
        signal_min=sig_min,
        signal_max=sig_max,
        signal_range=sig_range,
        signal_mean=sig_mean,
        signal_std=sig_std,
        is_flatline=is_flatline,
        has_nonfinite=has_nonfinite,
    )


def inspect_recording(
    recording: EEGRecording,
    trial_index: int = 0,
) -> SignalQualityReport:
    """Compute signal quality statistics for an EEG recording.

    Parameters
    ----------
    recording:
        The canonical recording to inspect.
    trial_index:
        For 3D (epoched) recordings, which trial to inspect.  Defaults to 0.
        Ignored for continuous (2D) recordings.

    Returns
    -------
    SignalQualityReport
    """
    array = recording.get_trial_view(trial_index)
    n_samples, n_channels = array.shape

    warnings: list[str] = []

    if recording.sampling_rate_hz is None:
        warnings.append(
            "Sampling rate is unknown; duration_seconds cannot be calculated."
        )
    duration = recording.duration_seconds  # already None when rate unknown

    per_channel: list[ChannelStats] = []
    for ch_idx in range(n_channels):
        name = (
            recording.channel_names[ch_idx]
            if recording.channel_names and ch_idx < len(recording.channel_names)
            else None
        )
        stats = _channel_stats(ch_idx, array[:, ch_idx], name)
        per_channel.append(stats)

    total_nonfinite = int(sum(s.n_nonfinite for s in per_channel))
    flatline_indices = [s.channel_index for s in per_channel if s.is_flatline]

    if total_nonfinite > 0:
        warnings.append(
            f"Recording contains {total_nonfinite} non-finite value(s) "
            f"(NaN or ±Inf) across {sum(1 for s in per_channel if s.has_nonfinite)} "
            "channel(s)."
        )

    if flatline_indices:
        warnings.append(
            f"Channel(s) {flatline_indices} appear to be flatline "
            f"(signal range ≤ {_FLATLINE_RANGE_THRESHOLD:.0e})."
        )

    if recording.n_trials is not None:
        warnings.append(
            f"This is a 3D (epoched) recording with {recording.n_trials} trial(s).  "
            f"Inspection was performed on trial index {trial_index}."
        )

    LOGGER.debug(
        "Signal inspection: recording_id=%s, trial=%s, n_ch=%d, n_samp=%d, "
        "nonfinite=%d, flatlines=%s",
        recording.recording_id,
        trial_index,
        n_channels,
        n_samples,
        total_nonfinite,
        flatline_indices,
    )

    return SignalQualityReport(
        recording_id=recording.recording_id,
        trial_index=trial_index if recording.n_trials is not None else None,
        n_channels=n_channels,
        n_samples=n_samples,
        sampling_rate_hz=recording.sampling_rate_hz,
        duration_seconds=duration,
        total_nonfinite_count=total_nonfinite,
        flatline_channel_indices=flatline_indices,
        channel_stats=per_channel,
        warnings=warnings,
    )
