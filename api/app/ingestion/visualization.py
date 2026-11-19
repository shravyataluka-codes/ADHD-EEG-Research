"""EEG signal visualization data generation for Drushti Detect Phase 2.

This module generates structured visualization data from actual uploaded EEG
samples.  It never fabricates or synthesises signal values.

Principles
----------
* All amplitude values in responses are drawn directly from the uploaded array.
* Downsampling (for display efficiency) is performed by strided sub-sampling.
  The original samples are never modified.  Downsampling is documented in
  every response.
* Time axis is in seconds when ``sampling_rate_hz`` is known, otherwise in
  sample indices (0-based integers), documented explicitly.
* PSD generation requires a known, positive sampling rate and uses
  ``scipy.signal.welch`` with documented parameters.
* Channel selection is validated against the recording's actual channel count.
* Time-window selection is clamped to the available recording duration.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.signal

from .recording import EEGRecording

LOGGER = logging.getLogger(__name__)

# Welch PSD defaults (documented and consistent).
_WELCH_NPERSEG_FRACTION = 0.5   # nperseg = int(n_samples * fraction)
_WELCH_MIN_NPERSEG = 64
_WELCH_MAX_NPERSEG = 1024
_WELCH_NOVERLAP_FRACTION = 0.5  # noverlap = nperseg // 2


# ---------------------------------------------------------------------------
# Request / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WaveformRequest:
    """Parameters for a waveform data request."""

    channel_indices: list[int]  # 0-based
    start_sample: int           # inclusive, 0-based
    end_sample: int             # exclusive, clamped to n_samples
    max_display_points: int     # max per channel in response
    trial_index: int = 0        # for 3D recordings


@dataclass
class WaveformChannel:
    """Waveform data for one channel."""

    channel_index: int
    channel_name: str | None
    time_values: list[float]    # seconds or sample indices
    amplitude_values: list[float]
    n_original_samples: int     # samples in the requested window
    stride: int                 # downsampling stride (1 = no downsampling)


@dataclass
class WaveformResponse:
    """Waveform data response for a recording."""

    recording_id: str
    trial_index: int | None
    time_axis_unit: Literal["seconds", "sample_index"]
    sampling_rate_hz: float | None
    time_axis_note: str
    downsampling_note: str
    channels: list[WaveformChannel]


@dataclass
class PSDChannel:
    """PSD data for one channel."""

    channel_index: int
    channel_name: str | None
    frequencies_hz: list[float]
    power_density: list[float]
    welch_nperseg: int
    welch_noverlap: int
    welch_window: str


@dataclass
class PSDResponse:
    """Power spectral density response for a recording."""

    recording_id: str
    trial_index: int | None
    sampling_rate_hz: float
    channels: list[PSDChannel]
    method: str = "welch"
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ChannelStatsSummary:
    """Channel statistics summary for visualization."""

    recording_id: str
    n_channels: int
    n_samples: int
    sampling_rate_hz: float | None
    duration_seconds: float | None
    channel_names: list[str | None]
    signal_min: list[float | None]
    signal_max: list[float | None]
    signal_mean: list[float | None]
    signal_std: list[float | None]
    is_flatline: list[bool]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_channel_indices(
    channel_indices: list[int],
    n_channels: int,
) -> list[int]:
    """Return validated channel indices; raise ValueError on bad input."""
    if not channel_indices:
        raise ValueError("channel_indices must not be empty.")
    invalid = [i for i in channel_indices if i < 0 or i >= n_channels]
    if invalid:
        raise ValueError(
            f"Channel index/indices {invalid} are out of range.  "
            f"This recording has {n_channels} channel(s) (indices 0–{n_channels - 1})."
        )
    # Deduplicate while preserving order.
    seen: set[int] = set()
    result: list[int] = []
    for idx in channel_indices:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def _validate_sample_window(
    start_sample: int,
    end_sample: int,
    n_samples: int,
) -> tuple[int, int]:
    """Clamp and validate sample window; raise ValueError on empty window."""
    start = max(0, int(start_sample))
    end = min(n_samples, int(end_sample))
    if start >= end:
        raise ValueError(
            f"Requested sample window [{start}, {end}) is empty or invalid.  "
            f"The recording has {n_samples} samples (indices 0–{n_samples - 1})."
        )
    return start, end


# ---------------------------------------------------------------------------
# Waveform generation
# ---------------------------------------------------------------------------


def _compute_stride(window_samples: int, max_display_points: int) -> int:
    """Return the display stride (≥ 1) so the response has ≤ max_display_points."""
    if max_display_points < 1:
        raise ValueError("max_display_points must be a positive integer.")
    if window_samples <= max_display_points:
        return 1
    return math.ceil(window_samples / max_display_points)


def get_waveform(
    recording: EEGRecording,
    channel_indices: list[int] | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    start_sample: int | None = None,
    end_sample: int | None = None,
    max_display_points: int = 2000,
    trial_index: int = 0,
) -> WaveformResponse:
    """Generate waveform display data from actual recording samples.

    Either (start_seconds / end_seconds) or (start_sample / end_sample)
    may be provided to define the time window.  Second-based parameters take
    priority when both are provided.  When neither is provided, the full
    recording is used (subject to max_display_points downsampling).

    Parameters
    ----------
    recording:
        Source recording.
    channel_indices:
        0-based channel indices to include.  Defaults to all channels.
    start_seconds / end_seconds:
        Time window in seconds (requires known sampling rate).
    start_sample / end_sample:
        Time window in sample indices (0-based, exclusive end).
    max_display_points:
        Maximum samples per channel in the response.  Original data is
        unchanged; only the display representation is reduced.
    trial_index:
        For 3D recordings, which trial to visualize.

    Returns
    -------
    WaveformResponse
    """
    array = recording.get_trial_view(trial_index)
    n_samples, n_channels = array.shape

    # Resolve channel indices.
    if channel_indices is None:
        indices = list(range(n_channels))
    else:
        indices = _validate_channel_indices(channel_indices, n_channels)

    # Resolve sample window.
    rate = recording.sampling_rate_hz

    if start_seconds is not None or end_seconds is not None:
        if rate is None:
            raise ValueError(
                "Cannot use second-based time window: sampling rate is unknown.  "
                "Use start_sample / end_sample instead."
            )
        ss = int((start_seconds or 0.0) * rate)
        es = int((end_seconds if end_seconds is not None else n_samples / rate) * rate)
    elif start_sample is not None or end_sample is not None:
        ss = start_sample if start_sample is not None else 0
        es = end_sample if end_sample is not None else n_samples
    else:
        ss, es = 0, n_samples

    ss, es = _validate_sample_window(ss, es, n_samples)
    window_samples = es - ss

    stride = _compute_stride(window_samples, max_display_points)

    # Time axis.
    sample_indices = np.arange(ss, es, stride, dtype=np.int64)

    if rate is not None:
        time_values = (sample_indices / rate).tolist()
        time_unit: Literal["seconds", "sample_index"] = "seconds"
        time_note = (
            f"Time axis in seconds (sampling_rate_hz={rate}).  "
            f"Window: {ss / rate:.4f}–{es / rate:.4f} s."
        )
    else:
        time_values = sample_indices.tolist()
        time_unit = "sample_index"
        time_note = (
            "Time axis in sample indices (0-based).  "
            "Sampling rate is unknown; seconds cannot be calculated."
        )

    if stride > 1:
        ds_note = (
            f"Display downsampling applied: stride={stride} "
            f"({window_samples} original samples → "
            f"{len(sample_indices)} display points per channel).  "
            "Original signal is preserved; downsampling is display-only."
        )
    else:
        ds_note = "No downsampling applied; all samples in the window are included."

    channels: list[WaveformChannel] = []
    for ch_idx in indices:
        ch_data = array[ss:es:stride, ch_idx]
        ch_name = (
            recording.channel_names[ch_idx]
            if recording.channel_names and ch_idx < len(recording.channel_names)
            else None
        )
        channels.append(
            WaveformChannel(
                channel_index=ch_idx,
                channel_name=ch_name,
                time_values=list(map(float, time_values)),
                amplitude_values=ch_data.tolist(),
                n_original_samples=window_samples,
                stride=stride,
            )
        )

    return WaveformResponse(
        recording_id=recording.recording_id,
        trial_index=trial_index if recording.n_trials is not None else None,
        time_axis_unit=time_unit,
        sampling_rate_hz=rate,
        time_axis_note=time_note,
        downsampling_note=ds_note,
        channels=channels,
    )


# ---------------------------------------------------------------------------
# PSD generation
# ---------------------------------------------------------------------------


def get_psd(
    recording: EEGRecording,
    channel_indices: list[int] | None = None,
    trial_index: int = 0,
) -> PSDResponse:
    """Compute Welch PSD for selected channels.

    Parameters
    ----------
    recording:
        Source recording.
    channel_indices:
        0-based channel indices.  Defaults to all channels.
    trial_index:
        Trial to use for 3D recordings.

    Returns
    -------
    PSDResponse

    Raises
    ------
    ValueError
        When sampling rate is unknown or the recording is too short.
    """
    rate = recording.sampling_rate_hz
    if rate is None or rate <= 0:
        raise ValueError(
            "PSD generation requires a known positive sampling rate.  "
            "The recording has no verified sampling rate."
        )

    array = recording.get_trial_view(trial_index)
    n_samples, n_channels = array.shape

    if channel_indices is None:
        indices = list(range(n_channels))
    else:
        indices = _validate_channel_indices(channel_indices, n_channels)

    # Welch parameters.
    nperseg = int(
        max(
            _WELCH_MIN_NPERSEG,
            min(_WELCH_MAX_NPERSEG, int(n_samples * _WELCH_NPERSEG_FRACTION)),
        )
    )
    noverlap = nperseg // 2

    if n_samples < nperseg:
        raise ValueError(
            f"Recording is too short for PSD estimation: {n_samples} samples "
            f"< nperseg={nperseg}.  Provide a longer recording or reduce nperseg."
        )

    assumptions = [
        f"Welch PSD parameters: nperseg={nperseg}, noverlap={noverlap}, "
        f"window='hann', detrend='constant', scaling='density'.",
        "PSD is computed from the uploaded samples without filtering.",
        "Frequencies are limited to [0, Nyquist = "
        f"{rate / 2:.1f} Hz] by the Nyquist criterion.",
    ]

    channels: list[PSDChannel] = []
    for ch_idx in indices:
        ch_data = array[:, ch_idx]
        finite_mask = np.isfinite(ch_data)
        if not finite_mask.all():
            assumptions.append(
                f"Channel {ch_idx}: {int((~finite_mask).sum())} non-finite value(s) "
                "set to 0 for PSD computation only."
            )
            ch_data = np.where(finite_mask, ch_data, 0.0)

        freqs, pxx = scipy.signal.welch(
            ch_data,
            fs=rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )

        ch_name = (
            recording.channel_names[ch_idx]
            if recording.channel_names and ch_idx < len(recording.channel_names)
            else None
        )
        channels.append(
            PSDChannel(
                channel_index=ch_idx,
                channel_name=ch_name,
                frequencies_hz=freqs.tolist(),
                power_density=pxx.tolist(),
                welch_nperseg=nperseg,
                welch_noverlap=noverlap,
                welch_window="hann",
            )
        )

    return PSDResponse(
        recording_id=recording.recording_id,
        trial_index=trial_index if recording.n_trials is not None else None,
        sampling_rate_hz=rate,
        channels=channels,
        assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# Channel statistics summary
# ---------------------------------------------------------------------------


def get_channel_stats(
    recording: EEGRecording,
    trial_index: int = 0,
) -> ChannelStatsSummary:
    """Return per-channel amplitude statistics for a recording.

    Uses finite values only for statistics; non-finite values are counted
    but excluded from min/max/mean/std.
    """
    array = recording.get_trial_view(trial_index)
    n_samples, n_channels = array.shape

    mins: list[float | None] = []
    maxs: list[float | None] = []
    means: list[float | None] = []
    stds: list[float | None] = []
    flatlines: list[bool] = []
    names: list[str | None] = []

    for ch_idx in range(n_channels):
        ch = array[:, ch_idx]
        name = (
            recording.channel_names[ch_idx]
            if recording.channel_names and ch_idx < len(recording.channel_names)
            else None
        )
        names.append(name)
        finite = ch[np.isfinite(ch)]
        if finite.size == 0:
            mins.append(None)
            maxs.append(None)
            means.append(None)
            stds.append(None)
            flatlines.append(True)
        else:
            vmin = float(finite.min())
            vmax = float(finite.max())
            mins.append(vmin)
            maxs.append(vmax)
            means.append(float(finite.mean()))
            stds.append(float(finite.std(ddof=0)))
            flatlines.append((vmax - vmin) <= 1e-10)

    return ChannelStatsSummary(
        recording_id=recording.recording_id,
        n_channels=n_channels,
        n_samples=n_samples,
        sampling_rate_hz=recording.sampling_rate_hz,
        duration_seconds=recording.duration_seconds,
        channel_names=names,
        signal_min=mins,
        signal_max=maxs,
        signal_mean=means,
        signal_std=stds,
        is_flatline=flatlines,
    )
