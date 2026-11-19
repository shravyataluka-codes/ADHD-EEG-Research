"""Tests for api.app.ingestion.visualization.

All test data is synthetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from api.app.ingestion.recording import EEGRecording
from api.app.ingestion.visualization import (
    WaveformResponse,
    get_channel_stats,
    get_psd,
    get_waveform,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_recording(
    n_samples: int = 500,
    n_channels: int = 3,
    sampling_rate_hz: float | None = 256.0,
    seed: int = 42,
) -> EEGRecording:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_samples, n_channels))
    return EEGRecording(
        recording_id="viz-test",
        original_filename="viz.csv",
        source_format="csv",
        provenance="csv",
        n_channels=n_channels,
        n_samples=n_samples,
        n_trials=None,
        sampling_rate_hz=sampling_rate_hz,
        channel_names=[f"ch{i}" for i in range(n_channels)],
        channel_units=None,
        signal_units_known=False,
        observed_time_axis=0,
        orientation_note="test",
        data_shape=(n_samples, n_channels),
        array=arr,
        array_3d=None,
        source_metadata={},
        import_status="ok",
        warnings=[],
        errors=[],
    )


def _make_3d_recording(
    n_trials: int = 5,
    n_samples: int = 100,
    n_channels: int = 4,
    sampling_rate_hz: float | None = 256.0,
    seed: int = 7,
) -> EEGRecording:
    rng = np.random.default_rng(seed)
    arr_3d = rng.standard_normal((n_trials, n_samples, n_channels))
    arr_2d = arr_3d[0]
    return EEGRecording(
        recording_id="viz-3d-test",
        original_filename="trials.mat",
        source_format="mat",
        provenance="mat_v5",
        n_channels=n_channels,
        n_samples=n_samples,
        n_trials=n_trials,
        sampling_rate_hz=sampling_rate_hz,
        channel_names=[f"ch{i}" for i in range(n_channels)],
        channel_units=None,
        signal_units_known=False,
        observed_time_axis=0,
        orientation_note="test",
        data_shape=(n_trials, n_samples, n_channels),
        array=arr_2d,
        array_3d=arr_3d,
        source_metadata={},
        import_status="ok",
        warnings=[],
        errors=[],
    )


# ---------------------------------------------------------------------------
# Waveform: values match source
# ---------------------------------------------------------------------------


def test_waveform_values_match_source():
    """The waveform endpoint must return actual uploaded samples, not fabricated data."""
    rec = _make_recording(n_samples=200, n_channels=2)
    response = get_waveform(rec, channel_indices=[0])
    # With no downsampling (200 ≤ 2000), all values should match.
    ch = response.channels[0]
    expected = rec.array[:, 0].tolist()
    assert ch.amplitude_values == expected


def test_waveform_stride_1_when_no_downsampling():
    rec = _make_recording(n_samples=100)
    response = get_waveform(rec, channel_indices=[0], max_display_points=2000)
    assert response.channels[0].stride == 1


def test_waveform_all_channels_returned_by_default():
    rec = _make_recording(n_channels=5)
    response = get_waveform(rec)
    assert len(response.channels) == 5


# ---------------------------------------------------------------------------
# Waveform: time axis
# ---------------------------------------------------------------------------


def test_waveform_time_axis_seconds_when_rate_known():
    rec = _make_recording(n_samples=256, sampling_rate_hz=256.0)
    response = get_waveform(rec)
    assert response.time_axis_unit == "seconds"
    # First time value should be 0.0.
    assert response.channels[0].time_values[0] == pytest.approx(0.0)
    # Last time value should be close to (n_samples - 1) / rate.
    assert response.channels[0].time_values[-1] == pytest.approx(255 / 256.0)


def test_waveform_time_axis_sample_index_when_rate_unknown():
    rec = _make_recording(sampling_rate_hz=None)
    response = get_waveform(rec, channel_indices=[0])
    assert response.time_axis_unit == "sample_index"
    assert response.channels[0].time_values[0] == 0


def test_waveform_time_note_mentions_rate_unknown_when_rate_missing():
    rec = _make_recording(sampling_rate_hz=None)
    response = get_waveform(rec)
    assert "unknown" in response.time_axis_note.lower()


# ---------------------------------------------------------------------------
# Waveform: downsampling
# ---------------------------------------------------------------------------


def test_waveform_downsampling_reduces_points():
    rec = _make_recording(n_samples=5000)
    response = get_waveform(rec, channel_indices=[0], max_display_points=100)
    assert len(response.channels[0].amplitude_values) <= 100


def test_waveform_downsampling_documented_in_note():
    rec = _make_recording(n_samples=5000)
    response = get_waveform(rec, channel_indices=[0], max_display_points=100)
    assert "stride" in response.downsampling_note.lower() or "downsampling" in response.downsampling_note.lower()


def test_waveform_original_sample_count_reported():
    rec = _make_recording(n_samples=5000)
    response = get_waveform(rec, channel_indices=[0], max_display_points=100)
    assert response.channels[0].n_original_samples == 5000


def test_waveform_stride_reported():
    rec = _make_recording(n_samples=5000)
    response = get_waveform(rec, channel_indices=[0], max_display_points=100)
    assert response.channels[0].stride > 1


def test_original_array_unchanged_after_waveform():
    rec = _make_recording(n_samples=5000)
    original = rec.array.copy()
    get_waveform(rec, channel_indices=[0], max_display_points=100)
    np.testing.assert_array_equal(rec.array, original)


# ---------------------------------------------------------------------------
# Waveform: channel selection
# ---------------------------------------------------------------------------


def test_waveform_specific_channel_selection():
    rec = _make_recording(n_channels=5)
    response = get_waveform(rec, channel_indices=[2, 4])
    assert len(response.channels) == 2
    assert response.channels[0].channel_index == 2
    assert response.channels[1].channel_index == 4


def test_waveform_invalid_channel_raises():
    rec = _make_recording(n_channels=3)
    with pytest.raises(ValueError, match="out of range"):
        get_waveform(rec, channel_indices=[99])


def test_waveform_negative_channel_raises():
    rec = _make_recording(n_channels=3)
    with pytest.raises(ValueError, match="out of range"):
        get_waveform(rec, channel_indices=[-1])


# ---------------------------------------------------------------------------
# Waveform: time window
# ---------------------------------------------------------------------------


def test_waveform_time_window_by_seconds():
    rec = _make_recording(n_samples=512, sampling_rate_hz=256.0)
    response = get_waveform(rec, channel_indices=[0], start_seconds=0.0, end_seconds=0.5)
    # 0.5 s × 256 Hz = 128 samples
    assert response.channels[0].n_original_samples == 128


def test_waveform_time_window_by_samples():
    rec = _make_recording(n_samples=500)
    response = get_waveform(rec, channel_indices=[0], start_sample=10, end_sample=100)
    assert response.channels[0].n_original_samples == 90


def test_waveform_window_clamped_to_recording():
    rec = _make_recording(n_samples=200)
    response = get_waveform(rec, channel_indices=[0], start_sample=0, end_sample=99999)
    assert response.channels[0].n_original_samples == 200


def test_waveform_empty_window_raises():
    rec = _make_recording(n_samples=200)
    with pytest.raises(ValueError, match="empty"):
        get_waveform(rec, channel_indices=[0], start_sample=100, end_sample=100)


def test_waveform_seconds_without_rate_raises():
    rec = _make_recording(sampling_rate_hz=None)
    with pytest.raises(ValueError, match="sampling rate"):
        get_waveform(rec, channel_indices=[0], start_seconds=0.0, end_seconds=1.0)


# ---------------------------------------------------------------------------
# Waveform: 3D recording trial selection
# ---------------------------------------------------------------------------


def test_waveform_3d_uses_trial_0_by_default():
    rec = _make_3d_recording()
    response = get_waveform(rec, channel_indices=[0])
    assert response.trial_index == 0


def test_waveform_3d_different_trials_different_data():
    rec = _make_3d_recording(n_trials=5, seed=99)
    r0 = get_waveform(rec, channel_indices=[0], trial_index=0)
    r1 = get_waveform(rec, channel_indices=[0], trial_index=1)
    # Trials have different data (synthetic random).
    assert r0.channels[0].amplitude_values != r1.channels[0].amplitude_values


def test_waveform_3d_invalid_trial_raises():
    rec = _make_3d_recording()
    with pytest.raises((ValueError, IndexError)):
        get_waveform(rec, channel_indices=[0], trial_index=9999)


# ---------------------------------------------------------------------------
# Channel statistics
# ---------------------------------------------------------------------------


def test_channel_stats_matches_numpy():
    rng = np.random.default_rng(5)
    arr = rng.standard_normal((200, 3))
    rec = _make_recording(n_samples=200, n_channels=3)
    # Replace array with our controlled one.
    object.__setattr__(rec, "array", arr)
    summary = get_channel_stats(rec)
    for i in range(3):
        np.testing.assert_almost_equal(summary.signal_mean[i], float(arr[:, i].mean()), decimal=10)
        np.testing.assert_almost_equal(summary.signal_std[i], float(arr[:, i].std(ddof=0)), decimal=10)


def test_channel_stats_flatline_detected():
    rec = _make_recording(n_samples=100, n_channels=2)
    # Make ch0 flatline
    arr = rec.array.copy()
    arr[:, 0] = 5.0
    object.__setattr__(rec, "array", arr)
    summary = get_channel_stats(rec)
    assert summary.is_flatline[0] is True
    assert summary.is_flatline[1] is False


def test_channel_stats_duration_correct():
    rec = _make_recording(n_samples=512, sampling_rate_hz=256.0)
    summary = get_channel_stats(rec)
    assert summary.duration_seconds == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------


def test_psd_requires_sampling_rate():
    rec = _make_recording(sampling_rate_hz=None)
    with pytest.raises(ValueError, match="sampling rate"):
        get_psd(rec)


def test_psd_frequencies_non_negative():
    rec = _make_recording(n_samples=512, sampling_rate_hz=256.0)
    result = get_psd(rec, channel_indices=[0])
    assert all(f >= 0 for f in result.channels[0].frequencies_hz)


def test_psd_max_frequency_is_nyquist():
    rec = _make_recording(n_samples=512, sampling_rate_hz=256.0)
    result = get_psd(rec, channel_indices=[0])
    freqs = result.channels[0].frequencies_hz
    assert max(freqs) == pytest.approx(128.0, abs=2.0)  # Nyquist = 256/2


def test_psd_power_density_non_negative():
    rec = _make_recording(n_samples=512, sampling_rate_hz=256.0)
    result = get_psd(rec, channel_indices=[0])
    assert all(p >= 0 for p in result.channels[0].power_density)


def test_psd_assumptions_documented():
    rec = _make_recording(n_samples=512, sampling_rate_hz=256.0)
    result = get_psd(rec)
    assert len(result.assumptions) > 0
    assert any("welch" in a.lower() or "nperseg" in a.lower() for a in result.assumptions)


def test_psd_does_not_modify_source():
    rec = _make_recording(n_samples=512, sampling_rate_hz=256.0)
    original = rec.array.copy()
    get_psd(rec)
    np.testing.assert_array_equal(rec.array, original)


def test_psd_too_short_raises():
    rec = _make_recording(n_samples=5, sampling_rate_hz=256.0)
    with pytest.raises(ValueError, match="short"):
        get_psd(rec)
