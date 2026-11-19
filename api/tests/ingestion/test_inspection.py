"""Tests for api.app.ingestion.inspection.

All test data is synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.app.ingestion.csv_reader import read_csv
from api.app.ingestion.inspection import SignalQualityReport, inspect_recording
from api.app.ingestion.mat_reader import read_mat
from api.app.ingestion.recording import EEGRecording


# ---------------------------------------------------------------------------
# Helper: build a recording directly from a numpy array
# ---------------------------------------------------------------------------


def _make_recording(
    array: np.ndarray,
    sampling_rate_hz: float | None = 256.0,
    n_trials: int | None = None,
    array_3d: np.ndarray | None = None,
) -> EEGRecording:
    """Build a minimal EEGRecording for inspection tests."""
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    n_samples, n_channels = array.shape
    return EEGRecording(
        recording_id="test-id",
        original_filename="test.csv",
        source_format="csv",
        provenance="csv",
        n_channels=n_channels,
        n_samples=n_samples,
        n_trials=n_trials,
        sampling_rate_hz=sampling_rate_hz,
        channel_names=None,
        channel_units=None,
        signal_units_known=False,
        observed_time_axis=0,
        orientation_note="test",
        data_shape=(n_samples, n_channels),
        array=array,
        array_3d=array_3d,
        source_metadata={},
        import_status="ok",
        warnings=[],
        errors=[],
    )


# ---------------------------------------------------------------------------
# Basic inspection
# ---------------------------------------------------------------------------


def test_inspect_produces_quality_report():
    arr = np.random.default_rng(1).standard_normal((100, 2))
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    assert isinstance(report, SignalQualityReport)


def test_correct_channel_count():
    arr = np.ones((50, 5))
    arr += np.arange(5) * 0.1  # make each channel slightly different
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    assert report.n_channels == 5


def test_correct_sample_count():
    arr = np.random.default_rng(2).standard_normal((200, 3))
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    assert report.n_samples == 200


# ---------------------------------------------------------------------------
# Channel statistics correctness
# ---------------------------------------------------------------------------


def test_channel_statistics_match_numpy():
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((100, 3))
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    for ch_idx in range(3):
        stats = report.channel_stats[ch_idx]
        np.testing.assert_almost_equal(stats.signal_min, float(arr[:, ch_idx].min()), decimal=10)
        np.testing.assert_almost_equal(stats.signal_max, float(arr[:, ch_idx].max()), decimal=10)
        np.testing.assert_almost_equal(stats.signal_mean, float(arr[:, ch_idx].mean()), decimal=10)
        np.testing.assert_almost_equal(
            stats.signal_std, float(arr[:, ch_idx].std(ddof=0)), decimal=10
        )


def test_signal_range_correct():
    arr = np.array([[0.0, 10.0], [5.0, -10.0], [2.0, 0.0]])
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    # ch0: range = 5.0 - 0.0 = 5.0
    assert report.channel_stats[0].signal_range == pytest.approx(5.0)
    # ch1: range = 10.0 - (-10.0) = 20.0
    assert report.channel_stats[1].signal_range == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Duration calculation
# ---------------------------------------------------------------------------


def test_duration_calculated_correctly():
    arr = np.ones((256, 2))
    rec = _make_recording(arr, sampling_rate_hz=256.0)
    report = inspect_recording(rec)
    assert report.duration_seconds == pytest.approx(1.0)


def test_duration_none_when_rate_unknown():
    arr = np.ones((100, 2))
    rec = _make_recording(arr, sampling_rate_hz=None)
    report = inspect_recording(rec)
    assert report.duration_seconds is None


def test_duration_warning_when_rate_unknown():
    arr = np.ones((100, 2))
    rec = _make_recording(arr, sampling_rate_hz=None)
    report = inspect_recording(rec)
    assert any("sampling rate" in w.lower() or "duration" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Flatline detection
# ---------------------------------------------------------------------------


def test_flatline_channel_detected():
    arr = np.zeros((50, 2))
    arr[:, 1] = np.linspace(0, 1, 50)  # ch1 varies, ch0 flatline
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    assert 0 in report.flatline_channel_indices


def test_non_flatline_channel_not_detected():
    arr = np.random.default_rng(10).standard_normal((100, 2))
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    # Gaussian noise should have non-zero range.
    assert report.flatline_channel_indices == []


def test_flatline_produces_quality_warning():
    arr = np.zeros((50, 2))
    arr[:, 1] = np.linspace(0, 1, 50)
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    assert any("flatline" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Non-finite value detection
# ---------------------------------------------------------------------------


def test_nonfinite_count_correct():
    arr = np.ones((50, 2))
    arr[5, 0] = float("nan")
    arr[10, 0] = float("inf")
    arr[20, 1] = float("-inf")
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    assert report.total_nonfinite_count == 3


def test_nonfinite_produces_warning():
    arr = np.ones((50, 2))
    arr[0, 0] = float("nan")
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    assert any("non-finite" in w.lower() or "nan" in w.lower() for w in report.warnings)


def test_channel_with_all_nonfinite():
    arr = np.ones((30, 2))
    arr[:, 0] = float("nan")
    rec = _make_recording(arr)
    report = inspect_recording(rec)
    # ch0 all NaN → stats should be None
    assert report.channel_stats[0].signal_min is None
    assert report.channel_stats[0].signal_max is None
    assert report.channel_stats[0].has_nonfinite


# ---------------------------------------------------------------------------
# 3D recording trial selection
# ---------------------------------------------------------------------------


def test_inspect_3d_recording_first_trial(mat_v5_3d):
    rec = read_mat(mat_v5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="trials")
    report = inspect_recording(rec, trial_index=0)
    assert report.trial_index == 0
    assert report.n_channels == rec.n_channels


def test_inspect_3d_recording_different_trials(mat_v5_3d):
    rec = read_mat(mat_v5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="trials")
    report_0 = inspect_recording(rec, trial_index=0)
    report_1 = inspect_recording(rec, trial_index=1)
    # Statistics may differ between trials (synthetic random data)
    assert report_0.n_channels == report_1.n_channels == rec.n_channels


def test_inspect_3d_out_of_range_raises(mat_v5_3d):
    rec = read_mat(mat_v5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="trials")
    with pytest.raises(IndexError):
        inspect_recording(rec, trial_index=9999)


# ---------------------------------------------------------------------------
# Inspection does not modify source data
# ---------------------------------------------------------------------------


def test_inspection_does_not_modify_array():
    arr = np.random.default_rng(99).standard_normal((100, 2))
    original = arr.copy()
    rec = _make_recording(arr)
    inspect_recording(rec)
    np.testing.assert_array_equal(rec.array, original)
# ---------------------------------------------------------------------------
# Regression: 3D recording memory accounting
# ---------------------------------------------------------------------------

def test_3d_recording_memory_is_not_double_counted():
    """The first-trial array is a view, not a separate allocation."""
    array_3d = np.random.default_rng(42).standard_normal(
        (10, 100, 4)
    ).astype(np.float64)

    rec = _make_recording(
        array=array_3d[0],
        n_trials=10,
        array_3d=array_3d,
    )

    assert np.shares_memory(rec.array, rec.array_3d)

    expected_memory = rec.array_3d.nbytes

    assert rec.memory_bytes == expected_memory


def test_2d_recording_memory_matches_array_size():
    """Continuous recordings should count their 2D array once."""
    array = np.random.default_rng(42).standard_normal(
        (100, 4)
    ).astype(np.float64)

    rec = _make_recording(array)

    assert rec.memory_bytes == array.nbytes


def test_trial_views_share_underlying_memory():
    """Accessing a trial must not create an unnecessary full copy."""
    array_3d = np.random.default_rng(42).standard_normal(
        (5, 100, 4)
    ).astype(np.float64)

    rec = _make_recording(
        array=array_3d[0],
        n_trials=5,
        array_3d=array_3d,
    )

    for trial_index in range(5):
        trial = rec.get_trial_view(trial_index)

        assert trial.shape == (100, 4)
        assert np.shares_memory(trial, array_3d)
