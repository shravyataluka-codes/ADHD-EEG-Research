"""Tests for api.app.ingestion.mat_reader.

All test data is synthetic.  No real research recordings are used.
"""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest
import scipy.io



from api.app.ingestion.mat_reader import (
    _build_canonical_array,
    read_mat,
)
from api.app.ingestion.recording import (
    AmbiguousStructureError,
    IngestionError,
)
from api.app.ingestion.mat_reader import (
    AmbiguousStructureError,
    IngestionError,
    UnsupportedFormatError,
    detect_mat_format,
    inspect_mat_structure,
    read_mat,
)
from api.app.ingestion.recording import EEGRecording


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_detect_mat_v5_format(mat_v5_single_var):
    assert detect_mat_format(mat_v5_single_var) == "mat_v5"


def test_detect_mat_hdf5_format(mat_hdf5_single_var):
    assert detect_mat_format(mat_hdf5_single_var) == "mat_v7.3_hdf5"


def test_detect_invalid_magic_raises(mat_invalid_magic):
    with pytest.raises(UnsupportedFormatError, match="supported MATLAB"):
        detect_mat_format(mat_invalid_magic)


# ---------------------------------------------------------------------------
# Standard MAT v5 reading
# ---------------------------------------------------------------------------


def test_mat_v5_single_var_reads_correctly(mat_v5_single_var):
    rec = read_mat(mat_v5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert isinstance(rec, EEGRecording)
    assert rec.source_format == "mat"
    assert rec.provenance == "mat_v5"
    assert rec.n_channels > 0
    assert rec.n_samples > 0


def test_mat_v5_single_var_shape(mat_v5_single_var):
    rec = read_mat(mat_v5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    # Shape (1000, 4) → samples=1000, channels=4
    assert rec.n_samples == 1000
    assert rec.n_channels == 4


def test_mat_v5_single_var_auto_selected_with_warning(mat_v5_single_var):
    rec = read_mat(mat_v5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    # Auto-selection should produce a warning.
    assert rec.import_status == "warnings"
    assert any("auto-selected" in w.lower() or "auto" in w.lower() for w in rec.warnings)


def test_mat_v5_array_dtype_is_float64(mat_v5_single_var):
    rec = read_mat(mat_v5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert rec.array.dtype == np.float64


def test_mat_v5_no_3d_for_2d_array(mat_v5_single_var):
    rec = read_mat(mat_v5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert rec.array_3d is None
    assert rec.n_trials is None


# ---------------------------------------------------------------------------
# MAT v5 — multiple variable ambiguity
# ---------------------------------------------------------------------------


def test_mat_v5_multi_var_raises_without_selection(mat_v5_multi_var):
    with pytest.raises(AmbiguousStructureError, match="candidate"):
        read_mat(mat_v5_multi_var, "eeg.mat", sampling_rate_hz=256.0)


def test_mat_v5_multi_var_explicit_selection_works(mat_v5_multi_var):
    rec = read_mat(
        mat_v5_multi_var, "eeg.mat", sampling_rate_hz=256.0, variable_name="eeg_a"
    )
    assert rec.n_channels == 2
    assert rec.n_samples == 2000


def test_mat_v5_missing_variable_raises(mat_v5_single_var):
    with pytest.raises(IngestionError, match="not found"):
        read_mat(mat_v5_single_var, "eeg.mat", variable_name="nonexistent")


# ---------------------------------------------------------------------------
# MAT v5 — no suitable numeric variable
# ---------------------------------------------------------------------------


def test_mat_v5_no_numeric_raises(mat_v5_no_numeric):
    with pytest.raises(IngestionError, match="no large numeric"):
        read_mat(mat_v5_no_numeric, "eeg.mat", sampling_rate_hz=256.0)


# ---------------------------------------------------------------------------
# MAT v5 — 3D array (epochs / trials)
# ---------------------------------------------------------------------------


def test_mat_v5_3d_array_stored(mat_v5_3d):
    rec = read_mat(mat_v5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="trials")
    assert rec.array_3d is not None
    assert rec.n_trials is not None
    assert rec.n_trials > 0


def test_mat_v5_3d_trial_view_correct_shape(mat_v5_3d):
    rec = read_mat(mat_v5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="trials")
    view = rec.get_trial_view(0)
    assert view.ndim == 2
    assert view.shape[1] == rec.n_channels


def test_mat_v5_3d_all_trials_accessible(mat_v5_3d):
    rec = read_mat(mat_v5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="trials")
    for i in range(rec.n_trials):
        view = rec.get_trial_view(i)
        assert view.ndim == 2


def test_mat_v5_3d_trial_view_does_not_modify_array(mat_v5_3d):
    rec = read_mat(mat_v5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="trials")
    original_shape = rec.array_3d.shape
    _ = rec.get_trial_view(0)
    assert rec.array_3d.shape == original_shape


def test_mat_v5_3d_out_of_range_raises(mat_v5_3d):
    rec = read_mat(mat_v5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="trials")
    with pytest.raises(IndexError):
        rec.get_trial_view(9999)


# ---------------------------------------------------------------------------
# MAT v7.3 HDF5 reading
# ---------------------------------------------------------------------------


def test_mat_hdf5_single_var_reads_correctly(mat_hdf5_single_var):
    rec = read_mat(mat_hdf5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert isinstance(rec, EEGRecording)
    assert rec.provenance == "mat_v7.3_hdf5"


def test_mat_hdf5_records_h5py_shape_in_metadata(mat_hdf5_single_var):
    rec = read_mat(mat_hdf5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert "h5py_shape" in rec.source_metadata
    assert "matlab_inferred_shape" in rec.source_metadata


def test_mat_hdf5_orientation_note_documented(mat_hdf5_single_var):
    rec = read_mat(mat_hdf5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert rec.orientation_note  # non-empty string
    assert any("h5py" in w.lower() or "hdf5" in w.lower() or "matlab" in w.lower()
               for w in rec.warnings)


def test_mat_hdf5_array_dtype_is_float64(mat_hdf5_single_var):
    rec = read_mat(mat_hdf5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert rec.array.dtype == np.float64


def test_mat_hdf5_multi_var_ambiguous_raises(mat_hdf5_multi_var):
    with pytest.raises(AmbiguousStructureError, match="candidate"):
        read_mat(mat_hdf5_multi_var, "eeg.mat", sampling_rate_hz=256.0)


def test_mat_hdf5_explicit_variable_selection(mat_hdf5_multi_var):
    rec = read_mat(mat_hdf5_multi_var, "eeg.mat", sampling_rate_hz=256.0, variable_name="eeg_a")
    assert rec.n_samples > 0


def test_mat_hdf5_no_numeric_raises(mat_hdf5_no_numeric):
    with pytest.raises(IngestionError, match="no large numeric"):
        read_mat(mat_hdf5_no_numeric, "eeg.mat", sampling_rate_hz=256.0)


def test_mat_hdf5_missing_variable_raises(mat_hdf5_single_var):
    with pytest.raises(IngestionError, match="not found"):
        read_mat(mat_hdf5_single_var, "eeg.mat", variable_name="nonexistent")


# ---------------------------------------------------------------------------
# MAT v7.3 — 3D array
# ---------------------------------------------------------------------------


def test_mat_hdf5_3d_stored_fully(mat_hdf5_3d):
    rec = read_mat(
        mat_hdf5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="eeg3d"
    )
    assert rec.array_3d is not None
    assert rec.n_trials is not None


def test_mat_hdf5_3d_trial_view_is_2d(mat_hdf5_3d):
    rec = read_mat(
        mat_hdf5_3d, "trials.mat", sampling_rate_hz=256.0, variable_name="eeg3d"
    )
    view = rec.get_trial_view(0)
    assert view.ndim == 2


# ---------------------------------------------------------------------------
# Explicit axis configuration
# ---------------------------------------------------------------------------


def test_explicit_time_and_channel_axis_honoured(mat_v5_single_var):
    """When time_axis=0 and channel_axis=1, orientation should match."""
    rec = read_mat(
        mat_v5_single_var, "eeg.mat", sampling_rate_hz=256.0,
        variable_name="eeg", time_axis=0, channel_axis=1
    )
    # (1000, 4) with time=0 → samples=1000, channels=4
    assert rec.n_samples == 1000
    assert rec.n_channels == 4


# ---------------------------------------------------------------------------
# Invalid / unsupported structure
# ---------------------------------------------------------------------------


def test_invalid_magic_raises(mat_invalid_magic):
    with pytest.raises(UnsupportedFormatError):
        read_mat(mat_invalid_magic, "bad.mat", sampling_rate_hz=256.0)


def test_empty_content_raises():
    with pytest.raises(IngestionError, match="empty"):
        read_mat(b"", "empty.mat", sampling_rate_hz=256.0)


# ---------------------------------------------------------------------------
# Sampling rate handling
# ---------------------------------------------------------------------------


def test_no_sampling_rate_produces_warning(mat_v5_single_var):
    rec = read_mat(mat_v5_single_var, "eeg.mat", sampling_rate_hz=None)
    assert rec.sampling_rate_hz is None
    assert any("sampling rate" in w.lower() for w in rec.warnings)


def test_negative_sampling_rate_raises(mat_v5_single_var):
    with pytest.raises(IngestionError, match="positive"):
        read_mat(mat_v5_single_var, "eeg.mat", sampling_rate_hz=-100.0, variable_name="eeg")


# ---------------------------------------------------------------------------
# inspect_mat_structure
# ---------------------------------------------------------------------------


def test_inspect_mat_v5_returns_candidates(mat_v5_single_var):
    result = inspect_mat_structure(mat_v5_single_var)
    assert result["format"] == "mat_v5"
    assert "eeg" in result["numeric_candidates"]


def test_inspect_mat_hdf5_returns_candidates(mat_hdf5_single_var):
    result = inspect_mat_structure(mat_hdf5_single_var)
    assert result["format"] == "mat_v7.3_hdf5"
    assert len(result["numeric_candidates"]) >= 1


def test_inspect_empty_raises():
    with pytest.raises(IngestionError, match="empty"):
        inspect_mat_structure(b"")


def test_inspect_invalid_magic_raises(mat_invalid_magic):
    with pytest.raises(UnsupportedFormatError):
        inspect_mat_structure(mat_invalid_magic)


# ---------------------------------------------------------------------------
# Source metadata preservation
# ---------------------------------------------------------------------------


def test_v5_source_metadata_contains_variable_info(mat_v5_single_var):
    rec = read_mat(mat_v5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert "variable_name" in rec.source_metadata
    assert "variable_shape" in rec.source_metadata


def test_hdf5_source_metadata_contains_shapes(mat_hdf5_single_var):
    rec = read_mat(mat_hdf5_single_var, "eeg.mat", sampling_rate_hz=256.0)
    assert "h5py_shape" in rec.source_metadata
    assert "matlab_inferred_shape" in rec.source_metadata
    # Both shapes should be recorded without silent transposition.
    h5_shape = rec.source_metadata["h5py_shape"]
    ml_shape = rec.source_metadata["matlab_inferred_shape"]
    assert list(reversed(h5_shape)) == ml_shape
# ---------------------------------------------------------------------------
# Regression: MAT orientation and trial preservation
# ---------------------------------------------------------------------------

def test_mat_3d_trial_data_is_preserved(mat_v5_3d):
    rec = read_mat(
        mat_v5_3d,
        "trials.mat",
        sampling_rate_hz=256.0,
        variable_name="trials",
    )

    assert rec.array_3d is not None

    for i in range(rec.n_trials):
        trial = rec.get_trial_view(i)

        assert trial.ndim == 2
        assert trial.shape == rec.array_3d.shape[1:]


def test_mat_3d_trial_view_contains_original_values(mat_v5_3d):
    rec = read_mat(
        mat_v5_3d,
        "trials.mat",
        sampling_rate_hz=256.0,
        variable_name="trials",
    )

    for i in range(rec.n_trials):
        np.testing.assert_array_equal(
            rec.get_trial_view(i),
            rec.array_3d[i],
        )


def test_mat_3d_trial_view_is_not_a_copy(mat_v5_3d):
    rec = read_mat(
        mat_v5_3d,
        "trials.mat",
        sampling_rate_hz=256.0,
        variable_name="trials",
    )

    trial = rec.get_trial_view(0)

    assert np.shares_memory(trial, rec.array_3d)


def test_mat_explicit_axes_reject_same_axis():
    array = np.random.default_rng(42).standard_normal(
        (10, 100, 4)
    )

    with pytest.raises(IngestionError):
        _build_canonical_array(
            array=array,
            ndim_expected=3,
            time_axis=1,
            channel_axis=1,
            provenance="mat_v5",
            warnings_out=[],
            metadata={},
        )


def test_mat_explicit_axes_reject_out_of_range():
    array = np.random.default_rng(42).standard_normal(
        (10, 100, 4)
    )

    with pytest.raises(IngestionError):
        _build_canonical_array(
            array=array,
            ndim_expected=3,
            time_axis=5,
            channel_axis=2,
            provenance="mat_v5",
            warnings_out=[],
            metadata={},
        )
