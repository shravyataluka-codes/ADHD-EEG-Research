"""Tests for api.app.ingestion.csv_reader.

All test data is synthetic.  No real research recordings are used.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.app.ingestion.csv_reader import read_csv
from api.app.ingestion.recording import EEGRecording, IngestionError


# ---------------------------------------------------------------------------
# Valid CSV — with header
# ---------------------------------------------------------------------------


def test_valid_csv_with_header_produces_correct_shape(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv", sampling_rate_hz=256.0)
    assert isinstance(rec, EEGRecording)
    assert rec.n_samples == 100
    assert rec.n_channels == 2
    assert rec.array.shape == (100, 2)


def test_valid_csv_with_header_names_extracted(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv", sampling_rate_hz=256.0)
    assert rec.channel_names == ["ch0", "ch1"]


def test_valid_csv_with_header_import_status_ok(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv", sampling_rate_hz=256.0)
    assert rec.import_status == "ok"
    assert rec.errors == []


# ---------------------------------------------------------------------------
# Valid CSV — without header
# ---------------------------------------------------------------------------


def test_valid_csv_no_header_produces_correct_shape(csv_no_header):
    rec = read_csv(csv_no_header, "test.csv", sampling_rate_hz=256.0)
    assert rec.n_samples == 100
    assert rec.n_channels == 2


def test_valid_csv_no_header_channel_names_none(csv_no_header):
    rec = read_csv(csv_no_header, "test.csv", sampling_rate_hz=256.0)
    assert rec.channel_names is None


def test_valid_csv_no_header_status_ok(csv_no_header):
    rec = read_csv(csv_no_header, "test.csv", sampling_rate_hz=256.0)
    assert rec.import_status == "ok"


# ---------------------------------------------------------------------------
# Multi-channel CSV
# ---------------------------------------------------------------------------


def test_multichannel_csv_correct_channels(csv_multichannel):
    rec = read_csv(csv_multichannel, "eeg8ch.csv", sampling_rate_hz=512.0)
    assert rec.n_channels == 8
    assert rec.array.shape == (200, 8)


def test_multichannel_csv_header_names_match(csv_multichannel):
    rec = read_csv(csv_multichannel, "eeg8ch.csv", sampling_rate_hz=512.0)
    assert rec.channel_names == [f"ch{i}" for i in range(8)]


# ---------------------------------------------------------------------------
# Invalid numeric data
# ---------------------------------------------------------------------------


def test_non_numeric_data_raises_ingestion_error(csv_non_numeric):
    with pytest.raises(IngestionError, match="not a valid number|non-numeric"):
        read_csv(csv_non_numeric, "bad.csv", sampling_rate_hz=256.0)


# ---------------------------------------------------------------------------
# Empty file
# ---------------------------------------------------------------------------


def test_empty_file_raises_ingestion_error(csv_empty):
    with pytest.raises(IngestionError, match="empty"):
        read_csv(csv_empty, "empty.csv", sampling_rate_hz=256.0)


def test_header_only_raises_ingestion_error(csv_header_only):
    with pytest.raises(IngestionError, match="no numeric"):
        read_csv(csv_header_only, "header_only.csv", sampling_rate_hz=256.0)


# ---------------------------------------------------------------------------
# Missing / NaN / Inf values
# ---------------------------------------------------------------------------


def test_missing_value_raises_ingestion_error(csv_missing_value):
    with pytest.raises(IngestionError):
        read_csv(csv_missing_value, "missing.csv", sampling_rate_hz=256.0)


def test_nan_value_raises_ingestion_error(csv_nan_value):
    with pytest.raises(IngestionError, match="NaN|non-finite|nan|not a valid number"):
        read_csv(csv_nan_value, "nan.csv", sampling_rate_hz=256.0)


def test_inf_value_raises_ingestion_error(csv_inf_value):
    with pytest.raises(IngestionError, match="NaN|infinite|non-finite|not a valid number"):
        read_csv(csv_inf_value, "inf.csv", sampling_rate_hz=256.0)


# ---------------------------------------------------------------------------
# Incorrect format (binary)
# ---------------------------------------------------------------------------


def test_binary_data_raises_ingestion_error(csv_binary):
    with pytest.raises(IngestionError):
        read_csv(csv_binary, "binary.bin", sampling_rate_hz=256.0)


# ---------------------------------------------------------------------------
# Column inconsistency
# ---------------------------------------------------------------------------


def test_inconsistent_column_count_raises_ingestion_error(csv_wrong_columns):
    with pytest.raises(IngestionError, match="column"):
        read_csv(csv_wrong_columns, "wrong_cols.csv", sampling_rate_hz=256.0)


# ---------------------------------------------------------------------------
# Sampling rate handling
# ---------------------------------------------------------------------------


def test_no_sampling_rate_produces_warning(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv", sampling_rate_hz=None)
    assert rec.sampling_rate_hz is None
    assert rec.import_status == "warnings"
    assert any("sampling rate" in w.lower() for w in rec.warnings)


def test_sampling_rate_stored_correctly(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv", sampling_rate_hz=512.0)
    assert rec.sampling_rate_hz == 512.0


def test_negative_sampling_rate_raises(csv_with_header):
    with pytest.raises(IngestionError, match="positive"):
        read_csv(csv_with_header, "test.csv", sampling_rate_hz=-1.0)


def test_zero_sampling_rate_raises(csv_with_header):
    with pytest.raises(IngestionError, match="positive"):
        read_csv(csv_with_header, "test.csv", sampling_rate_hz=0.0)


# ---------------------------------------------------------------------------
# Duration calculation
# ---------------------------------------------------------------------------


def test_duration_calculated_when_rate_known(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv", sampling_rate_hz=100.0)
    assert rec.duration_seconds == pytest.approx(1.0)  # 100 samples / 100 Hz


def test_duration_none_when_rate_unknown(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv", sampling_rate_hz=None)
    assert rec.duration_seconds is None


# ---------------------------------------------------------------------------
# Canonical representation
# ---------------------------------------------------------------------------


def test_array_dtype_is_float64(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv", sampling_rate_hz=256.0)
    assert rec.array.dtype == np.float64


def test_source_format_is_csv(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv")
    assert rec.source_format == "csv"
    assert rec.provenance == "csv"


def test_no_3d_array_for_csv(csv_with_header):
    rec = read_csv(csv_with_header, "test.csv")
    assert rec.array_3d is None
    assert rec.n_trials is None


def test_path_component_stripped_from_filename():
    content = b"ch0,ch1\n1.0,2.0\n3.0,4.0\n5.0,6.0\n"
    rec = read_csv(content, "/dangerous/../../path/attack.csv")
    assert "/" not in rec.original_filename
    assert "\\" not in rec.original_filename
    assert rec.original_filename == "attack.csv"


def test_values_match_source():
    """Verify that array values exactly match the uploaded CSV data."""
    csv_bytes = b"ch0,ch1\n1.5,2.5\n3.5,4.5\n5.5,6.5\n"
    rec = read_csv(csv_bytes, "exact.csv", sampling_rate_hz=256.0)
    expected = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    np.testing.assert_array_almost_equal(rec.array, expected)


def test_time_axis_is_rows():
    rec = read_csv(b"a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", "t.csv")
    assert rec.observed_time_axis == 0


def test_recording_id_assigned():
    rec = read_csv(b"a,b\n1.0,2.0\n3.0,4.0\n", "t.csv")
    assert rec.recording_id is not None
    assert len(rec.recording_id) > 0


def test_custom_recording_id_preserved():
    rec = read_csv(b"a,b\n1.0,2.0\n3.0,4.0\n", "t.csv", recording_id="my-id-123")
    assert rec.recording_id == "my-id-123"


def test_flatline_channel_allowed_in_csv(csv_flatline):
    """CSV reader does not reject flatline channels; inspection handles that."""
    rec = read_csv(csv_flatline, "flat.csv", sampling_rate_hz=256.0)
    assert rec.import_status in ("ok", "warnings")
    assert rec.n_channels == 2


def test_utf8_bom_handled():
    csv_bom = b"\xef\xbb\xbf" + b"a,b\n1.0,2.0\n3.0,4.0\n"
    rec = read_csv(csv_bom, "bom.csv")
    assert rec.n_channels == 2
