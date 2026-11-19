"""CSV EEG file reader for Drushti Detect Phase 2.

Supported CSV format
--------------------
* Encoding: UTF-8 (with or without BOM).
* Optional single non-numeric header row (auto-detected).
* One row per time sample; one column per EEG channel.
* All data values must be finite real numbers.
* Missing / NaN values are explicitly rejected (not silently discarded).
* Blank rows are skipped.
* Minimum 2 rows of numeric data required.
* Sampling rate must be supplied explicitly (not encoded in the file).

Limitations
-----------
* The first column is treated as EEG data if numeric.  There is no
  time-column support; a column labelled "time" or "t" in the header will
  be treated as a channel.
* No automatic resampling.
* No unit information — units are always marked as unknown.
* No automatic filtering, normalisation, or artifact removal.
"""

from __future__ import annotations

import csv
import logging
import uuid
from io import StringIO
from pathlib import Path
from typing import Sequence

import numpy as np

from .recording import AmbiguousStructureError, EEGRecording, IngestionError

LOGGER = logging.getLogger(__name__)

# Minimum number of numeric rows required to form a valid recording.
_MIN_ROWS = 2
# Maximum columns (channels) accepted without an explicit override warning.
_WARN_CHANNEL_THRESHOLD = 128


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_numeric_cell(value: str) -> bool:
    """Return True when *value* can be parsed as a finite float."""
    try:
        result = float(value.strip())
    except (ValueError, OverflowError):
        return False
    return True  # NaN/inf handled separately after parsing all rows


def _safe_float(value: str, line: int, col: int) -> float:
    """Parse *value* as float; raise :class:`IngestionError` on failure."""
    stripped = value.strip()
    if not stripped:
        raise IngestionError(
            f"CSV line {line}, column {col}: empty cell is not a valid numeric value."
        )
    try:
        result = float(stripped)
    except (ValueError, OverflowError) as exc:
        raise IngestionError(
            f"CSV line {line}, column {col}: {stripped!r} is not a valid number."
        ) from exc
    return result


def _decode_bytes(content: bytes) -> str:
    """Decode uploaded bytes as UTF-8 (with or without BOM)."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionError(
            "CSV file must be UTF-8 encoded (with or without BOM)."
        ) from exc


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------


def read_csv(
    content: bytes,
    filename: str,
    sampling_rate_hz: float | None = None,
    recording_id: str | None = None,
) -> EEGRecording:
    """Parse a CSV file into a canonical :class:`~.recording.EEGRecording`.

    Parameters
    ----------
    content:
        Raw file bytes.
    filename:
        Original filename (used only for metadata; path components are
        stripped).
    sampling_rate_hz:
        Sampling frequency in Hz.  Pass ``None`` if unknown — the recording
        will be created with ``sampling_rate_hz=None`` and a warning.
    recording_id:
        UUID to assign.  A new UUID is generated when omitted.

    Returns
    -------
    EEGRecording
        Import status is ``"ok"`` on clean success, ``"warnings"`` when
        assumptions were made.

    Raises
    ------
    IngestionError
        On fatal parse errors (empty file, non-numeric data, etc.).
    """
    rid = recording_id or str(uuid.uuid4())
    safe_name = Path(filename).name  # strip any path components
    warnings: list[str] = []

    if not content:
        raise IngestionError("CSV file is empty.")

    text = _decode_bytes(content)
    rows: list[list[float]] = []
    header_names: list[str] | None = None
    n_cols: int | None = None
    first_data_line: int | None = None

    try:
        reader = csv.reader(StringIO(text))
        for line_num, row in enumerate(reader, start=1):
            # Skip completely blank lines.
            if not row or all(not cell.strip() for cell in row):
                continue

            # Determine column count from first non-blank row.
            if n_cols is None:
                n_cols = len(row)

            if len(row) != n_cols:
                raise IngestionError(
                    f"CSV line {line_num} has {len(row)} column(s); "
                    f"expected {n_cols} (consistent with line 1)."
                )

            # Try to parse all cells as floats.
            parsed: list[float] = []
            failed_cells: list[int] = []
            for col_idx, cell in enumerate(row, start=1):
                if _is_numeric_cell(cell):
                    parsed.append(float(cell.strip()))
                else:
                    failed_cells.append(col_idx)

            if failed_cells:
                # First non-blank, non-numeric row: candidate header.
                if not rows and header_names is None:
                    header_names = [cell.strip() for cell in row]
                    LOGGER.debug(
                        "CSV line %d interpreted as header: %s", line_num, header_names
                    )
                    continue
                # Non-numeric row after data has started → error.
                raise IngestionError(
                    f"CSV line {line_num}: non-numeric value(s) in column(s) "
                    f"{failed_cells}.  All data rows must contain real numbers."
                )

            if first_data_line is None:
                first_data_line = line_num
            rows.append(parsed)

    except csv.Error as exc:
        raise IngestionError(f"CSV parse error: {exc}") from exc

    if not rows:
        raise IngestionError("CSV file contains no numeric data rows.")

    if len(rows) < _MIN_ROWS:
        raise IngestionError(
            f"CSV file contains only {len(rows)} data row(s); "
            f"at least {_MIN_ROWS} samples are required."
        )

    if n_cols is None or n_cols < 1:
        raise IngestionError("CSV file has no columns.")

    # Build array: shape [n_samples, n_channels].
    array = np.asarray(rows, dtype=np.float64)
    n_samples, n_channels = array.shape

    # Check for non-finite values.
    non_finite_mask = ~np.isfinite(array)
    if non_finite_mask.any():
        counts = non_finite_mask.sum(axis=0)
        bad_channels = np.flatnonzero(counts).tolist()
        raise IngestionError(
            f"CSV contains NaN or infinite value(s) in column(s) "
            f"{[c + 1 for c in bad_channels]}.  "
            "Missing and infinite values are not accepted."
        )

    # Sampling rate validation.
    if sampling_rate_hz is not None:
        if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
            raise IngestionError(
                f"sampling_rate_hz must be a positive finite number; "
                f"got {sampling_rate_hz!r}."
            )
    else:
        warnings.append(
            "No sampling rate was provided.  Time-axis calculations and "
            "duration reporting are unavailable."
        )

    # Channel name handling.
    channel_names: list[str] | None = None
    if header_names is not None:
        if len(header_names) == n_channels:
            channel_names = header_names
        else:
            warnings.append(
                f"Header row has {len(header_names)} name(s) but array has "
                f"{n_channels} channel(s); header names were discarded."
            )

    if n_channels > _WARN_CHANNEL_THRESHOLD:
        warnings.append(
            f"CSV contains {n_channels} channel columns, which is unusually large.  "
            "Verify that the file orientation is correct (rows = samples, "
            "columns = channels)."
        )

    import_status: str = "warnings" if warnings else "ok"

    LOGGER.info(
        "CSV ingestion complete: %s, shape=(%d, %d), rate=%s Hz, status=%s",
        safe_name,
        n_samples,
        n_channels,
        sampling_rate_hz,
        import_status,
    )

    return EEGRecording(
        recording_id=rid,
        original_filename=safe_name,
        source_format="csv",
        provenance="csv",
        n_channels=n_channels,
        n_samples=n_samples,
        n_trials=None,
        sampling_rate_hz=sampling_rate_hz,
        channel_names=channel_names,
        channel_units=None,
        signal_units_known=False,
        observed_time_axis=0,
        orientation_note=(
            "CSV rows are assumed to be time samples and columns are assumed "
            "to be channels.  No time column was present."
        ),
        data_shape=(n_samples, n_channels),
        array=array,
        array_3d=None,
        source_metadata={
            "header_row": header_names,
            "first_data_line": first_data_line,
            "n_rows_parsed": len(rows),
        },
        import_status=import_status,  # type: ignore[arg-type]
        warnings=warnings,
        errors=[],
    )
