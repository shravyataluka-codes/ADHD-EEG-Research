"""MATLAB MAT file reader for Drushti Detect Phase 2.

Supported formats
-----------------
* MATLAB v5 (classic .mat): read via ``scipy.io.loadmat``.
* MATLAB v7.3 (HDF5-based .mat): read via ``h5py``.

Format detection
----------------
Format is detected from magic bytes in the file header, **not** from the
filename extension.  This prevents misclassification when users rename files.

MATLAB v7.3 / HDF5 orientation note
-------------------------------------
HDF5 stores multidimensional arrays in C order (row-major).  MATLAB writes
arrays in Fortran/column-major order, so when h5py reads a MATLAB v7.3
dataset, the axis order is **reversed** relative to the MATLAB variable.

For example, a MATLAB array ``X`` of size ``[385 × 56 × 5000]`` is stored
in HDF5 as a dataset of shape ``(5000, 56, 385)`` (Python/h5py view).

This reader **preserves the h5py shape** as the ``observed_shape`` and records
both the h5py shape and the inferred MATLAB shape in ``source_metadata``.
Orientation is NOT blindly reversed.  Instead:

1. The h5py shape is inspected.
2. If the inferred MATLAB orientation is unambiguous (i.e. one axis is clearly
   channels and another is clearly samples), the array is reoriented to the
   canonical ``[n_samples, n_channels]`` format and a warning documents the
   assumption.
3. If orientation is ambiguous, ``observed_time_axis`` is set to ``"unknown"``
   and the array is left in h5py shape.  The user must supply
   ``time_axis`` and ``channel_axis`` explicitly.

Variable selection
------------------
1. All top-level variables are listed with their shapes and dtypes.
2. Candidate numeric arrays (not cell arrays, not pure scalar structs) are
   identified.
3. If exactly one candidate exists, it is selected automatically with a
   warning.
4. If multiple candidates exist and no ``variable_name`` was supplied, an
   :class:`~.recording.AmbiguousStructureError` is raised listing all
   candidates.

Limitations
-----------
* 3D arrays are stored in full.  The canonical ``array`` attribute holds a
  view of the first (or selected) trial.
* Cell arrays, nested structs, and function handles are not supported as the
  primary EEG variable.
* No automatic filtering, resampling, normalisation, or artifact removal.
* The 56-to-60 channel discrepancy in the ADHD dataset's ``chan.mat`` is
  not resolved here; ``chan.mat``-style structs are extracted as metadata only.
"""

from __future__ import annotations

import io
import logging
import struct
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io

from .recording import (
    AmbiguousStructureError,
    EEGRecording,
    IngestionError,
    UnsupportedFormatError,
)

LOGGER = logging.getLogger(__name__)

# MATLAB v5 magic string (first 128 bytes start with this).
_MAT5_MAGIC = b"MATLAB 5.0 MAT-file"
# HDF5 magic bytes at offset 0.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
# Maximum number of bytes inspected for magic detection.
_MAGIC_PEEK = 128

# Maximum element count before a numeric array is considered "too large" for
# a metadata-only struct (not used for rejection, only for classification).
_METADATA_MAX_ELEMENTS = 1024

# When one dimension is ≤ this value and others are larger, it is a candidate
# for the channel axis (heuristic, explicitly documented as an assumption).
_CHANNEL_AXIS_MAX = 512

# Minimum samples for a valid EEG recording.
_MIN_SAMPLES = 2


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_mat_format(content: bytes) -> str:
    """Return ``"mat_v5"`` or ``"mat_v7.3_hdf5"`` from magic bytes.

    Raises
    ------
    UnsupportedFormatError
        If the content does not look like any supported MAT variant.
    """
    if content[:8] == _HDF5_MAGIC:
        return "mat_v7.3_hdf5"
    if content[:len(_MAT5_MAGIC)] == _MAT5_MAGIC:
        return "mat_v5"
    raise UnsupportedFormatError(
        "File does not appear to be a supported MATLAB MAT file.  "
        "Expected either a MATLAB v5 or MATLAB v7.3 (HDF5) file."
    )


# ---------------------------------------------------------------------------
# Structure inspection helpers
# ---------------------------------------------------------------------------


def _is_numeric_dtype(dtype) -> bool:
    """Return True for real numeric dtypes (int*, uint*, float*, complex*)."""
    return np.issubdtype(dtype, np.number)


def _summarise_variable(name: str, value: Any) -> dict:
    """Produce a JSON-serialisable summary of a MATLAB variable."""
    if isinstance(value, np.ndarray):
        return {
            "name": name,
            "type": "numeric_array",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "is_numeric": _is_numeric_dtype(value.dtype),
            "n_elements": int(value.size),
        }
    if isinstance(value, np.void):
        return {"name": name, "type": "struct", "fields": list(value.dtype.names or [])}
    return {"name": name, "type": type(value).__name__}


def _h5_summarise_item(name: str, item) -> dict:
    """Summarise an h5py Dataset or Group."""
    if isinstance(item, h5py.Dataset):
        return {
            "name": name,
            "type": "dataset",
            "dtype": str(item.dtype),
            "shape": list(item.shape),
            "is_numeric": _is_numeric_dtype(item.dtype),
            "n_elements": int(item.size),
        }
    if isinstance(item, h5py.Group):
        return {"name": name, "type": "group", "keys": list(item.keys())}
    return {"name": name, "type": "unknown"}


# ---------------------------------------------------------------------------
# Orientation analysis
# ---------------------------------------------------------------------------


def _infer_orientation_2d(
    shape: tuple[int, int],
) -> tuple[int | str, int | str, str]:
    """Infer time and channel axes for a 2D array.

    Returns
    -------
    time_axis, channel_axis, note
        ``time_axis`` and ``channel_axis`` are integers (0 or 1) or the string
        ``"unknown"`` when orientation cannot be determined.
    """
    r, c = shape
    if r == c:
        note = (
            f"2D array is square ({r} × {c}).  Orientation is ambiguous; "
            "time_axis is unknown."
        )
        return "unknown", "unknown", note
    # Larger dimension is assumed to be time (documented assumption).
    if r > c:
        note = (
            f"2D array shape ({r}, {c}): rows > cols, so axis 0 is assumed "
            f"to be samples and axis 1 is assumed to be channels.  "
            "This is a heuristic, not a verified fact."
        )
        return 0, 1, note
    note = (
        f"2D array shape ({r}, {c}): cols > rows, so axis 1 is assumed "
        f"to be samples and axis 0 is assumed to be channels.  "
        "This is a heuristic, not a verified fact."
    )
    return 1, 0, note


def _infer_orientation_3d(
    shape: tuple[int, int, int],
    matlab_shape: tuple[int, int, int] | None = None,
) -> tuple[int | str, int | str, int | str, str]:
    """Infer trial, sample, and channel axes for a 3D array.

    Uses heuristics:
    * The smallest dimension (if ≤ _CHANNEL_AXIS_MAX) is treated as channels.
    * The largest remaining dimension is treated as trials (for epoched data).
    * The middle remaining dimension is treated as samples.

    Returns
    -------
    trial_axis, sample_axis, channel_axis, note
    """
    d0, d1, d2 = shape
    dims = sorted([(d, i) for i, d in enumerate(shape)])

    min_val, min_idx = dims[0]

    if min_val > _CHANNEL_AXIS_MAX:
        note = (
            f"3D array shape {shape}: all dimensions exceed {_CHANNEL_AXIS_MAX}; "
            "channel axis cannot be inferred heuristically.  Orientation is unknown."
        )
        return "unknown", "unknown", "unknown", note

    channel_axis = min_idx
    remaining = [(d, i) for d, i in dims[1:]]
    # Larger of remaining two → trials; smaller → samples.
    remaining_sorted = sorted(remaining, key=lambda x: x[0], reverse=True)
    trial_axis = remaining_sorted[0][1]
    sample_axis = remaining_sorted[1][1]

    matlab_note = ""
    if matlab_shape is not None:
        matlab_note = (
            f"  Inferred MATLAB shape (reversed from h5py): "
            f"{matlab_shape}."
        )

    note = (
        f"3D array shape {shape} (h5py/observed):{matlab_note}  "
        f"Heuristic assignment: "
        f"axis {channel_axis} = channels (smallest dim = {min_val}), "
        f"axis {trial_axis} = trials, "
        f"axis {sample_axis} = samples.  "
        "These are assumptions based on dimension sizes, not verified metadata."
    )
    return trial_axis, sample_axis, channel_axis, note


# ---------------------------------------------------------------------------
# MAT v5 reader
# ---------------------------------------------------------------------------


def _read_mat_v5(
    content: bytes,
    variable_name: str | None,
    time_axis: int | None,
    channel_axis: int | None,
) -> tuple[np.ndarray, str | None, dict, list[str], list[str]]:
    """Load a MAT v5 file and return ``(array, varname, metadata, warnings, candidates)``.

    ``candidates`` is the list of numeric variable names found.
    """
    warnings_out: list[str] = []
    try:
        mat = scipy.io.loadmat(io.BytesIO(content), squeeze_me=False)
    except Exception as exc:
        raise IngestionError(f"Could not read MAT v5 file: {exc}") from exc

    # Filter out MATLAB internal keys.
    public_keys = [k for k in mat if not k.startswith("__")]
    variable_summaries = {k: _summarise_variable(k, mat[k]) for k in public_keys}

    numeric_candidates = [
        k for k, s in variable_summaries.items()
        if s.get("is_numeric") and s.get("type") == "numeric_array"
        and s.get("n_elements", 0) > _METADATA_MAX_ELEMENTS
    ]

    if variable_name is None:
        if len(numeric_candidates) == 0:
            raise IngestionError(
                "MAT v5 file contains no large numeric arrays suitable as EEG data.  "
                f"Variables found: {list(public_keys)}"
            )
        if len(numeric_candidates) > 1:
            raise AmbiguousStructureError(
                f"MAT v5 file contains {len(numeric_candidates)} candidate numeric "
                f"arrays: {numeric_candidates}.  "
                "Provide variable_name to select the EEG variable."
            )
        variable_name = numeric_candidates[0]
        warnings_out.append(
            f"Variable '{variable_name}' was auto-selected (the only large numeric "
            f"array).  Candidates: {numeric_candidates}."
        )

    if variable_name not in mat:
        raise IngestionError(
            f"Variable '{variable_name}' not found in MAT v5 file.  "
            f"Available variables: {list(public_keys)}"
        )

    raw = mat[variable_name]
    if not isinstance(raw, np.ndarray):
        raise IngestionError(
            f"Variable '{variable_name}' is not a numeric array "
            f"(type={type(raw).__name__})."
        )
    if not _is_numeric_dtype(raw.dtype):
        raise IngestionError(
            f"Variable '{variable_name}' has dtype {raw.dtype}, which is not a "
            "supported real numeric type."
        )
    if np.iscomplexobj(raw):
        raise IngestionError(
            f"Variable '{variable_name}' contains complex values.  "
            "Only real-valued EEG data is supported."
        )

    array = raw.astype(np.float64, copy=False)

    metadata = {
        "mat_format": "mat_v5",
        "variable_name": variable_name,
        "variable_shape": list(array.shape),
        "variable_dtype": str(raw.dtype),
        "all_variables": variable_summaries,
        "numeric_candidates": numeric_candidates,
    }
    return array, variable_name, metadata, warnings_out, numeric_candidates


# ---------------------------------------------------------------------------
# MAT v7.3 / HDF5 reader
# ---------------------------------------------------------------------------


def _h5_read_numeric(ds: h5py.Dataset) -> np.ndarray:
    """Read an h5py dataset as float64, raising :class:`IngestionError` on failure."""
    try:
        raw = ds[()]
    except Exception as exc:
        raise IngestionError(f"Could not read HDF5 dataset: {exc}") from exc
    if not isinstance(raw, np.ndarray):
        raise IngestionError("HDF5 dataset did not return a numpy array.")
    if not _is_numeric_dtype(raw.dtype):
        raise IngestionError(
            f"HDF5 dataset dtype {raw.dtype} is not a supported real numeric type."
        )
    if np.iscomplexobj(raw):
        raise IngestionError("HDF5 dataset contains complex values.")
    return raw.astype(np.float64, copy=False)


def _read_mat_hdf5(
    content: bytes,
    variable_name: str | None,
    time_axis: int | None,
    channel_axis: int | None,
) -> tuple[np.ndarray, str | None, dict, list[str], list[str]]:
    """Load a MAT v7.3 (HDF5) file; return ``(array, varname, metadata, warnings, candidates)``."""
    warnings_out: list[str] = []
    try:
        fobj = io.BytesIO(content)
        hf = h5py.File(fobj, "r")
    except Exception as exc:
        raise IngestionError(f"Could not open MAT v7.3 HDF5 file: {exc}") from exc

    with hf:
        top_keys = [k for k in hf.keys() if not k.startswith("#")]
        variable_summaries = {k: _h5_summarise_item(k, hf[k]) for k in top_keys}

        numeric_candidates = [
            k for k, s in variable_summaries.items()
            if s.get("type") == "dataset"
            and s.get("is_numeric")
            and s.get("n_elements", 0) > _METADATA_MAX_ELEMENTS
        ]

        if variable_name is None:
            if len(numeric_candidates) == 0:
                raise IngestionError(
                    "MAT v7.3 file contains no large numeric datasets.  "
                    f"Top-level keys: {top_keys}"
                )
            if len(numeric_candidates) > 1:
                raise AmbiguousStructureError(
                    f"MAT v7.3 file contains {len(numeric_candidates)} candidate "
                    f"numeric datasets: {numeric_candidates}.  "
                    "Provide variable_name to select the EEG variable."
                )
            variable_name = numeric_candidates[0]
            warnings_out.append(
                f"Dataset '{variable_name}' was auto-selected (the only large numeric "
                f"dataset).  Candidates: {numeric_candidates}."
            )

        if variable_name not in hf:
            raise IngestionError(
                f"Variable '{variable_name}' not found in MAT v7.3 file.  "
                f"Top-level keys: {top_keys}"
            )

        item = hf[variable_name]
        if isinstance(item, h5py.Group):
            raise UnsupportedFormatError(
                f"Variable '{variable_name}' is an HDF5 Group (likely a MATLAB struct "
                "or cell array), not a dataset.  Only numeric datasets are supported."
            )
        if not isinstance(item, h5py.Dataset):
            raise UnsupportedFormatError(
                f"Variable '{variable_name}' is not an HDF5 Dataset."
            )

        h5py_shape = item.shape
        # MATLAB writes in column-major; h5py reads in row-major.
        # The axes are reversed relative to the MATLAB variable.
        # We record both shapes in metadata without silently reorienting.
        matlab_inferred_shape = tuple(reversed(h5py_shape))

        warnings_out.append(
            f"HDF5/MATLAB v7.3 axis note: variable '{variable_name}' has h5py "
            f"shape {h5py_shape}.  MATLAB column-major convention would imply "
            f"MATLAB shape {matlab_inferred_shape}.  "
            "The array is stored using the h5py shape; orientation is documented "
            "in source_metadata and must be verified before scientific use."
        )

        array = _h5_read_numeric(item)

        # Collect HDF5 attributes.
        h5_attrs = {}
        for attr_name, attr_val in item.attrs.items():
            try:
                if hasattr(attr_val, "tolist"):
                    h5_attrs[attr_name] = attr_val.tolist()
                else:
                    h5_attrs[attr_name] = str(attr_val)
            except Exception:
                h5_attrs[attr_name] = "<unreadable>"

        metadata = {
            "mat_format": "mat_v7.3_hdf5",
            "variable_name": variable_name,
            "h5py_shape": list(h5py_shape),
            "matlab_inferred_shape": list(matlab_inferred_shape),
            "variable_dtype": str(array.dtype),
            "hdf5_attributes": h5_attrs,
            "all_variables": variable_summaries,
            "numeric_candidates": numeric_candidates,
        }
    return array, variable_name, metadata, warnings_out, numeric_candidates


# ---------------------------------------------------------------------------
# Canonical array builder
# ---------------------------------------------------------------------------


def _build_canonical_array(
    array: np.ndarray,
    ndim_expected: int,
    time_axis: int | None,
    channel_axis: int | None,
    provenance: str,
    warnings_out: list[str],
    metadata: dict,
    matlab_shape: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int | str, str, int, int, int | None]:
    """Reorient *array* to canonical form and return recording dimensions.

    Returns
    -------
    array_2d, array_3d, observed_time_axis, orientation_note, n_samples,
    n_channels, n_trials
    """
    ndim = array.ndim
    if time_axis is not None or channel_axis is not None:
        if time_axis is None or channel_axis is None:
            raise IngestionError(
            "Both time_axis and channel_axis must be provided together."
        )

        if time_axis == channel_axis:
            raise IngestionError(
            "time_axis and channel_axis must be different."
        )

        if not (0 <= time_axis < ndim and 0 <= channel_axis < ndim):
            raise IngestionError(
            f"Axis indices must be between 0 and {ndim - 1}."
        )
    if ndim == 1:
        # Treat as a single-channel recording.
        warnings_out.append(
            "1D array detected; interpreted as a single-channel recording "
            "(n_samples=len, n_channels=1).  Verify this is correct."
        )
        array_2d = array.reshape(-1, 1)
        return (
            array_2d, None, 0,
            "1D array reshaped to [n_samples, 1].",
            array_2d.shape[0], 1, None,
        )

    if ndim == 2:
        r, c = array.shape
        if time_axis is not None and channel_axis is not None:
            # User-supplied explicit axes.
            if time_axis == 0:
                array_2d = array  # already [samples, channels]
                note = (
                    f"Orientation supplied by user: axis 0 = samples ({r}), "
                    f"axis 1 = channels ({c})."
                )
            else:
                array_2d = array.T
                note = (
                    f"Orientation supplied by user: axis 1 = samples ({c}), "
                    f"axis 0 = channels ({r}).  Array transposed."
                )
                warnings_out.append(note)
            return (
                array_2d, None, time_axis, note,
                array_2d.shape[0], array_2d.shape[1], None,
            )

        # Heuristic inference.
        t_ax, c_ax, note = _infer_orientation_2d((r, c))
        warnings_out.append(f"Orientation inferred heuristically: {note}")
        if t_ax == "unknown":
            # Return in observed shape; flag as unknown.
            return (
                array, None, "unknown", note,
                r, c, None,
            )
        if t_ax == 0:
            array_2d = array
        else:
            array_2d = array.T
        return (
            array_2d, None, t_ax, note,
            array_2d.shape[0], array_2d.shape[1], None,
        )

    if ndim == 3:
        d0, d1, d2 = array.shape
        if time_axis is not None and channel_axis is not None:
            # User-supplied explicit axes for 3D.
            axes = [0, 1, 2]
            remaining = [a for a in axes if a not in (time_axis, channel_axis)]
            trial_axis_inferred = remaining[0] if remaining else None
            if trial_axis_inferred is None:
                raise IngestionError(
                    "Could not determine trial axis from provided time_axis and "
                    "channel_axis for a 3D array."
                )
            # Reorder to [n_trials, n_samples, n_channels].
            perm = [trial_axis_inferred, time_axis, channel_axis]
            array_3d = np.transpose(array, perm)
            note = (
                f"User-supplied axes: time={time_axis}, channel={channel_axis}.  "
                f"3D array transposed to [trials, samples, channels]."
            )
            warnings_out.append(note)
            n_trials, n_samples, n_channels = array_3d.shape
            return (
                array_3d[0], array_3d, time_axis, note,
                n_samples, n_channels, n_trials,
            )

        # Heuristic inference.
        trial_ax, sample_ax, ch_ax, note = _infer_orientation_3d(
            (d0, d1, d2), matlab_shape=matlab_shape
        )
        warnings_out.append(f"3D orientation inferred heuristically: {note}")

        if any(ax == "unknown" for ax in (trial_ax, sample_ax, ch_ax)):
            raise AmbiguousStructureError(
        f"Cannot safely interpret 3D MAT array with shape {array.shape}. "
        "Trial, sample, and channel axes are ambiguous. "
        "Please provide explicit time_axis and channel_axis parameters."
    )

        # Reorder to [n_trials, n_samples, n_channels].
        perm = [int(trial_ax), int(sample_ax), int(ch_ax)]
        array_3d = np.transpose(array, perm)
        n_trials, n_samples, n_channels = array_3d.shape
        return (
            array_3d[0], array_3d, sample_ax, note,
            n_samples, n_channels, n_trials,
        )

    raise IngestionError(
        f"Array has {ndim} dimensions; only 1D, 2D, and 3D arrays are supported."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def read_mat(
    content: bytes,
    filename: str,
    sampling_rate_hz: float | None = None,
    variable_name: str | None = None,
    time_axis: int | None = None,
    channel_axis: int | None = None,
    recording_id: str | None = None,
) -> EEGRecording:
    """Parse a MAT file (v5 or v7.3) into a canonical :class:`~.recording.EEGRecording`.

    Parameters
    ----------
    content:
        Raw file bytes.
    filename:
        Original filename (used only for metadata).
    sampling_rate_hz:
        Sampling frequency in Hz.  Pass ``None`` if unknown.
    variable_name:
        Name of the MATLAB variable to use as the EEG array.  When ``None``,
        auto-selection is attempted (raises :class:`~.recording.AmbiguousStructureError`
        if multiple candidates exist).
    time_axis:
        Explicit time axis index (0-based).  When ``None``, orientation is
        inferred heuristically.
    channel_axis:
        Explicit channel axis index (0-based).  When ``None``, orientation is
        inferred heuristically.
    recording_id:
        UUID to assign.  A new UUID is generated when omitted.

    Returns
    -------
    EEGRecording

    Raises
    ------
    IngestionError
        On fatal errors.
    AmbiguousStructureError
        When the file contains multiple candidate variables and no
        ``variable_name`` was supplied.
    UnsupportedFormatError
        When the file is not a supported MAT variant.
    """
    rid = recording_id or str(uuid.uuid4())
    safe_name = Path(filename).name
    warnings_out: list[str] = []

    if not content:
        raise IngestionError("MAT file is empty.")

    provenance = detect_mat_format(content)

    if provenance == "mat_v5":
        raw_array, sel_var, metadata, rd_warnings, candidates = _read_mat_v5(
            content, variable_name, time_axis, channel_axis
        )
        matlab_shape = None  # MAT v5 doesn't have axis reversal
    else:
        raw_array, sel_var, metadata, rd_warnings, candidates = _read_mat_hdf5(
            content, variable_name, time_axis, channel_axis
        )
        # For HDF5, inferred MATLAB shape is the reverse of h5py shape.
        matlab_shape = tuple(reversed(raw_array.shape))

    warnings_out.extend(rd_warnings)

    if sampling_rate_hz is None:
        warnings_out.append(
            "No sampling rate was provided.  Time-axis calculations and duration "
            "reporting are unavailable."
        )
    elif not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise IngestionError(
            f"sampling_rate_hz must be a positive finite number; got {sampling_rate_hz!r}."
        )

    # Build canonical array representation.
    (
        array_2d,
        array_3d,
        observed_time_axis,
        orientation_note,
        n_samples,
        n_channels,
        n_trials,
    ) = _build_canonical_array(
        raw_array,
        ndim_expected=raw_array.ndim,
        time_axis=time_axis,
        channel_axis=channel_axis,
        provenance=provenance,
        warnings_out=warnings_out,
        metadata=metadata,
        matlab_shape=matlab_shape if provenance == "mat_v7.3_hdf5" else None,
    )

    # Validate minimum size.
    if n_samples < _MIN_SAMPLES:
        raise IngestionError(
            f"Recording has only {n_samples} sample(s); at least {_MIN_SAMPLES} are required."
        )

    # Check for non-finite values (sample the first trial for 3D arrays).
    check_array = array_2d if array_3d is None else array_3d[0]
    non_finite_count = int((~np.isfinite(check_array)).sum())
    if non_finite_count > 0:
        warnings_out.append(
            f"{non_finite_count} non-finite value(s) detected in the EEG array "
            "(first trial for 3D recordings).  Inspect the signal quality report."
        )

    # Update metadata with canonical shape.
    metadata["canonical_shape"] = (
        [n_samples, n_channels]
        if array_3d is None
        else [n_trials, n_samples, n_channels]
    )
    metadata["orientation_note"] = orientation_note

    import_status = "warnings" if warnings_out else "ok"

    LOGGER.info(
        "MAT ingestion complete: %s, format=%s, raw_shape=%s, "
        "canonical=(%d samples, %d channels, trials=%s), status=%s",
        safe_name, provenance, raw_array.shape,
        n_samples, n_channels, n_trials, import_status,
    )

    return EEGRecording(
        recording_id=rid,
        original_filename=safe_name,
        source_format="mat",
        provenance=provenance,
        n_channels=n_channels,
        n_samples=n_samples,
        n_trials=n_trials,
        sampling_rate_hz=sampling_rate_hz,
        channel_names=None,  # Phase 2: no automatic channel-name extraction
        channel_units=None,
        signal_units_known=False,
        observed_time_axis=observed_time_axis,
        orientation_note=orientation_note,
        data_shape=tuple(raw_array.shape),
        array=array_2d,
        array_3d=array_3d,
        source_metadata=metadata,
        import_status=import_status,  # type: ignore[arg-type]
        warnings=warnings_out,
        errors=[],
    )


# ---------------------------------------------------------------------------
# Inspection-only endpoint helper
# ---------------------------------------------------------------------------


def inspect_mat_structure(content: bytes) -> dict:
    """Return a summary of MAT file structure without loading EEG data.

    Useful for the ``/inspect-mat`` endpoint to let users choose a variable
    before committing a full upload.

    Returns
    -------
    dict with keys: ``format``, ``variables`` (list of summaries),
    ``numeric_candidates`` (variable names of large numeric arrays).
    """
    if not content:
        raise IngestionError("MAT file is empty.")

    provenance = detect_mat_format(content)

    if provenance == "mat_v5":
        try:
            mat = scipy.io.loadmat(io.BytesIO(content), squeeze_me=False)
        except Exception as exc:
            raise IngestionError(f"Could not read MAT v5 file: {exc}") from exc
        public_keys = [k for k in mat if not k.startswith("__")]
        variable_summaries = [_summarise_variable(k, mat[k]) for k in public_keys]
        candidates = [
            s["name"] for s in variable_summaries
            if s.get("is_numeric") and s.get("n_elements", 0) > _METADATA_MAX_ELEMENTS
        ]
        return {
            "format": provenance,
            "variables": variable_summaries,
            "numeric_candidates": candidates,
        }

    # HDF5
    try:
        fobj = io.BytesIO(content)
        hf = h5py.File(fobj, "r")
    except Exception as exc:
        raise IngestionError(f"Could not open MAT v7.3 HDF5 file: {exc}") from exc

    with hf:
        top_keys = [k for k in hf.keys() if not k.startswith("#")]
        variable_summaries = [_h5_summarise_item(k, hf[k]) for k in top_keys]
        candidates = [
            s["name"] for s in variable_summaries
            if s.get("type") == "dataset"
            and s.get("is_numeric")
            and s.get("n_elements", 0) > _METADATA_MAX_ELEMENTS
        ]
        return {
            "format": provenance,
            "variables": variable_summaries,
            "numeric_candidates": candidates,
        }
