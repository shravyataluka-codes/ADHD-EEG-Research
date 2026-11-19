"""Canonical EEG recording representation for Drushti Detect Phase 2.

This module defines the common internal representation that all file format
readers (CSV, MAT v5, MAT v7.3/HDF5) must produce. The Phase 3 model adapter
consumes this representation without coupling ingestion to any specific model.

Design rules
------------
* ``array`` is ALWAYS ``[n_samples, n_channels]`` for 2D recordings.
* For 3D (epoched) recordings, ``array_3d`` holds the full ``[n_trials,
  n_samples, n_channels]`` data.  A selected 2D trial view is obtained via
  :func:`get_trial_view` without modifying the stored source.
* When orientation cannot be established, ``observed_time_axis`` is set to
  ``"unknown"`` and no transposition is performed.
* All source metadata is preserved in ``source_metadata``.
* Warnings accumulate; they do not prevent ingestion unless ``import_status``
  is ``"rejected"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IngestionError(ValueError):
    """A file could not be ingested due to an unrecoverable problem."""


class AmbiguousStructureError(IngestionError):
    """The file structure is ambiguous and requires explicit configuration."""


class UnsupportedFormatError(IngestionError):
    """The file format or structure is not supported."""


# ---------------------------------------------------------------------------
# Canonical recording
# ---------------------------------------------------------------------------


@dataclass
class EEGRecording:
    """Canonical internal EEG recording representation.

    Both CSV and MAT readers produce this dataclass.  The Phase 3 model
    adapter inspects it without knowledge of the source file format.

    Attributes
    ----------
    recording_id:
        UUID assigned at ingestion time.
    original_filename:
        Sanitized original filename (basename only, no path components).
    source_format:
        ``"csv"`` or ``"mat"``.
    provenance:
        More specific format string, e.g. ``"mat_v5"`` or
        ``"mat_v7.3_hdf5"``.
    n_channels:
        Number of channels in the canonical 2D array (or in each trial of a
        3D recording).
    n_samples:
        Number of time samples per trial (or total samples for continuous
        recordings).
    n_trials:
        ``None`` for continuous 2D recordings.  Positive integer for epoched
        3D recordings.
    sampling_rate_hz:
        Sampling frequency in Hz.  ``None`` when not provided or unknown.
        When ``None``, time-axis calculations are impossible; callers must
        handle this case explicitly.
    channel_names:
        Ordered list of channel names, length ``n_channels``.  ``None`` when
        no names were available.  Individual entries may be ``"unknown"`` when
        partially specified.
    channel_units:
        Ordered list of unit strings, length ``n_channels``.  ``None`` when
        units are absent from the source file.  Do not assume microvolts.
    signal_units_known:
        ``True`` only when units were explicitly present in source metadata.
    observed_time_axis:
        The axis index (0 or 1 for 2D) of the time/samples dimension as
        observed in the source file, before any reorientation.
        ``"unknown"`` when orientation could not be established confidently.
    orientation_note:
        Human-readable explanation of how orientation was determined or why
        it is unknown.
    data_shape:
        Shape of the raw source array before any reorientation.  For 3D
        arrays this is the original ``(d0, d1, d2)`` tuple.
    array:
        Canonical 2D array ``[n_samples, n_channels]`` as ``float64``.
        For 3D recordings this is the view of the first (or selected) trial.
        Never modify this array in place; it may be a view of ``array_3d``.
    array_3d:
        Full ``[n_trials, n_samples, n_channels]`` array for epoched
        recordings.  ``None`` for 2D/continuous recordings.  Stored as
        ``float64``.
    source_metadata:
        All non-EEG metadata extracted from the file (variable names, MATLAB
        attributes, HDF5 attributes, etc.).  Values are JSON-serialisable
        primitives when possible.
    import_status:
        ``"ok"``: ingestion succeeded with no issues.
        ``"warnings"``: ingestion succeeded but assumptions were made.
        ``"rejected"``: ingestion failed; ``errors`` lists the reasons.
    warnings:
        Non-fatal issues discovered during ingestion.
    errors:
        Fatal issues that caused ``import_status == "rejected"``.
    """

    recording_id: str
    original_filename: str
    source_format: Literal["csv", "mat"]
    provenance: str  # e.g. "csv", "mat_v5", "mat_v7.3_hdf5"

    n_channels: int
    n_samples: int
    n_trials: int | None  # None = continuous 2D recording

    sampling_rate_hz: float | None  # None = unknown / not provided
    channel_names: list[str] | None
    channel_units: list[str] | None
    signal_units_known: bool

    observed_time_axis: int | Literal["unknown"]
    orientation_note: str
    data_shape: tuple[int, ...]

    array: np.ndarray          # [n_samples, n_channels], float64
    array_3d: np.ndarray | None  # [n_trials, n_samples, n_channels] or None

    source_metadata: dict

    import_status: Literal["ok", "warnings", "rejected"]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def duration_seconds(self) -> float | None:
        """Recording duration in seconds, or ``None`` if rate is unknown."""
        if self.sampling_rate_hz is None or self.sampling_rate_hz <= 0:
            return None
        return self.n_samples / self.sampling_rate_hz

    @property
    def memory_bytes(self) -> int:
        """Memory occupied by the retained signal array."""
        if self.array_3d is not None:
            return int(self.array_3d.nbytes)

        return int(self.array.nbytes)

    # ------------------------------------------------------------------
    # Trial access (3D recordings)
    # ------------------------------------------------------------------

    def get_trial_view(self, trial_index: int = 0) -> np.ndarray:
        """Return a 2D ``[n_samples, n_channels]`` view for a specific trial.

        Parameters
        ----------
        trial_index:
            Zero-based trial index.  For 2D recordings the only valid value
            is ``0``, which returns ``self.array``.

        Returns
        -------
        np.ndarray
            A ``[n_samples, n_channels]`` array.  For 3D recordings this is a
            view into ``array_3d``; it must not be modified in place.

        Raises
        ------
        IndexError
            When ``trial_index`` is out of range.
        """
        if self.array_3d is None:
            if trial_index != 0:
                raise IndexError(
                    f"This is a continuous (2D) recording; trial_index must be 0, "
                    f"got {trial_index}."
                )
            return self.array

        n = self.n_trials if self.n_trials is not None else 1
        if trial_index < 0 or trial_index >= n:
            raise IndexError(
                f"trial_index {trial_index} is out of range for a recording with "
                f"{n} trials."
            )
        return self.array_3d[trial_index]  # view, not copy

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        trials = f", n_trials={self.n_trials}" if self.n_trials is not None else ""
        rate = (
            f"{self.sampling_rate_hz} Hz"
            if self.sampling_rate_hz is not None
            else "unknown Hz"
        )
        return (
            f"EEGRecording(id={self.recording_id!r}, "
            f"fmt={self.provenance!r}, "
            f"shape=({self.n_samples}, {self.n_channels}){trials}, "
            f"rate={rate}, "
            f"status={self.import_status!r})"
        )
