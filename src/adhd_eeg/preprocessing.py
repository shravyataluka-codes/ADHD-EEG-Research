"""Reproducible, bounded-memory EEG preprocessing.

This module does not infer ICA artifacts. Processing requires explicit filter
settings and reviewed per-subject component decisions supplied by the caller.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import mne
import numpy as np
from mne.preprocessing import ICA

from .data_access import (
    DEFAULT_FILE_TRIAL_COUNTS,
    EEGTrialStore,
    TRIAL_DTYPE,
    TRIAL_SHAPE,
    TrialManifest,
    TrialMetadata,
)


BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 45.0
APPROVED_NOTCH_HZ = 50.0
PAPER_REPORTED_SAMPLING_HZ = 256.0


@dataclass(frozen=True)
class FilterSettings:
    """Explicit MNE filter-design parameters; no signal-filter defaults are used."""

    method: str
    phase: str
    fir_design: str
    filter_length: str | int
    l_trans_bandwidth: str | float
    h_trans_bandwidth: str | float
    fir_window: str
    pad: str
    notch_width_hz: float
    notch_transition_bandwidth_hz: float

    def validate(self) -> None:
        if not self.method or not self.phase or not self.fir_design or not self.pad:
            raise ValueError("Filter method, phase, FIR design, and padding must be explicit")
        if self.method != "fir":
            raise ValueError("Only the explicitly configured FIR filter path is implemented")
        if not self.fir_window:
            raise ValueError("FIR window must be explicit")
        if not _positive_setting(self.filter_length):
            raise ValueError("filter_length must be a positive value or a named MNE setting")
        if not _positive_setting(self.l_trans_bandwidth):
            raise ValueError("l_trans_bandwidth must be positive or a named MNE setting")
        if not _positive_setting(self.h_trans_bandwidth):
            raise ValueError("h_trans_bandwidth must be positive or a named MNE setting")
        if not np.isfinite(self.notch_width_hz) or self.notch_width_hz <= 0:
            raise ValueError("notch_width_hz must be positive and finite")
        if (
            not np.isfinite(self.notch_transition_bandwidth_hz)
            or self.notch_transition_bandwidth_hz <= 0
        ):
            raise ValueError("notch_transition_bandwidth_hz must be positive and finite")


@dataclass(frozen=True)
class ICASettings:
    """ICA fitting settings that materially affect the decomposition."""

    method: str
    n_components: int
    max_iter: int | str

    def validate(self) -> None:
        if not self.method:
            raise ValueError("ICA method must be explicit")
        if (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, int)
            or self.n_components <= 0
        ):
            raise ValueError("ICA n_components must be a positive integer")
        if isinstance(self.max_iter, bool) or (
            not isinstance(self.max_iter, int) and self.max_iter != "auto"
        ):
            raise ValueError("ICA max_iter must be a positive integer or 'auto'")
        if isinstance(self.max_iter, int) and self.max_iter <= 0:
            raise ValueError("ICA max_iter must be positive")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Complete, explicit configuration for one reproducible processing run."""

    sampling_frequency_hz: float
    notch_frequency_hz: float
    random_seed: int
    filters: FilterSettings
    ica: ICASettings

    def validate(self, channel_count: int = 56) -> None:
        if not np.isfinite(self.sampling_frequency_hz) or self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive and finite")
        if self.sampling_frequency_hz != PAPER_REPORTED_SAMPLING_HZ:
            raise ValueError(
                f"Project sampling frequency is paper-reported as "
                f"{PAPER_REPORTED_SAMPLING_HZ:g} Hz"
            )
        if self.notch_frequency_hz != APPROVED_NOTCH_HZ:
            raise ValueError(
                f"Project-approved notch frequency is {APPROVED_NOTCH_HZ:g} Hz"
            )
        nyquist_hz = self.sampling_frequency_hz / 2
        if BANDPASS_HIGH_HZ >= nyquist_hz:
            raise ValueError("Band-pass high cutoff must be below the Nyquist frequency")
        if not 0 < self.notch_frequency_hz < nyquist_hz:
            raise ValueError("Notch frequency must be below the Nyquist frequency")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise ValueError("random_seed must be in [0, 2**32 - 1]")
        self.filters.validate()
        if (
            self.notch_frequency_hz
            - self.filters.notch_width_hz / 2
            - self.filters.notch_transition_bandwidth_hz
            <= BANDPASS_HIGH_HZ
        ):
            raise ValueError("Notch transition band overlaps the 45 Hz band-pass cutoff")
        self.ica.validate()
        if self.ica.n_components > channel_count:
            raise ValueError(
                f"ICA n_components ({self.ica.n_components}) exceeds "
                f"channel count ({channel_count})"
            )


@dataclass(frozen=True)
class SubjectICAReview:
    """Externally reviewed component exclusions for a single subject."""

    exclude_components: tuple[int, ...]
    reviewer: str
    reviewed_at: str
    rationale: str

    def validate(self) -> None:
        values = (self.reviewer, self.reviewed_at, self.rationale)
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("ICA review requires reviewer, reviewed_at, and rationale")
        try:
            reviewed_at = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("reviewed_at must be an ISO 8601 timestamp") from error
        if reviewed_at.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in self.exclude_components):
            raise ValueError("ICA component indices must be integers")
        if len(set(self.exclude_components)) != len(self.exclude_components):
            raise ValueError("ICA component indices must not contain duplicates")


def _positive_setting(value: str | int | float) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return (
        not isinstance(value, bool)
        and np.isfinite(value)
        and value > 0
    )


def config_from_mapping(data: Mapping[str, Any]) -> PreprocessingConfig:
    """Build a config from JSON-compatible data without filling missing values."""
    try:
        filter_data = data["filters"]
        ica_data = data["ica"]
        config = PreprocessingConfig(
            sampling_frequency_hz=float(data["sampling_frequency_hz"]),
            notch_frequency_hz=float(data["notch_frequency_hz"]),
            random_seed=data["random_seed"],
            filters=FilterSettings(
                method=filter_data["method"],
                phase=filter_data["phase"],
                fir_design=filter_data["fir_design"],
                filter_length=filter_data["filter_length"],
                l_trans_bandwidth=filter_data["l_trans_bandwidth"],
                h_trans_bandwidth=filter_data["h_trans_bandwidth"],
                fir_window=filter_data["fir_window"],
                pad=filter_data["pad"],
                notch_width_hz=float(filter_data["notch_width_hz"]),
                notch_transition_bandwidth_hz=float(
                    filter_data["notch_transition_bandwidth_hz"]
                ),
            ),
            ica=ICASettings(
                method=ica_data["method"],
                n_components=ica_data["n_components"],
                max_iter=ica_data["max_iter"],
            ),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or malformed preprocessing setting: {error}") from error
    config.validate()
    return config


def reviews_from_mapping(data: Mapping[str, Any]) -> dict[str, SubjectICAReview]:
    """Parse explicit, reviewed ICA component decisions keyed by subject ID."""
    protocol = data.get("review_protocol")
    subjects = data.get("subjects")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("ICA review file must identify a non-empty review_protocol")
    if not isinstance(subjects, Mapping):
        raise ValueError("ICA review file must contain a subjects mapping")

    reviews: dict[str, SubjectICAReview] = {}
    for subject_id, entry in subjects.items():
        if not isinstance(subject_id, str) or not isinstance(entry, Mapping):
            raise ValueError("Each ICA review entry must be a subject-keyed object")
        if entry.get("reviewed") is not True:
            raise ValueError(f"ICA decision for {subject_id} is not marked reviewed")
        exclusions = entry.get("exclude_components")
        if not isinstance(exclusions, list):
            raise ValueError(f"ICA decision for {subject_id} needs exclude_components list")
        reviewer = entry.get("reviewer")
        reviewed_at = entry.get("reviewed_at")
        rationale = entry.get("rationale")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (reviewer, reviewed_at, rationale)
        ):
            raise ValueError(
                f"ICA decision for {subject_id} requires reviewer, reviewed_at, and rationale"
            )
        review = SubjectICAReview(
            exclude_components=tuple(exclusions),
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            rationale=f"{protocol}: {rationale}",
        )
        review.validate()
        reviews[subject_id] = review
    return reviews


def preprocess_trials(
    trials: np.ndarray,
    config: PreprocessingConfig,
    review: SubjectICAReview,
) -> tuple[np.ndarray, int]:
    """Filter and ICA-clean one subject's epochs, preserving ``(trials,time,channels)``."""
    data = np.asarray(trials)
    if data.ndim != 3 or data.shape[0] == 0:
        raise ValueError("trials must have shape (non-empty trials, time, channels)")
    if data.shape[1] != TRIAL_SHAPE[0]:
        raise ValueError(f"Expected {TRIAL_SHAPE[0]} time samples per trial, got {data.shape[1]}")
    if data.dtype != TRIAL_DTYPE:
        raise ValueError(f"Expected float32 trials, got {data.dtype}")
    if not np.isfinite(data).all():
        raise ValueError("Input EEG contains NaN or Inf values")
    config.validate(channel_count=data.shape[2])
    review.validate()

    channel_names = [f"EEG{index + 1:03d}" for index in range(data.shape[2])]
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=config.sampling_frequency_hz,
        ch_types=["eeg"] * data.shape[2],
    )
    epochs = mne.EpochsArray(
        data.transpose(0, 2, 1).astype(np.float64, copy=False),
        info,
        verbose="ERROR",
    )
    filter_settings = config.filters
    epochs.filter(
        l_freq=BANDPASS_LOW_HZ,
        h_freq=BANDPASS_HIGH_HZ,
        filter_length=filter_settings.filter_length,
        l_trans_bandwidth=filter_settings.l_trans_bandwidth,
        h_trans_bandwidth=filter_settings.h_trans_bandwidth,
        method=filter_settings.method,
        n_jobs=1,
        phase=filter_settings.phase,
        fir_window=filter_settings.fir_window,
        fir_design=filter_settings.fir_design,
        pad=filter_settings.pad,
        verbose="ERROR",
    )
    notched_data = mne.filter.notch_filter(
        epochs.get_data(copy=True),
        Fs=config.sampling_frequency_hz,
        freqs=[config.notch_frequency_hz],
        filter_length=filter_settings.filter_length,
        notch_widths=filter_settings.notch_width_hz,
        trans_bandwidth=filter_settings.notch_transition_bandwidth_hz,
        method=filter_settings.method,
        phase=filter_settings.phase,
        fir_window=filter_settings.fir_window,
        fir_design=filter_settings.fir_design,
        pad=filter_settings.pad,
        n_jobs=1,
        verbose="ERROR",
    )
    epochs = mne.EpochsArray(notched_data, info, verbose="ERROR")

    ica = ICA(
        n_components=config.ica.n_components,
        method=config.ica.method,
        max_iter=config.ica.max_iter,
        random_state=config.random_seed,
        fit_params={},
        verbose="ERROR",
    )
    ica.fit(
        epochs,
        decim=1,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )
    component_count = int(ica.n_components_)
    if any(index < 0 or index >= component_count for index in review.exclude_components):
        raise ValueError(
            f"ICA exclusions {review.exclude_components} are outside "
            f"[0, {component_count})"
        )
    ica.exclude = list(review.exclude_components)
    cleaned = ica.apply(epochs.copy(), verbose="ERROR").get_data(copy=True)
    result = cleaned.transpose(0, 2, 1).astype(np.float32, copy=False)
    if result.shape != data.shape:
        raise RuntimeError(f"Preprocessing changed shape from {data.shape} to {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Preprocessed EEG contains NaN or Inf values")
    return result, component_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics(trial: np.ndarray) -> dict[str, float]:
    values = trial.astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
    }


def _subject_trials(
    manifest: TrialManifest,
) -> dict[str, list[TrialMetadata]]:
    rows_by_subject: dict[str, list[TrialMetadata]] = {}
    for row in manifest.iter_metadata():
        rows_by_subject.setdefault(row.subject_id, []).append(row)
    return rows_by_subject


def _verify_reviews(
    reviews: Mapping[str, SubjectICAReview],
    subject_ids: set[str],
) -> None:
    if set(reviews) != subject_ids:
        missing = sorted(subject_ids - set(reviews))
        extra = sorted(set(reviews) - subject_ids)
        raise ValueError(f"ICA review subject mismatch; missing={missing}, extra={extra}")
    for review in reviews.values():
        review.validate()


def process_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    config: PreprocessingConfig,
    reviews: Mapping[str, SubjectICAReview],
    *,
    manifest: TrialManifest | None = None,
) -> dict[str, Any]:
    """Process subjects one at a time into separate derived HDF5 files.

    The output directory must not already exist. Source arrays are opened
    read-only, while output arrays retain each source's ``(time,channels,trials)``
    shape and float32 dtype.
    """
    source_dir = Path(dataset_dir)
    destination = Path(output_dir)
    trial_manifest = manifest or TrialManifest(source_dir)
    if trial_manifest.dataset_dir.resolve() != source_dir.resolve():
        raise ValueError("Manifest dataset_dir does not match dataset_dir")
    if destination.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("Derived output directory must be outside the raw dataset directory")
    if len(trial_manifest.file_trial_counts) != len(DEFAULT_FILE_TRIAL_COUNTS):
        raise ValueError("Expected exactly seven EEG source files")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite derived output directory: {destination}")
    config.validate()

    grouped_trials = _subject_trials(trial_manifest)
    _verify_reviews(reviews, set(grouped_trials))
    eeg_source_paths = [
        source_dir / f"d{file_number}.mat"
        for file_number in range(1, len(trial_manifest.file_trial_counts) + 1)
    ]
    if not all(path.is_file() for path in eeg_source_paths):
        missing = [path.name for path in eeg_source_paths if not path.is_file()]
        raise FileNotFoundError(f"Missing EEG source files: {missing}")
    raw_paths = sorted(source_dir.glob("*.mat"))
    source_hashes = {path.name: _sha256_file(path) for path in raw_paths}
    destination.parent.mkdir(parents=True, exist_ok=True)

    first_trial: tuple[np.ndarray, np.ndarray, TrialMetadata] | None = None
    subject_results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        output_files: dict[str, h5py.File] = {}
        output_datasets: dict[str, h5py.Dataset] = {}
        try:
            for file_number, trial_count in enumerate(trial_manifest.file_trial_counts, start=1):
                source_name = f"d{file_number}.mat"
                with h5py.File(source_dir / source_name, "r") as source_file:
                    dataset_name = Path(source_name).stem
                    if dataset_name not in source_file:
                        raise ValueError(f"Missing dataset {dataset_name} in {source_name}")
                    source_dataset = source_file[dataset_name]
                    expected_shape = (*TRIAL_SHAPE, trial_count)
                    if source_dataset.shape != expected_shape:
                        raise ValueError(
                            f"Unexpected shape in {source_name}: {source_dataset.shape}; "
                            f"expected {expected_shape}"
                        )
                    if source_dataset.dtype != TRIAL_DTYPE:
                        raise ValueError(
                            f"Unexpected dtype in {source_name}: {source_dataset.dtype}; "
                            f"expected {TRIAL_DTYPE}"
                        )
                    output_file = h5py.File(temporary_path / f"d{file_number}.h5", "x")
                    output_files[source_name] = output_file
                    output_datasets[source_name] = output_file.create_dataset(
                        dataset_name,
                        shape=expected_shape,
                        dtype=np.float32,
                        chunks=(min(192, expected_shape[0]), expected_shape[1], 1),
                    )

            with EEGTrialStore(trial_manifest) as store:
                for subject_id, rows in grouped_trials.items():
                    indices = [row.global_trial_index for row in rows]
                    raw_batch, returned_rows = store.get_batch(indices)
                    if returned_rows != tuple(rows):
                        raise RuntimeError(f"Trial metadata order changed for {subject_id}")
                    review = reviews[subject_id]
                    cleaned_batch, component_count = preprocess_trials(
                        raw_batch,
                        config,
                        review,
                    )
                    for batch_index, row in enumerate(rows):
                        output_datasets[row.source_file][:, :, row.within_file_trial_index] = (
                            cleaned_batch[batch_index]
                        )
                    subject_results[subject_id] = {
                        "trial_count": len(rows),
                        "ica_component_count": component_count,
                        "removed_component_indices": list(review.exclude_components),
                        "ica_review": asdict(review),
                    }
                    if first_trial is None:
                        first_trial = (raw_batch[0].copy(), cleaned_batch[0].copy(), rows[0])
        finally:
            for output_file in output_files.values():
                output_file.close()

        output_shapes: dict[str, list[int]] = {}
        for file_number, trial_count in enumerate(trial_manifest.file_trial_counts, start=1):
            output_path = temporary_path / f"d{file_number}.h5"
            with h5py.File(output_path, "r") as output_file:
                output_dataset = output_file[f"d{file_number}"]
                expected_shape = (*TRIAL_SHAPE, trial_count)
                if output_dataset.shape != expected_shape or output_dataset.dtype != TRIAL_DTYPE:
                    raise RuntimeError(f"Output validation failed for {output_path.name}")
                for chunk_slice in output_dataset.iter_chunks():
                    if not np.isfinite(output_dataset[chunk_slice]).all():
                        raise ValueError(f"Output contains NaN or Inf in {output_path.name}")
                output_shapes[output_path.name] = list(output_dataset.shape)

        trial_manifest.write_csv(temporary_path / "trial_manifest.csv")
        if first_trial is None:
            raise RuntimeError("No trials were processed")
        raw_trial, processed_trial, representative_metadata = first_trial
        np.savez_compressed(
            temporary_path / "inspection_trial.npz",
            raw=raw_trial,
            preprocessed=processed_trial,
        )
        unchanged = {path.name: source_hashes[path.name] == _sha256_file(path) for path in raw_paths}
        if not all(unchanged.values()):
            raise RuntimeError("Raw EEG file hash changed during preprocessing")

        metadata: dict[str, Any] = {
            "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": [
                {
                    "file": path.name,
                    "sha256": source_hashes[path.name],
                    "source_shape": [
                        *TRIAL_SHAPE,
                        trial_manifest.file_trial_counts[index],
                    ],
                    "output_file": f"{path.stem}.h5",
                    "output_shape": output_shapes[f"{path.stem}.h5"],
                }
                for index, path in enumerate(eeg_source_paths)
            ],
            "trial_count": trial_manifest.total_trials,
            "subject_count": len(grouped_trials),
            "sampling_frequency_hz": config.sampling_frequency_hz,
            "sampling_frequency_evidence": "He et al. paper-reported; not stored in MAT headers",
            "bandpass": {
                "low_hz": BANDPASS_LOW_HZ,
                "high_hz": BANDPASS_HIGH_HZ,
                **asdict(config.filters),
            },
            "notch": {
                "frequency_hz": config.notch_frequency_hz,
                "approved_frequency_hz": APPROVED_NOTCH_HZ,
                "width_hz": config.filters.notch_width_hz,
                "transition_bandwidth_hz": config.filters.notch_transition_bandwidth_hz,
                "n_jobs": 1,
            },
            "ica": {
                **asdict(config.ica),
                "random_seed": config.random_seed,
                "fit_scope": "per subject, across that subject's epochs",
                "fit_options": {
                    "decim": 1,
                    "reject": None,
                    "flat": None,
                    "reject_by_annotation": False,
                    "fit_params": {},
                },
                "component_selection": (
                    "externally reviewed per-subject indices; no automatic component removal"
                ),
                "subjects": subject_results,
            },
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "h5py": h5py.__version__,
                "mne": importlib.metadata.version("mne"),
            },
            "validation": {
                "shape_preserved": True,
                "dtype": "float32",
                "all_outputs_finite": True,
                "raw_files_unchanged": unchanged,
                "trial_manifest_written": True,
            },
            "limitations": [
                "The 56 EEG channel identities and EOG/ECG mapping are unresolved.",
                "ICA exclusions are decisions supplied by an external reviewer; "
                "this software does not validate their scientific correctness.",
                "The local files' upstream ICA state is unknown.",
                "Filtering is applied within already-epoched trials; continuous "
                "pre-epoch recordings are unavailable for edge-aware filtering.",
                "ICA is fitted independently per subject before any evaluation split; "
                "this component does not enforce fold-local fitting.",
                "The physical interpretation of the 385th sample remains unresolved.",
            ],
        }
        (temporary_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        inspection = {
            "representative_trial": asdict(representative_metadata),
            "raw_shape": list(raw_trial.shape),
            "preprocessed_shape": list(processed_trial.shape),
            "raw_statistics": _statistics(raw_trial),
            "preprocessed_statistics": _statistics(processed_trial),
            "shape_preserved": raw_trial.shape == processed_trial.shape,
            "trial_count": trial_manifest.total_trials,
            "subject_count": len(grouped_trials),
            "raw_files_unchanged": unchanged,
            "signal_arrays_file": "inspection_trial.npz",
        }
        (temporary_path / "inspection_report.json").write_text(
            json.dumps(inspection, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite derived output directory: {destination}")
        os.replace(temporary_path, destination)
    return metadata
