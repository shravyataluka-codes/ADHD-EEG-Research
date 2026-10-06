"""Read-only, bounded access to the established ADHD EEG trial contract."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import numpy as np


DEFAULT_FILE_TRIAL_COUNTS = (5000, 5000, 5000, 5000, 5000, 5000, 3902)
DEFAULT_GROUP_RANGES = (
    ("HC", 0, 10129),
    ("ADD", 10129, 23160),
    ("ADHD", 23160, 33902),
)
TRIAL_SHAPE = (385, 56)
TRIAL_DTYPE = np.dtype(np.float32)


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset"


@dataclass(frozen=True)
class TrialMetadata:
    """Index and validated subject/group metadata for one trial."""

    global_trial_index: int
    source_file: str
    within_file_trial_index: int
    subject_id: str
    clinical_group: str
    y_stim_column: int


class TrialManifest:
    """Create trial metadata from the established Sprint 1 crosswalk.

    The manifest reads only row 0 of ``y_stim`` (a small metadata vector); it
    never reads EEG samples. Indices are zero-based, and the default file/group
    ranges are those recorded in the Sprint 1 mapping report.
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        *,
        file_trial_counts: Sequence[int] = DEFAULT_FILE_TRIAL_COUNTS,
        group_ranges: Sequence[tuple[str, int, int]] = DEFAULT_GROUP_RANGES,
    ) -> None:
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else _default_dataset_dir()
        self.file_trial_counts = tuple(int(count) for count in file_trial_counts)
        if not self.file_trial_counts or any(count <= 0 for count in self.file_trial_counts):
            raise ValueError("file_trial_counts must contain positive trial counts")

        self._file_offsets = [0]
        for count in self.file_trial_counts:
            self._file_offsets.append(self._file_offsets[-1] + count)
        self.total_trials = self._file_offsets[-1]

        self.group_ranges = tuple((str(group), int(start), int(end)) for group, start, end in group_ranges)
        self._validate_group_ranges()

        metadata_path = self.dataset_dir / "y_stim.mat"
        with h5py.File(metadata_path, "r") as metadata_file:
            if "y_stim" not in metadata_file:
                raise ValueError(f"Missing 'y_stim' dataset in {metadata_path}")
            y_stim = metadata_file["y_stim"]
            if y_stim.ndim != 2 or y_stim.shape[0] != 4:
                raise ValueError(f"Expected y_stim shape (4, trials), got {y_stim.shape}")
            if y_stim.shape[1] != self.total_trials:
                raise ValueError(
                    f"y_stim has {y_stim.shape[1]} columns; expected {self.total_trials}"
                )
            subject_values = np.asarray(y_stim[0, :])

        if not np.isfinite(subject_values).all() or not np.equal(
            subject_values, np.floor(subject_values)
        ).all():
            raise ValueError("y_stim row 0 must contain finite integer subject indices")
        self._subject_indices = subject_values.astype(np.int64, copy=False)

    def _validate_group_ranges(self) -> None:
        if not self.group_ranges:
            raise ValueError("group_ranges must not be empty")
        expected_start = 0
        for group, start, end in self.group_ranges:
            if not group or start != expected_start or end <= start:
                raise ValueError("group_ranges must be named, contiguous, and non-empty")
            expected_start = end
        if expected_start != self.total_trials:
            raise ValueError("group_ranges must cover all trials exactly")

    def metadata_for(self, global_trial_index: int) -> TrialMetadata:
        """Return metadata for one zero-based global trial index."""
        if isinstance(global_trial_index, bool) or not isinstance(global_trial_index, (int, np.integer)):
            raise TypeError("global_trial_index must be an integer")
        index = int(global_trial_index)
        if index < 0 or index >= self.total_trials:
            raise IndexError(f"global trial index {index} outside [0, {self.total_trials})")

        file_number = next(
            offset_index
            for offset_index in range(len(self.file_trial_counts))
            if self._file_offsets[offset_index] <= index < self._file_offsets[offset_index + 1]
        )
        group, _, _ = next(
            group_range for group_range in self.group_ranges
            if group_range[1] <= index < group_range[2]
        )
        subject_index = int(self._subject_indices[index])
        if subject_index <= 0:
            raise ValueError(f"Invalid subject index {subject_index} at trial {index}")

        return TrialMetadata(
            global_trial_index=index,
            source_file=f"d{file_number + 1}.mat",
            within_file_trial_index=index - self._file_offsets[file_number],
            subject_id=f"{group}_{subject_index:02d}",
            clinical_group=group,
            y_stim_column=index,
        )

    def iter_metadata(self, indices: Iterable[int] | None = None) -> Iterator[TrialMetadata]:
        """Yield metadata in global order, or in the order of supplied indices."""
        selected = range(self.total_trials) if indices is None else indices
        for index in selected:
            yield self.metadata_for(index)

    def iter_subject(self, subject_id: str) -> Iterator[TrialMetadata]:
        """Yield all manifest rows for a composite Sprint 1 subject ID."""
        return (row for row in self.iter_metadata() if row.subject_id == subject_id)

    def iter_group(self, clinical_group: str) -> Iterator[TrialMetadata]:
        """Yield all manifest rows for one clinical group."""
        if clinical_group not in {group for group, _, _ in self.group_ranges}:
            raise KeyError(f"Unknown clinical group: {clinical_group}")
        return (
            row for row in self.iter_metadata()
            if row.clinical_group == clinical_group
        )

    def write_csv(self, output_path: str | Path) -> None:
        """Write a compact metadata-only manifest; no EEG values are included."""
        fields = tuple(TrialMetadata.__dataclass_fields__)
        with Path(output_path).open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            for row in self.iter_metadata():
                writer.writerow(asdict(row))


class EEGTrialStore:
    """Read individual trials or bounded batches from immutable HDF5 sources.

    Use as a context manager to reuse read-only HDF5 handles during iteration.
    The returned NumPy arrays contain only the requested trial/batch; the full
    EEG arrays are never loaded into memory.
    """

    def __init__(self, manifest: TrialManifest) -> None:
        self.manifest = manifest
        self._cache_enabled = False
        self._open_files: dict[str, h5py.File] = {}
        self._validated_files: set[str] = set()

    def __enter__(self) -> EEGTrialStore:
        self._cache_enabled = True
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close any cached read-only HDF5 handles."""
        for source_file in self._open_files.values():
            source_file.close()
        self._open_files.clear()
        self._validated_files.clear()
        self._cache_enabled = False

    def _validate_dataset(self, source_file: str, handle: h5py.File) -> h5py.Dataset:
        dataset_name = Path(source_file).stem
        if dataset_name not in handle:
            raise ValueError(f"Missing dataset '{dataset_name}' in {source_file}")
        dataset = handle[dataset_name]
        file_number = int(dataset_name[1:]) - 1
        expected_shape = (*TRIAL_SHAPE, self.manifest.file_trial_counts[file_number])
        if dataset.shape != expected_shape:
            raise ValueError(f"Unexpected shape in {source_file}: {dataset.shape}; expected {expected_shape}")
        if dataset.dtype != TRIAL_DTYPE:
            raise ValueError(f"Unexpected dtype in {source_file}: {dataset.dtype}; expected float32")
        self._validated_files.add(source_file)
        return dataset

    def _read_trial(self, metadata: TrialMetadata) -> np.ndarray:
        source_path = self.manifest.dataset_dir / metadata.source_file
        if self._cache_enabled:
            handle = self._open_files.get(metadata.source_file)
            if handle is None:
                handle = h5py.File(source_path, "r")
                self._open_files[metadata.source_file] = handle
            dataset = self._validate_dataset(metadata.source_file, handle)
            trial = np.asarray(dataset[:, :, metadata.within_file_trial_index])
        else:
            with h5py.File(source_path, "r") as handle:
                dataset = self._validate_dataset(metadata.source_file, handle)
                trial = np.asarray(dataset[:, :, metadata.within_file_trial_index])

        if trial.shape != TRIAL_SHAPE or trial.dtype != TRIAL_DTYPE:
            raise ValueError(
                f"Unexpected trial representation {trial.shape}/{trial.dtype}; "
                f"expected {TRIAL_SHAPE}/{TRIAL_DTYPE}"
            )
        return trial

    def get_trial(self, global_trial_index: int) -> tuple[np.ndarray, TrialMetadata]:
        """Read one trial and return its array with its metadata."""
        metadata = self.manifest.metadata_for(global_trial_index)
        return self._read_trial(metadata), metadata

    def get_batch(
        self, global_trial_indices: Sequence[int]
    ) -> tuple[np.ndarray, tuple[TrialMetadata, ...]]:
        """Read exactly the requested batch in the same order as its indices."""
        metadata = tuple(self.manifest.metadata_for(index) for index in global_trial_indices)
        if not metadata:
            raise ValueError("A batch must contain at least one trial index")
        trials = [self._read_trial(row) for row in metadata]
        return np.stack(trials, axis=0), metadata

    def iter_batches(
        self,
        batch_size: int,
        indices: Iterable[int] | None = None,
    ) -> Iterator[tuple[np.ndarray, tuple[TrialMetadata, ...]]]:
        """Yield batches bounded by ``batch_size``; defaults to all trials in order."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        selected = range(self.manifest.total_trials) if indices is None else iter(indices)
        while batch_indices := tuple(islice(selected, batch_size)):
            yield self.get_batch(batch_indices)