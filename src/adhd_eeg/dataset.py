"""PyTorch access to EEG trials through the bounded HDF5 data layer."""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data_access import EEGTrialStore, TrialManifest


TrialSourceMetadata: TypeAlias = dict[str, str | int]
EEGTrialSample: TypeAlias = tuple[Tensor, TrialSourceMetadata]


class EEGTrialDataset(Dataset[EEGTrialSample]):
    """Return one EEG tensor and source-local identifiers, without labels.

    The store is deliberately not entered as a context manager: its uncached
    reads open each HDF5 file read-only for one access and close it immediately.
    No live HDF5 handles are retained on this Dataset or inherited by workers.
    """

    def __init__(self, manifest: TrialManifest) -> None:
        self._manifest = manifest
        self._store = EEGTrialStore(manifest)

    def __len__(self) -> int:
        return self._manifest.total_trials

    def __getitem__(self, index: int) -> EEGTrialSample:
        signal, row = self._store.get_trial(index)
        tensor = torch.as_tensor(signal, dtype=torch.float32)
        source_filename = row.source_file
        within_file_trial_index = row.within_file_trial_index
        metadata: TrialSourceMetadata = {
            "trial_id": f"{source_filename}:{within_file_trial_index}",
            "source_filename": source_filename,
            "within_file_trial_index": within_file_trial_index,
        }
        return tensor, metadata
