"""Data module package initialization."""
from .dummy_generator import (
    generate_synthetic_eeg_trial,
    generate_synthetic_eeg_dataframe,
    STANDARD_TIMEPOINTS,
    STANDARD_CHANNELS,
    STANDARD_DTYPE
)

__all__ = [
    "generate_synthetic_eeg_trial",
    "generate_synthetic_eeg_dataframe",
    "STANDARD_TIMEPOINTS",
    "STANDARD_CHANNELS",
    "STANDARD_DTYPE"
]
