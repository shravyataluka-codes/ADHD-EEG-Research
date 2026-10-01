"""
================================================================================
SYNTHETIC / DUMMY EEG GENERATOR
================================================================================
Generates purely synthetic EEG signal tensors conforming to the ML contract:
- Shape: [385 time positions, 56 channels]
- Data Type: float32
- Deterministic when initialized with a random seed.

DISCLAIMER:
This data is synthetically generated for software prototype testing.
It does NOT represent real patient neurophysiological data.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple


STANDARD_TIMEPOINTS = 385  # ~1.5 seconds at 256 Hz
STANDARD_CHANNELS = 56
STANDARD_DTYPE = np.float32


def generate_synthetic_eeg_trial(
    seed: Optional[int] = 42,
    timepoints: int = STANDARD_TIMEPOINTS,
    channels: int = STANDARD_CHANNELS,
    scale: float = 0.5
) -> np.ndarray:
    """
    Generates a synthetic EEG trial matrix of shape [385, 56] and dtype float32.

    Args:
        seed: Random seed for deterministic reproducibility.
        timepoints: Number of temporal points (contract: 385).
        channels: Number of EEG electrode channels (contract: 56).
        scale: Amplitude scale factor.

    Returns:
        np.ndarray: Synthetic EEG tensor of shape [timepoints, channels], dtype float32.
    """
    rng = np.random.default_rng(seed)
    
    # Generate smooth synthetic waveforms across timepoints
    t = np.linspace(0, 1.5, timepoints, dtype=np.float32)
    data = np.zeros((timepoints, channels), dtype=STANDARD_DTYPE)
    
    for ch in range(channels):
        # Combine generic synthetic harmonic frequencies and small noise
        f1 = 4.0 + (ch % 8) * 1.5  # 4 - 16 Hz synthetic components
        f2 = 10.0 + (ch % 5) * 2.0
        phase = (ch * 0.1)
        
        harmonic_signal = (
            0.6 * np.sin(2.0 * np.pi * f1 * t + phase) +
            0.3 * np.sin(2.0 * np.pi * f2 * t + phase * 0.5)
        )
        noise = rng.normal(0, 0.15, size=timepoints)
        channel_data = (harmonic_signal + noise) * scale
        data[:, ch] = channel_data.astype(STANDARD_DTYPE)
        
    return data


def generate_synthetic_eeg_dataframe(
    seed: Optional[int] = 42,
    timepoints: int = STANDARD_TIMEPOINTS,
    channels: int = STANDARD_CHANNELS
) -> pd.DataFrame:
    """
    Generates a synthetic EEG DataFrame with a 'timestamp' column and 56 channel columns.

    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'ch_1', ..., 'ch_56']
    """
    tensor = generate_synthetic_eeg_trial(seed=seed, timepoints=timepoints, channels=channels)
    t = np.linspace(0.0, 1.5, timepoints, dtype=np.float32)
    
    col_names = [f"ch_{i+1}" for i in range(channels)]
    df = pd.DataFrame(tensor, columns=col_names, dtype=STANDARD_DTYPE)
    df.insert(0, "timestamp", np.round(t, 4))
    return df


if __name__ == "__main__":
    arr = generate_synthetic_eeg_trial(seed=42)
    print(f"Generated synthetic trial: shape={arr.shape}, dtype={arr.dtype}")
    df = generate_synthetic_eeg_dataframe(seed=42)
    print(f"Generated DataFrame: shape={df.shape}")
