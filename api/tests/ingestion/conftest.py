"""Shared pytest fixtures for Phase 2 ingestion tests.

All fixtures use synthetic data only.  No real research recordings are
referenced.  Fixture data is intentionally minimal and clearly labelled as
test data.
"""

from __future__ import annotations

import io
import struct

import h5py
import numpy as np
import pytest
import scipy.io


# ---------------------------------------------------------------------------
# Synthetic CSV fixtures
# ---------------------------------------------------------------------------


def _make_csv(
    n_samples: int = 50,
    n_channels: int = 2,
    header: bool = True,
    values: np.ndarray | None = None,
) -> bytes:
    """Return a minimal synthetic CSV as bytes."""
    rng = np.random.default_rng(42)
    if values is None:
        values = rng.standard_normal((n_samples, n_channels))

    lines: list[str] = []
    if header:
        lines.append(",".join(f"ch{i}" for i in range(n_channels)))
    for row in values:
        lines.append(",".join(f"{v:.6f}" for v in row))
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture
def csv_with_header() -> bytes:
    """Valid 2-channel CSV with a header row."""
    return _make_csv(n_samples=100, n_channels=2, header=True)


@pytest.fixture
def csv_no_header() -> bytes:
    """Valid 2-channel CSV without a header row."""
    return _make_csv(n_samples=100, n_channels=2, header=False)


@pytest.fixture
def csv_multichannel() -> bytes:
    """Valid 8-channel CSV with header."""
    return _make_csv(n_samples=200, n_channels=8, header=True)


@pytest.fixture
def csv_empty() -> bytes:
    return b""


@pytest.fixture
def csv_header_only() -> bytes:
    return b"ch0,ch1\n"


@pytest.fixture
def csv_non_numeric() -> bytes:
    return b"ch0,ch1\n1.0,not_a_number\n2.0,3.0\n"


@pytest.fixture
def csv_missing_value() -> bytes:
    """CSV with an empty cell (missing value)."""
    return b"ch0,ch1\n1.0,\n2.0,3.0\n"


@pytest.fixture
def csv_nan_value() -> bytes:
    """CSV with a NaN value."""
    return b"ch0,ch1\n1.0,nan\n2.0,3.0\n"


@pytest.fixture
def csv_inf_value() -> bytes:
    return b"ch0,ch1\n1.0,inf\n2.0,3.0\n"


@pytest.fixture
def csv_wrong_columns() -> bytes:
    return b"ch0,ch1,ch2\n1.0,2.0,3.0\n1.0,2.0\n"  # inconsistent


@pytest.fixture
def csv_binary() -> bytes:
    return bytes(range(256))


@pytest.fixture
def csv_flatline() -> bytes:
    """CSV where one channel is entirely constant (flatline)."""
    lines = ["ch0,ch1"]
    for _ in range(50):
        lines.append("0.5,1.0")  # ch0 is flatline at 0.5
    return ("\n".join(lines) + "\n").encode()


# ---------------------------------------------------------------------------
# Synthetic MAT v5 fixtures
# ---------------------------------------------------------------------------


def _make_mat_v5(variables: dict) -> bytes:
    buf = io.BytesIO()
    scipy.io.savemat(buf, variables)
    return buf.getvalue()


@pytest.fixture
def mat_v5_single_var() -> bytes:
    """MAT v5 with a single 2D EEG-like array [1000, 4]."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((1000, 4)).astype(np.float64)
    return _make_mat_v5({"eeg": data})


@pytest.fixture
def mat_v5_multi_var() -> bytes:
    """MAT v5 with two large numeric arrays (ambiguous)."""
    rng = np.random.default_rng(7)
    return _make_mat_v5({
        "eeg_a": rng.standard_normal((2000, 2)).astype(np.float64),
        "eeg_b": rng.standard_normal((2000, 3)).astype(np.float64),
    })


@pytest.fixture
def mat_v5_no_numeric() -> bytes:
    """MAT v5 with only small arrays (no large numeric candidate)."""
    return _make_mat_v5({"tiny": np.array([1.0, 2.0])})


@pytest.fixture
def mat_v5_3d() -> bytes:
    """MAT v5 with a 3D array [10 trials, 100 samples, 4 channels]."""
    rng = np.random.default_rng(99)
    data = rng.standard_normal((10, 100, 4)).astype(np.float64)
    return _make_mat_v5({"trials": data})


# ---------------------------------------------------------------------------
# Synthetic MAT v7.3 / HDF5 fixtures
# ---------------------------------------------------------------------------


def _make_mat_hdf5(datasets: dict) -> bytes:
    """Create a minimal HDF5 buffer with MATLAB v7.3 magic."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        # Write MATLAB version attribute so it looks like v7.3.
        hf.attrs["MATLAB_class"] = np.bytes_("double")
        for name, array in datasets.items():
            hf.create_dataset(name, data=array, compression="gzip")
    return buf.getvalue()


@pytest.fixture
def mat_hdf5_single_var() -> bytes:
    """MAT v7.3 HDF5 with a single 2D EEG array [1000, 4]."""
    rng = np.random.default_rng(13)
    data = rng.standard_normal((1000, 4)).astype(np.float64)
    return _make_mat_hdf5({"eeg": data})


@pytest.fixture
def mat_hdf5_multi_var() -> bytes:
    """MAT v7.3 HDF5 with two large numeric datasets (ambiguous)."""
    rng = np.random.default_rng(13)
    return _make_mat_hdf5({
        "eeg_a": rng.standard_normal((2000, 2)).astype(np.float64),
        "eeg_b": rng.standard_normal((2000, 3)).astype(np.float64),
    })


@pytest.fixture
def mat_hdf5_3d() -> bytes:
    """MAT v7.3 HDF5 with 3D [5, 100, 4] array (trials × samples × channels)."""
    rng = np.random.default_rng(55)
    data = rng.standard_normal((5, 100, 4)).astype(np.float64)
    return _make_mat_hdf5({"eeg3d": data})


@pytest.fixture
def mat_hdf5_no_numeric() -> bytes:
    """MAT v7.3 HDF5 with only a tiny dataset."""
    return _make_mat_hdf5({"small": np.array([1.0, 2.0])})


@pytest.fixture
def mat_invalid_magic() -> bytes:
    """Bytes that are neither MAT v5 nor HDF5."""
    return b"This is not a MAT file at all." + b"\x00" * 100
