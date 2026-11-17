"""Print a read-only structural report for the ADHD EEG MATLAB files."""

from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat


# Keep the script beside the data folder; the files are never opened for writing.
EXPECTED_FILES = ("FC.mat", "MC.mat", "FADHD.mat", "MADHD.mat")
SCRIPT_FOLDER = Path(__file__).resolve().parent
COMMON_CHANNEL_COUNTS = {8, 14, 16, 19, 21, 32, 64, 128, 256}


def locate_file(filename):
    """Find a dataset file beside this script or in the common data subfolder."""
    possible_paths = (
        SCRIPT_FOLDER / filename,
        SCRIPT_FOLDER / "Dataset files" / filename,
    )
    for path in possible_paths:
        if path.is_file():
            return path
    return None


def print_shape(label, value):
    """Print an object's Python type, shape, and dimension count if available."""
    shape = getattr(value, "shape", None)
    if shape is None:
        print(f"{label}: type={type(value).__name__}")
        return

    dimensions = len(shape)
    print(
        f"{label}: type={type(value).__name__}, "
        f"shape={shape}, dimensions={dimensions}"
    )


def print_recording_estimate(label, shape):
    """Report likely sample and channel axes for common EEG array shapes."""
    if len(shape) < 2:
        return

    if len(shape) == 2:
        small_axis = min(range(2), key=lambda axis: shape[axis])
        large_axis = 1 - small_axis
        if shape[small_axis] in COMMON_CHANNEL_COUNTS:
            print(
                f"{label}: likely samples={shape[large_axis]}, "
                f"channels={shape[small_axis]} "
                f"(estimated from a common EEG channel count)"
            )
        elif shape[small_axis] <= 256 and shape[large_axis] > shape[small_axis]:
            print(
                f"{label}: possible samples={shape[large_axis]}, "
                f"channels={shape[small_axis]} "
                "(axis roles are estimates; verify against dataset documentation)"
            )
        return

    if len(shape) == 3:
        # A common recording layout is subject x samples x channels/features.
        # Use the dimensions as a clue, but label the result as an estimate.
        sample_axis = max(range(3), key=lambda axis: shape[axis])
        other_axes = [axis for axis in range(3) if axis != sample_axis]
        channel_axis = min(other_axes, key=lambda axis: shape[axis])
        subject_axis = next(axis for axis in other_axes if axis != channel_axis)
        channel_label = (
            "channels" if shape[channel_axis] in COMMON_CHANNEL_COUNTS
            else "channels/features"
        )
        print(
            f"{label}: possible subjects={shape[subject_axis]}, "
            f"samples={shape[sample_axis]}, "
            f"{channel_label}={shape[channel_axis]} "
            "(axis roles estimated from array dimensions)"
        )


def inspect_scipy_value(name, value, indent="  "):
    """Describe a scipy-loaded variable and expand MATLAB cell arrays."""
    label = f"{indent}{name}"
    print_shape(label, value)
    if isinstance(value, np.ndarray):
        print(f"{indent}  dtype={value.dtype}")

        # MATLAB cell arrays are represented as NumPy object arrays by scipy.
        if value.dtype == object:
            cells = list(value.flat)
            if len(cells) == 11:
                print(f"{indent}  Cell array: inspecting all 11 recording conditions")
                for condition_number, cell in enumerate(cells, start=1):
                    condition_label = f"{indent}  Condition {condition_number}"
                    print_shape(condition_label, cell)
                    if isinstance(cell, np.ndarray) and cell.dtype == object:
                        subject_entries = list(cell.flat)
                        print(
                            f"{indent}    Nested cell entries="
                            f"{len(subject_entries)} (possible subject recordings)"
                        )
                        for subject_number, recording in enumerate(
                            subject_entries, start=1
                        ):
                            recording_label = (
                                f"{indent}    Subject/recording {subject_number}"
                            )
                            print_shape(recording_label, recording)
                            print_recording_estimate(
                                recording_label, recording.shape
                            )
                    elif isinstance(cell, np.ndarray):
                        print_recording_estimate(condition_label, cell.shape)
            elif cells and any(isinstance(cell, np.ndarray) for cell in cells):
                print(f"{indent}  Cell entries={len(cells)}")
                for entry_number, cell in enumerate(cells, start=1):
                    entry_label = f"{indent}  Cell entry {entry_number}"
                    print_shape(entry_label, cell)
                    print_recording_estimate(entry_label, cell.shape)
        else:
            print_recording_estimate(label, value.shape)


def is_matlab_v73(path):
    """Check the MAT-file header for the MATLAB 7.3 marker."""
    with path.open("rb") as mat_file:
        header = mat_file.read(128)
    return b"MATLAB 7.3 MAT-file" in header


def inspect_hdf5_file(path):
    """Inspect a MATLAB v7.3 file through h5py, including referenced cell data."""
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "This is a MATLAB v7.3 file, but h5py is not installed. "
            "Install h5py to inspect it."
        ) from error

    print("Loader: h5py (MATLAB v7.3 / HDF5)")
    seen = set()

    def inspect_hdf5_item(item, name, indent="  "):
        """Print a dataset or group and follow MATLAB cell references."""
        if isinstance(item, h5py.Group):
            print(f"{indent}{name}: type=Group, entries={len(item)}")
            for child_name, child in item.items():
                inspect_hdf5_item(child, child_name, indent + "  ")
            return

        print(
            f"{indent}{name}: type=Dataset, shape={item.shape}, "
            f"dimensions={item.ndim}, dtype={item.dtype}"
        )
        matlab_class = item.attrs.get("MATLAB_class")
        if matlab_class is not None:
            if isinstance(matlab_class, bytes):
                matlab_class = matlab_class.decode("utf-8", errors="replace")
            print(f"{indent}  MATLAB_class={matlab_class}")

        reference_type = h5py.check_dtype(ref=item.dtype)
        if reference_type is None:
            # Numeric datasets can still be assessed for likely EEG axes.
            if item.ndim in (2, 3) and item.size:
                print_recording_estimate(name, item.shape)
            return

        references = list(item[()].flat)
        if len(references) == 11:
            print(f"{indent}  Cell array: inspecting all 11 recording conditions")
        for index, reference in enumerate(references, start=1):
            if not reference:
                print(f"{indent}  Cell entry {index}: empty reference")
                continue
            target = item.file[reference]
            target_name = target.name
            if target_name in seen:
                print(f"{indent}  Cell entry {index}: {target_name} (already inspected)")
                continue
            seen.add(target_name)
            entry_label = (
                f"Condition {index}" if len(references) == 11
                else f"Cell entry {index}"
            )
            inspect_hdf5_item(target, entry_label, indent + "  ")

    with h5py.File(path, "r") as mat_file:
        public_names = [name for name in mat_file.keys() if not name.startswith("#")]
        if not public_names:
            print("  No top-level variables found.")
        for name in public_names:
            inspect_hdf5_item(mat_file[name], name)


def inspect_file(filename):
    """Load one MAT file and print all public variables."""
    path = locate_file(filename)
    print(f"\n{'=' * 72}\nFile: {filename}")
    if path is None:
        print("Status: not found beside the script or in the Dataset files folder.")
        return

    print(f"Path: {path}")
    version_73 = is_matlab_v73(path)
    try:
        variables = loadmat(path)
    except (NotImplementedError, ValueError, OSError) as error:
        if version_73:
            # MATLAB v7.3 uses HDF5, which scipy.io.loadmat does not support.
            print(f"scipy could not load the v7.3 file: {error}")
            inspect_hdf5_file(path)
            return
        print(f"Error loading MAT file with scipy: {error}")
        return

    print("Loader: scipy.io.loadmat")
    public_variables = {
        name: value
        for name, value in variables.items()
        if not name.startswith("__")
    }
    if not public_variables:
        print("  No MATLAB variables found.")
        return

    for name, value in public_variables.items():
        inspect_scipy_value(name, value)


def main():
    """Inspect each expected file without writing to the dataset."""
    print("ADHD EEG dataset inspection")
    print("Read-only report; no data files are changed.")
    for filename in EXPECTED_FILES:
        inspect_file(filename)
    print(f"\n{'=' * 72}\nInspection complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInspection interrupted.", file=sys.stderr)
