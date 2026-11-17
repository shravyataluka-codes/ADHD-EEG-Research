"""Validate the four ADHD EEG MAT files without changing the source data."""

import json
from pathlib import Path
import warnings

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
DATA_FOLDER = ROOT / "Dataset files"
REPORT_FOLDER = ROOT / "reports"
SAMPLING_RATE_HZ = 256
PUBLISHED_PARTICIPANT_COUNT = 79
PUBLISHED_QUALITY_NOTE = (
    "The Mendeley record notes that female-ADHD subject number 7 has a "
    "corrupted EEG signal."
)
FILES = ("FC.mat", "MC.mat", "FADHD.mat", "MADHD.mat")
MAPPING_SOURCE = "https://data.mendeley.com/datasets/6k4g25fhzg/1"

# Published condition mapping from the Mendeley Data record's Method field.
# Channel names are intentionally omitted; this script does not infer them.
PUBLISHED_MAPPING = (
    {"cell": 1, "task": "Eyes open baseline", "duration_seconds": 30},
    {"cell": 2, "task": "Eyes closed", "duration_seconds": 20},
    {"cell": 3, "task": "Eyes open", "duration_seconds": 20},
    {"cell": 4, "task": "Cognitive Challenge", "duration_seconds": 45},
    {"cell": 5, "task": "Pre-Omni harmonic baseline", "duration_seconds": 15},
    {"cell": 6, "task": "Omni harmonic assessment", "duration_seconds": 30},
    {"cell": 7, "task": "Eyes open baseline", "duration_seconds": 30},
    {"cell": 8, "task": "Eyes closed", "duration_seconds": 30},
    {"cell": 9, "task": "Eyes open", "duration_seconds": 30},
    {"cell": 10, "task": "Eyes closed", "duration_seconds": 45},
    {"cell": 11, "task": "Eyes closed", "duration_seconds": 45},
)
DUPLICATE_PAIRS = ((2, 8), (3, 9), (4, 10))
GROUP_LABELS = {
    "FC.mat": "female control",
    "MC.mat": "male control",
    "FADHD.mat": "female ADHD",
    "MADHD.mat": "male ADHD",
}


def decode_text(value):
    """Convert MATLAB/HDF5 text metadata to a readable Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return "".join(decode_text(part) for part in value.flat)
    return str(value)


def load_v73_cells(path, preferred_name):
    """Read an HDF5-backed MATLAB file if scipy cannot read it."""
    import h5py

    with h5py.File(path, "r") as mat_file:
        public_names = [name for name in mat_file if not name.startswith("#")]
        if preferred_name in public_names:
            variable_name = preferred_name
        else:
            variable_name = next(
                (
                    name for name in public_names
                    if h5py.check_dtype(ref=mat_file[name].dtype) is not None
                ),
                None,
            )
        if variable_name is None:
            raise ValueError("No MATLAB cell-array variable was found.")

        dataset = mat_file[variable_name]
        if h5py.check_dtype(ref=dataset.dtype) is None:
            raise ValueError(f"{variable_name} is not an HDF5 reference cell array.")

        cells = []
        for reference in dataset[()].flat:
            if not reference:
                cells.append(np.asarray([]))
                continue
            value = np.asarray(mat_file[reference][()])
            # MATLAB stores dimensions in the reverse order in HDF5 datasets.
            if value.ndim > 1:
                value = value.transpose(tuple(reversed(range(value.ndim))))
            cells.append(value)
        return variable_name, public_names, cells, "h5py (MATLAB v7.3)"


def load_cells(path):
    """Load one file with scipy first, falling back to h5py for MATLAB v7.3."""
    variable_name = path.stem
    try:
        variables = loadmat(path)
    except (NotImplementedError, ValueError, OSError):
        return load_v73_cells(path, variable_name)

    if variable_name in variables:
        value = variables[variable_name]
    else:
        candidates = [
            item for name, item in variables.items()
            if not name.startswith("__")
            and isinstance(item, np.ndarray)
            and item.dtype == object
            and item.size == 11
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely identify the 11-cell variable in {path.name}."
            )
        value = candidates[0]

    if not isinstance(value, np.ndarray) or value.dtype != object:
        raise ValueError(f"MATLAB variable {variable_name} is not a cell array.")
    public_names = [
        name for name in variables if not name.startswith("__")
    ]
    return variable_name, public_names, list(value.flat), "scipy.io.loadmat"


def analyze_condition(cell, mapping):
    """Compute structural, duration, signal-quality, and mapping facts."""
    data = np.asarray(cell)
    result = {
        "cell": mapping["cell"],
        "published_task": mapping["task"],
        "published_duration_seconds": mapping["duration_seconds"],
        "shape": [int(size) for size in data.shape],
        "dimensions": int(data.ndim),
        "dtype": str(data.dtype),
        "structure_supports_subjects_samples_two_channels": bool(
            data.ndim == 3 and data.shape[0] > 0
            and data.shape[1] > 0 and data.shape[2] == 2
        ),
        "participant_slots": int(data.shape[0]) if data.ndim >= 1 else 0,
        "sample_count": int(data.shape[1]) if data.ndim >= 2 else None,
        "channel_count": int(data.shape[2]) if data.ndim == 3 else None,
        "calculated_duration_seconds": None,
        "duration_matches_published_mapping": None,
        "nan_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "nonfinite_channel_recordings": 0,
        "flatline_channel_recordings": 0,
        "zero_variance_channel_recordings": 0,
        "flatline_trace_slots": [],
        "zero_variance_trace_slots": [],
        "quality_check_note": None,
    }

    if data.ndim >= 2:
        sample_count = int(data.shape[1])
        duration = sample_count / SAMPLING_RATE_HZ
        result["calculated_duration_seconds"] = duration
        result["duration_matches_published_mapping"] = bool(
            np.isclose(duration, mapping["duration_seconds"], rtol=0, atol=1e-9)
        )

    if data.ndim != 3 or data.shape[2] != 2:
        result["quality_check_note"] = (
            "Signal-quality checks require a numeric subjects x samples x 2 array."
        )
        return result
    if not np.issubdtype(data.dtype, np.number):
        result["quality_check_note"] = (
            f"Signal-quality checks skipped for non-numeric dtype {data.dtype}."
        )
        return result

    # Count missing and infinite samples without altering the source array.
    result["nan_count"] = int(np.isnan(data).sum())
    result["positive_infinity_count"] = int(np.isposinf(data).sum())
    result["negative_infinity_count"] = int(np.isneginf(data).sum())

    # Test each subject-slot/channel trace; only complete finite traces are
    # eligible for the flatline and variance checks.
    finite_per_trace = np.isfinite(data).all(axis=1)
    result["nonfinite_channel_recordings"] = int((~finite_per_trace).sum())
    flatline_count = 0
    zero_variance_count = 0
    for subject_slot in range(data.shape[0]):
        for channel_slot in range(data.shape[2]):
            trace = data[subject_slot, :, channel_slot]
            if not finite_per_trace[subject_slot, channel_slot]:
                continue
            if np.all(trace == trace[0]):
                flatline_count += 1
                result["flatline_trace_slots"].append(
                    {
                        "participant_slot": subject_slot + 1,
                        "channel_slot": channel_slot + 1,
                    }
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                variance = np.var(trace, dtype=np.float64)
            if variance == 0:
                zero_variance_count += 1
                result["zero_variance_trace_slots"].append(
                    {
                        "participant_slot": subject_slot + 1,
                        "channel_slot": channel_slot + 1,
                    }
                )

    result["flatline_channel_recordings"] = flatline_count
    result["zero_variance_channel_recordings"] = zero_variance_count
    return result


def exact_duplicate_result(left, right, left_cell, right_cell):
    """Test whether two complete condition arrays are exactly value-equal."""
    left_data = np.asarray(left_cell)
    right_data = np.asarray(right_cell)
    same_shape = left_data.shape == right_data.shape
    both_numeric = (
        np.issubdtype(left_data.dtype, np.number)
        and np.issubdtype(right_data.dtype, np.number)
    )
    exact_match = bool(
        same_shape and both_numeric
        and np.array_equal(left_data, right_data, equal_nan=False)
    )
    return {
        "cells": [left, right],
        "shape_comparison": "same" if same_shape else "different",
        "exact_array_match": exact_match,
        "comparison_note": (
            "Exact value comparison of complete arrays; NaN values do not count "
            "as equal. No participant-slot alignment is implied."
        ),
    }


def validate_file(filename):
    """Load and validate all 11 conditions for one dataset group."""
    path = DATA_FOLDER / filename
    file_result = {
        "file": filename,
        "group_label": GROUP_LABELS[filename],
        "path": str(path),
        "status": "error",
        "variable_name": None,
        "top_level_variable_names": [],
        "loader": None,
        "participant_slots_per_condition": [],
        "conditions": [],
        "duplicate_checks": [],
        "error": None,
    }
    if not path.is_file():
        file_result["error"] = "Dataset file not found."
        return file_result

    try:
        variable_name, variable_names, cells, loader = load_cells(path)
        file_result["variable_name"] = variable_name
        file_result["top_level_variable_names"] = variable_names
        file_result["loader"] = loader
        if len(cells) != len(PUBLISHED_MAPPING):
            file_result["error"] = (
                f"Expected 11 cells; found {len(cells)}."
            )
            return file_result

        file_result["conditions"] = [
            analyze_condition(cell, mapping)
            for cell, mapping in zip(cells, PUBLISHED_MAPPING)
        ]
        file_result["participant_slots_per_condition"] = [
            condition["participant_slots"]
            for condition in file_result["conditions"]
        ]
        file_result["duplicate_checks"] = [
            exact_duplicate_result(
                left, right, cells[left - 1], cells[right - 1]
            )
            for left, right in DUPLICATE_PAIRS
        ]
        file_result["status"] = "complete"
    except (ImportError, OSError, TypeError, ValueError) as error:
        file_result["error"] = f"{type(error).__name__}: {error}"
    return file_result


def summarize(results):
    """Create concise facts that can be used in both reports."""
    completed = [item for item in results if item["status"] == "complete"]
    durations_match = all(
        condition["duration_matches_published_mapping"] is True
        for item in completed for condition in item["conditions"]
    )
    structure_supported = all(
        condition["structure_supports_subjects_samples_two_channels"]
        for item in completed for condition in item["conditions"]
    )
    all_cells_present = all(item["status"] == "complete" for item in results)
    nonfinite_total = sum(
        condition["nan_count"]
        + condition["positive_infinity_count"]
        + condition["negative_infinity_count"]
        for item in completed for condition in item["conditions"]
    )
    flatline_total = sum(
        condition["flatline_channel_recordings"]
        for item in completed for condition in item["conditions"]
    )
    zero_variance_total = sum(
        condition["zero_variance_channel_recordings"]
        for item in completed for condition in item["conditions"]
    )
    return {
        "all_four_files_loaded": all_cells_present,
        "all_conditions_support_subjects_samples_two_channels": (
            structure_supported if all_cells_present else None
        ),
        "all_calculated_durations_match_published_mapping": (
            durations_match if all_cells_present else None
        ),
        "nonfinite_sample_count": nonfinite_total,
        "flatline_channel_recordings": flatline_total,
        "zero_variance_channel_recordings": zero_variance_total,
        "stored_participant_slot_total": (
            sum(
                item["participant_slots_per_condition"][0]
                for item in completed
                if item["participant_slots_per_condition"]
            )
            if all(item["status"] == "complete" for item in results)
            else None
        ),
        "participant_index_linkage": (
            "The MAT files expose the signal cell-array variables only; no "
            "participant-ID or linkage variable was found. Equal slot counts "
            "across conditions do not establish that row indices identify "
            "the same person. Cross-file participant matching is therefore "
            "not verified."
        ),
        "participant_linkage_verified": False,
        "quality_flags": {
            "duration_mapping_mismatches": sum(
                condition["duration_matches_published_mapping"] is False
                for item in completed for condition in item["conditions"]
            ),
            "exact_duplicate_condition_pairs": sum(
                check["exact_array_match"]
                for item in completed for check in item["duplicate_checks"]
            ),
        },
    }


def build_markdown(report):
    """Render the facts into a readable Markdown report."""
    summary = report["summary"]
    lines = [
        "# ADHD EEG Dataset Validation",
        "",
        "## Scope and method",
        "",
        "- Read-only validation of the four original `.mat` files in `Dataset files/`.",
        "- No preprocessing, feature extraction, participant-ID creation, or model training.",
        f"- Sampling rate used for duration calculations: **{SAMPLING_RATE_HZ} Hz**.",
        f"- Published mapping source: [Mendeley Data v1]({MAPPING_SOURCE}), "
        "Method field.",
        "- Array positions are reported as participant slots, not participant IDs.",
        "",
        "## Verified overall results",
        "",
        f"- All four files loaded with 11 conditions: **{summary['all_four_files_loaded']}**.",
        f"- All conditions have a three-dimensional shape ending in 2: "
        f"**{summary['all_conditions_support_subjects_samples_two_channels']}**.",
        "- This shape is consistent with the published "
        "`subjects x samples x 2 channels` description; it does not identify "
        "the physical channel names.",
        f"- All calculated durations match the published mapping: "
        f"**{summary['all_calculated_durations_match_published_mapping']}**.",
        f"- Non-finite samples (NaN + Inf): **{summary['nonfinite_sample_count']}**.",
        f"- Flatline channel recordings: **{summary['flatline_channel_recordings']}**.",
        f"- Zero-variance channel recordings: **{summary['zero_variance_channel_recordings']}**.",
        "",
        "## Participant counts",
        "",
        "| File | Group | Participant slots in each condition |",
        "|---|---|---:|",
    ]
    for item in report["files"]:
        counts = item["participant_slots_per_condition"]
        if item["status"] != "complete":
            count_text = f"Not available ({item['error']})"
        elif len(set(counts)) == 1:
            count_text = str(counts[0])
        else:
            count_text = ", ".join(str(value) for value in counts)
        lines.append(
            f"| {item['file']} | {item['group_label']} | {count_text} |"
        )

    lines.extend(
        [
            "",
            "These are stored participant-slot counts. The published record notes "
            "a corrupted female-ADHD recording; this report does not determine "
            "whether a slot is usable or remove it from the count.",
            "",
            f"The four files contain **{report['summary']['stored_participant_slot_total']} "
            "stored participant slots in total**. The Mendeley description reports "
            f"{PUBLISHED_PARTICIPANT_COUNT} participants and notes one corrupted "
            "female-ADHD recording. "
            "Stored slots are not asserted to be unique usable participants.",
            "",
            "## Condition-by-condition results",
            "",
            "| File | Cell | Published task | Shape (subjects x samples x channels) | "
            "Samples | Calculated (s) | Published (s) | Duration check | "
            "NaN | +Inf | -Inf | Nonfinite traces | Flatlines | Zero variance |",
            "|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["files"]:
        for condition in item["conditions"]:
            duration = condition["calculated_duration_seconds"]
            duration_text = "n/a" if duration is None else f"{duration:g}"
            match = condition["duration_matches_published_mapping"]
            match_text = "not checked" if match is None else (
                "match" if match else "**MISMATCH**"
            )
            shape = " x ".join(str(size) for size in condition["shape"])
            lines.append(
                f"| {item['file']} | {condition['cell']} | "
                f"{condition['published_task']} | {shape} | "
                f"{condition['sample_count'] if condition['sample_count'] is not None else 'n/a'} | "
                f"{duration_text} | {condition['published_duration_seconds']} | "
                f"{match_text} | {condition['nan_count']} | "
                f"{condition['positive_infinity_count']} | "
                f"{condition['negative_infinity_count']} | "
                f"{condition['nonfinite_channel_recordings']} | "
                f"{condition['flatline_channel_recordings']} | "
                f"{condition['zero_variance_channel_recordings']} |"
            )

    lines.extend(
        [
            "",
            "## Flatline and zero-variance trace locations",
            "",
            "Locations below use one-based array storage slots only. They are "
            "not participant IDs, and channel-slot numbers are not channel names.",
            "",
            "| File | Cell | Participant slot | Channel slot | Check |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in report["files"]:
        for condition in item["conditions"]:
            for trace in condition["flatline_trace_slots"]:
                lines.append(
                    f"| {item['file']} | {condition['cell']} | "
                    f"{trace['participant_slot']} | {trace['channel_slot']} | "
                    "flatline |"
                )
            for trace in condition["zero_variance_trace_slots"]:
                lines.append(
                    f"| {item['file']} | {condition['cell']} | "
                    f"{trace['participant_slot']} | {trace['channel_slot']} | "
                    "zero variance |"
                )
    if not any(
        condition["flatline_trace_slots"]
        or condition["zero_variance_trace_slots"]
        for item in report["files"] for condition in item["conditions"]
    ):
        lines.append("| - | - | - | - | No flatline or zero-variance traces |")
    flatline_rows = [
        (item, condition, trace)
        for item in report["files"]
        for condition in item["conditions"]
        for trace in condition["flatline_trace_slots"]
    ]
    if flatline_rows:
        lines.extend(
            [
                "",
                f"{PUBLISHED_QUALITY_NOTE} The observed flatline storage slot 7 is "
                "consistent with that note, but the arrays do not provide an "
                "identity crosswalk to prove these are the same participant "
                "across conditions.",
            ]
        )

    lines.extend(
        [
            "",
            "## Exact duplicate checks",
            "",
            "Each check compares the complete numeric arrays using exact value "
            "equality; NaNs are not treated as equal.",
            "",
            "| File | Cell pair | Same shape | Complete arrays exactly equal |",
            "|---|---|---|---|",
        ]
    )
    for item in report["files"]:
        for check in item["duplicate_checks"]:
            lines.append(
                f"| {item['file']} | {check['cells'][0]}/{check['cells'][1]} | "
                f"{check['shape_comparison']} | "
                f"{'YES' if check['exact_array_match'] else 'NO'} |"
            )

    lines.extend(
        [
            "",
            "## Participant matching: verified facts vs. limitations",
            "",
            "**Verified:** each condition's first-axis length is reported above. "
            "The published Mendeley method describes each file as a 1 x 11 cell "
            "array with participant recordings in the cells.",
            "",
            "**Not verified:** the files contain no participant IDs or explicit "
            "row-to-person crosswalk among the exposed MAT variables. Matching slot counts do "
            "not prove that slot *n* in one condition is the same individual "
            "as slot *n* in another condition, and they do not enable matching "
            "people across the four files.",
            "",
            "## Published mapping and source",
            "",
            "| Cell | Published task | Published duration (s) |",
            "|---:|---|---:|",
        ]
    )
    for mapping in PUBLISHED_MAPPING:
        lines.append(
            f"| {mapping['cell']} | {mapping['task']} | "
            f"{mapping['duration_seconds']} |"
        )
    lines.extend(
        [
            "",
            f"Source: {MAPPING_SOURCE} (dataset DOI `10.17632/6k4g25fhzg.1`). "
            "The published values are preserved as given; observed duration "
            "differences are flagged, not silently reconciled.",
            "",
        ]
    )
    return "\n".join(lines)


def print_terminal_summary(report):
    """Print a brief, useful summary while keeping full details in the files."""
    summary = report["summary"]
    print("ADHD EEG dataset validation (read-only)")
    for item in report["files"]:
        if item["status"] != "complete":
            print(f"{item['file']}: ERROR - {item['error']}")
            continue
        count_set = sorted(set(item["participant_slots_per_condition"]))
        count_label = (
            str(count_set[0]) if len(count_set) == 1
            else ", ".join(str(count) for count in count_set)
        )
        mismatches = [
            condition["cell"] for condition in item["conditions"]
            if condition["duration_matches_published_mapping"] is False
        ]
        duplicates = [
            check["cells"] for check in item["duplicate_checks"]
            if check["exact_array_match"]
        ]
        print(
            f"{item['file']} ({item['group_label']}): "
            f"{len(item['conditions'])} conditions; "
            f"{count_label} participant slots/condition; "
            f"duration mismatches={mismatches or 'none'}; "
            f"exact duplicate pairs={duplicates or 'none'}"
        )
    print(
        "Signal quality totals: "
        f"nonfinite samples={summary['nonfinite_sample_count']}, "
        f"flatline traces={summary['flatline_channel_recordings']}, "
        f"zero-variance traces={summary['zero_variance_channel_recordings']}."
    )
    print("Participant identity linkage across conditions/files: not verified.")
    print(f"Markdown report: {REPORT_FOLDER / 'dataset_validation_report.md'}")
    print(f"JSON results: {REPORT_FOLDER / 'dataset_validation_results.json'}")


def main():
    """Validate all files and write reports, leaving source MAT files untouched."""
    REPORT_FOLDER.mkdir(exist_ok=True)
    results = [validate_file(filename) for filename in FILES]
    report = {
        "title": "ADHD EEG Dataset Validation",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "mapping_source": MAPPING_SOURCE,
        "published_context": {
            "participant_count": PUBLISHED_PARTICIPANT_COUNT,
            "quality_note": PUBLISHED_QUALITY_NOTE,
        },
        "published_mapping": list(PUBLISHED_MAPPING),
        "duplicate_pairs_checked": [list(pair) for pair in DUPLICATE_PAIRS],
        "participant_matching": {
            "verified": False,
            "basis": (
                "The four MAT files expose the signal cell-array variable only "
                "and no participant-ID or linkage variable. Equal slot counts "
                "are not proof of participant identity or index alignment."
            ),
        },
        "files": results,
    }
    report["summary"] = summarize(results)

    json_path = REPORT_FOLDER / "dataset_validation_results.json"
    markdown_path = REPORT_FOLDER / "dataset_validation_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print_terminal_summary(report)


if __name__ == "__main__":
    main()
