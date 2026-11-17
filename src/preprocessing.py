"""Run the configured, read-only EEG preprocessing pipeline."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, sosfiltfilt

from validate_dataset import load_cells

from .epoching import (
    assign_grouped_folds,
    candidate_epoch_count,
    epoch_recording,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingRejectedError(ValueError):
    """A recording cannot safely be passed to the configured filter."""

    def __init__(self, reason, channel_slots=None, detail=None):
        super().__init__(detail or reason)
        self.reason = reason
        self.channel_slots = channel_slots or []


def bandpass_filter(
    recording,
    sampling_rate_hz=256,
    low_cutoff_hz=0.5,
    high_cutoff_hz=40.0,
    order=4,
):
    """Apply a zero-phase Butterworth band-pass filter along the sample axis."""
    data = np.asarray(recording)
    if data.ndim != 2:
        raise ValueError("A recording must have samples x channels dimensions.")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("A recording must contain numeric values.")
    if np.iscomplexobj(data):
        raise ValueError("A recording must contain real-valued EEG samples.")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("A recording must contain samples and channels.")
    if not np.isfinite(data).all():
        raise RecordingRejectedError("nonfinite_input")

    flatline_channels = np.flatnonzero(np.ptp(data, axis=0) == 0) + 1
    if flatline_channels.size:
        raise RecordingRejectedError(
            "flatline_input_channel",
            channel_slots=flatline_channels.tolist(),
        )
    if sampling_rate_hz <= 0:
        raise ValueError("Sampling rate must be positive.")
    if not 0 < low_cutoff_hz < high_cutoff_hz < sampling_rate_hz / 2:
        raise ValueError("Cutoffs must satisfy 0 < low < high < Nyquist.")
    if order < 1:
        raise ValueError("Butterworth filter order must be at least one.")

    # Copy to float64 so filtering cannot alter or alias source data.
    filter_input = np.array(data, dtype=np.float64, copy=True)
    second_order_sections = butter(
        order,
        (low_cutoff_hz, high_cutoff_hz),
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    try:
        filtered = sosfiltfilt(second_order_sections, filter_input, axis=0)
    except ValueError as error:
        raise ValueError(f"Zero-phase filtering failed: {error}") from error
    if not np.isfinite(filtered).all():
        raise RecordingRejectedError("nonfinite_filtered_output")
    return filtered


def load_config(config_path):
    """Read the YAML configuration and resolve all paths from its location."""
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("The preprocessing YAML must contain a mapping.")

    required_keys = {
        "sampling_rate_hz",
        "filter",
        "notch_filter",
        "edge_trim_seconds",
        "epoch_duration_seconds",
        "selected_cells",
        "excluded_storage_slots",
        "class_by_file",
        "cv_folds",
        "paths",
    }
    missing_keys = required_keys - config.keys()
    if missing_keys:
        raise ValueError(
            f"Missing required configuration keys: {sorted(missing_keys)}"
        )
    if config["notch_filter"].get("enabled", False):
        raise ValueError(
            "Notch filtering is disabled unless separate diagnostics justify it."
        )
    if config["filter"].get("family") != "butterworth":
        raise ValueError("This pipeline currently requires a Butterworth filter.")
    if config["filter"].get("zero_phase") is not True:
        raise ValueError("This pipeline requires zero-phase filtering.")
    if config.get("epoch_overlap_seconds") != 0:
        raise ValueError("This pipeline requires non-overlapping epochs.")

    sampling_rate = config["sampling_rate_hz"]
    low_cutoff = config["filter"]["low_cutoff_hz"]
    high_cutoff = config["filter"]["high_cutoff_hz"]
    if not 0 < low_cutoff < high_cutoff < sampling_rate / 2:
        raise ValueError("Filter cutoffs must satisfy 0 < low < high < Nyquist.")
    if config["filter"]["order"] < 1:
        raise ValueError("Butterworth filter order must be at least one.")
    candidate_epoch_count(
        0,
        sampling_rate,
        config["edge_trim_seconds"],
        config["epoch_duration_seconds"],
    )
    selected_cells = config["selected_cells"]
    if (
        not selected_cells
        or len(set(selected_cells)) != len(selected_cells)
        or any(cell not in range(1, 12) for cell in selected_cells)
    ):
        raise ValueError("Selected cells must be unique values from 1 through 11.")

    paths = config["paths"]
    for path_key in ("input_directory", "output_directory", "report_markdown", "report_json"):
        if path_key not in paths:
            raise ValueError(f"Missing paths.{path_key} in preprocessing YAML.")
        paths[path_key] = (config_path.parent / paths[path_key]).resolve()
    config["_config_path"] = config_path
    return config


def _count_nonfinite(data):
    """Count NaN and infinite input samples for quality-control reporting."""
    return {
        "nan_count": int(np.isnan(data).sum()),
        "positive_infinity_count": int(np.isposinf(data).sum()),
        "negative_infinity_count": int(np.isneginf(data).sum()),
    }


def _record_rejections(
    rejected_epochs,
    cell_number,
    filename,
    class_label,
    storage_slot,
    group_key,
    candidate_count,
    reason,
    channel_slots=None,
    detail=None,
):
    """Record every candidate epoch rejected with a recording-level reason."""
    for epoch_index in range(candidate_count):
        rejected_epochs.append(
            {
                "cell": cell_number,
                "source_file": filename,
                "class_label": class_label,
                "participant_slot": storage_slot,
                "group_key": group_key,
                "epoch_index": epoch_index + 1,
                "reason": reason,
                "channel_slots": channel_slots or [],
                "detail": detail,
            }
        )


def _add_cell_epoch(
    cell_number,
    filename,
    class_label,
    storage_slot,
    group_key,
    raw_recording,
    config,
    cell_state,
):
    """Filter and epoch one participant-slot recording."""
    sampling_rate = config["sampling_rate_hz"]
    expected_epochs = candidate_epoch_count(
        raw_recording.shape[0],
        sampling_rate,
        config["edge_trim_seconds"],
        config["epoch_duration_seconds"],
    )
    quality = _count_nonfinite(raw_recording)
    flatline_slots = []
    if np.isfinite(raw_recording).all():
        flatline_slots = (
            np.flatnonzero(np.ptp(raw_recording, axis=0) == 0) + 1
        ).tolist()

    participant_result = {
        "cell": cell_number,
        "source_file": filename,
        "class_label": class_label,
        "participant_slot": storage_slot,
        "group_key": group_key,
        "status": "processed",
        "candidate_epoch_count": expected_epochs,
        "accepted_epoch_count": 0,
        "rejected_epoch_count": 0,
        "edge_trim_samples_per_side": round(
            config["edge_trim_seconds"] * sampling_rate
        ),
        "discarded_tail_samples": 0,
        "raw_quality_control": {
            **quality,
            "flatline_channel_slots": flatline_slots,
        },
        "filtered_output_finite": None,
    }

    try:
        filtered = bandpass_filter(
            raw_recording,
            sampling_rate_hz=sampling_rate,
            low_cutoff_hz=config["filter"]["low_cutoff_hz"],
            high_cutoff_hz=config["filter"]["high_cutoff_hz"],
            order=config["filter"]["order"],
        )
    except RecordingRejectedError as error:
        participant_result["status"] = "rejected_recording"
        participant_result["recording_rejection_reason"] = error.reason
        participant_result["rejected_epoch_count"] = expected_epochs
        _record_rejections(
            cell_state["rejected_epochs"],
            cell_number,
            filename,
            class_label,
            storage_slot,
            group_key,
            expected_epochs,
            error.reason,
            error.channel_slots,
        )
        cell_state["participant_results"].append(participant_result)
        return
    except ValueError as error:
        participant_result["status"] = "rejected_recording"
        participant_result["recording_rejection_reason"] = "filter_error"
        participant_result["filter_error"] = str(error)
        participant_result["rejected_epoch_count"] = expected_epochs
        _record_rejections(
            cell_state["rejected_epochs"],
            cell_number,
            filename,
            class_label,
            storage_slot,
            group_key,
            expected_epochs,
            "filter_error",
            detail=str(error),
        )
        cell_state["participant_results"].append(participant_result)
        return

    participant_result["filtered_output_finite"] = bool(
        np.isfinite(filtered).all()
    )
    epoch_result = epoch_recording(
        filtered,
        sampling_rate_hz=sampling_rate,
        edge_trim_seconds=config["edge_trim_seconds"],
        epoch_duration_seconds=config["epoch_duration_seconds"],
    )
    participant_result["candidate_epoch_count"] = (
        epoch_result["candidate_epoch_count"]
    )
    participant_result["accepted_epoch_count"] = (
        epoch_result["accepted_epoch_count"]
    )
    participant_result["rejected_epoch_count"] = len(
        epoch_result["rejected_epochs"]
    )
    participant_result["discarded_tail_samples"] = (
        epoch_result["discarded_tail_samples"]
    )

    for rejection in epoch_result["rejected_epochs"]:
        cell_state["rejected_epochs"].append(
            {
                "cell": cell_number,
                "source_file": filename,
                "class_label": class_label,
                "participant_slot": storage_slot,
                "group_key": group_key,
                **rejection,
            }
        )

    if epoch_result["accepted_epoch_count"]:
        count = epoch_result["accepted_epoch_count"]
        cell_state["epochs"].append(epoch_result["epochs"])
        cell_state["class_labels"].extend([class_label] * count)
        cell_state["group_keys"].extend([group_key] * count)
        cell_state["source_files"].extend([filename] * count)
        cell_state["participant_slots"].extend([storage_slot] * count)
        cell_state["epoch_indices"].extend(
            epoch_result["accepted_epoch_indices"]
        )
    cell_state["participant_results"].append(participant_result)


def _summarize_cell(cell_number, cell_state, config):
    """Build class counts and grouped fold assignments for a single cell."""
    if not cell_state["epochs"]:
        raise ValueError(f"Cell {cell_number} produced no accepted epochs.")

    epochs = np.concatenate(cell_state["epochs"], axis=0)
    labels = cell_state["class_labels"]
    groups = cell_state["group_keys"]
    folds = assign_grouped_folds(
        groups, labels, n_splits=config["cv_folds"]
    )

    # Verify the leakage-prevention invariant before writing any output.
    group_fold_ids = {}
    for group_key, fold_id in zip(groups, folds.tolist()):
        group_fold_ids.setdefault(group_key, set()).add(fold_id)
    if any(len(group_folds) != 1 for group_folds in group_fold_ids.values()):
        raise RuntimeError("A provisional participant-slot group crossed folds.")

    per_class = {}
    for class_label in sorted(set(labels)):
        class_groups = {
            group_key for group_key, label in zip(groups, labels)
            if label == class_label
        }
        per_class[class_label] = {
            "participant_slots_with_epochs": len(class_groups),
            "epoch_count": labels.count(class_label),
        }

    summary = {
        "cell": cell_number,
        "shape": [int(size) for size in epochs.shape],
        "epoch_samples": int(epochs.shape[1]),
        "channel_count": int(epochs.shape[2]),
        "class_counts": per_class,
        "total_epoch_count": int(len(epochs)),
        "participant_slot_count_with_epochs": len(set(groups)),
        "rejected_epoch_count": len(cell_state["rejected_epochs"]),
        "grouped_fold_count": int(config["cv_folds"]),
        "groups_split_across_folds": False,
        "tail_samples_discarded_total": sum(
            result["discarded_tail_samples"]
            for result in cell_state["participant_results"]
        ),
    }

    output_path = (
        config["paths"]["output_directory"]
        / f"cell_{cell_number:02d}_epochs.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        epochs=epochs,
        class_labels=np.asarray(labels),
        group_keys=np.asarray(groups),
        source_files=np.asarray(cell_state["source_files"]),
        participant_slots=np.asarray(cell_state["participant_slots"], dtype=np.int64),
        epoch_indices=np.asarray(cell_state["epoch_indices"], dtype=np.int64),
        fold_ids=folds,
    )
    summary["output_file"] = str(output_path.relative_to(PROJECT_ROOT))
    return summary


def build_report(report):
    """Render the preprocessing results and limitations as a Markdown report."""
    config = report["configuration"]
    lines = [
        "# EEG Preprocessing Run Report",
        "",
        "## Scope and reproducibility",
        "",
        "- Original MAT files were read only and were not modified.",
        "- No model was trained and no features were extracted.",
        "- The five selected cells are processed and saved separately.",
        "- Group keys are provisional source-file/storage-slot keys, not "
        "participant IDs.",
        "- Filter/epoch tests passing does not establish scientific validity.",
        "",
        "## Configuration used",
        "",
        f"- Sampling rate: {config['sampling_rate_hz']} Hz.",
        f"- Filter: {config['filter']['order']}-order Butterworth SOS "
        f"band-pass, {config['filter']['low_cutoff_hz']} to "
        f"{config['filter']['high_cutoff_hz']} Hz.",
        "- Filtering is zero-phase (`sosfiltfilt`) along the sample axis.",
        "- Notch filter: **not applied**; no diagnostics in this run justified "
        "changing the configured decision.",
        f"- Edge trim: {config['edge_trim_seconds']} s from each end, after "
        "filtering.",
        f"- Epoching: {config['epoch_duration_seconds']} s, non-overlapping; "
        "incomplete trailing samples are discarded.",
        f"- Selected cells: {', '.join(str(cell) for cell in config['selected_cells'])}.",
        "- FADHD storage slot 7 is excluded in every selected cell before "
        "filtering, as requested based on the validation result.",
        "- The fourth-order filter order is an explicit implementation choice; "
        "the request did not specify an order.",
        "",
        "## Epoch counts by cell and class",
        "",
        "| Cell | Control slots | Control epochs | ADHD slots | ADHD epochs | "
        "Total epochs | Rejected epochs |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        control = cell["class_counts"].get(
            "Control", {"participant_slots_with_epochs": 0, "epoch_count": 0}
        )
        adhd = cell["class_counts"].get(
            "ADHD", {"participant_slots_with_epochs": 0, "epoch_count": 0}
        )
        lines.append(
            f"| {cell['cell']} | {control['participant_slots_with_epochs']} | "
            f"{control['epoch_count']} | "
            f"{adhd['participant_slots_with_epochs']} | {adhd['epoch_count']} | "
            f"{cell['total_epoch_count']} | {cell['rejected_epoch_count']} |"
        )

    lines.extend(
        [
            "",
            "Epoch arrays are saved in separate NPZ files:",
            "",
        ]
    )
    for cell in report["cells"]:
        lines.append(f"- `{cell['output_file']}`: shape `{cell['shape']}`.")

    lines.extend(
        [
            "",
            "## Participant-slot epoch counts",
            "",
            "One-based storage slots are listed below. They are local grouping "
            "keys only and do not assert identity across conditions or files.",
            "",
            "| Cell | File | Class | Storage slot | Status | Candidate | "
            "Accepted | Rejected | Tail samples discarded |",
            "|---:|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for participant in report["participant_level_counts"]:
        lines.append(
            f"| {participant['cell']} | {participant['source_file']} | "
            f"{participant['class_label']} | {participant['participant_slot']} | "
            f"{participant['status']} | {participant['candidate_epoch_count']} | "
            f"{participant['accepted_epoch_count']} | "
            f"{participant['rejected_epoch_count']} | "
            f"{participant['discarded_tail_samples']} |"
        )

    lines.extend(
        [
            "",
            "## Exclusions, rejected epochs, and quality control",
            "",
            f"- Whole-recording exclusions: **{len(report['excluded_recordings'])}** "
            "(FADHD storage slot 7, one exclusion for each selected cell).",
            f"- Rejected candidate epochs: **{len(report['rejected_epochs'])}**.",
            f"- NaN/Inf input samples in processed recordings: "
            f"**{report['quality_control']['nonfinite_sample_count']}**.",
            f"- Flatline channel traces among processed recordings: "
            f"**{report['quality_control']['flatline_channel_recordings']}**.",
            f"- Filter outputs finite: **{report['quality_control']['all_filter_outputs_finite']}**.",
            "- Per-participant input QC, rejection details, exclusion reasons, "
            "and all counts are also recorded in "
            "`reports/preprocessing_results.json`.",
        ]
    )
    if report["rejected_epochs"]:
        lines.extend(
            [
                "",
                "| Cell | File | Class | Slot | Epoch | Reason | Channel slots |",
                "|---:|---|---|---:|---:|---|---|",
            ]
        )
        for rejected in report["rejected_epochs"]:
            lines.append(
                f"| {rejected['cell']} | {rejected['source_file']} | "
                f"{rejected['class_label']} | {rejected['participant_slot']} | "
                f"{rejected['epoch_index']} | {rejected['reason']} | "
                f"{rejected.get('channel_slots', [])} |"
            )
    else:
        lines.append("- No candidate epochs were rejected.")

    lines.extend(
        [
            "",
            "## CV fold grouping (no model training)",
            "",
            f"Within each cell, epochs are assigned to {config['cv_folds']} "
            "deterministic folds by provisional group key and class. Every epoch "
            "for a given `(source file, storage slot)` key stays in one fold. "
            "Since cross-cell identity linkage is unverified, fold assignments "
            "are made independently per cell and must not be read as person "
            "matching across cells.",
            "",
            "## Scientific limitations",
            "",
            "This is a deterministic implementation of the requested settings, "
            "not evidence that the cutoffs, filter order, edge trim, epoch length, "
            "or rejection rules are scientifically optimal. No power-line "
            "diagnostic was used to justify a notch filter, so none was applied. "
            "Review the data provenance, channel mapping, participant linkage, "
            "and preprocessing choices before scientific interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pipeline(config_path):
    """Run preprocessing and write separate epoch arrays and run reports."""
    config = load_config(config_path)
    input_directory = config["paths"]["input_directory"]
    cell_states = {
        cell: {
            "epochs": [],
            "class_labels": [],
            "group_keys": [],
            "source_files": [],
            "participant_slots": [],
            "epoch_indices": [],
            "participant_results": [],
            "rejected_epochs": [],
        }
        for cell in config["selected_cells"]
    }
    participant_results = []
    excluded_recordings = []

    for filename, class_label in config["class_by_file"].items():
        variable_name, _, conditions, _ = load_cells(
            input_directory / filename
        )
        if variable_name != Path(filename).stem:
            raise ValueError(
                f"Unexpected MATLAB variable {variable_name!r} in {filename}."
            )
        if len(conditions) != 11:
            raise ValueError(f"{filename} must contain exactly 11 conditions.")

        excluded_slots = set(
            config["excluded_storage_slots"].get(filename, [])
        )
        for cell_number in config["selected_cells"]:
            recording_array = np.asarray(conditions[cell_number - 1])
            if recording_array.ndim != 3 or recording_array.shape[2] != 2:
                raise ValueError(
                    f"{filename} cell {cell_number} is not "
                    "subjects x samples x 2 channels."
                )
            cell_state = cell_states[cell_number]
            for slot_index in range(recording_array.shape[0]):
                storage_slot = slot_index + 1
                group_key = f"{filename}:slot-{storage_slot:03d}"
                raw_recording = recording_array[slot_index]
                candidate_count = candidate_epoch_count(
                    raw_recording.shape[0],
                    config["sampling_rate_hz"],
                    config["edge_trim_seconds"],
                    config["epoch_duration_seconds"],
                )

                if storage_slot in excluded_slots:
                    exclusion = {
                        "cell": cell_number,
                        "source_file": filename,
                        "class_label": class_label,
                        "participant_slot": storage_slot,
                        "group_key": group_key,
                        "reason": "excluded per request: flatline recording",
                    }
                    excluded_recordings.append(exclusion)
                    participant_result = {
                        **exclusion,
                        "status": "excluded",
                        "candidate_epoch_count": candidate_count,
                        "accepted_epoch_count": 0,
                        "rejected_epoch_count": 0,
                        "edge_trim_samples_per_side": round(
                            config["edge_trim_seconds"]
                            * config["sampling_rate_hz"]
                        ),
                        "discarded_tail_samples": 0,
                        "raw_quality_control": {
                            "not_reprocessed": True,
                            "note": (
                                "Excluded before filtering based on the prior "
                                "dataset validation."
                            ),
                        },
                        "filtered_output_finite": None,
                    }
                    participant_results.append(participant_result)
                    cell_state["participant_results"].append(participant_result)
                    continue

                before_count = len(cell_state["participant_results"])
                _add_cell_epoch(
                    cell_number,
                    filename,
                    class_label,
                    storage_slot,
                    group_key,
                    raw_recording,
                    config,
                    cell_state,
                )
                participant_results.append(
                    cell_state["participant_results"][before_count]
                )

    cell_summaries = [
        _summarize_cell(cell_number, cell_states[cell_number], config)
        for cell_number in config["selected_cells"]
    ]
    all_filter_status = [
        result["filtered_output_finite"]
        for result in participant_results
        if result["filtered_output_finite"] is not None
    ]
    raw_quality = [
        result["raw_quality_control"]
        for result in participant_results
        if result["status"] != "excluded"
    ]
    report = {
        "title": "EEG Preprocessing Run",
        "configuration": {
            key: value for key, value in config.items()
            if not key.startswith("_") and key != "paths"
        },
        "input_directory": str(input_directory),
        "participant_identity_note": (
            "Storage slots are provisional group keys. No cross-cell or "
            "cross-file participant identity linkage is asserted."
        ),
        "cells": cell_summaries,
        "participant_level_counts": participant_results,
        "excluded_recordings": excluded_recordings,
        "rejected_epochs": [
            rejection
            for cell in config["selected_cells"]
            for rejection in cell_states[cell]["rejected_epochs"]
        ],
        "quality_control": {
            "nonfinite_sample_count": sum(
                quality.get("nan_count", 0)
                + quality.get("positive_infinity_count", 0)
                + quality.get("negative_infinity_count", 0)
                for quality in raw_quality
            ),
            "flatline_channel_recordings": sum(
                len(quality.get("flatline_channel_slots", []))
                for quality in raw_quality
            ),
            "all_filter_outputs_finite": bool(all(all_filter_status)),
            "filter_output_recording_count": len(all_filter_status),
            "notch_filter_applied": False,
            "source_mat_files_modified": False,
        },
    }

    json_path = config["paths"]["report_json"]
    markdown_path = config["paths"]["report_markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_report(report), encoding="utf-8")
    print_terminal_summary(report, markdown_path, json_path)
    return report


def print_terminal_summary(report, markdown_path, json_path):
    """Print actual class-by-cell counts and report paths."""
    print("EEG preprocessing complete (no model training).")
    for cell in report["cells"]:
        class_counts = cell["class_counts"]
        control = class_counts.get("Control", {}).get("epoch_count", 0)
        adhd = class_counts.get("ADHD", {}).get("epoch_count", 0)
        print(
            f"Cell {cell['cell']}: Control={control} epochs, "
            f"ADHD={adhd} epochs, total={cell['total_epoch_count']}; "
            f"rejected={cell['rejected_epoch_count']}."
        )
    print(
        f"Excluded recordings={len(report['excluded_recordings'])}; "
        f"rejected epochs={len(report['rejected_epochs'])}; "
        f"nonfinite samples="
        f"{report['quality_control']['nonfinite_sample_count']}."
    )
    print(f"Markdown report: {markdown_path}")
    print(f"JSON details: {json_path}")


def main():
    """Command-line entry point; run from the project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "preprocessing.yaml"),
        help="Path to the YAML preprocessing configuration.",
    )
    arguments = parser.parse_args()
    run_pipeline(arguments.config)


if __name__ == "__main__":
    main()
