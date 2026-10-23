"""Command-line entry point for the gated EEG preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .preprocessing import (
    config_from_mapping,
    process_dataset,
    reviews_from_mapping,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ica-decisions", type=Path, required=True)
    arguments = parser.parse_args()

    config = config_from_mapping(_read_json(arguments.config))
    reviews = reviews_from_mapping(_read_json(arguments.ica_decisions))
    metadata = process_dataset(
        arguments.dataset_dir,
        arguments.output_dir,
        config,
        reviews,
    )
    print(
        f"Processed {metadata['trial_count']} trials from "
        f"{metadata['subject_count']} subjects into {arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
