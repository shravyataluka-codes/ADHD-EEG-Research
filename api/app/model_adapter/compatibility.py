"""Technical and scientific compatibility checks for Phase 2 recordings."""

import numpy as np

from .config import CHANNEL_COUNT, EPOCH_SAMPLES, EDGE_TRIM_SECONDS, SAMPLING_RATE_HZ


def validate_recording_compatibility(recording) -> list[str]:
    errors: list[str] = []
    if recording.import_status == "rejected":
        errors.append("Recording import was rejected: " + "; ".join(recording.errors))
    if recording.sampling_rate_hz is None:
        errors.append("Sampling rate is missing; provide 256 Hz at ingestion.")
    elif float(recording.sampling_rate_hz) != SAMPLING_RATE_HZ:
        errors.append(f"Sampling rate must be {SAMPLING_RATE_HZ} Hz.")
    if recording.n_channels != CHANNEL_COUNT:
        errors.append(f"Exactly {CHANNEL_COUNT} channels are required.")
    # Phase 2 readers normalize arrays to samples-by-channels while retaining
    # the source axis value (0 or 1) as provenance. Only unknown orientation is
    # unsafe; do not reinterpret or transpose the canonical array here.
    if recording.observed_time_axis not in (0, 1):
        errors.append("Time/sample orientation is unknown; re-ingest with an explicit time axis.")
    try:
        data = np.asarray(recording.array_3d if recording.array_3d is not None else recording.array)
        expected_ndim = 3 if recording.array_3d is not None else 2
        if data.ndim != expected_ndim or data.shape[-1] != CHANNEL_COUNT:
            errors.append("Signal must be [samples, 2] or [trials, samples, 2] in canonical orientation.")
        elif not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data) or not np.isfinite(data).all():
            errors.append("Signal samples must be finite real numeric values.")
        elif data.shape[-2] < 2 * EDGE_TRIM_SECONDS * SAMPLING_RATE_HZ + EPOCH_SAMPLES:
            errors.append("Each trial must contain at least 1,536 samples (6 seconds) for trimming and one epoch.")
    except (TypeError, ValueError):
        errors.append("Signal array is unavailable or invalid.")
    return errors
