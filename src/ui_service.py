"""Input parsing and thin orchestration for the research UI."""

import csv
from dataclasses import dataclass
from io import BytesIO, StringIO

import numpy as np

from src.epoching import candidate_epoch_count
from src.explainability import RecordingExplanation, explain_recording
from src.inference import RecordingInferenceResult, predict_recording


class UIInputError(ValueError):
    """A user-provided file or sampling rate is invalid for this model."""


@dataclass(frozen=True)
class UIAnalysisResult:
    """Prediction and explanation outputs without user diagnosis labels."""

    inference: RecordingInferenceResult
    explanation: RecordingExplanation
    sampling_rate_hz: float


def _uploaded_bytes(upload):
    if isinstance(upload, bytes):
        return upload
    if isinstance(upload, bytearray):
        return bytes(upload)
    if hasattr(upload, "getvalue"):
        content = upload.getvalue()
    elif hasattr(upload, "read"):
        content = upload.read()
    else:
        raise UIInputError("The uploaded file could not be read.")
    if not isinstance(content, (bytes, bytearray)):
        raise UIInputError("The uploaded file must provide binary content.")
    return bytes(content)


def _validate_recording_array(recording):
    values = np.asarray(recording)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        raise UIInputError(
            "Recording must have shape [samples, 2] with at least one sample."
        )
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise UIInputError("Recording must contain real numeric values.")
    try:
        values = values.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise UIInputError("Recording values cannot be represented as real numbers.") from error
    if not np.isfinite(values).all():
        raise UIInputError("Recording contains NaN or infinite values.")
    return values


def _is_numeric_cell(value):
    try:
        float(value.strip())
    except (AttributeError, ValueError, OverflowError):
        return False
    return True


def parse_uploaded_recording(filename, upload):
    """Parse CSV or NPY input into a samples-by-two-channel-slots array.

    CSV files may have one optional, non-numeric header row. MAT files and
    object/pickled NPY data are intentionally unsupported.
    """
    suffix = str(filename).lower().rsplit(".", 1)[-1] if "." in str(filename) else ""
    content = _uploaded_bytes(upload)
    if suffix == "npy":
        try:
            values = np.load(BytesIO(content), allow_pickle=False)
        except (ValueError, EOFError, OSError) as error:
            raise UIInputError(f"Could not read NPY recording: {error}") from error
        if not isinstance(values, np.ndarray):
            close = getattr(values, "close", None)
            if close is not None:
                close()
            raise UIInputError("Upload a single NPY array, not an NPZ archive.")
        return _validate_recording_array(values)

    if suffix != "csv":
        raise UIInputError("Unsupported file type. Upload a .csv or .npy recording.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UIInputError("CSV file must be UTF-8 encoded.") from error

    rows = []
    first_content_row_seen = False
    try:
        reader = csv.reader(StringIO(text))
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                raise UIInputError(
                    f"CSV line {line_number} must contain exactly two columns."
                )
            try:
                numeric_row = [float(value.strip()) for value in row]
            except ValueError as error:
                if not first_content_row_seen and not rows:
                    if all(value.strip() for value in row) and not any(
                        _is_numeric_cell(value) for value in row
                    ):
                        first_content_row_seen = True
                        continue
                raise UIInputError(
                    f"CSV line {line_number} must contain two numeric values."
                ) from error
            first_content_row_seen = True
            rows.append(numeric_row)
    except csv.Error as error:
        raise UIInputError(f"Could not parse CSV file: {error}") from error

    if not rows:
        raise UIInputError("CSV file contains no numeric recording rows.")
    return _validate_recording_array(np.asarray(rows, dtype=np.float64))


def parse_sampling_rate(value):
    """Parse a required user-entered sampling rate in Hz."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise UIInputError("Enter the recording sampling rate in Hz.")
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise UIInputError("Sampling rate must be a numeric value in Hz.") from error
    if not np.isfinite(rate) or rate <= 0:
        raise UIInputError("Sampling rate must be a positive finite value in Hz.")
    return rate


def validate_recording_for_model(recording, sampling_rate_hz, predictor):
    """Validate shape, values, rate, duration, and flatline channel slots."""
    values = _validate_recording_array(recording)
    try:
        rate = parse_sampling_rate(sampling_rate_hz)
    except UIInputError:
        raise

    preprocessing = getattr(predictor, "preprocessing_configuration", None)
    if not isinstance(preprocessing, dict):
        raise UIInputError(
            "The loaded model does not provide preprocessing metadata."
        )
    configured_rate = preprocessing.get("sampling_rate_hz")
    if rate != float(configured_rate):
        raise UIInputError(
            f"Sampling rate must match this model's preprocessing rate "
            f"({configured_rate} Hz)."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        ranges = np.ptp(values, axis=0)
        variances = np.var(values, axis=0)
    if not np.isfinite(ranges).all() or not np.isfinite(variances).all():
        raise UIInputError(
            "Recording channel values exceed the supported numeric range."
        )
    invalid_slots = np.flatnonzero((ranges == 0) | (variances == 0)) + 1
    if invalid_slots.size:
        slots = ", ".join(str(int(slot)) for slot in invalid_slots)
        raise UIInputError(
            f"Flatline or zero-variance channel slot(s): {slots}. "
            "Both channel slots must contain varying EEG values."
        )

    try:
        candidate_count = candidate_epoch_count(
            values.shape[0],
            configured_rate,
            preprocessing["edge_trim_seconds"],
            preprocessing["epoch_duration_seconds"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise UIInputError(
            "The checkpoint contains invalid recording-duration settings."
        ) from error
    if candidate_count < 1:
        minimum_seconds = (
            2 * preprocessing["edge_trim_seconds"]
            + preprocessing["epoch_duration_seconds"]
        )
        raise UIInputError(
            f"Recording is too short. At least {minimum_seconds:g} seconds "
            f"are required at {configured_rate} Hz to produce one complete epoch."
        )
    return values


def analyze_recording(
    predictor,
    recording,
    sampling_rate_hz,
    integration_steps=128,
):
    """Run existing prediction and XAI entry points without using labels."""
    values = validate_recording_for_model(
        recording,
        sampling_rate_hz,
        predictor,
    )
    inference = predict_recording(
        predictor,
        values,
        sampling_rate_hz=sampling_rate_hz,
    )
    explanation = explain_recording(
        predictor,
        values,
        sampling_rate_hz=sampling_rate_hz,
        integration_steps=integration_steps,
    )
    if len(inference.epoch_predictions) != len(explanation.epoch_explanations):
        raise RuntimeError(
            "Prediction and explanation returned different accepted epoch counts."
        )
    for prediction, epoch_explanation in zip(
        inference.epoch_predictions,
        explanation.epoch_explanations,
    ):
        if prediction.epoch_index != epoch_explanation.epoch_index:
            raise RuntimeError(
                "Prediction and explanation epoch indices do not align."
            )
        expected_probabilities = np.asarray(
            [prediction.probability_control, prediction.probability_adhd]
        )
        if not np.allclose(
            expected_probabilities,
            epoch_explanation.probabilities,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Prediction and explanation probabilities do not agree."
            )
    return UIAnalysisResult(
        inference=inference,
        explanation=explanation,
        sampling_rate_hz=float(sampling_rate_hz),
    )
