"""Fold-specific inference for the Cell 1 EEG and feature-fusion model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from src.deep_fusion import EEGTransformerFusion
from src.features import extract_epoch_features
from src.epoching import candidate_epoch_count, epoch_recording
from src.preprocessing import bandpass_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "deep_learning"
    / "cell_01"
    / "feature_fusion"
    / "full"
    / "run_20261115T110540461128Z"
    / "C_transformer_plus_handcrafted"
    / "outer_fold_0"
    / "best_checkpoint.pt"
)
DEFAULT_FUSION_CONFIG = PROJECT_ROOT / "configs" / "deep_fusion_cell1.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "deep_models_cell1.yaml"
CLASS_MAPPING = {"Control": 0, "ADHD": 1}
ADHD_THRESHOLD = 0.5


@dataclass(frozen=True)
class InferenceResult:
    """Two-class probabilities and thresholded class predictions."""

    probabilities: np.ndarray
    predicted_labels: np.ndarray


@dataclass(frozen=True)
class EpochPrediction:
    """Prediction for one accepted, one-based epoch in the input recording."""

    epoch_index: int
    start_sample: int
    stop_sample: int
    probability_control: float
    probability_adhd: float
    predicted_label: int
    predicted_class: str


@dataclass(frozen=True)
class RecordingInferenceResult:
    """Accepted epoch predictions and a descriptive recording-level summary."""

    status: str
    epoch_predictions: tuple
    summary: dict
    candidate_epoch_count: int
    rejected_epochs: tuple
    edge_trim_samples_per_side: int
    epoch_samples: int
    discarded_tail_samples: int | None


@dataclass(frozen=True)
class PreparedRecording:
    """Preprocessed and epoched recording shared by prediction and XAI."""

    epochs: np.ndarray
    candidate_epoch_count: int
    accepted_epoch_indices: tuple
    rejected_epochs: tuple
    edge_trim_samples_per_side: int
    epoch_samples: int
    discarded_tail_samples: int | None


def _read_yaml(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)
    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration at {path} must contain a mapping.")
    return configuration


def _validated_checkpoint_vector(checkpoint, key, expected_size):
    if key not in checkpoint:
        raise ValueError(f"Checkpoint is missing required field {key!r}.")
    values = np.asarray(checkpoint[key], dtype=np.float64)
    if values.shape != (expected_size,) or not np.isfinite(values).all():
        raise ValueError(
            f"Checkpoint field {key!r} must contain {expected_size} finite values."
        )
    return values


class FoldFusionInference:
    """Load one trained fold model and predict from preprocessed EEG epochs."""

    def __init__(
        self,
        model,
        checkpoint,
        device,
        sampling_rate_hz=256,
        fold_id=0,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.fold_id = int(fold_id)
        self.raw_mean = _validated_checkpoint_vector(
            checkpoint, "raw_normalization_mean", 2
        )
        self.raw_scale = _validated_checkpoint_vector(
            checkpoint, "raw_normalization_scale", 2
        )
        self.feature_mean = _validated_checkpoint_vector(
            checkpoint, "feature_normalization_mean", 40
        )
        self.feature_scale = _validated_checkpoint_vector(
            checkpoint, "feature_normalization_scale", 40
        )
        if np.any(self.raw_scale <= 0) or np.any(self.feature_scale <= 0):
            raise ValueError("Checkpoint normalization scales must be positive.")
        constant_indices = np.asarray(
            checkpoint.get("constant_feature_indices", []), dtype=np.int64
        )
        if (
            constant_indices.ndim != 1
            or np.any(constant_indices < 0)
            or np.any(constant_indices >= 40)
            or np.unique(constant_indices).size != constant_indices.size
        ):
            raise ValueError("Checkpoint has invalid constant-feature indices.")
        if np.any(self.feature_scale[constant_indices] != 1.0):
            raise ValueError(
                "Constant feature columns must retain the training-time unit scale."
            )
        self.constant_feature_indices = constant_indices

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path=DEFAULT_CHECKPOINT,
        device="cpu",
        fusion_config_path=DEFAULT_FUSION_CONFIG,
        model_config_path=DEFAULT_MODEL_CONFIG,
        expected_fold=0,
    ):
        """Construct the configured architecture and strictly load a fold checkpoint."""
        device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        required_fields = {
            "model_state_dict",
            "raw_normalization_mean",
            "raw_normalization_scale",
            "feature_normalization_mean",
            "feature_normalization_scale",
            "constant_feature_indices",
            "metadata",
        }
        missing_fields = required_fields - set(checkpoint)
        if missing_fields:
            raise ValueError(
                f"Checkpoint is missing fields: {sorted(missing_fields)}."
            )
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint metadata must be a mapping.")
        if metadata.get("outer_fold") != expected_fold:
            raise ValueError(
                f"Expected outer fold {expected_fold}, got "
                f"{metadata.get('outer_fold')!r}."
            )

        fusion_configuration = _read_yaml(fusion_config_path)["fusion"]
        model_configuration = _read_yaml(model_config_path)
        transformer_configuration = model_configuration["models"][
            "eeg_transformer"
        ]
        if fusion_configuration["feature_count"] != 40:
            raise ValueError("Fusion configuration must specify 40 features.")
        if bool(fusion_configuration["freeze_transformer_backbone"]) != bool(
            metadata.get("freeze_transformer_backbone")
        ):
            raise ValueError(
                "Fusion configuration does not match checkpoint backbone metadata."
            )

        model = EEGTransformerFusion(
            in_channels=2,
            n_samples=1024,
            num_classes=2,
            feature_count=fusion_configuration["feature_count"],
            fusion_dim=fusion_configuration["projection_dim"],
            head_hidden_dim=fusion_configuration["classifier_hidden_dim"],
            dropout=fusion_configuration["dropout"],
            freeze_transformer_backbone=fusion_configuration[
                "freeze_transformer_backbone"
            ],
            transformer_config=transformer_configuration,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return cls(
            model=model,
            checkpoint=checkpoint,
            device=device,
            sampling_rate_hz=model_configuration["data"]["sampling_rate_hz"],
            fold_id=expected_fold,
        )

    def _prepare_model_inputs(self, epochs):
        """Build the exact normalized tensors used by prediction."""
        values = np.asarray(epochs)
        if values.ndim != 3 or values.shape[1:] != (1024, 2):
            raise ValueError(
                "EEG epochs must have shape [N, 1024, 2]; "
                f"got {values.shape}."
            )
        if values.shape[0] == 0:
            raise ValueError("At least one EEG epoch is required.")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError("EEG epochs must contain real numeric samples.")
        if not np.isfinite(values).all():
            raise ValueError("EEG epochs contain NaN or infinite samples.")

        feature_matrix, _ = extract_epoch_features(
            values,
            sampling_rate_hz=self.sampling_rate_hz,
        )
        normalized_raw = (
            (values.astype(np.float64, copy=False) - self.raw_mean)
            / self.raw_scale
        )
        model_raw = torch.from_numpy(
            np.transpose(normalized_raw.astype(np.float32), (0, 2, 1)).copy()
        ).to(self.device)

        normalized_features = (
            (feature_matrix - self.feature_mean) / self.feature_scale
        ).astype(np.float32)
        model_features = torch.from_numpy(normalized_features).to(self.device)
        return model_raw, model_features

    def predict(self, epochs, batch_size=16):
        """Predict labels and probabilities for [N, 1024, 2] preprocessed epochs."""
        values = np.asarray(epochs)
        model_raw, model_features = self._prepare_model_inputs(values)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        probability_batches = []
        with torch.no_grad():
            for start in range(0, values.shape[0], batch_size):
                logits = self.model(
                    model_raw[start : start + batch_size],
                    model_features[start : start + batch_size],
                )
                probability_batches.append(
                    torch.softmax(logits, dim=1).cpu().numpy()
                )
        probabilities = np.concatenate(probability_batches, axis=0)

        if probabilities.shape != (values.shape[0], 2):
            raise RuntimeError(
                "Fusion model returned an unexpected probability shape: "
                f"{probabilities.shape}."
            )
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Fusion model returned non-finite probabilities.")
        if not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
        ):
            raise FloatingPointError("Two-class probabilities do not sum to one.")

        predicted_labels = (
            probabilities[:, CLASS_MAPPING["ADHD"]] >= ADHD_THRESHOLD
        ).astype(np.int64)
        return InferenceResult(
            probabilities=probabilities,
            predicted_labels=predicted_labels,
        )


class FinalFusionInference(FoldFusionInference):
    """Load a self-describing all-data final fusion checkpoint."""

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cpu"):
        device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        required_fields = {
            "artifact_type",
            "model_state_dict",
            "raw_normalization_mean",
            "raw_normalization_scale",
            "feature_normalization_mean",
            "feature_normalization_scale",
            "constant_feature_indices",
            "feature_names",
            "class_mapping",
            "model_configuration",
            "fusion_configuration",
            "preprocessing_configuration",
            "dataset_metadata",
            "training_metadata",
            "random_seed",
            "metadata",
        }
        missing_fields = required_fields - set(checkpoint)
        if missing_fields:
            raise ValueError(
                f"Final checkpoint is missing fields: {sorted(missing_fields)}."
            )
        if checkpoint["artifact_type"] != "final_cell1_transformer_handcrafted_fusion":
            raise ValueError("Checkpoint is not a final Cell 1 fusion artifact.")
        if checkpoint["class_mapping"] != CLASS_MAPPING:
            raise ValueError("Final checkpoint class mapping is not Control=0/ADHD=1.")
        feature_names = checkpoint["feature_names"]
        if (
            not isinstance(feature_names, list)
            or len(feature_names) != 40
            or len(set(feature_names)) != 40
        ):
            raise ValueError("Final checkpoint must contain 40 unique feature names.")

        model_configuration = checkpoint["model_configuration"]
        fusion_configuration = checkpoint["fusion_configuration"]
        if (
            model_configuration.get("in_channels") != 2
            or model_configuration.get("n_samples") != 1024
            or model_configuration.get("num_classes") != 2
        ):
            raise ValueError("Final checkpoint has incompatible model input dimensions.")
        model = EEGTransformerFusion(
            in_channels=model_configuration["in_channels"],
            n_samples=model_configuration["n_samples"],
            num_classes=model_configuration["num_classes"],
            feature_count=fusion_configuration["feature_count"],
            fusion_dim=fusion_configuration["projection_dim"],
            head_hidden_dim=fusion_configuration["classifier_hidden_dim"],
            dropout=fusion_configuration["dropout"],
            freeze_transformer_backbone=fusion_configuration[
                "freeze_transformer_backbone"
            ],
            transformer_config=model_configuration["transformer"],
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        predictor = cls(
            model=model,
            checkpoint=checkpoint,
            device=device,
            sampling_rate_hz=model_configuration["sampling_rate_hz"],
            fold_id=-1,
        )
        predictor.feature_names = tuple(feature_names)
        predictor.class_mapping = dict(checkpoint["class_mapping"])
        predictor.random_seed = int(checkpoint["random_seed"])
        predictor.preprocessing_configuration = dict(
            checkpoint["preprocessing_configuration"]
        )
        return predictor


def _empty_recording_summary(rejected_count=0):
    return {
        "valid_epoch_count": 0,
        "rejected_epoch_count": int(rejected_count),
        "predicted_epoch_counts": {"Control": 0, "ADHD": 0},
        "majority_vote_label": None,
        "majority_vote_class": None,
        "mean_probabilities": None,
    }


def prepare_recording(predictor, recording, sampling_rate_hz=256):
    """Apply the checkpoint preprocessing and return accepted unlabeled epochs."""
    preprocessing = getattr(predictor, "preprocessing_configuration", None)
    if not isinstance(preprocessing, dict):
        raise TypeError(
            "Recording inference requires a final checkpoint with embedded "
            "preprocessing configuration."
        )

    data = np.asarray(recording)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != 2:
        raise ValueError(
            "Raw recording must have shape [samples, 2] with at least one sample; "
            f"got {data.shape}."
        )
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise ValueError("Raw recording must contain real numeric samples.")
    if not np.isfinite(data).all():
        raise ValueError("Raw recording contains NaN or infinite samples.")
    if (
        isinstance(sampling_rate_hz, (bool, np.bool_))
        or not isinstance(sampling_rate_hz, (int, float, np.number))
        or not np.isfinite(sampling_rate_hz)
    ):
        raise ValueError("Sampling rate must be a finite numeric value in Hz.")

    configured_rate = preprocessing["sampling_rate_hz"]
    if float(sampling_rate_hz) != float(configured_rate):
        raise ValueError(
            f"Sampling rate must match the checkpoint preprocessing configuration "
            f"({configured_rate} Hz); got {sampling_rate_hz} Hz."
        )
    filter_configuration = preprocessing["filter"]
    if (
        filter_configuration.get("family") != "butterworth"
        or filter_configuration.get("zero_phase") is not True
        or filter_configuration.get("implementation") != "sosfiltfilt"
        or preprocessing.get("notch_filter", {}).get("enabled", False)
        or preprocessing.get("epoch_overlap_seconds") != 0
    ):
        raise ValueError(
            "Checkpoint preprocessing configuration is incompatible with the "
            "implemented zero-phase, non-overlapping pipeline."
        )

    edge_trim_seconds = preprocessing["edge_trim_seconds"]
    epoch_duration_seconds = preprocessing["epoch_duration_seconds"]
    sampling_rate = int(configured_rate)
    trim_samples = int(round(edge_trim_seconds * sampling_rate))
    epoch_samples = int(round(epoch_duration_seconds * sampling_rate))
    candidate_count = candidate_epoch_count(
        data.shape[0],
        sampling_rate,
        edge_trim_seconds,
        epoch_duration_seconds,
    )
    if candidate_count == 0:
        return PreparedRecording(
            epochs=np.empty((0, epoch_samples, 2), dtype=np.float64),
            candidate_epoch_count=0,
            accepted_epoch_indices=(),
            rejected_epochs=(),
            edge_trim_samples_per_side=trim_samples,
            epoch_samples=epoch_samples,
            discarded_tail_samples=None,
        )

    filtered = bandpass_filter(
        data,
        sampling_rate_hz=sampling_rate,
        low_cutoff_hz=filter_configuration["low_cutoff_hz"],
        high_cutoff_hz=filter_configuration["high_cutoff_hz"],
        order=filter_configuration["order"],
    )
    epoch_result = epoch_recording(
        filtered,
        sampling_rate_hz=sampling_rate,
        edge_trim_seconds=edge_trim_seconds,
        epoch_duration_seconds=epoch_duration_seconds,
    )
    return PreparedRecording(
        epochs=epoch_result["epochs"],
        candidate_epoch_count=int(epoch_result["candidate_epoch_count"]),
        accepted_epoch_indices=tuple(epoch_result["accepted_epoch_indices"]),
        rejected_epochs=tuple(epoch_result["rejected_epochs"]),
        edge_trim_samples_per_side=int(
            epoch_result["edge_trim_samples_per_side"]
        ),
        epoch_samples=int(epoch_result["epoch_samples"]),
        discarded_tail_samples=int(epoch_result["discarded_tail_samples"]),
    )


def predict_recording(
    predictor,
    recording,
    sampling_rate_hz=256,
    batch_size=16,
):
    """Preprocess one unlabeled samples-by-two-slots recording and predict epochs.

    Channel-slot identities are unverified. Returned labels are model outputs,
    not clinical assessments.
    """
    prepared = prepare_recording(
        predictor,
        recording,
        sampling_rate_hz=sampling_rate_hz,
    )
    if prepared.candidate_epoch_count == 0:
        return RecordingInferenceResult(
            status="no_valid_epochs",
            epoch_predictions=(),
            summary=_empty_recording_summary(),
            candidate_epoch_count=0,
            rejected_epochs=(),
            edge_trim_samples_per_side=prepared.edge_trim_samples_per_side,
            epoch_samples=prepared.epoch_samples,
            discarded_tail_samples=None,
        )

    if prepared.epochs.shape[0]:
        predictions = predictor.predict(prepared.epochs, batch_size=batch_size)
        epoch_predictions = []
        for row, epoch_index in enumerate(prepared.accepted_epoch_indices):
            start_sample = prepared.edge_trim_samples_per_side + (
                epoch_index - 1
            ) * prepared.epoch_samples
            predicted_label = int(predictions.predicted_labels[row])
            predicted_class = (
                "ADHD" if predicted_label == CLASS_MAPPING["ADHD"] else "Control"
            )
            epoch_predictions.append(
                EpochPrediction(
                    epoch_index=int(epoch_index),
                    start_sample=int(start_sample),
                    stop_sample=int(start_sample + prepared.epoch_samples),
                    probability_control=float(predictions.probabilities[row, 0]),
                    probability_adhd=float(predictions.probabilities[row, 1]),
                    predicted_label=predicted_label,
                    predicted_class=predicted_class,
                )
            )
        counts = {
            "Control": sum(
                prediction.predicted_label == CLASS_MAPPING["Control"]
                for prediction in epoch_predictions
            ),
            "ADHD": sum(
                prediction.predicted_label == CLASS_MAPPING["ADHD"]
                for prediction in epoch_predictions
            ),
        }
        if counts["Control"] == counts["ADHD"]:
            majority_label = None
            majority_class = None
        else:
            majority_class = max(counts, key=counts.get)
            majority_label = CLASS_MAPPING[majority_class]
        summary = {
            "valid_epoch_count": len(epoch_predictions),
            "rejected_epoch_count": len(prepared.rejected_epochs),
            "predicted_epoch_counts": counts,
            "majority_vote_label": majority_label,
            "majority_vote_class": majority_class,
            "mean_probabilities": {
                "Control": float(np.mean(predictions.probabilities[:, 0])),
                "ADHD": float(np.mean(predictions.probabilities[:, 1])),
            },
        }
    else:
        epoch_predictions = []
        summary = _empty_recording_summary(
            rejected_count=len(prepared.rejected_epochs)
        )

    return RecordingInferenceResult(
        status=(
            "ok"
            if prepared.epochs.shape[0]
            else "no_valid_epochs"
        ),
        epoch_predictions=tuple(epoch_predictions),
        summary=summary,
        candidate_epoch_count=prepared.candidate_epoch_count,
        rejected_epochs=prepared.rejected_epochs,
        edge_trim_samples_per_side=prepared.edge_trim_samples_per_side,
        epoch_samples=prepared.epoch_samples,
        discarded_tail_samples=prepared.discarded_tail_samples,
    )
