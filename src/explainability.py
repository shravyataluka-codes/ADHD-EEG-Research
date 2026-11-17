"""Exploratory Integrated Gradients explanations for final fusion inference."""

from dataclasses import dataclass

import numpy as np
import torch

from src.inference import PreparedRecording, prepare_recording


ZERO_NORMALIZED_BASELINE = (
    "Zero-valued normalized raw EEG and handcrafted features; this corresponds "
    "to the saved training normalization means. Attributions explain the model "
    "inputs after preprocessing and do not attribute the filtering operation."
)


@dataclass(frozen=True)
class EpochExplanation:
    """Prediction and joint-input attributions for one accepted epoch."""

    epoch_index: int
    start_sample: int
    stop_sample: int
    probabilities: np.ndarray
    predicted_label: int
    predicted_class: str
    logit_margin: float
    raw_attributions: np.ndarray
    raw_attribution_magnitudes: np.ndarray
    feature_attributions: np.ndarray
    feature_attribution_magnitudes: np.ndarray
    completeness_residual: float


@dataclass(frozen=True)
class RecordingExplanation:
    """Epoch explanations plus recording-level attribution summaries."""

    status: str
    feature_names: tuple
    epoch_explanations: tuple
    mean_signed_feature_attribution: np.ndarray | None
    mean_absolute_feature_attribution: np.ndarray | None
    mean_signed_raw_attribution: np.ndarray | None
    mean_absolute_raw_attribution: np.ndarray | None
    candidate_epoch_count: int
    accepted_epoch_count: int
    rejected_epoch_count: int
    rejected_epochs: tuple
    edge_trim_samples_per_side: int
    epoch_samples: int
    discarded_tail_samples: int | None
    integration_steps: int
    baseline_description: str


def _validate_integration_steps(integration_steps):
    if (
        isinstance(integration_steps, (bool, np.bool_))
        or not isinstance(integration_steps, (int, np.integer))
        or integration_steps < 1
    ):
        raise ValueError("integration_steps must be a positive integer.")
    return int(integration_steps)


def _recording_without_attributions(
    prepared: PreparedRecording,
    feature_names,
    integration_steps,
):
    return RecordingExplanation(
        status="no_valid_epochs",
        feature_names=tuple(feature_names),
        epoch_explanations=(),
        mean_signed_feature_attribution=None,
        mean_absolute_feature_attribution=None,
        mean_signed_raw_attribution=None,
        mean_absolute_raw_attribution=None,
        candidate_epoch_count=prepared.candidate_epoch_count,
        accepted_epoch_count=0,
        rejected_epoch_count=len(prepared.rejected_epochs),
        rejected_epochs=prepared.rejected_epochs,
        edge_trim_samples_per_side=prepared.edge_trim_samples_per_side,
        epoch_samples=prepared.epoch_samples,
        discarded_tail_samples=prepared.discarded_tail_samples,
        integration_steps=integration_steps,
        baseline_description=ZERO_NORMALIZED_BASELINE,
    )


def _margin(model, raw_eeg, handcrafted_features):
    logits = model(raw_eeg, handcrafted_features)
    if logits.ndim != 2 or logits.shape != (raw_eeg.shape[0], 2):
        raise RuntimeError(
            "Fusion model returned unexpected logits with shape "
            f"{tuple(logits.shape)}."
        )
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("Fusion model returned non-finite logits.")
    return logits[:, 1] - logits[:, 0], logits


def _explain_epoch(
    model,
    raw_eeg,
    handcrafted_features,
    epoch_index,
    start_sample,
    epoch_samples,
    integration_steps,
):
    raw_input = raw_eeg.detach()
    feature_input = handcrafted_features.detach()
    raw_delta = raw_input
    feature_delta = feature_input
    raw_gradient_sum = torch.zeros_like(raw_delta)
    feature_gradient_sum = torch.zeros_like(feature_delta)

    with torch.no_grad():
        actual_margin_tensor, actual_logits = _margin(
            model, raw_input, feature_input
        )
        baseline_raw = torch.zeros_like(raw_input)
        baseline_features = torch.zeros_like(feature_input)
        baseline_margin_tensor, _ = _margin(
            model, baseline_raw, baseline_features
        )
        probabilities = torch.softmax(actual_logits, dim=1)

    if not torch.isfinite(probabilities).all().item():
        raise FloatingPointError("Fusion model returned non-finite probabilities.")
    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise FloatingPointError("Two-class probabilities do not sum to one.")

    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
        integration_steps
    )
    alphas = torch.as_tensor(
        (quadrature_nodes + 1.0) / 2.0,
        device=raw_input.device,
        dtype=raw_input.dtype,
    )
    weights = (quadrature_weights / 2.0).tolist()
    for alpha, weight in zip(alphas, weights):
        interpolated_raw = (
            raw_delta * alpha
        ).detach().requires_grad_(True)
        interpolated_features = (
            feature_delta * alpha
        ).detach().requires_grad_(True)
        margin_tensor, _ = _margin(
            model,
            interpolated_raw,
            interpolated_features,
        )
        raw_gradient, feature_gradient = torch.autograd.grad(
            margin_tensor.sum(),
            (interpolated_raw, interpolated_features),
            create_graph=False,
            retain_graph=False,
        )
        raw_gradient_sum += raw_gradient * weight
        feature_gradient_sum += feature_gradient * weight
    raw_attributions = raw_delta * (
        raw_gradient_sum
    )
    feature_attributions = feature_delta * (
        feature_gradient_sum
    )
    raw_attributions = raw_attributions[0].detach().cpu().numpy().astype(
        np.float64
    )
    feature_attributions = (
        feature_attributions[0].detach().cpu().numpy().astype(np.float64)
    )
    probabilities_array = probabilities[0].detach().cpu().numpy().astype(
        np.float64
    )
    logit_margin = float(actual_margin_tensor[0].item())
    margin_difference = float(
        (actual_margin_tensor - baseline_margin_tensor)[0].item()
    )
    completeness_residual = margin_difference - float(
        raw_attributions.sum() + feature_attributions.sum()
    )
    if not (
        np.isfinite(raw_attributions).all()
        and np.isfinite(feature_attributions).all()
        and np.isfinite(probabilities_array).all()
        and np.isfinite(logit_margin)
        and np.isfinite(completeness_residual)
    ):
        raise FloatingPointError("Integrated Gradients produced non-finite values.")

    probability_adhd = float(probabilities_array[1])
    predicted_label = int(probability_adhd >= 0.5)
    predicted_class = "ADHD" if predicted_label else "Control"
    return EpochExplanation(
        epoch_index=int(epoch_index),
        start_sample=int(start_sample),
        stop_sample=int(start_sample + epoch_samples),
        probabilities=probabilities_array,
        predicted_label=predicted_label,
        predicted_class=predicted_class,
        logit_margin=logit_margin,
        raw_attributions=raw_attributions,
        raw_attribution_magnitudes=np.abs(raw_attributions),
        feature_attributions=feature_attributions,
        feature_attribution_magnitudes=np.abs(feature_attributions),
        completeness_residual=completeness_residual,
    )


def explain_recording(
    predictor,
    recording,
    sampling_rate_hz=256,
    integration_steps=128,
):
    """Explain unlabeled recording predictions using joint-input Integrated Gradients.

    Gauss-Legendre integration uses 128 steps by default. The baseline is zero
    in normalized model-input space. Probabilities are model outputs; attribution
    magnitudes are not confidence, and neither output is causal or clinical
    evidence.
    """
    steps = _validate_integration_steps(integration_steps)
    feature_names = tuple(getattr(predictor, "feature_names", ()))
    if len(feature_names) != 40 or len(set(feature_names)) != 40:
        raise ValueError(
            "The final checkpoint must provide 40 unique feature names."
        )
    class_mapping = getattr(predictor, "class_mapping", None)
    if class_mapping != {"Control": 0, "ADHD": 1}:
        raise ValueError("The predictor must use Control=0 and ADHD=1.")

    prepared = prepare_recording(
        predictor,
        recording,
        sampling_rate_hz=sampling_rate_hz,
    )
    if not prepared.epochs.shape[0]:
        return _recording_without_attributions(
            prepared,
            feature_names,
            steps,
        )

    raw_inputs, feature_inputs = predictor._prepare_model_inputs(
        prepared.epochs
    )
    if not (
        torch.isfinite(raw_inputs).all().item()
        and torch.isfinite(feature_inputs).all().item()
    ):
        raise FloatingPointError("Normalized model inputs contain non-finite values.")

    model = predictor.model
    was_training = model.training
    model.eval()
    epoch_explanations = []
    try:
        for row, epoch_index in enumerate(prepared.accepted_epoch_indices):
            epoch_start = prepared.edge_trim_samples_per_side + (
                epoch_index - 1
            ) * prepared.epoch_samples
            epoch_explanations.append(
                _explain_epoch(
                    model,
                    raw_inputs[row : row + 1],
                    feature_inputs[row : row + 1],
                    epoch_index=epoch_index,
                    start_sample=epoch_start,
                    epoch_samples=prepared.epoch_samples,
                    integration_steps=steps,
                )
            )
    finally:
        model.train(was_training)

    raw_attribution_stack = np.stack(
        [item.raw_attributions for item in epoch_explanations],
        axis=0,
    )
    feature_attribution_stack = np.stack(
        [item.feature_attributions for item in epoch_explanations],
        axis=0,
    )
    return RecordingExplanation(
        status="ok",
        feature_names=feature_names,
        epoch_explanations=tuple(epoch_explanations),
        mean_signed_feature_attribution=np.mean(
            feature_attribution_stack, axis=0
        ),
        mean_absolute_feature_attribution=np.mean(
            np.abs(feature_attribution_stack), axis=0
        ),
        mean_signed_raw_attribution=np.mean(raw_attribution_stack, axis=0),
        mean_absolute_raw_attribution=np.mean(
            np.abs(raw_attribution_stack), axis=0
        ),
        candidate_epoch_count=prepared.candidate_epoch_count,
        accepted_epoch_count=len(epoch_explanations),
        rejected_epoch_count=len(prepared.rejected_epochs),
        rejected_epochs=prepared.rejected_epochs,
        edge_trim_samples_per_side=prepared.edge_trim_samples_per_side,
        epoch_samples=prepared.epoch_samples,
        discarded_tail_samples=prepared.discarded_tail_samples,
        integration_steps=steps,
        baseline_description=ZERO_NORMALIZED_BASELINE,
    )
