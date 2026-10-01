"""
================================================================================
MOCK MODEL SERVICE FOR PROTOTYPE DEVELOPMENT
================================================================================
Provides deterministic, synthetic predictions conforming to the ML contract:
- Validates trial shape [385, 56]
- Validates data type (float32)
- Returns class ("HC", "ADD", "ADHD"), class_index, probabilities, confidence,
  demo flag, and an optional interpretability explanation dictionary.

MEDICAL DISCLAIMER:
This mock model produces purely synthetic values for interface development.
It does NOT perform clinical diagnosis or real EEG analysis.
================================================================================
"""

import numpy as np
from typing import Dict, Any, Optional, Union

# Class definitions for the 3-class ADHD classification task
CLASS_NAMES = ["HC", "ADD", "ADHD"]
CLASS_INDICES = {"HC": 0, "ADD": 1, "ADHD": 2}

EXPECTED_SHAPE = (385, 56)
EXPECTED_DTYPE = np.float32


class MockEEGTransformerModel:
    """
    Mock prediction service implementing the established ML contract.
    """

    def __init__(self, default_class: str = "ADHD"):
        if default_class not in CLASS_NAMES:
            raise ValueError(f"Invalid default class '{default_class}'. Must be one of {CLASS_NAMES}")
        self.default_class = default_class

    def validate_trial(self, trial_data: np.ndarray) -> np.ndarray:
        """
        Validates that the input tensor adheres to [385, 56] float32 contract.

        Args:
            trial_data: Input EEG trial array.

        Returns:
            np.ndarray: Validated array with dtype float32.

        Raises:
            ValueError: If shape does not match (385, 56).
            TypeError: If input cannot be converted to float32.
        """
        if not isinstance(trial_data, np.ndarray):
            try:
                trial_data = np.asarray(trial_data, dtype=EXPECTED_DTYPE)
            except Exception as e:
                raise TypeError(f"Could not convert input to numpy array: {e}")

        if trial_data.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Invalid EEG trial shape {trial_data.shape}. "
                f"Expected exact shape {EXPECTED_SHAPE} (385 timepoints, 56 channels)."
            )

        if trial_data.dtype != EXPECTED_DTYPE:
            # Cast with warning to adhere to float32 contract
            trial_data = trial_data.astype(EXPECTED_DTYPE)

        return trial_data

    def predict(
        self,
        trial_data: np.ndarray,
        include_optional_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Performs mock inference on a single EEG trial.

        Args:
            trial_data: Array of shape [385, 56] and dtype float32.
            include_optional_explanation: Whether to attach optional interpretability info.

        Returns:
            Dict[str, Any] conforming to the ML contract:
            {
                "class": "ADHD",
                "class_index": 2,
                "probabilities": {"HC": 0.05, "ADD": 0.10, "ADHD": 0.85},
                "confidence": 0.85,
                "is_demo": True,
                "status_label": "DEMO / MOCK PREDICTION",
                "explanation": Optional[Dict]
            }
        """
        # 1. Enforce strict shape and type validation
        validated_arr = self.validate_trial(trial_data)

        # 2. Derive deterministic dummy probabilities from input signal statistics
        # (This ensures the UI reacts predictably to different synthetic inputs)
        signal_mean = float(np.mean(validated_arr))
        signal_std = float(np.std(validated_arr))

        # Modulate mock probability distribution deterministically
        if signal_std > 0.35:
            # Higher variance pattern -> ADHD demo profile
            probs = {"HC": 0.05, "ADD": 0.10, "ADHD": 0.85}
        elif signal_mean < -0.05:
            # Negative shift pattern -> ADD demo profile
            probs = {"HC": 0.15, "ADD": 0.70, "ADHD": 0.15}
        else:
            # Default or baseline pattern -> Healthy Control demo profile
            probs = {"HC": 0.80, "ADD": 0.12, "ADHD": 0.08}

        # Normalize probabilities so they sum to 1.0
        total_p = sum(probs.values())
        probs = {k: round(v / total_p, 4) for k, v in probs.items()}

        # Identify predicted class and confidence
        predicted_class = max(probs, key=probs.get)
        confidence = probs[predicted_class]
        class_index = CLASS_INDICES[predicted_class]

        # 3. Construct optional explanation if requested
        explanation = None
        if include_optional_explanation:
            # 56 synthetic channel importance scores for optional prototype visualization
            synthetic_importance = [
                round(float(0.1 + 0.9 * abs(np.sin(i * 0.2))), 3)
                for i in range(EXPECTED_SHAPE[1])
            ]
            explanation = {
                "type": "Optional Synthetic Attention & Channel Importance",
                "channel_importance": synthetic_importance,
                "note": "Optional demo field. Real Transformer may provide attention topomaps."
            }

        response = {
            "class": predicted_class,
            "class_index": class_index,
            "probabilities": probs,
            "confidence": confidence,
            "is_demo": True,
            "status_label": "DEMO / MOCK PREDICTION",
            "disclaimer": "This is a synthetic mock prediction for prototype validation. Not a medical diagnosis.",
            "explanation": explanation
        }

        return response
