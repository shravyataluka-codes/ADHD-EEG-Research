"""
================================================================================
MODEL INTERFACE & ADAPTER LAYER
================================================================================
Decouples the Prototype UI from the underlying deep learning model implementation.

Communication Flow:
Prototype UI ──► ModelAdapter.predict(eeg_trial) ──► MockModel (Current)
                                                  ──► RealTransformer (Future)

Input Contract:
- Dictionary: {"eeg": np.ndarray with shape [385, 56], dtype float32}
- Or directly: np.ndarray [385, 56], dtype float32

Prediction Response Contract:
{
    "class": "ADHD",          # One of ["HC", "ADD", "ADHD"]
    "class_index": 2,         # Integer 0 (HC), 1 (ADD), 2 (ADHD)
    "probabilities": {
        "HC": 0.05,
        "ADD": 0.10,
        "ADHD": 0.85
    },
    "confidence": 0.85,
    "is_demo": True,          # Flag indicating demo/synthetic prediction
    "status_label": "DEMO / MOCK PREDICTION",
    "explanation": { ... }    # OPTIONAL. UI handles None/missing seamlessly.
}
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union

from .mock_model import MockEEGTransformerModel, EXPECTED_SHAPE, EXPECTED_DTYPE
from . import trained_model_placeholder

# Global switch: Set to True once the ML teammate supplies trained model weights
USE_REAL_MODEL: bool = False

# Singleton mock model instance for low-overhead inference
_MOCK_INSTANCE = MockEEGTransformerModel()
_REAL_MODEL_INSTANCE = None


class ModelAdapter:
    """
    Standardized adapter service dispatching inference requests to either
    the mock model (prototype phase) or real Transformer (production phase).
    """

    @staticmethod
    def extract_eeg_array(eeg_input: Union[Dict[str, Any], np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Extracts and normalizes the raw EEG array from diverse supported input formats.

        Args:
            eeg_input: Dict with 'eeg' key, or raw 2D numpy array, or pandas DataFrame.

        Returns:
            np.ndarray: Array formatted for inference.

        Raises:
            ValueError: If input format or shape cannot be parsed.
        """
        if isinstance(eeg_input, dict):
            if "eeg" not in eeg_input:
                raise ValueError("Input dictionary must contain the key 'eeg'.")
            arr = np.asarray(eeg_input["eeg"])
        elif isinstance(eeg_input, pd.DataFrame):
            # Exclude time/timestamp columns if present
            non_time_cols = [c for c in eeg_input.columns if c.lower() not in ['timestamp', 'time']]
            arr = eeg_input[non_time_cols].values
        elif isinstance(eeg_input, np.ndarray):
            arr = eeg_input
        else:
            try:
                arr = np.asarray(eeg_input)
            except Exception as e:
                raise ValueError(f"Unsupported input type '{type(eeg_input)}': {e}")

        # Ensure 2D tensor
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D EEG matrix of shape [385, 56], but got {arr.ndim}D shape {arr.shape}.")

        # Enforce float32
        if arr.dtype != EXPECTED_DTYPE:
            arr = arr.astype(EXPECTED_DTYPE)

        return arr

    @classmethod
    def predict(
        cls,
        eeg_input: Union[Dict[str, Any], np.ndarray, pd.DataFrame],
        include_optional_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Executes prediction through the active model service.

        Args:
            eeg_input: Input conforming to the ML contract.
            include_optional_explanation: Toggle for optional explainability payload.

        Returns:
            Dict[str, Any]: Standardized prediction response.
        """
        global _REAL_MODEL_INSTANCE

        # 1. Parse and validate input tensor shape
        trial_arr = cls.extract_eeg_array(eeg_input)

        if trial_arr.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Contract Shape Mismatch: Received shape {trial_arr.shape}. "
                f"Expected exactly {EXPECTED_SHAPE} (385 time positions x 56 channels)."
            )

        # 2. Dispatch to Real Model if enabled, otherwise fallback to Mock Model
        if USE_REAL_MODEL:
            try:
                if _REAL_MODEL_INSTANCE is None:
                    _REAL_MODEL_INSTANCE = trained_model_placeholder.load_trained_model()
                return trained_model_placeholder.predict_trained_model(_REAL_MODEL_INSTANCE, trial_arr)
            except NotImplementedError:
                # Fallback to mock service during prototype testing
                pass

        # 3. Default: Execute Mock Service
        return _MOCK_INSTANCE.predict(trial_arr, include_optional_explanation=include_optional_explanation)

    @classmethod
    def predict_safe(
        cls,
        eeg_input: Union[Dict[str, Any], np.ndarray, pd.DataFrame],
        include_optional_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Safe wrapper for UI components that catches validation exceptions
        and returns an error response dictionary rather than raising an unhandled exception.
        """
        try:
            res = cls.predict(eeg_input, include_optional_explanation=include_optional_explanation)
            res["status"] = "success"
            return res
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "is_demo": True,
                "status_label": "INPUT VALIDATION ERROR",
                "class": None,
                "class_index": None,
                "confidence": 0.0,
                "probabilities": {"HC": 0.0, "ADD": 0.0, "ADHD": 0.0},
                "explanation": None
            }

    @staticmethod
    def get_service_status() -> Dict[str, Any]:
        """Returns metadata about the active model service."""
        return {
            "is_real_model_active": USE_REAL_MODEL,
            "mode": "Real Transformer Active" if USE_REAL_MODEL else "DEMO / MOCK PREDICTION MODE",
            "expected_shape": list(EXPECTED_SHAPE),
            "expected_dtype": str(EXPECTED_DTYPE),
            "classes": ["HC", "ADD", "ADHD"],
            "disclaimer": "This is a research prototype. Predictions are synthetic outputs."
        }


def predict(eeg_trial: Union[Dict[str, Any], np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """Top-level functional interface for simple caller convenience."""
    return ModelAdapter.predict(eeg_trial)
