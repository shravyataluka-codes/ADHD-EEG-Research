"""
================================================================================
PLACEHOLDER FOR FUTURE REAL EEG TRANSFORMER
================================================================================
This file is reserved for the ML engineer to integrate the trained Transformer.

DO NOT IMPLEMENT THE REAL TRANSFORMER IN THIS BRANCH.
Real model weights, training scripts, and preprocessing pipelines are developed
in parallel on dedicated ML research branches.

When the real Transformer is ready:
1. Provide the serialized model weights (e.g., eeg_transformer_best.pt).
2. Implement load_trained_model() using PyTorch / TorchScript / ONNX.
3. Implement predict_trained_model() to accept [385, 56] float32 input and return
   the standardized prediction dictionary.
================================================================================
"""

from typing import Dict, Any, Optional
import numpy as np

MODEL_PATH: Optional[str] = None


def load_trained_model(model_path: Optional[str] = None):
    """
    Loads the trained EEG Transformer model from disk.
    
    Raises:
        NotImplementedError: Real model is not implemented in this branch.
    """
    raise NotImplementedError(
        "Trained EEG Transformer has not been connected yet. "
        "The real model pipeline is being developed on a separate ML branch."
    )


def predict_trained_model(model: Any, trial_data: np.ndarray) -> Dict[str, Any]:
    """
    Executes inference using the real trained Transformer model.
    
    Args:
        model: Loaded model instance.
        trial_data: Array of shape [385, 56], dtype float32.
        
    Raises:
        NotImplementedError: Real model is not implemented in this branch.
    """
    raise NotImplementedError(
        "Trained EEG Transformer inference routine is pending ML pipeline completion."
    )
