"""Model package initialization."""
from .mock_model import MockEEGTransformerModel, CLASS_NAMES, CLASS_INDICES, EXPECTED_SHAPE, EXPECTED_DTYPE
from .model_interface import ModelAdapter, predict, USE_REAL_MODEL
from . import trained_model_placeholder

__all__ = [
    "MockEEGTransformerModel",
    "ModelAdapter",
    "predict",
    "CLASS_NAMES",
    "CLASS_INDICES",
    "EXPECTED_SHAPE",
    "EXPECTED_DTYPE",
    "USE_REAL_MODEL",
    "trained_model_placeholder"
]
