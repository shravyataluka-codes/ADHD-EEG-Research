"""
================================================================================
INTEGRATION & CONTRACT TESTS FOR PROTOTYPE MODEL INTERFACE
================================================================================
Tests the contract boundary between the Prototype UI and Model Service.

Verifications:
1. Exact [385, 56] input is accepted.
2. Wrong shapes (e.g. [100, 10], [385, 4], 3D tensors) are rejected.
3. Wrong dtype (e.g., float64, int) is properly handled and converted/documented.
4. Probability values are valid [0.0, 1.0].
5. Probabilities sum to approximately 1.0.
6. Returned class matches the highest probability class (argmax).
7. Demo flag (is_demo == True) is present.
8. Missing/null optional explanation does not break the response schema.
9. Safe prediction wrapper returns error status without crashing.
================================================================================
"""

import sys
import os
import unittest
import numpy as np

# Ensure prototype package is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prototype.data.dummy_generator import generate_synthetic_eeg_trial
from prototype.model.mock_model import MockEEGTransformerModel, EXPECTED_SHAPE, EXPECTED_DTYPE
from prototype.model.model_interface import ModelAdapter, predict


class TestModelInterfaceContract(unittest.TestCase):
    """Test suite verifying adherence to the established ML interface contract."""

    def setUp(self):
        # Generate synthetic test trial conforming to [385, 56] float32 contract
        self.valid_synthetic_trial = generate_synthetic_eeg_trial(seed=42)

    def test_1_valid_contract_input_accepted(self):
        """1. Verify exact [385, 56] float32 input is accepted."""
        self.assertEqual(self.valid_synthetic_trial.shape, (385, 56))
        self.assertEqual(self.valid_synthetic_trial.dtype, np.float32)
        
        response = predict(self.valid_synthetic_trial)
        self.assertIsInstance(response, dict)
        self.assertIn("class", response)
        self.assertIn("probabilities", response)

    def test_2_wrong_shape_rejected(self):
        """2. Verify non-contract shapes are explicitly rejected."""
        wrong_shapes = [
            np.zeros((100, 56), dtype=np.float32),   # Truncated timepoints
            np.zeros((385, 16), dtype=np.float32),   # Insufficient channels
            np.zeros((385, 64), dtype=np.float32),   # Excess channels
            np.zeros((385,), dtype=np.float32),       # 1D array
            np.zeros((1, 385, 56), dtype=np.float32)  # 3D array
        ]
        
        for bad_input in wrong_shapes:
            with self.subTest(shape=bad_input.shape):
                # Standard predict should raise ValueError
                with self.assertRaises(ValueError):
                    ModelAdapter.predict(bad_input)

    def test_3_wrong_dtype_handled_and_cast(self):
        """3. Verify non-float32 dtypes (e.g. float64) are converted to float32 without crashing."""
        float64_input = np.zeros((385, 56), dtype=np.float64)
        extracted = ModelAdapter.extract_eeg_array(float64_input)
        self.assertEqual(extracted.dtype, np.float32)
        
        response = predict(float64_input)
        self.assertIn("class", response)

    def test_4_probabilities_are_valid_range(self):
        """4. Verify each class probability is within [0.0, 1.0]."""
        response = predict(self.valid_synthetic_trial)
        probs = response["probabilities"]
        
        for class_name, p in probs.items():
            self.assertGreaterEqual(p, 0.0, f"Probability for {class_name} is negative: {p}")
            self.assertLessEqual(p, 1.0, f"Probability for {class_name} exceeds 1.0: {p}")

    def test_5_probabilities_sum_to_approximately_one(self):
        """5. Verify probabilities sum to approximately 1.0."""
        response = predict(self.valid_synthetic_trial)
        total_p = sum(response["probabilities"].values())
        self.assertAlmostEqual(total_p, 1.0, places=2)

    def test_6_predicted_class_matches_highest_probability(self):
        """6. Verify returned class matches the argmax of probabilities."""
        response = predict(self.valid_synthetic_trial)
        probs = response["probabilities"]
        expected_class = max(probs, key=probs.get)
        
        self.assertEqual(response["class"], expected_class)
        self.assertEqual(response["confidence"], probs[expected_class])

    def test_7_demo_flag_is_present(self):
        """7. Verify the is_demo flag is explicitly True and warning label is present."""
        response = predict(self.valid_synthetic_trial)
        self.assertTrue(response.get("is_demo"), "Expected 'is_demo' flag to be True")
        self.assertIn("DEMO", response.get("status_label", ""))

    def test_8_missing_optional_explanation_does_not_break(self):
        """8. Verify missing or disabled explanation does not break response schema."""
        model = MockEEGTransformerModel()
        response_without_exp = model.predict(self.valid_synthetic_trial, include_optional_explanation=False)
        
        self.assertIsNone(response_without_exp["explanation"])
        self.assertIn("class", response_without_exp)
        self.assertIn("probabilities", response_without_exp)

    def test_9_safe_wrapper_handles_errors_gracefully(self):
        """9. Verify ModelAdapter.predict_safe catches errors without crashing UI."""
        bad_tensor = np.zeros((10, 10), dtype=np.float32)
        safe_response = ModelAdapter.predict_safe(bad_tensor)
        
        self.assertEqual(safe_response["status"], "error")
        self.assertIn("Contract Shape Mismatch", safe_response["error_message"])
        self.assertIsNone(safe_response["class"])


if __name__ == "__main__":
    unittest.main()
