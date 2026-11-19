"""Verified contract and scientific limitations for the final Cell 1 model."""

SAMPLING_RATE_HZ = 256
CHANNEL_COUNT = 2
EPOCH_SECONDS = 4
EPOCH_SAMPLES = 1024
EDGE_TRIM_SECONDS = 1
FEATURE_COUNT = 40
CLASS_MAPPING = {"Control": 0, "ADHD": 1}
ADHD_THRESHOLD = 0.5
ARTIFACT_TYPE = "final_cell1_transformer_handcrafted_fusion"
INTEGRATION_STEPS = 128
LIMITATIONS = [
    "The two input channel slots represent array positions; their physical electrode identities are unverified.",
    "The model was trained on 553 Cell 1 epochs.",
    "Participant-independent generalization has not been established.",
    "No independent test evaluation has been established.",
    "This system is not a clinical diagnostic tool.",
    "Predicted probabilities are model outputs, not calibrated diagnostic confidence.",
    "Integrated Gradients are exploratory attributions, not causal or diagnostic evidence.",
]
