# Prototype ML Model Integration Contract

> **Research & Prototype Notice:**  
> This specification documents the boundary between the **ADHD EEG Research Prototype (Application Layer)** and the **Machine Learning Model Service**.  
> **All outputs in the current prototype phase are synthetic/mock values for software interface stabilization.** This prototype does not perform clinical diagnosis.

---

## 1. Purpose

The objective of this contract is to allow the application team to develop, test, and stabilize the interactive research prototype independently while the ML research engineer develops the EEG signal preprocessing, artifact rejection, and EEG Transformer architecture in parallel.

By establishing an explicit input/output contract:
- The UI communicates strictly with an abstraction layer (`ModelAdapter`).
- The ML engineer can train the model independently and integrate it without requiring UI rewrites.
- The prototype works out-of-the-box using synthetic/dummy EEG data.

---

## 2. Input Contract

The model interface accepts a single EEG trial representing an individual segment/epoch of neuroelectric recording.

### 2.1 EEG Tensor Specifications

| Property | Value / Specification | Notes |
| :--- | :--- | :--- |
| **Shape** | `[385, 56]` | 385 sequential time positions $\times$ 56 electrode channels |
| **Data Type** | `float32` | Standard 32-bit floating point |
| **Dimension 0** | Time sequence ($T = 385$) | Corresponds to ~1.5 seconds at 256 Hz |
| **Dimension 1** | Spatial channels ($C = 56$) | 56 scalp electrodes |
| **Data Structure** | Python `dict` or NumPy array | When dict is used, key is `"eeg"` |

### 2.2 Input Format Options

```json
{
    "eeg": [
        [0.124, -0.045, "...", 0.098],
        [0.140, -0.038, "...", 0.105],
        "..."
    ]
}
```
*(Array shape must be exactly 385 rows by 56 columns of float32 values. No real EEG data is included in this document).*

---

## 3. Prediction Response Contract

The model service responds with a JSON-compatible dictionary adhering to the schema below.

### 3.1 Standard Response Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `class` | `str` | Predicted diagnostic class: `"HC"`, `"ADD"`, or `"ADHD"`. |
| `class_index` | `int` | Integer encoding: `0` (HC), `1` (ADD), `2` (ADHD). |
| `probabilities` | `dict[str, float]` | Normalized class probabilities summing to $1.0 \pm 0.01$. |
| `confidence` | `float` | Highest probability value (`max(probabilities.values())`). |
| `is_demo` | `bool` | `true` in prototype/mock mode; `false` when real model is active. |
| `status_label` | `str` | Descriptive label (e.g., `"DEMO / MOCK PREDICTION"`). |
| `disclaimer` | `str` | Mandatory research and non-clinical disclaimer. |
| `explanation` | `dict` or `null` | **OPTIONAL** explainability payload (see Section 4). |

### 3.2 Target Diagnostic Classes

1. **`HC` (Healthy Control):** Index `0` — Neurotypical control profile.
2. **`ADD` (Attention Deficit Disorder):** Index `1` — Inattentive subtype pattern.
3. **`ADHD` (Attention Deficit/Hyperactivity Disorder):** Index `2` — Combined/hyperactive subtype pattern.

---

## 4. Optional Explainability Payload

The `explanation` field is completely **optional**. The prototype handles `null`, `None`, or missing explanation fields gracefully without throwing errors.

When implemented by the future Transformer, it may optionally contain:

```json
{
    "explanation": {
        "attention_summary": "Top-activated attention heads across timepoints",
        "channel_importance": [0.85, 0.42, "...", 0.61],
        "temporal_saliency": null
    }
}
```

*Note: The prototype does not require this field to function.*

---

## 5. Error Response Contract

If the input trial fails shape or type validation, the interface rejects the request with a structured error response:

```json
{
    "status": "error",
    "error_message": "Contract Shape Mismatch: Received shape (200, 10). Expected exactly (385, 56).",
    "is_demo": true,
    "status_label": "INPUT VALIDATION ERROR",
    "class": null,
    "class_index": null,
    "confidence": 0.0,
    "probabilities": {
        "HC": 0.0,
        "ADD": 0.0,
        "ADHD": 0.0
    },
    "explanation": null
}
```

---

## 6. Future Transformer Integration Point

To replace the `MockModel` with the real `EEGTransformerModel` once trained:

1. Place the serialized weights file (`.pt`, `.onnx`, or `.h5`) in `prototype/model/`.
2. Open `prototype/model/trained_model_placeholder.py`.
3. Complete `load_trained_model()` to load the PyTorch/TensorFlow model:
   ```python
   import torch
   def load_trained_model():
       model = MyTransformerArchitecture(num_classes=3, in_channels=56, seq_len=385)
       model.load_state_dict(torch.load("prototype/model/eeg_transformer_best.pt", map_location="cpu"))
       model.eval()
       return model
   ```
4. Complete `predict_trained_model()` to convert the `[385, 56]` NumPy array to a tensor, perform `model.forward()`, apply `Softmax`, and return the response dictionary.
5. In `prototype/model/model_interface.py`, set:
   ```python
   USE_REAL_MODEL = True
   ```
6. **Zero UI modifications are required.** The research prototype dashboard will automatically consume the real model predictions.

---

## 7. Concrete Request / Response Example

### Example Request
```python
import numpy as np
from prototype.model import predict

# Synthetic demo input adhering to contract [385, 56], float32
synthetic_trial = np.zeros((385, 56), dtype=np.float32)

# Execute prediction through model adapter
result = predict(synthetic_trial)
```

### Example Response
```json
{
    "class": "ADHD",
    "class_index": 2,
    "probabilities": {
        "HC": 0.05,
        "ADD": 0.10,
        "ADHD": 0.85
    },
    "confidence": 0.85,
    "is_demo": true,
    "status_label": "DEMO / MOCK PREDICTION",
    "disclaimer": "This is a synthetic mock prediction for prototype validation. Not a medical diagnosis.",
    "explanation": null
}
```

---

## 8. Summary of Non-Assumptions

The prototype deliberately does **not** assume or hardcode:
- Specific bandpass filter boundaries (e.g. 0.5–45 Hz)
- Powerline notch filter specifics (50 Hz vs 60 Hz)
- ICA decomposition algorithms
- TBR or spectral ratio thresholds
- Transformer encoder hyperparameters (number of heads, layers, or projection dimension)
- Subject train/test validation splits

These parameters are being investigated and finalized independently by the ML research workflow.
