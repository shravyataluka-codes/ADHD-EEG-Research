# ADHD vs Healthy Control (HC) Benchmark Test Samples

This directory contains 4 standardized, 2-channel EEG test recordings extracted from the Mendeley ADHD-HC dataset (`FC.mat`, `FADHD.mat`, `MC.mat`, `MADHD.mat`).

Each file corresponds to **Cell 1: Eyes open baseline recording (30 seconds @ 256 Hz = 7,680 samples, 2 channel slots)** with standard CSV headers (`channel_slot_1,channel_slot_2`).

---

## Sample Summary Table

| Sample File | Demographic & Group | Source MAT File | Subject # | Ground Truth | Deployed Fusion Model Prediction | Control Probability | ADHD Probability |
|---|---|---|---|---|---|---|---|
| `sample_female_control_HC.csv` | Female Healthy Control | `FC.mat` (Cell 1) | Subject 01 | **Control** | **Control** | **72.8%** | **27.2%** |
| `sample_female_adhd.csv` | Female ADHD | `FADHD.mat` (Cell 1) | Subject 01 | **ADHD** | **ADHD** | **29.3%** | **70.7%** |
| `sample_male_control_HC.csv` | Male Healthy Control | `MC.mat` (Cell 1) | Subject 01 | **Control** | **Control** | **96.1%** | **3.9%** |
| `sample_male_adhd.csv` | Male ADHD | `MADHD.mat` (Cell 1) | Subject 01 | **ADHD** | **ADHD** | **0.7%** | **99.3%** |

---

## File Specifications
- **Format**: Standard comma-separated values (CSV)
- **Columns**: `channel_slot_1`, `channel_slot_2`
- **Rows**: 7,680 numeric rows (excluding header)
- **Sampling Rate**: 256 Hz
- **Duration**: 30.00 seconds
- **Finite Check**: All samples verified 100% finite (0 NaN, 0 Inf)
- **Signal Quality**: Verified non-zero variance and free of flatlines/corruptions

---

## How to Test
1. **Via Drushti Detect Web Interface**:
   - In the upload section, click any of the 4 buttons in the **"Mendeley Dataset Benchmark Samples"** card.
   - The recording will be loaded instantly into the analysis pipeline.
   - Click **"Analyze EEG recording"** to view the multi-tab interactive results.

2. **Via Python / Backend API**:
   ```bash
   curl -X POST "http://localhost:8000/analyze" \
     -F "file=@test_samples/sample_female_control_HC.csv" \
     -F "sampling_rate_hz=256"
   ```
