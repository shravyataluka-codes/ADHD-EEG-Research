# Raw Dataset Investigation Viewer

This directory contains a lightweight Python workflow to visually inspect the raw EEG dataset (`d1.mat`) without applying any preprocessing, filtering, or modifications.

## Purpose

The main goal of this script is strictly observational:
**“See the raw EEG data as it actually exists before doing any preprocessing or modeling.”**

It strictly follows the following scientific rules for this project:
- Separates what is DIRECTLY OBSERVED, STRONGLY SUPPORTED, INFERRED, and UNKNOWN.
- Does not silently resolve missing or conflicting information (e.g., the 56 vs 60 channels mapping).
- Does not guess or infer biological meaning without metadata (e.g., physical unit of amplitude).
- Does not preprocess, normalize, or train models.

## Prerequisites

You need a standard Python environment with the following dependencies:
```bash
pip install h5py numpy matplotlib
```

## How to Run

1. Ensure you have the raw data (`d1.mat`) located in the root of the repository or the directory where you are running the script.
2. Execute the script from the command line:

```bash
python scripts/dataset-investigation/view_eeg.py
```

## What This Workflow Demonstrates

When you run `view_eeg.py`, it performs the following steps:

1. **Loads the `.mat` file via HDF5:** Confirms the exact shapes, dimensions, dtypes, and compression stats directly from the raw format.
2. **Computes Global Statistics:** Calculates the global min, max, mean, and standard deviation to understand the data's raw scale.
3. **Inspects a Single Trial:** Extracts a single trial (e.g., Trial 0) to demonstrate the multidimensional structure (Channels × Time).
4. **Plots a Single Channel (Output: `single_channel.png`):** Shows a raw waveform for one channel over 385 sample positions. 
5. **Plots Multiple Channels (Output: `multiple_channels.png`):** Displays a subset of 5 channels offset vertically. Channel names are deliberately left as numerical indices since the mapping to `chan.mat` is currently UNRESOLVED.
6. **Heatmap Visualization (Output: `heatmap.png`):** Renders all 56 channels against the 385 sample positions for a single trial, allowing visual inspection of spatial and temporal structure.
7. **Cross-Trial Inspection:** Inspects multiple trials (0, 1, 10, 100, 1000, 4999) to evaluate consistency across the dataset.

No data is written back or altered, ensuring the dataset remains purely in its raw form for observation.
