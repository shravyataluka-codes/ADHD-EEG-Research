# Minimal EEG preprocessing component

This document describes the isolated preprocessing implementation. It does not
change the loader, model, classifier, or evaluation code. The selected signal
operations are 0.5–45 Hz band-pass filtering, a 50 Hz notch, and MNE ICA.

## Current execution gate

The component is intentionally fail-closed. It does not infer missing settings
or select ICA components automatically. Before processing the real dataset, the
operator must provide:

- The paper-reported sampling frequency (256 Hz) and the approved 50 Hz notch.
- Explicit pass-band and notch filter design, transition, width, phase, padding,
  and filter-length settings in a JSON run configuration.
- An integer random seed and explicit ICA method, component count, and iteration
  limit.
- A reviewed ICA decision for every subject, including reviewer, review time,
  rationale, and the excluded component indices. An empty list is permitted only
  when the review explicitly determines that no components should be excluded.

The repository does not establish the filter design details, ICA component
criteria, or random seed. The 56 channels have no validated names or EOG/ECG
mapping. Consequently, no scientifically validated artifact-identification rule
can be inferred from the local data. The software records supplied decisions,
but does not certify them. No processing of the seven source files has been
authorized until a complete configuration and reviewed decisions are supplied.

The historical audit documents record the former deferred/no-transform state.
This new implementation follows the explicitly selected pipeline request; it
does not rewrite those historical evidence records.

## Memory, representation, and safety

`process_dataset` groups only trial metadata in memory and loads one subject's
epochs at a time. Its derived HDF5 files retain each source array's
`(time, channels, trials)` representation and `float32` dtype. The source MAT
files are opened read-only and SHA-256 hashes are compared before and after the
run. Existing output directories are never overwritten.

The subject/trial crosswalk is saved as `trial_manifest.csv`. A successful run
also writes `metadata.json`, `inspection_report.json`, and
`inspection_trial.npz`; the latter contains one actual raw trial and its
preprocessed counterpart. The inspection report includes shape/count checks and
basic before/after statistics. No plot is generated.

Filtering operates on already-epoched data because continuous recordings are
not present. Edge effects and the unknown meaning of the 385th sample remain
limitations. The implementation preserves all samples and channels, does not
read `y_stim` rows 1–3, and does not perform a train/test split.

ICA is fitted independently per subject before any evaluation split. This
component does not enforce fold-local fitting; a later evaluation must address
the resulting use of each subject's EEG in the unsupervised decomposition and
must not claim leakage-safe evaluation based solely on this preprocessing run.

## Configuration and review files

The JSON run configuration requires these top-level keys:

```json
{
  "sampling_frequency_hz": 256.0,
  "notch_frequency_hz": 50.0,
  "random_seed": null,
  "filters": {
    "method": null,
    "phase": null,
    "fir_design": null,
    "filter_length": null,
    "l_trans_bandwidth": null,
    "h_trans_bandwidth": null,
    "fir_window": null,
    "pad": null,
    "notch_width_hz": null,
    "notch_transition_bandwidth_hz": null
  },
  "ica": {
    "method": null,
    "n_components": null,
    "max_iter": null
  }
}
```

The null fields are **required inputs**, not defaults. The example is not
directly runnable and must not be treated as an approved protocol.

The ICA review JSON has a top-level `review_protocol` and a `subjects` object
keyed by every subject ID in the manifest. Each entry contains `reviewed: true`,
`reviewer`, `reviewed_at`, `rationale`, and an `exclude_components` integer
array. The parser rejects missing subjects, unreviewed entries, and malformed
indices. This format captures external decisions; it does not establish an
artifact-identification criterion.

## Commands

From the repository root, install the project dependencies and run tests:

```powershell
python -m pip install -e ".[preprocessing]"
python -m unittest discover -s tests -v
```

After the required settings and all per-subject ICA review decisions have been
approved and written to JSON, run:

```powershell
python -m adhd_eeg.preprocess_cli --dataset-dir dataset --output-dir data/preprocessed/run-001 --config CONFIG.json --ica-decisions ICA-DECISIONS.json
```

`CONFIG.json` and `ICA-DECISIONS.json` are inputs to be supplied after the
remaining methodological decisions; the illustrative settings above are not
approved values. The CLI refuses to overwrite an existing output directory.
