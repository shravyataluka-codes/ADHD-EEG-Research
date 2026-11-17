# EEG Preprocessing Run Report

## Scope and reproducibility

- Original MAT files were read only and were not modified.
- No model was trained and no features were extracted.
- The five selected cells are processed and saved separately.
- Group keys are provisional source-file/storage-slot keys, not participant IDs.
- Filter/epoch tests passing does not establish scientific validity.

## Configuration used

- Sampling rate: 256 Hz.
- Filter: 4-order Butterworth SOS band-pass, 0.5 to 40.0 Hz.
- Filtering is zero-phase (`sosfiltfilt`) along the sample axis.
- Notch filter: **not applied**; no diagnostics in this run justified changing the configured decision.
- Edge trim: 1 s from each end, after filtering.
- Epoching: 4 s, non-overlapping; incomplete trailing samples are discarded.
- Selected cells: 1, 5, 6, 7, 11.
- FADHD storage slot 7 is excluded in every selected cell before filtering, as requested based on the validation result.
- The fourth-order filter order is an explicit implementation choice; the request did not specify an order.

## Epoch counts by cell and class

| Cell | Control slots | Control epochs | ADHD slots | ADHD epochs | Total epochs | Rejected epochs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 294 | 37 | 259 | 553 | 0 |
| 5 | 42 | 126 | 37 | 111 | 237 | 0 |
| 6 | 42 | 294 | 37 | 259 | 553 | 0 |
| 7 | 42 | 294 | 37 | 259 | 553 | 0 |
| 11 | 42 | 420 | 37 | 370 | 790 | 0 |

Epoch arrays are saved in separate NPZ files:

- `processed\cell_01_epochs.npz`: shape `[553, 1024, 2]`.
- `processed\cell_05_epochs.npz`: shape `[237, 1024, 2]`.
- `processed\cell_06_epochs.npz`: shape `[553, 1024, 2]`.
- `processed\cell_07_epochs.npz`: shape `[553, 1024, 2]`.
- `processed\cell_11_epochs.npz`: shape `[790, 1024, 2]`.

## Participant-slot epoch counts

One-based storage slots are listed below. They are local grouping keys only and do not assert identity across conditions or files.

| Cell | File | Class | Storage slot | Status | Candidate | Accepted | Rejected | Tail samples discarded |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | FC.mat | Control | 1 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 2 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 3 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 4 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 5 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 6 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 7 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 8 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 9 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 10 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 11 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 12 | processed | 7 | 7 | 0 | 0 |
| 1 | FC.mat | Control | 13 | processed | 7 | 7 | 0 | 0 |
| 5 | FC.mat | Control | 1 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 2 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 3 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 4 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 5 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 6 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 7 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 8 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 9 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 10 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 11 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 12 | processed | 3 | 3 | 0 | 256 |
| 5 | FC.mat | Control | 13 | processed | 3 | 3 | 0 | 256 |
| 6 | FC.mat | Control | 1 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 2 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 3 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 4 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 5 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 6 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 7 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 8 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 9 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 10 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 11 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 12 | processed | 7 | 7 | 0 | 0 |
| 6 | FC.mat | Control | 13 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 1 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 2 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 3 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 4 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 5 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 6 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 7 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 8 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 9 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 10 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 11 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 12 | processed | 7 | 7 | 0 | 0 |
| 7 | FC.mat | Control | 13 | processed | 7 | 7 | 0 | 0 |
| 11 | FC.mat | Control | 1 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 2 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 3 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 4 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 5 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 6 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 7 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 8 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 9 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 10 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 11 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 12 | processed | 10 | 10 | 0 | 768 |
| 11 | FC.mat | Control | 13 | processed | 10 | 10 | 0 | 768 |
| 1 | MC.mat | Control | 1 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 2 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 3 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 4 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 5 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 6 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 7 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 8 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 9 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 10 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 11 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 12 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 13 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 14 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 15 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 16 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 17 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 18 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 19 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 20 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 21 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 22 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 23 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 24 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 25 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 26 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 27 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 28 | processed | 7 | 7 | 0 | 0 |
| 1 | MC.mat | Control | 29 | processed | 7 | 7 | 0 | 0 |
| 5 | MC.mat | Control | 1 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 2 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 3 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 4 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 5 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 6 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 7 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 8 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 9 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 10 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 11 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 12 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 13 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 14 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 15 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 16 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 17 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 18 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 19 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 20 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 21 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 22 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 23 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 24 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 25 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 26 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 27 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 28 | processed | 3 | 3 | 0 | 256 |
| 5 | MC.mat | Control | 29 | processed | 3 | 3 | 0 | 256 |
| 6 | MC.mat | Control | 1 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 2 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 3 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 4 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 5 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 6 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 7 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 8 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 9 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 10 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 11 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 12 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 13 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 14 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 15 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 16 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 17 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 18 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 19 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 20 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 21 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 22 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 23 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 24 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 25 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 26 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 27 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 28 | processed | 7 | 7 | 0 | 0 |
| 6 | MC.mat | Control | 29 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 1 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 2 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 3 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 4 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 5 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 6 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 7 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 8 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 9 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 10 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 11 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 12 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 13 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 14 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 15 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 16 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 17 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 18 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 19 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 20 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 21 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 22 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 23 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 24 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 25 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 26 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 27 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 28 | processed | 7 | 7 | 0 | 0 |
| 7 | MC.mat | Control | 29 | processed | 7 | 7 | 0 | 0 |
| 11 | MC.mat | Control | 1 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 2 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 3 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 4 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 5 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 6 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 7 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 8 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 9 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 10 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 11 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 12 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 13 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 14 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 15 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 16 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 17 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 18 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 19 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 20 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 21 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 22 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 23 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 24 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 25 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 26 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 27 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 28 | processed | 10 | 10 | 0 | 768 |
| 11 | MC.mat | Control | 29 | processed | 10 | 10 | 0 | 768 |
| 1 | FADHD.mat | ADHD | 1 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 2 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 3 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 4 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 5 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 6 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 7 | excluded | 7 | 0 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 8 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 9 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 10 | processed | 7 | 7 | 0 | 0 |
| 1 | FADHD.mat | ADHD | 11 | processed | 7 | 7 | 0 | 0 |
| 5 | FADHD.mat | ADHD | 1 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 2 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 3 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 4 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 5 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 6 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 7 | excluded | 3 | 0 | 0 | 0 |
| 5 | FADHD.mat | ADHD | 8 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 9 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 10 | processed | 3 | 3 | 0 | 256 |
| 5 | FADHD.mat | ADHD | 11 | processed | 3 | 3 | 0 | 256 |
| 6 | FADHD.mat | ADHD | 1 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 2 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 3 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 4 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 5 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 6 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 7 | excluded | 7 | 0 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 8 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 9 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 10 | processed | 7 | 7 | 0 | 0 |
| 6 | FADHD.mat | ADHD | 11 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 1 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 2 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 3 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 4 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 5 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 6 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 7 | excluded | 7 | 0 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 8 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 9 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 10 | processed | 7 | 7 | 0 | 0 |
| 7 | FADHD.mat | ADHD | 11 | processed | 7 | 7 | 0 | 0 |
| 11 | FADHD.mat | ADHD | 1 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 2 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 3 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 4 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 5 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 6 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 7 | excluded | 10 | 0 | 0 | 0 |
| 11 | FADHD.mat | ADHD | 8 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 9 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 10 | processed | 10 | 10 | 0 | 768 |
| 11 | FADHD.mat | ADHD | 11 | processed | 10 | 10 | 0 | 768 |
| 1 | MADHD.mat | ADHD | 1 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 2 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 3 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 4 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 5 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 6 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 7 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 8 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 9 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 10 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 11 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 12 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 13 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 14 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 15 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 16 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 17 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 18 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 19 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 20 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 21 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 22 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 23 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 24 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 25 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 26 | processed | 7 | 7 | 0 | 0 |
| 1 | MADHD.mat | ADHD | 27 | processed | 7 | 7 | 0 | 0 |
| 5 | MADHD.mat | ADHD | 1 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 2 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 3 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 4 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 5 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 6 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 7 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 8 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 9 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 10 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 11 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 12 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 13 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 14 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 15 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 16 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 17 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 18 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 19 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 20 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 21 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 22 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 23 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 24 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 25 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 26 | processed | 3 | 3 | 0 | 256 |
| 5 | MADHD.mat | ADHD | 27 | processed | 3 | 3 | 0 | 256 |
| 6 | MADHD.mat | ADHD | 1 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 2 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 3 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 4 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 5 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 6 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 7 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 8 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 9 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 10 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 11 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 12 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 13 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 14 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 15 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 16 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 17 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 18 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 19 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 20 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 21 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 22 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 23 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 24 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 25 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 26 | processed | 7 | 7 | 0 | 0 |
| 6 | MADHD.mat | ADHD | 27 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 1 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 2 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 3 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 4 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 5 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 6 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 7 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 8 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 9 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 10 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 11 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 12 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 13 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 14 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 15 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 16 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 17 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 18 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 19 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 20 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 21 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 22 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 23 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 24 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 25 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 26 | processed | 7 | 7 | 0 | 0 |
| 7 | MADHD.mat | ADHD | 27 | processed | 7 | 7 | 0 | 0 |
| 11 | MADHD.mat | ADHD | 1 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 2 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 3 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 4 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 5 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 6 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 7 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 8 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 9 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 10 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 11 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 12 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 13 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 14 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 15 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 16 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 17 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 18 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 19 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 20 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 21 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 22 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 23 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 24 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 25 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 26 | processed | 10 | 10 | 0 | 768 |
| 11 | MADHD.mat | ADHD | 27 | processed | 10 | 10 | 0 | 768 |

## Exclusions, rejected epochs, and quality control

- Whole-recording exclusions: **5** (FADHD storage slot 7, one exclusion for each selected cell).
- Rejected candidate epochs: **0**.
- NaN/Inf input samples in processed recordings: **0**.
- Flatline channel traces among processed recordings: **0**.
- Filter outputs finite: **True**.
- Per-participant input QC, rejection details, exclusion reasons, and all counts are also recorded in `reports/preprocessing_results.json`.
- No candidate epochs were rejected.

## CV fold grouping (no model training)

Within each cell, epochs are assigned to 5 deterministic folds by provisional group key and class. Every epoch for a given `(source file, storage slot)` key stays in one fold. Since cross-cell identity linkage is unverified, fold assignments are made independently per cell and must not be read as person matching across cells.

## Scientific limitations

This is a deterministic implementation of the requested settings, not evidence that the cutoffs, filter order, edge trim, epoch length, or rejection rules are scientifically optimal. No power-line diagnostic was used to justify a notch filter, so none was applied. Review the data provenance, channel mapping, participant linkage, and preprocessing choices before scientific interpretation.
