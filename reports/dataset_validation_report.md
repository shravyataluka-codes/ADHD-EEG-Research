# ADHD EEG Dataset Validation

## Scope and method

- Read-only validation of the four original `.mat` files in `Dataset files/`.
- No preprocessing, feature extraction, participant-ID creation, or model training.
- Sampling rate used for duration calculations: **256 Hz**.
- Published mapping source: [Mendeley Data v1](https://data.mendeley.com/datasets/6k4g25fhzg/1), Method field.
- Array positions are reported as participant slots, not participant IDs.

## Verified overall results

- All four files loaded with 11 conditions: **True**.
- All conditions have a three-dimensional shape ending in 2: **True**.
- This shape is consistent with the published `subjects x samples x 2 channels` description; it does not identify the physical channel names.
- All calculated durations match the published mapping: **False**.
- Non-finite samples (NaN + Inf): **0**.
- Flatline channel recordings: **11**.
- Zero-variance channel recordings: **11**.

## Participant counts

| File | Group | Participant slots in each condition |
|---|---|---:|
| FC.mat | female control | 13 |
| MC.mat | male control | 29 |
| FADHD.mat | female ADHD | 11 |
| MADHD.mat | male ADHD | 27 |

These are stored participant-slot counts. The published record notes a corrupted female-ADHD recording; this report does not determine whether a slot is usable or remove it from the count.

The four files contain **80 stored participant slots in total**. The Mendeley description reports 79 participants and notes one corrupted female-ADHD recording. Stored slots are not asserted to be unique usable participants.

## Condition-by-condition results

| File | Cell | Published task | Shape (subjects x samples x channels) | Samples | Calculated (s) | Published (s) | Duration check | NaN | +Inf | -Inf | Nonfinite traces | Flatlines | Zero variance |
|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| FC.mat | 1 | Eyes open baseline | 13 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 2 | Eyes closed | 13 x 5120 x 2 | 5120 | 20 | 20 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 3 | Eyes open | 13 x 5120 x 2 | 5120 | 20 | 20 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 4 | Cognitive Challenge | 13 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 5 | Pre-Omni harmonic baseline | 13 x 3840 x 2 | 3840 | 15 | 15 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 6 | Omni harmonic assessment | 13 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 7 | Eyes open baseline | 13 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 8 | Eyes closed | 13 x 5120 x 2 | 5120 | 20 | 30 | **MISMATCH** | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 9 | Eyes open | 13 x 5120 x 2 | 5120 | 20 | 30 | **MISMATCH** | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 10 | Eyes closed | 13 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FC.mat | 11 | Eyes closed | 13 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 1 | Eyes open baseline | 29 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 2 | Eyes closed | 29 x 5120 x 2 | 5120 | 20 | 20 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 3 | Eyes open | 29 x 5120 x 2 | 5120 | 20 | 20 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 4 | Cognitive Challenge | 29 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 5 | Pre-Omni harmonic baseline | 29 x 3840 x 2 | 3840 | 15 | 15 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 6 | Omni harmonic assessment | 29 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 7 | Eyes open baseline | 29 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 8 | Eyes closed | 29 x 5120 x 2 | 5120 | 20 | 30 | **MISMATCH** | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 9 | Eyes open | 29 x 5120 x 2 | 5120 | 20 | 30 | **MISMATCH** | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 10 | Eyes closed | 29 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MC.mat | 11 | Eyes closed | 29 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| FADHD.mat | 1 | Eyes open baseline | 11 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 2 | Eyes closed | 11 x 5120 x 2 | 5120 | 20 | 20 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 3 | Eyes open | 11 x 5120 x 2 | 5120 | 20 | 20 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 4 | Cognitive Challenge | 11 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 5 | Pre-Omni harmonic baseline | 11 x 3840 x 2 | 3840 | 15 | 15 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 6 | Omni harmonic assessment | 11 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 7 | Eyes open baseline | 11 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 8 | Eyes closed | 11 x 5120 x 2 | 5120 | 20 | 30 | **MISMATCH** | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 9 | Eyes open | 11 x 5120 x 2 | 5120 | 20 | 30 | **MISMATCH** | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 10 | Eyes closed | 11 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| FADHD.mat | 11 | Eyes closed | 11 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 1 | 1 |
| MADHD.mat | 1 | Eyes open baseline | 27 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 2 | Eyes closed | 27 x 5120 x 2 | 5120 | 20 | 20 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 3 | Eyes open | 27 x 5120 x 2 | 5120 | 20 | 20 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 4 | Cognitive Challenge | 27 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 5 | Pre-Omni harmonic baseline | 27 x 3840 x 2 | 3840 | 15 | 15 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 6 | Omni harmonic assessment | 27 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 7 | Eyes open baseline | 27 x 7680 x 2 | 7680 | 30 | 30 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 8 | Eyes closed | 27 x 5120 x 2 | 5120 | 20 | 30 | **MISMATCH** | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 9 | Eyes open | 27 x 5120 x 2 | 5120 | 20 | 30 | **MISMATCH** | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 10 | Eyes closed | 27 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |
| MADHD.mat | 11 | Eyes closed | 27 x 11520 x 2 | 11520 | 45 | 45 | match | 0 | 0 | 0 | 0 | 0 | 0 |

## Flatline and zero-variance trace locations

Locations below use one-based array storage slots only. They are not participant IDs, and channel-slot numbers are not channel names.

| File | Cell | Participant slot | Channel slot | Check |
|---|---:|---:|---:|---|
| FADHD.mat | 1 | 7 | 2 | flatline |
| FADHD.mat | 1 | 7 | 2 | zero variance |
| FADHD.mat | 2 | 7 | 2 | flatline |
| FADHD.mat | 2 | 7 | 2 | zero variance |
| FADHD.mat | 3 | 7 | 2 | flatline |
| FADHD.mat | 3 | 7 | 2 | zero variance |
| FADHD.mat | 4 | 7 | 2 | flatline |
| FADHD.mat | 4 | 7 | 2 | zero variance |
| FADHD.mat | 5 | 7 | 2 | flatline |
| FADHD.mat | 5 | 7 | 2 | zero variance |
| FADHD.mat | 6 | 7 | 2 | flatline |
| FADHD.mat | 6 | 7 | 2 | zero variance |
| FADHD.mat | 7 | 7 | 2 | flatline |
| FADHD.mat | 7 | 7 | 2 | zero variance |
| FADHD.mat | 8 | 7 | 2 | flatline |
| FADHD.mat | 8 | 7 | 2 | zero variance |
| FADHD.mat | 9 | 7 | 2 | flatline |
| FADHD.mat | 9 | 7 | 2 | zero variance |
| FADHD.mat | 10 | 7 | 2 | flatline |
| FADHD.mat | 10 | 7 | 2 | zero variance |
| FADHD.mat | 11 | 7 | 2 | flatline |
| FADHD.mat | 11 | 7 | 2 | zero variance |

The Mendeley record notes that female-ADHD subject number 7 has a corrupted EEG signal. The observed flatline storage slot 7 is consistent with that note, but the arrays do not provide an identity crosswalk to prove these are the same participant across conditions.

## Exact duplicate checks

Each check compares the complete numeric arrays using exact value equality; NaNs are not treated as equal.

| File | Cell pair | Same shape | Complete arrays exactly equal |
|---|---|---|---|
| FC.mat | 2/8 | same | YES |
| FC.mat | 3/9 | same | YES |
| FC.mat | 4/10 | same | YES |
| MC.mat | 2/8 | same | YES |
| MC.mat | 3/9 | same | YES |
| MC.mat | 4/10 | same | YES |
| FADHD.mat | 2/8 | same | YES |
| FADHD.mat | 3/9 | same | YES |
| FADHD.mat | 4/10 | same | YES |
| MADHD.mat | 2/8 | same | YES |
| MADHD.mat | 3/9 | same | YES |
| MADHD.mat | 4/10 | same | YES |

## Participant matching: verified facts vs. limitations

**Verified:** each condition's first-axis length is reported above. The published Mendeley method describes each file as a 1 x 11 cell array with participant recordings in the cells.

**Not verified:** the files contain no participant IDs or explicit row-to-person crosswalk among the exposed MAT variables. Matching slot counts do not prove that slot *n* in one condition is the same individual as slot *n* in another condition, and they do not enable matching people across the four files.

## Published mapping and source

| Cell | Published task | Published duration (s) |
|---:|---|---:|
| 1 | Eyes open baseline | 30 |
| 2 | Eyes closed | 20 |
| 3 | Eyes open | 20 |
| 4 | Cognitive Challenge | 45 |
| 5 | Pre-Omni harmonic baseline | 15 |
| 6 | Omni harmonic assessment | 30 |
| 7 | Eyes open baseline | 30 |
| 8 | Eyes closed | 30 |
| 9 | Eyes open | 30 |
| 10 | Eyes closed | 45 |
| 11 | Eyes closed | 45 |

Source: https://data.mendeley.com/datasets/6k4g25fhzg/1 (dataset DOI `10.17632/6k4g25fhzg.1`). The published values are preserved as given; observed duration differences are flagged, not silently reconciled.
