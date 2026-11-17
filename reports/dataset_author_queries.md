# Questions for the ADHD EEG Dataset Authors

## Purpose

This document separates observations verified from the distributed MATLAB files
and published dataset description from questions that require author
confirmation. Storage slots are array positions, not verified participant IDs.
No participant identity or correspondence is inferred from matching slot
numbers.

## Verified observations

1. **Exact duplicate condition arrays:** In each of `FC.mat`, `MC.mat`,
   `FADHD.mat`, and `MADHD.mat`, the complete arrays for cells 2 and 8, cells 3
   and 9, and cells 4 and 10 are exactly equal.
2. **Duration discrepancy:** At 256 Hz, cells 8 and 9 contain 5,120 samples per
   recording, corresponding to 20 seconds. The published mapping gives each a
   duration of 30 seconds. This discrepancy occurs in all four files.
3. **Participant identity metadata:** The exposed MATLAB variables provide
   cell arrays and recording data but no participant IDs or explicit
   row-to-person crosswalk across conditions or files. Matching row counts or
   storage-slot positions do not establish identity.
4. **Storage-slot count:** The files contain 80 stored slots in total:
   13 in `FC.mat`, 29 in `MC.mat`, 11 in `FADHD.mat`, and 27 in `MADHD.mat`.
   The published dataset description reports 79 participants and mentions a
   corrupted female-ADHD recording. Stored slots are not asserted to represent
   80 distinct usable participants.
5. **Female ADHD flatline:** `FADHD.mat` storage slot 7 has a flatline,
   zero-variance channel in all 11 cells (channel slot 2; the physical channel
   name is not established). This is consistent with the published note about
   a corrupted female-ADHD signal, but does not independently establish the
   participant's identity or cross-cell correspondence.
6. **Current modeling scope:** The initial binary-classification work is
   restricted to cells 1, 5, 6, 7, and 11, kept separate. Preprocessing
   excludes `FADHD.mat` storage slot 7. These choices do not establish that the
   selected recordings are suitable for subject-independent evaluation.

## Questions requiring author confirmation

### 1. Duplicate condition arrays

Are the exact duplicates between cells 2/8, 3/9, and 4/10 in all four files
intentional? If so, which condition labels should be used for these recordings,
and should each duplicated pair be interpreted as one recording or as distinct
conditions? If not, can corrected files or an explanation of the duplication
be provided?

### 2. Published durations for cells 8 and 9

The published mapping lists 30 seconds for cells 8 and 9, but the distributed
arrays contain 5,120 samples (20 seconds at 256 Hz) in every file. Which
duration is correct, and should the published mapping or the distributed
arrays be corrected?

### 3. Participant-ID crosswalk

Can you provide a verified participant-ID crosswalk covering all four MATLAB
files and all 11 cells? In particular, please clarify whether storage-slot
ordering is consistent across cells within a file and whether any participants
occur in more than one file. Please identify any exceptions or missing
recordings explicitly; matching storage-slot numbers alone cannot establish
these correspondences.

### 4. 80 storage slots versus 79 published participants

How should the 80 stored slots be reconciled with the 79 participants reported
in the dataset description? Which specific slot is corrupted, duplicated,
extra, or otherwise excluded from the participant count, and what is the
verified participant ID for that slot, if available?

### 5. Corrupted female ADHD recording

Can you confirm whether `FADHD.mat` storage slot 7 is the corrupted
female-ADHD recording mentioned in the dataset description? Is that same
participant represented by slot 7 in all 11 cells, and what is the verified
participant ID? The observed flatline is present in all 11 cells, but the
MATLAB files do not establish identity correspondence.

### 6. Suitability of selected cells for subject-independent classification

Can cells 1, 5, 6, 7, and 11 be used for subject-independent binary
classification of ADHD versus Healthy Controls? Please provide the verified
participant IDs for every selected cell in each of `FC.mat`, `MC.mat`,
`FADHD.mat`, and `MADHD.mat`, including any absent or excluded recordings.
Please also confirm whether the participant populations overlap across these
cells and files, and whether the selected conditions are comparable for this
analysis.

## Existing evidence

- [Dataset validation report](dataset_validation_report.md) documents the
  MATLAB structures, shapes, duration calculations, duplicate comparisons,
  slot counts, and identity limitations.
- [Preprocessing report](preprocessing_report.md) documents the current
  preprocessing choices, separate cell outputs, and excluded FADHD storage
  slot.
- [Mendeley Data page](https://data.mendeley.com/datasets/6k4g25fhzg/1)
  contains the published dataset description and condition mapping.
