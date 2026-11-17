# Participant Identity and Recording Correspondence Audit

## Executive conclusion

**STOP: verified-subject five-fold cross-validation is not currently possible.** The MAT files expose signal arrays but no participant IDs or row-to-person crosswalk. Existing filename/storage-slot groups and folds do not establish verified participant identity. No training was run and no existing data/results were modified.

## Scope and method

- Read-only inspection of all four MAT files, local metadata, relevant source/configuration, and available processed archives.
- Inventoried 880 recording arrays and evaluated all 386,760 unordered recording pairs; the CSV contains one row per pair.
- Pairwise metrics: exact signal equality, Pearson correlation on common prefixes, zero-lag normalized cross-correlation, fixed-grid Welch PSD cosine, relative band-power Jensen-Shannon divergence and L1 distance, log theta/beta distance, and robust-scaled statistical fingerprint cosine.
- For same-length pairs, also computed max-absolute lagged NCC after anti-aliased 4:1 reduction to 64 Hz over ±32 reduced samples (±0.5 s). This bounded search can miss longer shifts.
- The similarity analysis did not use diagnostic labels to infer identity. Similarity thresholds are screening rules, not calibrated identity probabilities. No clustering was used.
- Weak candidate flag definition: non-exact/non-prefix pair with mean same-length lagged NCC ≥0.80, or mean relative-PSD cosine ≥0.97, or mean relative-band-power L1 distance ≤0.08. This intentionally broad screen is not a match probability.
- Sampling rate 256 Hz is from the published Mendeley record, not embedded in the MAT files.

## 1. MAT metadata inspection

Each file contains one public MATLAB variable (`FC`, `MC`, `FADHD`, or `MADHD`) with shape 1×11 and class cell. Each cell contains numeric float64 arrays shaped storage-slots × samples × 2. No embedded participant strings/IDs, timestamps, structures, event labels, channel labels, or acquisition metadata fields were found. MAT header/version values are format metadata only.

| File | Public variable | Recording slots in each condition |
|---|---|---:|
| FC.mat | `FC` (`1×11` cell) | 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13 |
| MC.mat | `MC` (`1×11` cell) | 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29 |
| FADHD.mat | `FADHD` (`1×11` cell) | 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 |
| MADHD.mat | `MADHD` (`1×11` cell) | 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27 |

| Cell | Samples | Duration at 256 Hz | Channels | Counts FC / MC / FADHD / MADHD |
|---:|---:|---:|---:|---|
| 1 | 7680 | 30 s | 2 | 13 / 29 / 11 / 27 |
| 2 | 5120 | 20 s | 2 | 13 / 29 / 11 / 27 |
| 3 | 5120 | 20 s | 2 | 13 / 29 / 11 / 27 |
| 4 | 11520 | 45 s | 2 | 13 / 29 / 11 / 27 |
| 5 | 3840 | 15 s | 2 | 13 / 29 / 11 / 27 |
| 6 | 7680 | 30 s | 2 | 13 / 29 / 11 / 27 |
| 7 | 7680 | 30 s | 2 | 13 / 29 / 11 / 27 |
| 8 | 5120 | 20 s | 2 | 13 / 29 / 11 / 27 |
| 9 | 5120 | 20 s | 2 | 13 / 29 / 11 / 27 |
| 10 | 11520 | 45 s | 2 | 13 / 29 / 11 / 27 |
| 11 | 11520 | 45 s | 2 | 13 / 29 / 11 / 27 |

Cell 8 and 9 arrays have 5120 samples (20 s), while the published mapping says 30 s. The published description gives condition-specific channel pairs, so channel slot 1 is not a stable physical channel across all cells. Comparisons across channel slots are descriptive.

The public Mendeley record reports 79 participants (42 HC, 37 ADHD) and a corrupted female-ADHD signal. Stored slot totals are 13+29+11+27=80. Removing FADHD storage slot 7 yields 79 slots, but does not establish one-to-one identity. FADHD slot 7 channel slot 2 is constant in all 11 cells; non-finite recordings found: 0.

## 2–5. Recording correspondence, duplicates, and fingerprints

- Evaluated **386,760** unordered pairs. Full metrics are in `recording_correspondence.csv`.
- Exact full-array equality: **240 pairs** across **240 duplicate signal groups**; exact groups spanning files: **0**; exact repeats across different slots within one file/condition: **0**.
- In addition, **560 pairs** have byte-for-byte equal common prefixes despite different full lengths. Together, these are confirmed signal-reuse correspondences, not verified participant identities.
- Whole-array copies are conditions 2/8, 3/9, and 4/10. Exact shared-prefix pairs are 2/7, 3/4, 3/10, 4/9, 5/11, 7/8, and 9/10 within each file; each pair is present at the same storage slot.
- Strict non-exact candidates: **0**; weak similarity flags: **102923**.
- Cross-file pairs: **274,670**, including **10,406** same-numbered slot comparisons. Cross-file exact full/prefix signal matches: **0**. Weak-similarity flags were 56081 across all cross-file pairs and 2841 among same-numbered cross-file slots; these weak flags do not support common participant ordering.
- Within-file same-slot cross-condition comparisons: 4,400; 240 are exact whole-array copies and 560 are exact shorter-recording prefix copies.
- Threshold sensitivity (both channel lagged-NCC thresholds, plus PSD threshold): strict ≥0.98/0.98: 0; moderate ≥0.95/0.98: 0; loose ≥0.90/0.95: 0. The candidate counts depend on arbitrary screening thresholds.
- Welch fingerprints used a common 1–40 Hz grid (`nperseg=1024`), five bands (0.5–4, 4–8, 8–13, 13–30, 30–40 Hz), theta/beta ratio, and per-channel mean, standard deviation, RMS, skew, excess kurtosis, and Hjorth activity/mobility/complexity.
- Different-length comparisons use the common prefix and assume sample-start alignment. PSD/fingerprint resemblance can reflect state, montage, filtering, or noise and cannot identify people.

### Confirmed signal duplicates (not participant IDs)

| Condition pair | Exact arrays | Cross-file? | Provisional key |
|---|---:|---:|---|
| 2/8 in each file | Yes, all slots | No | Same filename:slot for a row |
| 3/9 in each file | Yes, all slots | No | Same filename:slot for a row |
| 4/10 in each file | Yes, all slots | No | Same filename:slot for a row |

### Exact shared-prefix recordings (different full lengths)

| Condition pair | Shared exact prefix | Pair instances | Provisional key relation |
|---|---:|---:|---|
| 2/7 | 20 s | 80 | Same filename:slot |
| 3/4 | 20 s | 80 | Same filename:slot |
| 3/10 | 20 s | 80 | Same filename:slot |
| 4/9 | 20 s | 80 | Same filename:slot |
| 5/11 | 15 s | 80 | Same filename:slot |
| 7/8 | 20 s | 80 | Same filename:slot |
| 9/10 | 20 s | 80 | Same filename:slot |

These are byte-identical signal segments at the common prefix across both channel slots, not merely high correlations. Same-slot matches occur in all 80 stored rows (including the corrupted FADHD slot); only 79 are retained in processed archives after the FADHD exclusion. This strongly confirms repeated signal content at corresponding within-file storage positions, but does not prove those positions identify the same people.

Cross-file same-numbered slots were compared rather than presumed matched: no exact full-array or exact common-prefix signal matches were found across files. The 10,406 cross-file same-slot comparisons include 2841 weak fingerprint flags; their median and tail similarities are recorded row-by-row in the pairwise CSV. This does not establish a shared ordering.

### Highest signal-similarity candidates (not identities)

No non-exact pair crossed the strict candidate threshold.
Top weak candidates (all flagged pairs are retained in the CSV):
- FC.mat, cell 3, slot 2 <> MC.mat, cell 9, slot 3 (lag-NCC mean=0.3469, PSD cosine=0.9062)
- FC.mat, cell 3, slot 2 <> MC.mat, cell 3, slot 3 (lag-NCC mean=0.3469, PSD cosine=0.9062)
- FC.mat, cell 9, slot 2 <> MC.mat, cell 9, slot 3 (lag-NCC mean=0.3469, PSD cosine=0.9062)
- FC.mat, cell 9, slot 2 <> MC.mat, cell 3, slot 3 (lag-NCC mean=0.3469, PSD cosine=0.9062)
- MC.mat, cell 8, slot 2 <> MC.mat, cell 8, slot 3 (lag-NCC mean=0.3743, PSD cosine=0.8622)
- MC.mat, cell 2, slot 2 <> MC.mat, cell 8, slot 3 (lag-NCC mean=0.3743, PSD cosine=0.8622)
- MC.mat, cell 2, slot 2 <> MC.mat, cell 2, slot 3 (lag-NCC mean=0.3743, PSD cosine=0.8622)
- MC.mat, cell 2, slot 3 <> MC.mat, cell 8, slot 2 (lag-NCC mean=0.3743, PSD cosine=0.8622)

**Interpretation:** Exact equal arrays and prefixes confirm repeated signal content across the specified condition cells. The exhaustive scan found no exact cross-file full-array or prefix matches and no exact repeats across different slots within a cell. Absence of such matches does not imply different participants.

## 6. Existing group-ID and fold audit

`src/preprocessing.py` creates `filename:slot-NNN`; condition number is omitted. The same file/slot key is reused across conditions, while the same slot in different files gets a different key. This is a location key, not an actual participant ID. `src/epoching.py` assigns each provisional key to one fold; `src/modeling_data.py` checks only that saved keys do not cross folds. Configs explicitly mark participant IDs unverified. Cell 1 has 553 epochs and 79 distinct provisional keys after excluding FADHD slot 7.

| Processed archive | Epochs | Provisional groups | Folds | Group crossing? |
|---|---:|---:|---|---:|
| cell_01_epochs.npz | 553 | 79 | [0, 1, 2, 3, 4] | No |
| cell_05_epochs.npz | 237 | 79 | [0, 1, 2, 3, 4] | No |
| cell_06_epochs.npz | 553 | 79 | [0, 1, 2, 3, 4] | No |
| cell_07_epochs.npz | 553 | 79 | [0, 1, 2, 3, 4] | No |
| cell_11_epochs.npz | 790 | 79 | [0, 1, 2, 3, 4] | No |

- Identical processed epoch hashes crossing saved folds: **0**.
- Exact full-array pairs with different provisional IDs: **0**; with the same provisional ID: **240**.
- Exact common-prefix pairs: **560**; pairs with different provisional IDs: **0**; same ID: **560**.
- Whole-array duplicate pairs where both conditions have saved processed archives: **0**. The duplicate pairs are not in current selected/modelled cells 1, 5, 6, 7, and 11.
- Exact-prefix pairs with both conditions in saved archives: **80**. Exact processed epoch matches across archives: none (0 same-fold and 0 cross-fold exact matches). Separately, the 237 aligned Cell 5/11 processed epoch pairs are highly correlated and all have the same fold assignment; see Section 8–9.
- Thus no exact duplicate cross-fold leakage was found in the existing processed archives. This does not rule out one person appearing under multiple file/slot keys.

## 7. Cross-file correspondence

Slot counts differ (13, 29, 11, 27), and the files contain no common participant ordering metadata. There are no exact cross-file signal duplicates, but this is not evidence that the cohorts are disjoint. Signal similarity scores do not establish row ordering or participant correspondence. Preprocessing skips FADHD slot 7 without renumbering later slots; its cross-condition/cross-file identity remains unknown.

The complete candidate correspondence table is `recording_correspondence.csv`; `duplicate_recordings.csv` contains exact duplicates and strict near-duplicate candidates. “No threshold match” does not mean “different person.”

## 8–9. Leakage and subject-wise feasibility

The exact full copies and shared prefixes use the same within-file filename/slot key; a group-aware fold assignment based on that key keeps those copies together. The Cell 5/11 exact raw prefix is present in the selected processed archives. Aligning the processed epochs by provisional group key and epoch index gives 237 pairs; **0 are exactly equal**, but their Pearson correlations are high (median 0.999999995 and 0.999999995 for the two channel slots; minima 0.987359 and 0.996985). All 237 aligned pairs have the same saved fold assignment. Filtering recordings of different lengths can alter samples even when their raw prefixes match, so these are highly similar processed epochs, not exact processed duplicates. Current folds are group-disjoint only for the provisional keys, not verified people.

| Evidence class | Finding | Identity interpretation |
|---|---|---|
| A. Explicit verified identity | None found | Not available |
| B. Strong signal candidate | 0 strict non-exact pairs | Not verified people |
| C. Weak/ambiguous signal similarity | 102923 pairs | Not identity evidence |
| D. Unknown relationship | Cross-cell and cross-file linkage | Unknown |

**Verified subject-wise five-fold CV is not possible with the current files and metadata.** Only explicit, author-verified participant identity evidence can establish this.

## Exact dataset-author questions

1. Can you provide stable pseudonymous participant IDs for every row in all 11 cells of all four files, including corrupted, missing, excluded, or reordered recordings?
2. Is row order consistent across conditions within each file? If not, what is the exact row-to-person mapping for each cell?
3. Are participants mutually exclusive across FC, MC, FADHD, and MADHD? If not, which IDs recur across files?
4. How do 80 stored slots reconcile with 79 published participants? Which verified ID is the corrupted female-ADHD recording?
5. Are exact condition copies 2/8, 3/9, and 4/10 intentional? Which condition/record should be used, and are row identities the same?
6. Is 256 Hz confirmed for every file and condition, and is any acquisition metadata absent from the distributed MAT files?

## Limitations

- Similarity, exact equality, and storage-slot ordering do not prove participant identity.
- Lagged NCC is limited to ±0.5 s at 64 Hz; different-length pairs use common-prefix comparison.
- Condition montages and recording states vary.
- No clustering was used; no identity labels were inferred.
- Existing fold checks verify provisional-key separation only.
- No models were trained and no existing artifacts were changed.

## New audit outputs

- `audit_report.md`
- `recording_correspondence.csv` — 386,760 pairs
- `duplicate_recordings.csv`
- `group_id_audit.json`
- `fold_leakage_audit.json`
- `subjectwise_feasibility.json`
