# 04 — Preprocessing Decision Audit

**Status:** Evidence review only; no preprocessing choice is made or authorized  
**Purpose:** Explain what is known about possible EEG transformations in this project, what each would change, and what still needs a decision.  
**Audience:** A Computer Science student who is new to EEG and signal processing.

> This is not a standard EEG recipe. A method being common elsewhere is not evidence that it is appropriate for this dataset. Do not implement a transformation on the basis of this document. The preprocessing architecture and final choices are to be decided separately by the researcher.

## How to read this audit

### Status labels

These labels describe the **evidence and current project position**, not a recommendation or authorization:

- **SUPPORTED** — There is direct project or paper evidence for the narrowly stated fact. This does not automatically support applying a new transformation to the local files.
- **REASONABLE BUT NEEDS A DECISION** — There is a plausible project-specific reason to consider the operation, but the evidence does not settle whether or how to use it.
- **NOT CURRENTLY JUSTIFIED** — No project-specific evidence presently supports applying it.
- **CONFLICTS WITH VERIFIED INFORMATION** — Applying the proposed operation would contradict a verified data fact or a documented preservation constraint.
- **UNKNOWN** — The available evidence is insufficient even to settle a central fact about the operation or data.

### Leakage language

**Information leakage** means that information from validation/test participants, or from their labels or outcomes, influences a transformation or a choice that should have been made without that information. A fixed operation does not automatically leak data. Fitting its parameters, selecting its settings after examining held-out results, or splitting related examples across partitions can create leakage. The project methodology requires any fitted or data-selected operation to be limited to training data/folds and requires all trials from one participant to stay in the same partition.

### Evidence boundary

- The local files are verified to contain 385 positions by 56 columns per trial, stored as `float32`; this is a statement about the released arrays, **not** proof that the arrays are untouched acquisition recordings.
- He et al. report that the source EEG was preprocessed using ICA before trials were made. The paper does not specify enough detail to reproduce that ICA, and the local files' exact preprocessing history is unresolved.
- The later Gate 1 mapping report documents validation of the sequential file-to-metadata ordering using boundary signal checks. Older notes that call this join only inferred are superseded on that specific point; the original subject-filename ordinal association remains a separate caveat. This evidence validates a metadata crosswalk, not a signal-processing choice.
- Existing project state documents say no signal preprocessing has been run, and the preprocessing gate is not passed.

## Operation-by-operation evidence

### 1. No additional signal modification

- **Meaning:** Use each released trial's values as they are, without filtering, scaling, subtracting a baseline, dropping samples/channels, or otherwise changing the signal.
- **What it changes:** Nothing in the EEG values or `(385, 56)` trial shape. Attaching verified metadata is not itself a signal transformation.
- **Why someone might consider it:** It provides a clear “as released” comparison and avoids adding undocumented signal-processing choices.
- **Evidence for this project:** Project audits report that no preprocessing has been run; the arrays' shape, dtype, and finite values were checked. The project-state audit records preserving sample/channel columns and `float32` as constraints.
- **Evidence against / limitation:** The paper says ICA was applied to the source data. We cannot establish whether the local OSF files are exactly those ICA outputs or reproduce their processing history. Thus “no additional modification” must not be called “raw EEG” or “unprocessed EEG.”
- **Risk:** Unverified upstream processing could be mistaken for known-clean or truly raw data. It also leaves open whether artifacts or scale differences affect later analysis.
- **Shape effect:** None.
- **Leakage possibility:** None from a true no-op itself. Choosing it because of held-out performance would still contaminate the evaluation.
- **Clearly used by original paper?** No. The paper reports ICA upstream, so it does not describe an untouched-signal pipeline.
- **Exact implementation known?** In-project no-op means preserve source values; the source files' upstream processing is not known exactly.
- **Status: SUPPORTED** for a no-additional-transformation reference condition and preservation of source values; **not** evidence that the source data was never processed.

### 2. Detrending / DC removal

- **Meaning:** Remove a constant offset (DC removal) or a slow straight-line drift (detrending) from a signal segment.
- **What it changes:** Changes the value at one or more time positions in each channel. A per-trial subtraction or line-fit typically keeps the same number of positions and channels.
- **Why someone might consider it:** A sensor offset or slow drift may obscure faster changes in voltage. Whether that applies here depends on the signal and acquisition history.
- **Evidence for this project:** No audit, data header, or paper detail establishes a DC offset or drift that needs correction. A detrend option appears as a possible setting for proposed Welch spectral analysis, not as an established preprocessing step.
- **Evidence against / limitation:** The signal units, reference, upstream processing, and appropriate time scale are unresolved. Removing a trend can also remove genuine slow EEG activity.
- **Risk:** Changes low-frequency/amplitude information; the outcome depends on whether subtraction is per trial, per channel, or over a longer recording.
- **Shape effect:** Usually none for a same-length per-channel operation.
- **Leakage possibility:** A fixed per-trial operation has no cross-participant fitting by itself. Estimating a shared trend/statistic across the dataset, or choosing the rule using test outcomes, can leak information.
- **Clearly used by original paper?** Not specified.
- **Exact implementation known?** No. The project does not define constant subtraction versus linear detrending, the time span, or edge handling.
- **Status: NOT CURRENTLY JUSTIFIED.**

### 3. Normalization / standardization

- **Meaning:** Rescale numbers, for example by subtracting a mean and dividing by a standard deviation. “Per-trial,” “per-channel,” and “dataset-level” scaling are different procedures.
- **What it changes:** Alters EEG amplitudes and offsets. A scale-only transformation usually keeps `(385, 56)`, but using different axes produces different values.
- **Why someone might consider it:** A model may train more easily when its input values have a suitable numerical scale. That is an engineering possibility, not evidence this dataset needs a particular scaling rule.
- **Evidence for this project:** The project guide proposed scaling; existing audits say the paper does not specify it. Project records explicitly prohibit calculating normalization statistics from the full dataset; if scaling is chosen, fitted statistics must use training data only.
- **Evidence against / limitation:** Per-trial scaling can remove absolute amplitude differences; global scaling can expose held-out data; and the physical units/reference are unresolved. The paper does not determine whether or how the original model was scaled.
- **Risk:** Loss of potentially meaningful amplitude information, inconsistent train/test handling, or leakage from statistics calculated using held-out participants.
- **Shape effect:** Normally none for raw-value scaling; engineered feature scaling acts on the chosen feature dimensions instead.
- **Leakage possibility:** Yes, if means, standard deviations, clipping limits, or other fitted values use validation/test data. A separately computed rule for each sample avoids cross-sample fitting but can still change the scientific meaning of the values.
- **Clearly used by original paper?** Not specified.
- **Exact implementation known?** No: method, axes, fitting population, and when scaling would occur are all undecided.
- **Status: REASONABLE BUT NEEDS A DECISION.**

### 4. Band-pass filtering

- **Meaning:** Keep signal changes inside a chosen low-to-high frequency range and reduce changes outside it.
- **What it changes:** Changes channel values over time and suppresses some slower/faster components. The output can retain the same sample count, but filter start/end effects can alter boundary values.
- **Why someone might consider it:** A researcher might want to limit frequencies outside a stated scientific question or remove known unwanted frequency content. The relevant range must be justified for this data and question.
- **Evidence for this project:** A project guide proposed a 0.5–45 Hz, fourth-order Butterworth filter. The base-paper audit finds no specified passband or reproducible filter design in the paper.
- **Evidence against / limitation:** The proposed cutoffs/design are not corroborated by the paper or data documentation. It is unknown whether the released files were already filtered. A filter can remove real signal content and make a reproduction diverge from the paper.
- **Risk:** Meaningful frequency content can be removed; filter phase and boundary behavior can alter timing or shape of waveforms.
- **Shape effect:** Commonly the sample/channel dimensions are retained, but this depends on implementation; filtering does not mean “no change” just because the shape is unchanged.
- **Leakage possibility:** A fixed, preregistered filter is not inherently a cross-participant leak. Selecting its parameters based on test performance or test-data inspection contaminates evaluation.
- **Clearly used by original paper?** No reproducible filter details are reported.
- **Exact implementation known?** No. Cutoffs, filter family/order, phase mode, padding, and edge handling are not established.
- **Status: NOT CURRENTLY JUSTIFIED.**

### 5. Notch filtering

- **Meaning:** Strongly reduce a narrow frequency band, often to target a known electrical-line interference frequency.
- **What it changes:** Alters values across time by suppressing content around the selected frequency; can affect nearby frequencies and produce boundary/ringing effects.
- **Why someone might consider it:** Only if there is project-specific evidence that a particular narrow-band interference is present and a decision that addressing it is appropriate.
- **Evidence for this project:** A guide proposed 50/60 Hz alternatives. The reviewed paper/project evidence does not establish the local line frequency or demonstrate mains contamination.
- **Evidence against / limitation:** Which frequency is relevant is unknown; a “50 or 60” proposal is not an evidence-based parameter choice. Nearby signal content could be removed.
- **Risk:** Unnecessary signal modification, ringing, and inconsistent processing if the wrong frequency or bandwidth is used.
- **Shape effect:** Usually retains the time/channel shape but changes the sample values.
- **Leakage possibility:** A fixed notch is not inherently leaky; deciding its frequency after examining held-out data or outcomes is test-set contamination.
- **Clearly used by original paper?** Not specified.
- **Exact implementation known?** No: target frequency, width, design, and edge policy are unknown.
- **Status: NOT CURRENTLY JUSTIFIED.**

### 6. ICA / artifact removal

- **Meaning:** Independent Component Analysis (ICA) tries to represent recorded channels as mixtures of simpler signal sources. A person then identifies components thought to be artifacts and removes them before reconstructing the channels.
- **What it changes:** Reconstructed channel values change. If removed components are projected back into the same channels, dimensions can stay the same; exact effects depend on the decomposition and component choices.
- **Why someone might consider it:** The base paper says ICA was used for noise removal. It could be relevant to understanding or attempting to reproduce that paper's upstream cleaning.
- **Evidence for this project:** ICA/noise removal is explicitly reported by He et al. before trial division. The same paper does not identify the ICA algorithm, software, component count, or criteria for removing components.
- **Evidence against / limitation:** The state of the exact downloaded arrays is unresolved. Rerunning ICA may repeat an operation already applied, and “remove ocular/cardiac artifacts” is more specific than what the paper establishes. The project has no verified ocular/ECG channel map to support that claim.
- **Risk:** Removing genuine brain activity or retaining artifacts; inconsistent human component decisions; changing the paper's data twice; and using information from held-out participants when fitting a decomposition.
- **Shape effect:** It may retain 56 channels and 385 positions after reconstruction, but component representation itself has a different component axis; shape depends on whether output is reconstructed EEG or components.
- **Leakage possibility:** Potentially yes. ICA is fitted to data. Fitting it across training and held-out participants makes held-out signal information part of the learned transform; fitting separately within training folds avoids that form of cross-partition fitting. Component selection based on labels/test scores is also leakage.
- **Clearly used by original paper?** Yes at the high level: the paper says the original data were preprocessed with ICA to remove noise.
- **Exact implementation known?** No. Algorithm/software, fitting scope, rank, component selection, artifact classes, and reconstruction are not specified. Whether local arrays embody that step is unknown.
- **Status: SUPPORTED** for the limited claim that the paper reports ICA upstream; **UNKNOWN** for the local files' ICA state and exact method. This does not support an automatic rerun.

### 7. Baseline correction

- **Meaning:** Choose a time interval as a reference and subtract its average from the rest of the signal, often separately for each channel and trial.
- **What it changes:** Shifts values based on the chosen interval; generally retains sample/channel count.
- **Why someone might consider it:** If a trial has a meaningful pre-event baseline, subtraction may express activity relative to that interval.
- **Evidence for this project:** The available project/paper audits do not identify a baseline interval or state that baseline correction was performed.
- **Evidence against / limitation:** These arrays are already trial-shaped; the time origin and whether any portion is a valid baseline are unknown. The unexplained 385th position does not identify a baseline.
- **Risk:** Subtracting a non-baseline interval or a signal that contains task activity can distort values; it can also erase a meaningful between-trial amplitude offset.
- **Shape effect:** Usually none.
- **Leakage possibility:** A fixed within-trial baseline does not inherently use other participants' data. Learning/selecting a baseline rule from held-out results can contaminate evaluation.
- **Clearly used by original paper?** Not specified.
- **Exact implementation known?** No baseline bounds, reference method, or correction scope are reported.
- **Status: NOT CURRENTLY JUSTIFIED.**

### 8. Cropping 385 samples to 384

- **Meaning:** Delete one of the 385 stored time positions from each trial, usually the final position, to create a 384-position input.
- **What it changes:** Removes an observed EEG value from every trial and shortens the time axis.
- **Why someone might consider it:** The paper reports 1.5 seconds at 256 Hz, whose product is 384, while the data and architecture table report 385 positions. Someone may try to reconcile the mismatch by cropping.
- **Evidence for this project:** Binary inspection documents 385 positions in every trial. He et al.'s architecture table reports a length of 385. Existing project records explicitly state that all 385 positions are to be preserved pending clarification.
- **Evidence against / limitation:** Neither source explains the final position or establishes that it is extra padding. The arithmetic alone does not identify which sample, if any, should be removed.
- **Risk:** Irrecoverable loss of signal and an unsupported interpretation of the data; implementation may cease to match the paper's reported model length.
- **Shape effect:** Yes: changes trial shape from `(385, 56)` to `(384, 56)`.
- **Leakage possibility:** Cropping itself does not estimate a statistic. Choosing whether/which sample to crop based on test outcomes is still test-set contamination.
- **Clearly used by original paper?** The model input length of 385 is clearly reported; a crop to 384 is not reported.
- **Exact implementation known?** The proposed crop is mechanically clear, but its scientific justification and correct sample are unknown.
- **Status: CONFLICTS WITH VERIFIED INFORMATION.** Cropping conflicts with the verified stored shape and documented preservation constraint; this status does not claim the 385th position's physical meaning is known.

### 9. Resampling

- **Meaning:** Convert a signal to a different sample rate by calculating values on a new time grid.
- **What it changes:** Changes the number and values of time positions. Upsampling adds estimated positions; downsampling removes positions and needs care to avoid frequency aliasing.
- **Why someone might consider it:** A different model or comparison might require a different time resolution, or a verified sampling-rate mismatch might need correction.
- **Evidence for this project:** The paper reports 256 Hz, and the project uses that as the reported acquisition rate. The EEG headers do not encode a sampling rate, but no project requirement for a different rate has been established.
- **Evidence against / limitation:** The physical time definition of the 385 positions is unresolved. Resampling before resolving that coordinate convention could assign incorrect time intervals or silently combine two different changes.
- **Risk:** Loss or invention of temporal detail, aliasing, changes to spectral content, and inability to compare directly with the released samples.
- **Shape effect:** Yes in the time dimension in most uses; exact result depends on target rate, duration, and endpoint convention.
- **Leakage possibility:** A fixed resampling ratio is not inherently leaky. Choosing the ratio using held-out outcomes or estimating global properties from test participants can contaminate the evaluation.
- **Clearly used by original paper?** Not specified; the paper reports 256 Hz.
- **Exact implementation known?** No target rate, resampling method, anti-alias filter, or handling of the 385th point is defined.
- **Status: NOT CURRENTLY JUSTIFIED.**

### 10. Temporal windowing / re-epoching

- **Meaning:** Cut a recording into shorter or overlapping time segments (“windows” or “epochs”). The released arrays are already arranged as trials.
- **What it changes:** Can change the time length and create more examples from each original trial. Overlapping windows can share many of the same EEG values.
- **Why someone might consider it:** A later experiment might ask about shorter time patterns or use multiple segments. The base paper itself reports trials/segments.
- **Evidence for this project:** Local arrays are already trial-shaped, with 385 positions per trial. The base paper reports 33,902 trials of nominal 1.5 seconds and a model input length of 385.
- **Evidence against / limitation:** The continuous recordings, precise trial boundaries, time origin, overlap policy, and 385-position convention are not available in the reviewed evidence. There is no basis to reconstruct or subdivide the original acquisition epochs precisely.
- **Risk:** Incorrect boundaries, artificial duplication, changing the prediction unit, and highly related examples appearing to be independent.
- **Shape effect:** Yes: re-epoching changes positions per example and often the number of examples.
- **Leakage possibility:** High if windows from one participant or overlapping portions of the same source are divided across train/test. Keep all windows/trials from one participant in the same partition; any window settings selected from test outcomes also leak.
- **Clearly used by original paper?** Trial division is reported, but the exact boundary construction and windowing implementation are not.
- **Exact implementation known?** No event timing, start/end times, overlap, or segmentation rule is established.
- **Status: NOT CURRENTLY JUSTIFIED** for additional windowing or re-epoching. Existing released trial boundaries can be described as supplied, without claiming how they were created.

### 11. Channel removal / reordering

- **Meaning:** Drop selected channel columns or change their order. A physical electrode map would additionally assign names/locations to columns.
- **What it changes:** Dropping a channel removes its signal; reordering changes which column is associated with each position in any downstream operation.
- **Why someone might consider it:** A future analysis might require a documented subset or named electrode positions. That requires a verified mapping first.
- **Evidence for this project:** EEG arrays have 56 channel columns; `chan.mat` has 60 entries. Their exact correspondence and the four unmatched entries are unknown. Current project records require retaining all 56 columns in stored order and prohibit assuming physical identities.
- **Evidence against / limitation:** The mismatch prevents evidence-based selection or electrode-specific interpretation. The source paper reports 56 channels but does not provide the data-column ordering.
- **Risk:** Wrong electrode identity, accidental misalignment, lost information, or unjustified spatial/physiological claims.
- **Shape effect:** Removal changes the channel dimension; reordering preserves shape but changes column meaning/order.
- **Leakage possibility:** A fixed, externally specified mapping does not inherently leak. Choosing channels based on all participants, labels, or test performance can leak evaluation information.
- **Clearly used by original paper?** The use of 56 channels is reported. Removal/reordering procedure and name-to-column mapping are not.
- **Exact implementation known?** No verified mapping from 56 data columns to 60 metadata entries.
- **Status: CONFLICTS WITH VERIFIED INFORMATION** for dropping or reordering columns under current project constraints. The physical channel map itself is **UNKNOWN**.

## Other preprocessing or representation operations already raised in project documents

### 12. Artifact rejection, repair, or channel interpolation

- **Meaning:** Reject a trial judged too noisy, alter/repair a trial, or estimate a missing/bad channel from other channels.
- **What it changes:** Rejection removes examples; repair/interpolation changes signal values and possibly channel data.
- **Why someone might consider it:** A documented quality-control rule could address known corrupt recordings/channels.
- **Evidence for this project:** Existing audits note no quality labels, thresholds, trial-rejection criteria, or repair procedure in the reviewed sources. The paper's broad ICA statement does not establish trial rejection or interpolation.
- **Evidence against / limitation:** No definition of “bad” or verified channel map exists; selective removal may change group representation.
- **Risk:** Data loss, group-dependent selection, altered channel covariance, and biased results if thresholds are chosen after seeing held-out data.
- **Shape effect:** Rejection changes example count; interpolation generally preserves channel count but changes values.
- **Leakage possibility:** Yes, if quality thresholds or repair decisions are fitted/selected using held-out distribution, labels, or scores. Trial rejection can also cause participant representation to differ across folds.
- **Clearly used by original paper?** Not specified.
- **Exact implementation known?** No thresholds, criteria, or interpolation method.
- **Status: NOT CURRENTLY JUSTIFIED.**

### 13. Re-referencing

- **Meaning:** Recalculate each channel relative to a different voltage reference, such as a selected electrode or an average of electrodes.
- **What it changes:** Changes values across channels because the chosen reference is subtracted or otherwise applied.
- **Why someone might consider it:** EEG voltage is measured relative to a reference, so a known acquisition/reference issue or a pre-specified scientific comparison might motivate changing it.
- **Evidence for this project:** Available audits say the reference scheme for the 56 columns is unresolved. The 56-to-60 channel mapping is also unresolved.
- **Evidence against / limitation:** Without knowing the recorded reference and channel identities, a new reference cannot be fully interpreted or reliably constructed.
- **Risk:** Can change all channel signals, introduce shared dependencies between channels, or produce a misleading physical interpretation.
- **Shape effect:** Usually none if only values are recomputed across the same channel set.
- **Leakage possibility:** A fixed reference formula is not inherently leaky, but a data-selected reference or one chosen using held-out performance can contaminate evaluation.
- **Clearly used by original paper?** Not specified.
- **Exact implementation known?** No acquisition reference, target reference, or electrode list is verified.
- **Status: UNKNOWN.**

### 14. Spectral power features and theta/beta ratio

- **Meaning:** Convert a time signal into summaries of frequency content. Welch power spectral density (PSD) is one proposed way to estimate power by averaging frequency estimates over segments. Theta/beta ratio divides estimated power in two selected frequency ranges.
- **What it changes:** Replaces or supplements the time series with frequency-band summaries/ratios. It is feature construction, not simply cleaning.
- **Why someone might consider it:** The project guide raises band power and TBR as candidate experiments. They could test a distinct scientific/model representation if the question and parameters are decided.
- **Evidence for this project:** These features appear in project proposals only. The base-paper description reviewed here does not establish them as required inputs to its EEG-Transformer or as a validated biomarker for this cohort.
- **Evidence against / limitation:** Frequency bands, PSD window/overlap/detrending, units, denominator handling, and interpretation are unspecified. Feature decisions can encode assumptions not settled by this dataset's evidence.
- **Risk:** Different PSD settings produce different features; ratio instability; loss of time detail; and confirmation bias if TBR is treated as a proven diagnostic marker.
- **Shape effect:** Yes: time positions are replaced by frequency bins/bands or derived features, changing the input representation.
- **Leakage possibility:** Yes if frequency/band/feature selection is informed by all data or test outcomes. Any learned feature selection must be done within training folds.
- **Clearly used by original paper?** The reviewed paper audit does not establish these as the Transformer input preprocessing.
- **Exact implementation known?** No.
- **Status: NOT CURRENTLY JUSTIFIED** as a mandatory transformation; the proposal can be considered separately as a research decision.

### 15. Loading strategy, dtype, and derived-data writing (not signal filters)

- **Meaning:** Choose how files are read and stored, whether values are converted from `float32`, and where any future transformed copy is written.
- **What it changes:** Filewise/bounded loading should not change EEG values. Changing dtype can round values. Writing a derived copy changes storage, not necessarily signal meaning.
- **Why someone might consider it:** The source consists of seven large files; project documents set memory and source-preservation constraints.
- **Evidence for this project:** Arrays are documented as `float32`; source files are read-only; filewise/bounded reading is an engineering constraint. Existing audits report that no derived preprocessed data has been created.
- **Evidence against / limitation:** A memory strategy is not evidence for any signal transformation. Silent dtype conversion or overwriting source files would violate stated project constraints.
- **Risk:** Unintentional precision loss, large memory copies, source-data mutation, or confusing derived data with the original.
- **Shape effect:** Loading/writing alone: none. Dtype conversion: shape generally unchanged.
- **Leakage possibility:** Loading alone does not leak. Computing global statistics during a full-dataset scan can leak if those statistics are later used for transformation.
- **Clearly used by original paper?** Exact local file/loading implementation is not established by the paper.
- **Exact implementation known?** Project constraints are documented; a future I/O implementation is not part of this evidence audit.
- **Status: SUPPORTED** as data-integrity and engineering constraints only; not a signal-preprocessing decision.

## Summary by current evidence status

| Status | Operations / claims |
|---|---|
| **SUPPORTED** | No additional signal modification is supported as a reproducible reference condition, not as proof that the released data is untouched. The paper's high-level statement that ICA was applied upstream is supported. Preserving source values/dtype and using bounded reads are documented constraints. |
| **REASONABLE BUT NEEDS A DECISION** | Normalization/standardization: there is a reason to discuss possible model scaling, but no evidence chooses a scaling method or axes. |
| **NOT CURRENTLY JUSTIFIED** | Detrending/DC removal, new band-pass or notch filters, baseline correction, resampling, additional temporal windowing/re-epoching, artifact rejection/repair/interpolation, and spectral power/TBR feature construction. Rerunning ICA is not justified merely by the paper's upstream ICA statement. |
| **CONFLICTS WITH VERIFIED INFORMATION** | Cropping 385 to 384; dropping or reordering the 56 stored channel columns under the current preservation constraint. |
| **UNKNOWN** | Local arrays' exact upstream ICA/filtering/reference history; re-referencing details; physical channel mapping; exact implementation of the paper's preprocessing; and any transformation parameters not documented above. |

## A. Operations we can confidently discuss

- The project can compare an **as-released, no-additional-transformation** condition with clearly specified alternatives later, while explicitly avoiding the unsupported label “unprocessed raw EEG.”
- The paper reports ICA/noise removal before trial division, but only at a high level. This supports discussion of what the paper claims, not faithful reproduction or an automatic second ICA pass.
- The stored shape is `(385, 56)` per trial. Preserving all positions and columns is the current documented constraint; the physical meaning of position 385 and the channel identities are not thereby resolved.
- The distinction between fixed operations and fitted/data-selected operations is important: fitted means, ICA decompositions, quality thresholds, and feature selection need training-only rules to avoid leakage.
- The local trial-to-subject grouping crosswalk has a later dedicated validation report, but that does not settle signal processing.

## B. Operations requiring an explicit decision

- Whether the first experiment should be strictly “as released” or should test any additional transformation.
- Whether normalization is needed, and if so, its method, axis, fitting population, and application timing.
- Whether the project's goal includes exact reproduction of upstream ICA, despite the missing ICA details and unknown local file state.
- Whether further investigation can establish the local source-processing history, acquisition reference, physical time coordinates, or channel mapping.
- Whether any future experiment will use re-referencing, artifact rejection/repair, spectral power/TBR, resampling, or additional temporal windows—and what exact scientific question would justify it.
- For any selected adaptive/data-dependent operation: its complete parameters, training-fold fitting scope, participant grouping, and held-out evaluation protections.

## C. Operations currently unsupported

The available evidence does not currently justify automatically applying detrending/DC removal, 0.5–45 Hz band-pass filtering, 50/60 Hz notch filtering, baseline correction, resampling, additional windowing/re-epoching, artifact rejection/interpolation, spectral power/TBR, or a new reference. The paper's ICA statement does not justify rerunning ICA without establishing the local state and specifying the procedure.

Cropping to 384 and channel removal/reordering do not merely lack evidence: they conflict with the current verified-data preservation constraint. This audit does not decide whether that constraint should ever change; any change needs new evidence and an explicit decision.

## D. Questions to answer before preprocessing can be frozen

1. Is the first experiment intended to use released values without additional signal changes, or to test a specified transformation? How will the distinction from the unknown upstream processing be documented?
2. Is exact reproduction of the paper's ICA step a goal, or is the paper used only as context? What evidence, if any, can establish whether the exact local arrays already underwent ICA?
3. What does the paper mean by ICA “to remove noise,” and can its algorithm, fitting scope, and component-removal criteria be recovered? If not, will exact reproduction be declared unavailable?
4. Is normalization needed for the selected model/experiment? If so, which axis and rule, and will all fitted statistics use training participants only?
5. Is there evidence in these files for drift, electrical-line contamination, or a valid baseline interval that would motivate detrending, notch, or baseline correction?
6. Can the sample-time convention and the 385th position be resolved from source documentation? Until then, should all positions remain in the representation?
7. Can a verified mapping from the 56 data columns to physical channel names and a verified acquisition reference be obtained? Until then, which analyses must treat columns as anonymous?
8. Is any new temporal windowing, resampling, artifact rejection, repair, or frequency-feature branch part of the research question? What exact parameters and shape changes would it require?
9. For each operation with fitted or data-selected settings, how will fitting and choices be confined to training folds, and how will all examples from a participant remain in one partition?
10. What objective evidence and explicit approval are required to alter the existing preservation constraints?

## Sources reviewed

- [Dataset investigation](../dataset-investigation.md)
- [Dataset and preprocessing audit](../ADHD_EEG_PREPROCESSING_AUDIT.md)
- [Earlier dataset-preprocessing audit](../dataset-preprocessing-audit.md)
- [Base-paper analysis](../base-paper-analysis.md)
- [ML methodology and research-gap audit](../ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md)
- [Dataset and data-contract audit](02_DATASET_AND_DATA_CONTRACT_AUDIT.md)
- [Project-state audit](01_PROJECT_STATE_AUDIT.md)
- [Base-paper and research-evidence audit](03_BASE_PAPER_AND_RESEARCH_EVIDENCE_AUDIT.md)
- [Sprint 2 preprocessing and data contract](../SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md)
- [Trial/subject mapping validation report](../trial_subject_mapping_validation.md)
- He et al. (2023), “Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model,” *Journal of Neural Engineering*, 20 (2023), 056013. DOI: [10.1088/1741-2552/acf7f5](https://doi.org/10.1088/1741-2552/acf7f5).

This audit records evidence and open questions only. It does not select a final preprocessing pipeline or authorize implementation.
