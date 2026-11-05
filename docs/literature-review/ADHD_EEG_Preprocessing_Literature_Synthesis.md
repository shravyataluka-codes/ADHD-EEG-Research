# ADHD EEG Preprocessing Literature Synthesis

## Purpose

This document turns the literature review into a preprocessing-focused summary for the project. It records what is known from the prior cohort papers, what remains unresolved, and what the current repository is allowed to treat as an approved preprocessing contract.

## Evidence from the literature

The critical literature review establishes the following points for the Dresden corpus used by the core ADHD/ADD/HC studies:

- The cohort is the same 144-child Dresden event-related EEG dataset reported by Vahid et al. (2019), Wang et al. (2022), and He et al. (2023).
- The reported cohort counts are 44 healthy controls (HC), 52 ADD, and 48 ADHD children.
- The EEG representation is reported as 56 channels and a 256 Hz sampling rate.
- The trial tensor is reported as 385 time points, consistent with a nominal 1.5-second window.
- The project stores the released sample arrays as `(385, 56, trials)` with `float32` dtype.
- The papers describe event-related EEG acquired during a stimulus-locked task, not a resting-state recording.
- The original paper states that the source data were preprocessed using ICA to remove noise, and Vahid et al. also describe artifact rejection and baseline correction in the source-study pipeline.

The literature review also makes clear that the exact preprocessing details are not fully reproducible from the papers alone. The project cannot currently infer a single exact implementation from the published descriptions.

## What the survey does and does not support

### Supported facts for the project

- The local dataset is associated with the Dresden EEG cohort used by the reviewed papers.
- The project should preserve the released 385-sample trial structure unless a later, explicitly approved contract decides otherwise.
- A 256 Hz sampling rate is the paper-reported contract for the downstream EEG tensor.
- The literature supports a signal-processing context involving artifact rejection, ICA-based denoising, and stimulus-locked trial processing.
- Participant-wise evaluation is not a unique contribution to the field; it has already been reported in the cohort literature.

### Unresolved or weakly supported facts

The literature review does not establish the following with enough detail to authorize implementation:

- the exact ICA configuration used by the original source data pipeline,
- the precise filtering parameters,
- whether a notch filter was used at 50 Hz or 60 Hz,
- how the four missing channels from the 60-electrode metadata were handled,
- the exact interpretation of the 385th sample position,
- whether the local `.mat` files are already post-ICA processed data or raw acquisitions,
- the patient-to-file mapping and clinical diagnosis provenance at the per-trial level.

These gaps mean the repository should treat literature review findings as design context, not as a finished implementation recipe.

## Preprocessing implications for this repository

The project should not treat the literature survey as evidence that a default EEG pipeline is allowed. Instead, it should use the survey to frame the following constraints:

1. Raw EEG evidence must remain immutable until an explicit preprocessing contract is approved.
2. Any signal-changing step (filtering, notch, ICA, amplitude scaling, epoch trimming, baseline correction, or feature engineering) must be documented and frozen before execution.
3. Learned preprocessing operations must be fit only on training data, never on all participants or the held-out test set.
4. Any provenance or mapping assumption must be explicitly validated before it is used in a model contract.
5. The review supports a conservative operational baseline: preserve the existing sample tensor and only apply a documented, approved transformation later.

## Project-specific decision summary

The current repository status is intentionally conservative:

- No signal-changing preprocessing is currently approved for the cohort.
- Existing preprocessing code is treated as a configurable implementation scaffold, not evidence that the pipeline is scientifically valid.
- The literature survey supports the need for rigorous methodological control and leakage avoidance, but does not authorize a blind application of a generic ERP cleaning pipeline.

The practical project decision is therefore:

- preserve the raw dataset,
- keep the project contract at the released 385 x 56 representation,
- require explicit approval for any changes to the signal,
- use training-only fitting and participant-disjoint evaluation for any future chosen preprocessing or feature pipeline.

## Implementation contract to keep for future work

If a future protocol chooses to run preprocessing, the following should be treated as the minimum documentation requirement before code execution:

- exact source file hashes and timestamps,
- explicit filter parameters,
- explicit ICA settings and review protocol,
- subject-level exclusion decisions, not automatic component removal,
- train/test partition isolation,
- pre/post-processing manifest for each subject or file,
- final artifact hashes and output paths.

## Bottom line

The literature survey supports a disciplined, evidence-based preprocessing stance: EEG preprocessing is a real methodological decision in this cohort, not a default convenience. The papers establish a likely EEG pipeline context (ICA, artifact rejection, ERP windows), but they do not provide enough reproducible detail to justify a single inherited implementation. The repository should therefore keep preprocessing explicit, auditable, and approval-gated.
