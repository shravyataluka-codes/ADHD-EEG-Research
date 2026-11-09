# ADHD EEG Research: Sprint 3-8 Execution Plan

**Status:** Prospective execution plan. No model, training loop, real-data split, or inference command is implemented by this document.

**Current roadmap authority:** This plan records the user-confirmed four-phase, eight-sprint roadmap and the selected compact EEG Conformer-style primary architecture candidate. It supersedes conflicting *prospective architecture/model-direction recommendations* in older planning documents for Sprints 3-8. Historical audits and decision records are preserved as evidence of prior decisions; they are not edited here. The selected architecture is a research-design choice, not evidence that it outperforms EEGNet.

**Evidence boundary:** Sprint 1 engineering and Sprint 2 engineering checkpoints are treated as completed. This does not promote their candidate participant/diagnosis mappings to verified status or pass scientific gates for real-data training or participant-independent evaluation.

## 1. Executive plan

| Phase | Sprint | Scope | Can begin now? | Real-data status |
|---|---|---|---|---|
| 2 | 3. EEG input representation and encoder | Fixed `(B, 385, 56)` contract; temporal convolutional tokenizer; compact Conformer encoder; synthetic tests. | Yes, synthetic-only. | No real training. Use identity signal handling only unless the preprocessing gate is approved. |
| 2 | 4. Classification head and model integration | Three-logit head and model wrapper; adapter around the existing unlabeled Dataset without changing its contract. | Yes, synthetic-only. | No real labels or predictions. |
| 2 | 5. Training pipeline and validation | Reproducible loop, synthetic validation, checkpoint format, finite/shape guards. | Yes, synthetic-only. | Real training requires verified labels/participants, approved input protocol, and split rules. |
| 3 | 6. Participant-independent evaluation | Freeze and implement approved participant-level protocol; generate auditable folds only after verified mapping and labels. | Protocol/test scaffolding can proceed synthetically; real assignments cannot. | Blocked until mapping, diagnosis, preprocessing/input, and protocol gates pass. |
| 3 | 7. Ablations and analysis | Matched Conformer/EEGNet and component ablations, uncertainty and error analysis. | Design and synthetic tests can proceed. | Requires valid Sprint 6 results and approved real-data experiment. |
| 4 | 8. Results, reproducibility, and publication | Lock outputs, rerun checks, result tables, methods and limitations. | Artifact schema and templates can be drafted. | Requires completed, valid experiments; no publishable result exists yet. |

## 2. Current repository contracts and constraints

### Existing code to reuse

- [data_access.py](../src/adhd_eeg/data_access.py): `TrialManifest`, `EEGTrialStore`, `TRIAL_SHAPE`, `TRIAL_DTYPE`; read-only bounded access to HDF5-backed MAT files. Manifest participant/group fields are explicitly candidate mappings.
- [dataset.py](../src/adhd_eeg/dataset.py): `EEGTrialDataset` yields a `torch.float32` signal of `(385, 56)` and a separate source-local metadata dictionary. It intentionally yields no target, participant ID, clinical group, or verified diagnosis.
- [evaluation.py](../src/adhd_eeg/evaluation.py): generic group-disjoint train/validation/test utility. It only separates caller-supplied group IDs; it does not verify identities, stratify at participant level, provide nested CV, or authorize real-data splits.
- [preprocessing.py](../src/adhd_eeg/preprocessing.py) and [preprocess_cli.py](../src/adhd_eeg/preprocess_cli.py): executable filter/notch/ICA pipeline, currently **not approved for research use** by the Sprint 2 data contract. Do not call it from the model path by default.

### Input/output contract for Sprints 3-5

- Encoder input: `torch.float32`, shape `(B, 385, 56)`, finite values, stored time-index order, and stored channel-column order.
- The 56 columns are anonymous fixed-order features. Do not infer electrode labels, montage coordinates, or scalp geometry.
- Preserve all 385 input positions. The encoder's documented convolutional tokenization changes sequence length internally; it does not crop, resample, or rewrite the source trial.
- Encoder output: `(B, 97, 64)` sequence embeddings.
- Classifier output: three **logits** in fixed index order `HC`, `ADD`, `ADHD`; apply softmax only for inference/reporting. Logits are used with cross-entropy loss.
- Dataset contract remains unchanged: sample tensor and source-local metadata only. For synthetic supervised training, use a test-only fixture that returns fabricated targets. For any future real experiment, join targets through a separate verified manifest/wrapper; never promote the existing candidate fields.

### What can and cannot wait

| Limitation or decision | Synthetic engineering | Real experiment consequence |
|---|---|---|
| Trial-to-participant join is unverified | Does not block encoder, head, or fake-label training tests. | Blocks verified participant-level splits and unseen-participant claims. |
| Diagnosis-label provenance is unresolved; `y_stim` rows 1-3 are not clinical labels | Does not block testing class indices 0/1/2 on fabricated samples. | Blocks all real supervised classification and clinical-group claims. |
| Meaning of sample 385 is unresolved | Does not block using fixed tensor index positions and preserving all input positions. | Blocks physical-time interpretation, cropping, and any temporal claim dependent on the endpoint. |
| 56-vs-60 channel mapping is unresolved | Does not block treating 56 columns as consistently ordered anonymous features. | Blocks electrode-specific, spatial, topographic, or named-channel claims; requires consistent column order across inputs. |
| Original preprocessing state is uncertain | Does not block identity-path tests. | Must be described as unknown for use-as-released; blocks a faithful preprocessing reproduction or re-ICA without evidence. |
| Existing filter/notch/ICA implementation is not automatically approved | Does not block the model code. | Any real signal transformation needs an explicit approved protocol and fold-safe fit scope. The default planned path is identity/no new transform. |
| Split and model-selection protocol not frozen | Does not block generic synthetic split/training tests. | Blocks real split generation, model selection, training, and evaluation. |
| No external cohort/clinical validation | Does not block research code testing. | Limits claims; results on this cohort alone are not clinical or external validation. |

No task in this plan requires replacing the dataset or restarting the already documented structural/provenance investigations. Do not contact dataset authors as a default action; only reconsider source recovery if the research team chooses to reopen that evidence question.

## 3. Sprint 3: EEG Input Representation and Encoder

**Objective:** Implement and verify the Conformer encoder as a tensor-to-sequence module using the frozen stored trial shape, with synthetic inputs only.

**Engineering that can begin now:** Yes. This sprint does not require diagnosis labels, participant identities, real-data splits, filtering, or model training on the cohort.

**Tasks in execution order:**

1. Add the shape/dtype/finite-value interface contract and architecture configuration constants.
2. Implement the temporal convolutional tokenizer described in Section 4; preserve time-index coverage with explicit padding and stride.
3. Implement fixed sinusoidal positional encoding over the resulting 97 token indices. These are sample-index positions, not calibrated physical timestamps.
4. Implement two compact Conformer-style encoder blocks with explicit residual, normalization, dropout, attention, and local convolution operations.
5. Return sequence embeddings only; do not put a class head, class labels, Dataset label join, or training loop in this sprint.
6. Add stage-shape and gradient smoke tests on deterministic synthetic batches.

**Existing files to reuse:** [data_access.py](../src/adhd_eeg/data_access.py), [dataset.py](../src/adhd_eeg/dataset.py), [pyproject.toml](../pyproject.toml), [test_dataset.py](../tests/test_dataset.py).

**New files expected:**

- `src/adhd_eeg/conformer.py`: tokenizer, positional encoding, encoder block, and `EEGConformerEncoder`.
- `tests/test_conformer.py`: encoder interface, tensor shapes, invalid input, determinism, finite output, and gradient tests.

**Dependencies:** PyTorch optional extra; no MNE, scikit-learn, labels, or real data required. The current `pytorch = ["torch"]` extra is unpinned; keep imports out of the mandatory base data-access path. The current Dataset tests use `unittest.skipIf(torch is None, ...)`, which is the local convention.

**Unit and integration tests:**

- Input `[B, 385, 56]` produces `[B, 97, 64]` for `B=1` and `B>1`.
- Reject wrong rank, wrong sample/channel dimensions, non-float32 inputs (unless the interface explicitly documents a conversion), empty batch, NaN, and Inf.
- Verify exact intermediate shape sequence from Section 4.
- Verify positional encoding is deterministic, finite, and does not add trainable parameters.
- Verify output and gradients are finite on a small synthetic signal.
- Verify no source metadata/label object is accepted by or concatenated into the model input.
- Optional integration: read a temporary HDF5 fixture through `EEGTrialDataset`, discard its metadata explicitly, batch signals, and call encoder; preserve the Dataset return structure.

**Acceptance criteria:**

- The documented input/output contract is enforced and all stage shapes match.
- Model parameter count matches the implementation within the documented accounting (actual implemented encoder: 185,664 trainable parameters, asserted in unit tests; proposed combined model with prospective Sprint 4 classification head: 194,115 trainable parameters).
- Repeated `eval()` forward passes with identical input are identical within floating-point tolerance.
- `train()` produces finite gradients for trainable encoder parameters.
- No real labels, splits, dataset-wide normalization, or signal transformations are introduced.

**Research limitations and blockers:** The design does not establish that the 56 columns represent known electrodes or that each sample index equals a known physical time. It assumes channel columns are consistently ordered across trials. These limitations do not block synthetic encoder implementation but constrain later real inputs and interpretation.

**Definition of done:** Encoder module and synthetic tests pass; the layer-by-layer shape/parameter record in this plan is consistent with the implemented module; no classifier or training code is added under Sprint 3.

## 4. Selected Conformer candidate and tensor-shape specification

This is the specified **primary architecture candidate**, not an experimentally selected winner. It is intentionally compact and temporal. It does not use electrode coordinates, spatial convolution across a presumed scalp geometry, handcrafted spectral/TBR features, or channel names.

### Layer specification

| Stage | Operation | Shape | Parameters / rationale |
|---|---|---|---|
| 0 | Input validation | `(B, 385, 56)` | Fixed source contract; requires float32, finite values, unchanged stored order. |
| 1 | Add singleton feature-map axis and permute | `(B, 1, 385, 56)` | Conv2d convention: batch, feature-map, time, anonymous channel-column. A layout change only; no values are cropped or reordered along the channel axis. |
| 2 | Temporal Conv2d | `1 -> 16` maps; kernel `(15, 1)`, stride `(4, 1)`, padding `(7, 0)`, bias enabled; output `(B, 16, 97, 56)` | Same 15-index temporal kernel is applied independently across channel columns. Padding 7 and stride 4 produce `ceil(385/4)=97` tokens and cover both ends. Kernel sizes are in sample-index units; the paper-reported 256 Hz is not treated as independently confirmed acquisition metadata. |
| 3 | Permute and flatten feature map/channel values within each time token | `(B, 97, 56 * 16) = (B, 97, 896)` | Preserves a consistent feature ordering. The model learns from anonymous channel-column values; it does not attach scalp positions. |
| 4 | Linear projection `896 -> 64`, LayerNorm(64), GELU, dropout 0.1 | `(B, 97, 64)` | Projects each token to `d_model=64`. Projection mixes the 56 fixed-order channel features and temporal filter maps without claiming electrode identity. |
| 5 | Fixed sinusoidal positional encoding | `(B, 97, 64)` | Adds order information at the 97 token positions; no learnable positional parameters and no physical-time interpretation. |
| 6 | Two Conformer-style blocks, `d_model=64`, 4 attention heads, FFN width 128, conv kernel 15, dropout 0.1 | `(B, 97, 64)` | Two heads per 32 dimensions? Precisely 4 heads × 16 dimensions each. Macaron-style half-step FFNs model channel-wise token transforms; self-attention relates distant tokens; depthwise temporal convolution supplies local token context. Residuals and pre-normalization support stable optimization. |
| 7 | Concatenate temporal mean and temporal max | `(B, 128)` | Mean summarizes distributed evidence; max retains salient token activations. It does not claim that a peak is physiologically meaningful. Pooling choice remains prespecified for matched experiments, not justified by superiority. |
| 8 | Classification head: Linear `128 -> 64`, GELU, dropout 0.2, Linear `64 -> 3` | logits `(B, 3)` | Fixed index mapping: `0=HC`, `1=ADD`, `2=ADHD`. Softmax is for inference scores only; `CrossEntropyLoss` receives logits. |

### Conformer block definition

For each block, keep the token length and feature dimension unchanged. Use four attention heads and dropout 0.1 throughout. The planned block has:

1. Pre-normalized feed-forward network `64 -> 128 -> 64`, GELU, dropout 0.1; residual scale 0.5.
2. Pre-normalized multi-head self-attention with four heads, no causal mask, output dropout 0.1; residual connection.
3. Pre-normalized convolution module: pointwise `Conv1d(64, 128, kernel_size=1)` without bias, GLU to 64 features, depthwise `Conv1d(64, 64, kernel_size=15, padding=7, groups=64)`, BatchNorm1d(64), SiLU, pointwise `Conv1d(64, 64, kernel_size=1)`, dropout 0.1; residual connection. BatchNorm running statistics are model state and must only be updated in training mode on the applicable training partition.
4. Second pre-normalized feed-forward network `64 -> 128 -> 64`, GELU, dropout 0.1; residual scale 0.5.
5. Final LayerNorm(64).

Padding is only along the token/time axis. No causal mask is needed for classifying a complete, already segmented trial. Attention cost is proportional to `97^2` per head per block rather than `385^2`; this is a compute reduction, not a claim about accuracy or participant-level generalization.

### Parameter accounting

Trainable parameter accounting for the implemented encoder and proposed classification head:

- Temporal Conv2d: `16 * 1 * 15 * 1 + 16 = 256`.
- Projection: `896 * 64 + 64 = 57,408`.
- Projection LayerNorm: `2 * 64 = 128`.
- Tokenizer total: `57,792`.
- Fixed positional encoding: `0` trainable parameters.
- One Conformer block: `63,936` parameters, including two FFNs (2 × 16,704 = 33,408), Q/K/V/output self-attention projections (16,768), convolution module (13,632), and final LayerNorm (128).
- Two Conformer blocks: `2 * 63,936 = 127,872`.
- **Actual implemented encoder: 185,664 trainable parameters** (`57,792 + 127,872 = 185,664`), as verified and asserted in `tests/test_conformer.py`.
- **Proposed Sprint 4 classification head: 8,451 parameters** (`(128 * 64 + 64) + (64 * 3 + 3) = 8,451`). Note: this head is a prospective future design and is not implemented in Sprint 3.
- **Proposed encoder + head total: 194,115 trainable parameters** (`185,664 + 8,451 = 194,115`).

*Note on preliminary estimate:* The original 194,371 estimate was preliminary (which assumed 64,064 per block, or 128,128 across two blocks) and differed by 256 parameters because the first pointwise convolution in each Conformer block is configured with `bias=False` (`Conv1d(64, 128, kernel_size=1, bias=False)`), saving 128 bias parameters per block ($2 \times 128 = 256$).

The 144 paper-reported participants, not 33,902 repeated trials, are the relevant scale for generalization. A ~194k-parameter model can still overfit; the compact specification is a bounded engineering proposal, not a guarantee of data efficiency.

## 5. Sprint 4: Classification Head and Model Integration

**Objective:** Assemble the encoder and classification head into a complete logits-producing classifier and connect it to existing unlabeled samples without changing `EEGTrialDataset`.

**Engineering that can begin now:** Yes, on synthetic data only.

**Tasks in execution order:**

1. Implement a top-level classifier that accepts `(B, 385, 56)` and calls `EEGConformerEncoder`.
2. Add the `128 -> 64 -> 3` head after mean/max aggregation and expose a stable class-order constant.
3. Keep `forward()` output as logits; provide a separate prediction helper or Sprint 8 inference wrapper that applies softmax.
4. Define a synthetic labeled fixture/wrapper in tests only. Do not add candidate IDs or `y_stim` fields as real targets.
5. Integration-test DataLoader batches from a temporary synthetic MAT fixture into the model, while metadata remains outside the forward call.

**Existing files to reuse:** [dataset.py](../src/adhd_eeg/dataset.py), [data_access.py](../src/adhd_eeg/data_access.py), [test_dataset.py](../tests/test_dataset.py), [pyproject.toml](../pyproject.toml).

**New files expected:**

- `src/adhd_eeg/classifier.py`: `EEGConformerClassifier`, logits contract, class order, and prediction-score helper if warranted.
- `tests/test_classifier.py`: outputs, labels/order, loss compatibility, and Dataset integration.

**Dependencies:** Sprint 3 encoder; optional `pytorch` extra. No MNE/scikit-learn requirement.

**Unit and integration tests:**

- Logits shape exactly `(B, 3)`, finite, and in fixed class order.
- `CrossEntropyLoss` accepts integer synthetic labels in `[0, 2]`; verify backward pass.
- Softmax scores sum to 1 within tolerance, are finite, and map to declared label order. Do not call them calibrated confidence.
- Model input accepts only signal tensors; metadata is not fed into the module.
- Temporary `EEGTrialDataset` integration confirms output tensors pass through unchanged and metadata remains separately available.
- Save/load a temporary state dict and check identical `eval()` outputs.

**Acceptance criteria:** Model/head contract is independent of the Dataset implementation; output class order is tested; no real group labels or claims are introduced; parameter count for full classifier matches the architecture record.

**Research limitations and blockers:** Clinical labels and participant mappings remain blocked. Synthetic labels are solely API fixtures. The model output class names are an intended future order, not evidence that local trials already have those targets.

**Definition of done:** Encoder plus classifier run on synthetic batches, produce logits/scores under tested interfaces, and integrate with the existing unlabeled Dataset without modifying its return contract.

## 6. Sprint 5: Training Pipeline and Validation

**Objective:** Build a deterministic, auditable training loop and verify it end-to-end on synthetic data; prepare but do not execute the real-data path.

**Engineering that can begin now:** Synthetic loop, loss, optimizer, logging, temporary checkpoints, and failure handling. Real EEG training is not authorized.

**Tasks in execution order:**

1. Define a versioned training configuration: seed, batch size, epoch limit, optimizer settings, loss, validation metric, early-stopping rule, and checkpoint selection rule. Values are engineering defaults for synthetic smoke tests until a research protocol is approved.
2. Implement deterministic seeding for Python/NumPy/PyTorch (and CUDA when available); record library and hardware information.
3. Implement train/evaluate epoch functions using logits and `CrossEntropyLoss`; validation runs under evaluation mode and no gradients.
4. Implement early stopping/checkpoint selection based only on the synthetic validation set in tests. Real use requires a participant-grouped validation protocol.
5. Save checkpoint plus model configuration, class order, source/protocol IDs, and software versions. Test round-trip loading from a temporary directory.
6. Add a tiny synthetic end-to-end run that decreases training loss enough to confirm the loop executes; do not make a performance claim from this fixture.
7. Keep identity preprocessing as the default. Do not import or call `preprocessing.py` from training. Any approved future transform must be fold-fitted and configured before the real run.

**Existing files to reuse:** [classifier.py](../src/adhd_eeg/classifier.py) (Sprint 4), [dataset.py](../src/adhd_eeg/dataset.py), [pyproject.toml](../pyproject.toml); do not use the current preprocessing CLI as an implicit model transform.

**New files expected:**

- `src/adhd_eeg/training.py`: training/evaluation loop and seeding.
- `configs/conformer_synthetic.json`: explicit synthetic smoke-test configuration only; never use as a real cohort protocol.
- `tests/test_training.py`: deterministic unit and tiny end-to-end tests.

**Dependencies:** Sprint 4 model; PyTorch optional extra. Add no other package unless implementation needs it and the dependency is justified. PyTorch version should be pinned or recorded in the run manifest before a scientific run.

**Unit and integration tests:**

- Loss, gradients, optimizer update, finite metrics, and validation `no_grad`/`eval()` behavior.
- Repeat synthetic run with the same seed/config and compare split/config and predictions within a declared tolerance.
- Check checkpoint state, config, class order, and reload equivalence.
- Check missing/invalid config values fail closed.
- Confirm held-out synthetic validation data does not update parameters or BatchNorm running stats during evaluation.
- Confirm the runner does not access the real dataset path unless explicitly configured and gates are satisfied.

**Acceptance criteria:** Tiny synthetic training and checkpoint reload pass; configuration and seed are recorded; validation does not update model state; no raw source files are modified; no real-data run or result is produced.

**Research limitations and blockers:** Synthetic validation cannot select real hyperparameters, estimate generalization, validate label provenance, or prove participant separation. Real preprocessing options remain gated. Real training is blocked until verified labels/participants, protocol, and fold construction exist.

**Definition of done:** Reproducible synthetic training pipeline and checkpoint contract pass tests; real-run entry point refuses to proceed without the future protocol/provenance gate.

## 7. Sprint 6: Participant-independent Evaluation

**Objective:** Implement and run the approved participant-independent evaluation only after the independent mapping, diagnosis, and protocol gates pass.

**Engineering that can begin now:** Draft interfaces, schemas, and synthetic disjointness/leakage tests. Do not create real fold assignments using the current candidate mappings.

**Tasks in execution order:**

1. Before real split code is enabled, verify the authoritative trial locator-to-participant and participant-to-diagnosis records through the existing Sprint 2 entry criteria; this is a gate, not a repeat of the completed structural audit.
2. Freeze the estimand, outer/inner strategy, fold count or holdout, class/participant balance, seed/repeats, metric/aggregation, inclusion rules, and permitted preprocessing. Record team approval and protocol version.
3. Implement a verified metadata manifest separate from EEG tensors. Retain source hashes, trial locator, verified participant key, verified target, mapping provenance, protocol ID, and split assignment.
4. Implement the approved group split over a participant-level table and expand assignments to trials. The existing [evaluation.py](../src/adhd_eeg/evaluation.py) helper may be reused only for what it actually guarantees; it is not stratified and does not validate groups.
5. Add assertions for complete coverage, unique trial locators, one target per participant, no participant overlap, all trials of a participant in the same partition, and deterministic assignments.
6. Generate real manifests only after approvals; review realized fold distributions before training. Fit all learned transforms/weights and selection only in their allowed training partitions.
7. Run the predeclared evaluation using the Sprint 5 runner and persist out-of-fold predictions with participant-level aggregation and uncertainty metadata.

**Existing files to reuse:** [data_access.py](../src/adhd_eeg/data_access.py), [dataset.py](../src/adhd_eeg/dataset.py), [evaluation.py](../src/adhd_eeg/evaluation.py), [SPRINT_2_EVALUATION_READINESS_AUDIT.md](SPRINT_2_EVALUATION_READINESS_AUDIT.md), [data_contract.md](data_contract.md).

**New files expected (after gates):**

- `src/adhd_eeg/splits.py`: approved split strategy and assertions.
- `src/adhd_eeg/experiment_data.py`: verified-label sidecar/wrapper that composes with, rather than changes, `EEGTrialDataset`.
- `tests/test_splits.py` and `tests/test_experiment_data.py`: synthetic unit and integration coverage.
- `configs/conformer_real.json`: only after protocol approval; no default real config before then.

**Dependencies:** Sprints 1/2 engineering checkpoints; Sprint 5 runner; verified participant/diagnosis evidence; approved preprocessing/input and evaluation decisions. `scikit-learn` is optional only if the approved splitter uses it; record exact version. A standard-library splitter remains possible for a fully specified design, but do not replace a justified protocol with convenience.

**Unit and integration tests:**

- Synthetic all-trial coverage, group/target consistency, no overlap, deterministic replay, missing/duplicate key failures, and class distribution reporting.
- Fold-local preprocessing and weighting sentinel tests: validation/test values cannot influence train-fitted parameters, stopping, or selection.
- Model inputs exclude participant IDs, labels, filenames, and candidate metadata.
- Test real manifest schema and source hash handling using fabricated records only until all gates pass.

**Acceptance criteria:** No real split until signed evidence and protocol are present. After approval: all trials map one-to-one; every participant remains in one fold; exact manifest is reproducible and reviewed; split and selection boundaries are enforced and tested; participant-level outputs are reconstructible from saved predictions.

**Research limitations and blockers:** This is the first sprint whose main scientific deliverable depends on verified identity and diagnosis. Current candidate mappings are not admissible. `y_stim` indicators remain prohibited as clinical targets. The current generic splitter alone is insufficient.

**Definition of done:** Approved, tested participant-level assignments and out-of-fold prediction artifacts exist for the frozen protocol, or the real-data branch is explicitly marked blocked with synthetic tests only and no performance claims.

## 8. Sprint 7: Ablation Studies and Analysis

**Objective:** Measure the contribution and sensitivity of the predeclared Conformer components and compare against EEGNet under matched conditions, without post-hoc selection on test results.

**Engineering that can begin now:** Ablation configuration schema and synthetic tests. No scientific ablation result can be produced from synthetic data.

**Tasks in execution order:**

1. Before results are viewed, register a small one-factor-at-a-time ablation set: (a) remove temporal self-attention while retaining the tokenizer/conv pathway; (b) remove the local Conformer convolution module; (c) replace mean+max pooling with mean-only; (d) EEGNet baseline. Avoid unbounded hyperparameter sweeps.
2. Match data partitions, trial inclusion, input path, training budget, class order, model selection, and metrics across models. Parameter/compute differences must be reported, not silently matched by giving one architecture more tuning.
3. Run only after Sprint 6 gate and frozen protocol. Keep final test folds out of ablation selection; select only with the approved inner validation procedure.
4. Analyze participant-level metrics, class-wise errors, fold/seed variability, and uncertainty. Label the unit and aggregation rule explicitly.
5. Use attribution only if separately approved and stable; do not infer named channels or a causal/clinical biomarker from anonymous columns or attention weights.

**Existing files to reuse:** [classifier.py](../src/adhd_eeg/classifier.py), [training.py](../src/adhd_eeg/training.py), Sprint 6 split/manifest modules, [evaluation.py](../src/adhd_eeg/evaluation.py), and the literature audit [ADHD_EEG_Literature_Review.md](literature-review/ADHD_EEG_Literature_Review.md).

**New files expected:**

- `src/adhd_eeg/ablation.py` or a small model/config registry in the existing model module, depending on duplication after implementation.
- `configs/ablation_*.json` for only the frozen ablations.
- `tests/test_ablation_config.py` and `tests/test_analysis.py` for dispatch, metric, and artifact validation.

**Dependencies:** Sprint 6 valid out-of-fold pipeline and approved real experiment. EEGNet comparison must use identical participant splits and selection budgets.

**Unit and integration tests:**

- Ablation config changes exactly one named component.
- Same fold IDs/data inclusion across model variants.
- Deterministic output manifests; no test-fold use for selection.
- Metrics recomputed from saved predictions; participant weighting and class ordering verified.
- Synthetic known-output tests for aggregation, uncertainty summaries, and error tables.

**Acceptance criteria:** Every reported ablation corresponds to a preregistered contrast; all model variants use the same eligible participants and folds; all test predictions are out-of-fold; limitations and uncertainty are reported; no architecture superiority claim without matched evidence.

**Research limitations and blockers:** No real ablation is valid before Sprint 6 gates and Sprint 6 execution. A single cohort cannot establish external generalization. Unresolved channel identities prohibit scalp topography/electrode interpretation. Null or weak results must be reported rather than rescued by new post-hoc variants.

**Definition of done:** Prespecified ablation and EEGNet comparisons are completed on valid splits, with reproducible configs/predictions and a result table whose metrics can be recomputed from stored artifacts.

## 9. Sprint 8: Results, Reproducibility, and Publication

**Objective:** Make completed experiments auditable and prepare a bounded research report/manuscript; do not equate implementation with publication merit.

**Engineering that can begin now:** Artifact schema, methods templates, and checklists can be drafted. Final results/release cannot be completed before Sprints 6-7 results exist.

**Tasks in execution order:**

1. Freeze the source snapshot ID/hashes, code revision, model/config versions, protocol approvals, and environment lock/versions.
2. Reproduce tables and figures from saved predictions through deterministic analysis scripts; do not hand-enter scores.
3. Audit all label, split, preprocessing, selection, and uncertainty claims against the evidence artifacts.
4. Document negative/null findings, deviations, missing data, failed runs, and the limitations in Section 10.
5. Prepare a publication package: methods, architecture and parameter count, input representation, split/selection details, metrics, uncertainty, baselines, ablations, data availability, and exact limits on clinical/external claims.
6. Review public artifact privacy/consent/licensing and ensure no identifying or restricted metadata is emitted. Keep raw EEG out of the repository unless its license explicitly permits distribution.

**Existing files to reuse:** [README.md](../README.md), [docs/README.md](README.md), [base-paper-analysis.md](base-paper-analysis.md), [ADHD_EEG_Literature_Review.md](literature-review/ADHD_EEG_Literature_Review.md), Sprints 5-7 configs, logs, manifests, and result artifacts.

**New files expected (only when results exist):**

- `scripts/reproduce_results.py` or equivalent deterministic table/figure generator.
- `docs/RESULTS_AND_REPRODUCIBILITY_REPORT.md` and a methods/results manuscript artifact as approved.
- Run manifests, prediction tables, and model checkpoints in a versioned artifacts location, not in `dataset/`.

**Dependencies:** Valid Sprints 5-7 training/evaluation and approved data/result release. A paper claim cannot be made from architecture code, synthetic tests, or paper-reported scores.

**Unit and integration tests:**

- Recompute every reported table from saved OOF predictions.
- Verify manifest/prediction coverage, class order, hashes, and run metadata.
- Re-run the approved analysis pipeline and compare deterministic outputs.
- Claim-to-evidence checklist tests/validation for no participant overlap, no candidate labels, no unsupported preprocessing, and correct uncertainty scope.

**Acceptance criteria:** A reviewer can trace each sample eligibility, input transformation, split, selection decision, prediction, metric, and reported figure to versioned artifacts; reported limitations are explicit; no unsupported clinical, causal, external-validation, or architecture-superiority claim appears.

**Research limitations and blockers:** Publication suitability is not guaranteed. Novelty remains conditional on the actual results and literature comparisons. One-cohort performance is not clinical validity or external validation. Interpretability is not causal explanation.

**Definition of done:** The final report and all numerical claims are reproducible from preserved, approved artifacts, limitations are reviewed, and no further analysis choices are made using an untouched test set.

## 10. Scientific gates and approved preprocessing boundary

The following are carried forward from Sprint 1/2 records; this plan does not reopen or repeat their investigations:

1. **Mapping and labels:** Trial-to-participant and participant-to-diagnosis links remain unverified. No real labels, participant keys, or group assignments may be inferred from `TrialManifest` candidate fields, filenames, aggregate counts, or `y_stim` rows 1-3.
2. **Real split/training:** No real participant-independent folds or model training until verified mappings/labels and a reviewed split/validation/metric protocol exist. Generic splitting on fabricated groups is engineering only.
3. **Preprocessing:** The existing filter/notch/ICA implementation is separate legacy software, not approval. The planned model path uses identity/no new transform unless the research team approves an explicit alternative with rationale, parameters, and training-fold fitting rules. Do not invoke the current preprocessing CLI for model training by default.
4. **Tensor/channel limitations:** Keep 385 positions and 56 columns in their existing order. Treat them as indices/features only; sample-385 meaning, physical units/reference, upstream processing, and the 56/60 mapping remain unresolved.
5. **Architecture claims:** Conformer is the selected primary candidate per the current user-directed plan. It is not proven superior, novel, or clinically meaningful. EEGNet remains a matched comparison baseline.

## 11. Synthetic versus real-data workflows

### Workflow A: synthetic development and testing (available now)

`Synthetic NumPy/Torch batch and fabricated class indices` → input validation → Conformer tokenizer/encoder → classifier logits → `CrossEntropyLoss` → optimizer step → synthetic validation → temporary checkpoint → reload/inference smoke test.

Use deterministic artificial tensors generated in tests or temporary fixtures. Any three-class labels in this flow are arbitrary interface fixtures with the declared output order; they are not sourced from the Dresden files and provide no clinical evidence. Synthetic tests must not load or access the repository `dataset/` directory.

### Workflow B: real-data training and evaluation (gated, not implemented)

`Immutable .mat source snapshot` → `TrialManifest` / `EEGTrialStore` read-only slice → existing `EEGTrialDataset` `(trial, metadata)` → verified-manifest wrapper joins authorized target separately → approved transform (identity by default; any fitted transform belongs inside its training fold) → participant-aware DataLoader → Conformer logits → training-only loss/optimizer → validation-only selection → fold checkpoint → untouched outer-fold predictions → participant aggregation/metrics → versioned result artifacts.

The existing Dataset API remains unchanged. A future wrapper should compose it, look up labels by a **verified** source-local `trial_id`, and return `(signal, target, metadata)` for the trainer. Until then, real targets and this wrapper are absent. Split assignments must be made before any fold-fitted transform or model selection. No real-data split/training/inference command currently exists.

## 12. Proposed inference interface (future; not implemented)

The release dataset consists of trial arrays, while the system input specification for a new subject/session is not yet a complete `.mat` schema. Therefore define the stable model-facing API first and defer a user-facing `.mat` parser until its keys, axis order, and preprocessing equivalence are specified.

**Proposed internal call:**

```python
predict_trials(
    trials: np.ndarray,  # (N, 385, 56), float32, finite, stored order
    checkpoint_path: Path,
    preprocessing_id: str,
) -> PredictionResult
```

Validation requirements: non-empty `N`; exact `(N, 385, 56)` trailing dimensions; float32; finite; channel column order must match the model training contract; checkpoint architecture/version and class order must match; apply only the exact preprocessing recorded in the checkpoint (currently identity/no new transform is the plan). Do not silently crop, reorder, scale, or refit normalization at inference.

**Proposed result fields:**

- `predicted_class`: one of `HC`, `ADD`, `ADHD`.
- `class_scores`: mapping in fixed class order from softmax scores; do not label them calibrated probabilities/confidence unless calibration is separately evaluated.
- `trial_count`, model/checkpoint ID, architecture/config version, input-contract version, preprocessing ID, and inference timestamp.
- Optional per-trial scores in a CSV sidecar and one session-level JSON summary. Do not include inferred participant IDs or clinical target labels.

For a session-level decision from multiple trials, mean-probability aggregation is a candidate only; the rule must be frozen and validated before real evaluation. A single-trial call can return a trial score but must not be presented as a verified subject-level decision. No working CLI, checkpoint, trained model, JSON/CSV prediction output, or calibrated confidence exists today.

## 13. Sprint 3 preparation: exact first implementation task

**First implementation task:** Create `src/adhd_eeg/conformer.py` containing only the input validator, temporal Conv2d tokenizer, fixed sinusoidal position encoding, and two-block `EEGConformerEncoder`; add `tests/test_conformer.py` and synthetic-only shape/gradient tests. Do not add the classification head, training code, preprocessing, or labels in that first change.

**Why this is first:** It is the smallest layer that validates the selected architecture’s central assumption and exposes tensor-shape/optional-dependency issues. It can be falsified entirely with synthetic data while keeping existing data and Dataset contracts untouched.

**Sprint 3 code/test conventions to follow:**

- Match the existing flat `src/adhd_eeg/` module layout and `unittest` style.
- Follow `tests/test_dataset.py` for optional PyTorch handling and temporary fixtures.
- Keep PyTorch imports inside PyTorch-specific modules; importing `adhd_eeg.data_access` must not require torch.
- Reuse HDF5 fixtures only for the optional Dataset-to-encoder smoke test. Unit tests should use in-memory synthetic tensors.
- Do not use MNE preprocessing in the encoder.
- Keep trial metadata out of the forward signature.

**Sprint 3 preparation acceptance:** Before expanding to Sprint 4, run the focused encoder test and confirm exact `(B, 97, 64)` output, parameter count, finite gradients, and no contract change to `EEGTrialDataset`.

## 14. Repository references and status

The controlling evidence for this plan is [ADHD_PRE_TRANSFORMER_AUDIT.md](ADHD_PRE_TRANSFORMER_AUDIT.md), [SPRINT_1_IMPLEMENTATION_AUDIT.md](SPRINT_1_IMPLEMENTATION_AUDIT.md), [SPRINT_2_EVIDENCE_RECONCILIATION_AUDIT.md](SPRINT_2_EVIDENCE_RECONCILIATION_AUDIT.md), [SPRINT_2_EVALUATION_READINESS_AUDIT.md](SPRINT_2_EVALUATION_READINESS_AUDIT.md), [SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md](SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md), [ADHD_EEG_Literature_Review.md](literature-review/ADHD_EEG_Literature_Review.md), and [base-paper-analysis.md](base-paper-analysis.md).

The older [RESEARCH_DECISION_AND_IMPLEMENTATION_PLAN.md](RESEARCH_DECISION_AND_IMPLEMENTATION_PLAN.md) records an earlier EEGNet-primary reproducibility proposal. The current plan follows the subsequently user-confirmed Conformer-primary eight-sprint direction while retaining EEGNet as a baseline. The older proposal remains in the repository as decision history and must not be read as the current Sprints 3-8 architecture decision.
