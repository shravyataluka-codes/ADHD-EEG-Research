# Research Decision and Implementation Plan

**Decision date:** 2026-10-28

**Scope:** Research decision and executable plan; this document does not authorize model training or signal preprocessing.
**Decision status:** Conditional. The study is feasible only after the diagnostic-label provenance gate below is passed.

## Five direct answers

1. **What are we building?** A reproducible, participant-level reanalysis of the Dresden 144-child, three-group EEG cohort, with a locked, nested participant-disjoint evaluation and a matched trial-random control. This is a reproducibility/empirical evaluation study, not a new-architecture paper.
2. **What exact research gap are we addressing?** Not whether participant-held-out ADHD EEG evaluation is possible—it has already been done. We will test whether the large trial-split versus participant-held-out performance discrepancy reported on this exact cohort can be independently reproduced and quantified as child-level three-class performance, with verified label provenance, identical model families, fold-contained selection, and out-of-fold uncertainty.
3. **What will be different from existing papers?** A prespecified benchmark will reconcile the competing evaluation estimands in one implementation, report one out-of-fold prediction per child with equal-child metrics and participant-bootstrap intervals, and release the split manifest, trial predictions, configuration, and run metadata. This is an incremental reproducibility contribution, not a claim that the split or model is novel.
4. **Why might the paper merit publication?** The same Dresden cohort has near-ceiling random-trial results and low, highly variable held-out-child results. A transparent, independently reproducible estimate could clarify what those results do—and do not—say about the cohort. The novelty threshold is modest and conditional; a paper is not justified if it merely reimplements prior LOSO results without resolving a material reproducibility or reporting issue.
5. **What is the first implementation step?** Add a fail-closed protocol/data-contract gate and keep the current filter/ICA CLI outside this benchmark. The gate must refuse to generate splits or train until the label-to-diagnosis provenance is independently confirmed. Do not start with the network.

## 1. Decision and evidence standard

### Selected direction

**Working paper title:** *Reproducing ADHD EEG Classification on the Dresden Cohort: Participant-Level Performance, Split Sensitivity, and Uncertainty*

**Research problem:** Published three-class results on the Dresden cohort are difficult to compare: papers use different split units, input representations, preprocessing descriptions, and metric summaries. Participant-exclusive evaluation is already in the literature, but an independent, auditable reanalysis with a single frozen protocol and child-level outputs could establish whether the published performance discrepancy survives strict pipeline control.

**Research question:** On the Dresden cohort, what participant-level three-class performance remains when every held-out child's trials are excluded from model fitting, preprocessing estimates, model selection, and checkpoint decisions, and how does that estimate compare with a trial-random evaluation of the same model family?

**Falsifiable primary hypothesis:** For the primary EEGNet model, the trial-random control will exceed participant-disjoint evaluation by at least **0.10 absolute macro-balanced-accuracy points** when both are summarized as one prediction per child using the same aggregation and metric definition. Support requires the observed difference to be at least 0.10 and the paired participant-bootstrap 95% interval's lower bound to exceed 0.10. The participant-disjoint estimate must also be reported against the 1/3 chance reference; it is not assumed to be clinically useful. The 0.10 threshold is a prespecified decision threshold for this study, not a result claimed from prior papers. Report published accuracy values as context, not as a direct numerical comparison with this different metric.

**Primary contribution:** An independently reproducible estimate of the size and uncertainty of the split-unit performance gap, using verified class provenance, nested participant grouping, equal-child metrics, and released out-of-fold predictions. The model is an instrument for that question, not the contribution.

**Novelty verdict:** The research direction does **not yet establish a strong novel-method claim**. It reaches only a narrow, conditional reproducibility-study threshold if the work resolves a documented comparability/provenance issue and produces an independently useful, reproducible participant-level estimate that is not already available in prior artifacts. If it only reproduces existing LOSO point estimates, the publication case is weak and the team should not inflate the novelty claim.

### Why this direction, and why not a Transformer

- Vahid et al. (2019) already used EEGNet with leave-one-child-out evaluation on the same three groups.
- Wang et al. (2022) already compared random trial five-fold evaluation with leave-one-child-out evaluation on the same cohort and reported sharply different estimates.
- He et al. (2023) already proposed a Transformer on the same cohort. A new Transformer, CNN-Transformer hybrid, attention map, or higher score would not resolve the reproducibility question.
- Participant-independent evaluation, trial/participant score differences, and EEG explainability are not new by themselves. A fair matched protocol and reliable child-level outputs—not a new layer type—are the only defensible incremental value here.
- Therefore, **do not implement a Transformer**. Use compact EEGNet as the prespecified main reference architecture, with matched established CNN baselines. Do not make an architecture-superiority claim.

## 2. Closest competing evidence

Evidence labels used below:

- **Verified:** stated in an inspected primary-paper full text or supported by the repository's direct checks.
- **Reported but not independently reproduced:** a paper's stated result/protocol.
- **Inferred:** a cautious interpretation of the stated protocol.
- **Unknown:** not established by the available full text or records. An unknown is not evidence that an analysis was omitted or that leakage occurred.

| Study | Task, cohort, labels and counts | Model and protocol | Results, aggregation and uncertainty | Ablations, reproducibility, explicit limits, and remaining issue |
|---|---|---|---|---|
| **Vahid et al. (2019), “Deep Learning Based on Event-Related EEG Differentiates Children with ADHD from Healthy Controls”** ([DOI](https://doi.org/10.3390/jcm8071055); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679086/)) | **Verified:** Dresden event-related EEG; 144 children, HC 44 / ADD 52 / ADHD-combined 48. The paper describes 60 recorded electrodes, 56 analyzed channels, and its own ERP preprocessing. | EEGNet. **Verified participant-held-out LOSO:** one child held out at a time; validation subjects were selected on the training side. This already establishes that subject-independent evaluation has been attempted on the cohort. | Whole-cohort mean single-trial three-class accuracy: **39% (SD 30%)**; individual results ranged widely. The paper also reports **69% (SD 28%)** for a subset selected for above-chance individual performance; this conditional subset is not the whole-cohort estimate. It reports high binary results on selected participants and near-chance ADD-versus-ADHD discrimination. | Reports saliency/source analyses, not a causal biomarker. The SD across participants is reported; a participant-bootstrap CI is not established by the evidence reviewed here. The reviewed literature record says the study links the data and EEGNet implementation, not a complete modern pipeline record. The selected-subset result and reported participant-count inconsistencies limit interpretation. **Still open:** reproduce the published per-held-out-child single-trial result under the local tensor/label mapping, then report a prespecified one-probability-per-child benchmark separately. |
| **Wang et al. (2022), “Towards high-accuracy classifying attention-deficit/hyperactivity disorders using CNN-LSTM model”** ([DOI](https://doi.org/10.1088/1741-2552/ac7f5d); [publisher full text](https://iopscience.iop.org/article/10.1088/1741-2552/ac7f5d)) | **Verified:** same OSF/Dresden cohort, 144 children, HC/ADD/ADHD counts 44/52/48; paper describes 33,902 trials, 56 channels, and 385 time points. | CNN-LSTM, compared with EEGNet, ShallowConvNet, and DeepConvNet. Reports **random trial-based five-fold** evaluation and separate **LOSO** evaluation. Random trial splitting is not participant-exclusive; the paper also explicitly discusses this distinction. | CNN-LSTM: **98.23%** accuracy under random trial splitting; **39.59% (SD 28.11%)** under LOSO. The paper reports strong subject-to-subject variability. These are different estimands, not contradictory estimates of one deployment setting. No separate child-level probability aggregation or participant-bootstrap interval is established by the reviewed evidence. | Includes model/feature visualizations. The paper's result-comparison significance claim is not accompanied by a clear reported model-comparison test or confidence intervals in the reviewed account; averaging conventions and exact reproduction details remain incomplete. **Still open:** reproduce under a single controlled implementation and metric contract, not claim this contrast is new. |
| **He et al. (2023), “Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model”** ([DOI](https://doi.org/10.1088/1741-2552/acf7f5); repository copy: [`papers/base-paper.pdf`](../papers/base-paper.pdf)) | **Verified in the paper:** same Dresden cohort; 144 children, 44/52/48; 33,902 trials. | EEG-Transformer with six blocks and six heads; compares EEGNet, ShallowConvNet, and DeepConvNet. The paper describes a five-fold trial-level evaluation using 6,000 test trials per fold; it does not establish participant-exclusive testing. **This is split ambiguity, not proof of leakage.** | The full-text main comparison reports **95.58% ± 0.52%** and AUC **0.9926**; the abstract reports **95.85%**. The meaning of “±”, the unit of aggregation, and the per-child prediction rule are unresolved. | Reports component/pooling and block/head ablations, but tables and prose contain conflicting values. The paper has additional internal inconsistencies, including a 144-versus-300 participant statement and parameter counts. No external validation or clinically valid biomarker is established. **Still open:** participant-level generalization, exact reproducibility, and a consistent result definition. |
| **Hu et al. (2025), SCANet** ([DOI](https://doi.org/10.1109/JSEN.2025.3560349)) | DOI/publisher metadata verifies the paper. Its abstract reports a public three-class ADHD/ADD/HC dataset and a private binary dataset; **exact public cohort and participant count are not verified** from the accessible evidence. | Multiscale selective channel-attention network; abstract reports comparison with EEGNet, EEG-Transformer, and CNN-LSTM, and Grad-CAM. **Split unit, participant overlap, and tuning protocol unknown.** | Abstract reports **99.78%** on the public three-class data. Child/trial aggregation, uncertainty, and per-class results are not verified from the accessible record. | The abstract establishes that attention/Grad-CAM are not new ideas. It does not establish that the result is on Dresden or that it is participant-exclusive. **Do not treat it as a confirmed Dresden replication** until full text/protocol establishes cohort and split details. |
| **Sagar & S (2026), GAFNet** ([DOI](https://doi.org/10.1109/CONIT69683.2026.11620713)) | Crossref/DOI metadata verifies a 2026 CONIT proceedings record. The repository literature review reports that the record claims use of Nasrabadi 121 and Vahid 144 participants; exact cohort construction and valid three-class label mapping remain unverified from a primary full-text examination. | Title claims a graph-augmented fusion Transformer and cross-dataset detection. Exact train/test construction, subject independence, domain-adaptation boundary, and tuning are **unknown**. | The repository review records claimed 99.1% binary and 88.7% three-class results, but cohort-specific metric definitions, aggregation, and uncertainty have not been independently checked. | Publication metadata is verified; reported method/results are **not independently verified** here. Treat it as a serious novelty counterclaim to resolve before submission, not as confirmed proof of Dresden participant-level generalization. |

**Author-stated limitations versus audit inference:** Vahid et al.'s abstract explicitly says more work is needed to improve reliability; the above-chance subset is a paper-defined conditional analysis, not a population-wide result. He et al. explicitly acknowledge the limited participant sample and need for larger samples before clinical application. For Wang et al., this evidence summary confirms the reported split contrast and high participant variability, but does not assert an author-stated limitations section that was not separately verified. For SCANet and GAFNet, accessible abstract/metadata evidence is insufficient to assess author-stated limitations, ablations beyond the abstract description, or reproducibility artifacts. These unknowns are not treated as omissions.

**Reproducibility availability in the reviewed repository evidence:** Vahid links the data and EEGNet implementation but not a full modern pipeline record; Wang links the OSF data and a code repository; He links the data but has no study-specific code repository documented in the review ([`ADHD_EEG_Literature_Review.md:128`](./literature-review/ADHD_EEG_Literature_Review.md#L128)). These availability statements are not claims that the linked code was executed in this project.

**Evidence basis in the repository:** Vahid, Wang, and He are summarized at [`ADHD_EEG_Literature_Review.md:54`](./literature-review/ADHD_EEG_Literature_Review.md#L54), [`ADHD_EEG_Literature_Review.md:55`](./literature-review/ADHD_EEG_Literature_Review.md#L55), and [`ADHD_EEG_Literature_Review.md:56`](./literature-review/ADHD_EEG_Literature_Review.md#L56); the Vahid and Wang full-text evidence is further summarized at [`Elicit research-gap review:29`](./literature-review/Elicit%20-%20Reproducible%20ADHDADD%20classification%20on%20the%20Dresden%20EEG%20cohort%20evidence%20and%20research%20gaps.md#L29) and [`Elicit research-gap review:37`](./literature-review/Elicit%20-%20Reproducible%20ADHDADD%20classification%20on%20the%20Dresden%20EEG%20cohort%20evidence%20and%20research%20gaps.md#L37). SCANet is summarized at [`ADHD_EEG_Literature_Review.md:81`](./literature-review/ADHD_EEG_Literature_Review.md#L81), and GAFNet at [`ADHD_EEG_Literature_Review.md:97`](./literature-review/ADHD_EEG_Literature_Review.md#L97). He et al.'s reporting conflicts and evaluation boundary are summarized at [`base-paper-analysis.md:1082`](./base-paper-analysis.md#L1082) and [`03_BASE_PAPER_AND_RESEARCH_EVIDENCE_AUDIT.md:232`](./architecture-preparation/03_BASE_PAPER_AND_RESEARCH_EVIDENCE_AUDIT.md#L232).

### What is established, inferred, and unknown

- **Established:** Vahid and Wang performed participant-held-out evaluations on this cohort. Therefore, “first unseen-child evaluation” is false.
- **Established:** Wang directly reports a large random-trial versus LOSO result difference. Therefore, merely reporting that trial-level performance can exceed subject-level performance is not new.
- **Established:** He et al. published a Transformer on this cohort. A Transformer or CNN-Transformer combination is not novel here.
- **Inferred, not proven:** Trial-random splitting in Wang and He permits participant overlap; He et al.'s actual overlap is not proven by a protocol description alone.
- **Unknown:** Whether SCANet uses Dresden, its split unit, and its participant-level aggregation; whether GAFNet's advertised domain adaptation preserves an untouched participant-level test; whether the accessible EEG tensor is preprocessing-identical to any of the papers.
- **Unknown / hard gate:** Whether the local HC/ADD/ADHD block ordering is independently linked to original clinical diagnosis records. A block pattern and count match are not themselves a clinical-label provenance record.

## 3. Frozen research specification

| Item | Decision |
|---|---|
| **Research question** | What child-level, three-class performance remains under participant-disjoint nested evaluation, and how large is the difference from a trial-random split that permits each child's other trials in training? |
| **Hypothesis** | For EEGNet, the trial-random control will exceed participant-disjoint child-level macro-balanced accuracy by at least 0.10 absolute points under the same metric and aggregation rule. Support requires the observed difference to be at least 0.10 and the paired participant-bootstrap 95% interval's lower bound to exceed 0.10. |
| **Primary claim** | This work provides an auditable independent estimate of split sensitivity and participant-level performance/uncertainty for the released Dresden tensor after source-label confirmation. It does not claim a new architecture, new split concept, diagnostic test, or external validity. |
| **Population/task** | The released Dresden cohort only; three dataset groups HC, ADD, ADHD. The term “clinical diagnosis” may be used only after the label-source gate passes; otherwise describe the target as dataset group labels and stop short of diagnosis claims. |
| **Primary outcome** | Participant-level **macro-balanced accuracy**: average of the three class recalls, with one out-of-fold prediction per child. Compare participant-disjoint primary protocol with the trial-random control. Chance reference is 1/3. |
| **Secondary outcomes** | Participant-level macro-F1, ordinary accuracy, per-class recall and specificity, confusion matrix, one-vs-rest macro ROC-AUC where class scores permit, and fold/seed variability. Trial-level scores are secondary and labeled as such. |
| **Aggregation** | For each child, arithmetic mean of that child's held-out trial softmax probabilities, then argmax once. Each child contributes one row and equal weight to all participant-level metrics. The aggregation rule is frozen before model runs. |
| **Cohort prerequisite** | Target group counts must reconcile to 44 HC, 52 ADD, and 48 ADHD; every trial must map to exactly one participant. Local mapping evidence is documented, but block-to-diagnosis provenance is not independently established by the existing binary metadata alone. Use the reviewed block-to-label crosswalk, not `y_stim` rows 1–3 as a model target, until the conflicting repository contract language is reconciled. |
| **Primary split** | Five-fold stratification over a one-row-per-participant table, using the composite participant ID to assign every trial to its participant fold; fixed split seed `1729`. Every child appears in one outer test fold only; all trials remain together. Use nested four-fold participant-level stratification on each outer-training partition for model/config selection. This avoids letting high-trial participants control fold balance. No separate untouched 20% holdout: with 144 children, nested outer-fold predictions provide a more data-efficient estimate. |
| **Model selection** | Four configurations per neural model: Adam learning rate in `{0.0003, 0.001}` crossed with weight decay in `{0.0001, 0.001}`; batch size 64, maximum 100 epochs, early-stopping patience 10, and architecture/dropout fixed as specified. Select by mean inner-fold participant-level macro-balanced accuracy; exact ties resolve by a fixed lexicographic config order. Use the median best epoch across inner folds when refitting on all outer-training participants. No outer-fold scores, test participants, or final aggregate predictions may influence settings, thresholds, or reporting choices. |
| **Training loss weighting** | Compute from each fit's training partition only. Give each training participant equal total weight within class, then give each class equal aggregate weight (per-trial weight proportional to `1 / (training participants in class × that participant's training trial count)`, normalized to mean 1). Do not oversample or estimate weights from validation/test folds. |
| **Main architecture** | Compact EEGNet-style temporal/spatial convolutional classifier (details below), not a Transformer. It is selected as a small-cohort reference with direct prior-art and EEGNet-comparison context, not as a claimed novelty. |
| **Baselines** | (1) ShallowConvNet; (2) CNN-LSTM with a deliberately small fixed capacity; (3) majority-class dummy for sanity only. EEGNet is the proposed main model. Report identical splits, outer predictions, aggregation, and metrics. Do not claim a model family is superior without paired child-level analysis. |
| **Preprocessing** | Preserve the stored `(385, 56)` trials and `float32`; no filter, notch, ICA, baseline correction, channel remapping, crop, resample, artifact rejection, spectral feature, or full-dataset normalization. The network receives raw values. Any BatchNorm parameters are learned within each training fold. Baseline feature scaling, if used, lives inside a fold-fitted sklearn pipeline. |
| **Robustness** | Refit fixed primary split with four additional training seeds (1730–1733); report variability, not the best seed. Repeat participant aggregation using 30 randomly sampled trials per child (all mapped children reportedly have at least 46 trials), with 20 fixed subsampling seeds. Report majority-vote aggregation as a prespecified sensitivity analysis only. |
| **Uncertainty** | 2,000 stratified participant bootstrap resamples of the pooled out-of-fold child predictions, preserving class counts. Report 95% percentile intervals for each model and paired model/split differences. Label these intervals **conditional on the fitted outer-fold models**; they do not capture full model-refitting uncertainty. Also show outer-fold and training-seed variability. |
| **Leakage controls** | Keep participant IDs and labels out of model inputs; compute loss weights and all fitting/scaling/stopping/selection within the relevant training/validation partitions; check no subject intersection in the primary protocol; never use the outer fold to choose a model, seed, threshold, or aggregation. Store out-of-fold predictions before computing the final metrics. |
| **Claims excluded** | Clinical diagnosis, clinical utility, biomarker, causal explanation, universal child/population generalization, external-site transfer, or “leakage proven” in prior studies. |

### Proposed main architecture: compact EEGNet

Input is `(batch, 1, 56, 385)`, obtained only by axis permutation from stored `(trial, time, channel)`; channel columns remain anonymous and in stored order.

1. Temporal `Conv2d` with 8 filters and a 64-position kernel; batch normalization.
2. Depthwise spatial convolution spanning the 56 stored channels, depth multiplier 2; batch normalization and ELU.
3. Average pooling by 4 and dropout 0.25.
4. Separable temporal convolution with 16 output maps; batch normalization and ELU.
5. Average pooling by 8, dropout 0.25, temporal global average pooling, and a three-logit linear head.

The kernel is specified in **sample positions**, not milliseconds: the stored sample-time convention and 385th position are unresolved. No scalp topography, named electrode interpretation, or spatial explanation is permitted because channel identities/order are unresolved. All hyperparameters above are starting fixed values and must be recorded as the prespecified configuration; only the declared small regularization grid may be chosen by inner folds.

### Essential experiments

| Experiment | Exact contrast | What it can support / cannot support |
|---|---|---|
| **E1: Primary participant-disjoint evaluation** | EEGNet, ShallowConvNet, and CNN-LSTM under the same 5x4 nested participant-grouped protocol. | Estimates held-out-child performance on this one cohort. Does not establish external or clinical validity. |
| **E2: Split-unit control** | Same architecture and search budget, but a separate random stratified trial five-fold partition with trial-random inner validation. Ensure each trial receives one out-of-fold prediction and aggregate those predictions per child using the same probability mean as E1. | Tests whether the split-unit gap replicates. The random-trial result is explicitly a seen-participant control: the participant appears in training/validation when their held-out trials are scored, so it is not an unseen-child estimate. |
| **E3: Architecture comparison** | Compare EEGNet with ShallowConvNet and CNN-LSTM on identical outer folds, with all tuning confined to inner folds. | Shows whether the compact EEGNet reference is competitive; not an architecture-novelty claim. |
| **E4: Aggregation sensitivity** | Mean-probability primary aggregation versus majority vote; equal-count 30-trial participant subsampling. | Tests sensitivity to per-child trial count and aggregation. No aggregation is selected after examining outer test results. |
| **E5: Uncertainty/reproducibility** | Stratified participant bootstrap; fixed-seed reruns; publish child-level out-of-fold predictions, split IDs and run config. | Makes the estimate auditable; intervals must not be interpreted as external-population uncertainty. |

No filter/ICA ablation is included: the current tensors' upstream processing state and the original paper's preprocessing cannot be reconstructed sufficiently, and the project contract does not approve signal transformations. No Transformer ablation is included because there is no Transformer research claim. An optional EEGNet temporal-only ablation is not required for the primary paper and must not displace the provenance, leakage, or benchmark controls.

## 4. Data and label gate

The local mapping report concludes that filename block prefixes and count matches align with the paper's groups and maps blocks to HC/ADD/ADHD ([`trial_subject_mapping_validation.md:206`](./trial_subject_mapping_validation.md#L206)). However, the research-design audit says the supplied data do not contain an explicit participant diagnosis field and that the local mapping cannot independently verify diagnoses ([`05_RESEARCH_DESIGN_AND_EVALUATION_AUDIT.md:27`](./architecture-preparation/05_RESEARCH_DESIGN_AND_EVALUATION_AUDIT.md#L27)). Treat this as a provenance discrepancy to resolve, not as a reason to repeat the already completed tensor audit. The prerequisite is an independently reviewable source linking the local block/group encoding to the published HC/ADD/ADHD clinical groups.

**Hard stop:** Until this provenance is recorded and independently reviewed, code may validate structure and emit a metadata-only mapping report, but it must not create labeled folds, fit a classifier, or write results described as ADHD diagnosis performance. If provenance cannot be established, either relabel the project honestly as exploratory prediction of *dataset group codes* with explicit limitation and obtain approval, or stop the proposed clinical-group study. Do not infer the meaning from trial counts alone.

## 5. Implementation architecture and repository map

Existing reusable code:

- [`src/adhd_eeg/data_access.py`](../src/adhd_eeg/data_access.py): `TrialManifest`, `TrialMetadata`, and bounded `EEGTrialStore` already cover trial metadata and input access. Extend only to add explicit provenance status or stronger contract validation; do not duplicate the HDF5 reader.
- [`tests/test_data_access.py`](../tests/test_data_access.py): current synthetic data-access contract tests.
- [`src/adhd_eeg/preprocessing.py:35`](../src/adhd_eeg/preprocessing.py#L35) and [`src/adhd_eeg/preprocess_cli.py:34`](../src/adhd_eeg/preprocess_cli.py#L34): existing path uses 0.5–45 Hz filtering, 50 Hz notch, and ICA ([`preprocessing.py:259`](../src/adhd_eeg/preprocessing.py#L259)); its processing metadata says ICA is fitted per subject before any evaluation split ([`preprocessing.py:590`](../src/adhd_eeg/preprocessing.py#L590)). This conflicts with the frozen no-transformation contract ([`SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md:48`](./SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md#L48)) and the upstream preprocessing state is unresolved. **Do not import or call it from this benchmark.** Keep it as a separate, unapproved legacy path; do not silently change its output or treat its current configuration as approval.

| Component | Proposed file(s) | Responsibility |
|---|---|---|
| Protocol and provenance gate | Add `src/adhd_eeg/research_protocol.py`; modify `src/adhd_eeg/data_access.py` only if needed to expose provenance distinctly from group code. | Validate that label provenance is confirmed, trial IDs are unique, expected `(385,56)`/`float32` contract holds, and only the approved identity input path is selected. Fail explicitly if a gate is missing. |
| Trial/participant manifest | Reuse `TrialManifest`; add `src/adhd_eeg/experiment_manifest.py` only for immutable run/fold metadata. | Persist a metadata-only CSV/JSON with trial index, source file, composite participant key, dataset group, label-provenance identifier, and split/fold. Never include EEG values in the manifest. |
| Participant-disjoint folds | Add `src/adhd_eeg/splits.py`. | Generate deterministic nested stratified folds from one row per participant, then map participant assignments to all trial rows; assert all trials of a participant are together and every participant is outer-test exactly once. Save fold assignments before fitting. |
| Fold-contained transforms | Add `src/adhd_eeg/fold_transforms.py` only if a feature baseline needs a scaler. | Identity transform for neural input. Baseline `StandardScaler` is inside a fit/transform pipeline trained independently per inner/outer fit. No reusable global statistics. |
| Model interface and models | Add `src/adhd_eeg/models.py`. | Common `fit`/`predict_proba` interface; compact EEGNet, ShallowConvNet, small CNN-LSTM, and dummy baseline. Return three-class probabilities with fixed class order. |
| Training/model selection | Add `src/adhd_eeg/training.py`. | Seed control; training-fold-only participant/class-balanced loss weights; fixed four-point learning-rate/weight-decay grid; inner-fold scoring; early stopping only on inner validation units; refit on outer train for median selected epoch; and checkpointing. |
| Trial and participant predictions | Add `src/adhd_eeg/predictions.py`. | Write one row per held-out trial with trial ID, participant ID, true label, model, fold, seed, and three class probabilities; derive a separate one-row-per-participant aggregation table. |
| Metrics and uncertainty | Add `src/adhd_eeg/evaluation.py`. | Participant metrics, confusion matrices, trial secondary metrics, stratified participant bootstrap, paired differences, and explicit conditional-interval labeling. |
| Experiment orchestration | Add `configs/dresden_reproducibility.json` and `scripts/run_experiment.py`. | Validate protocol, run all models/experiments under fixed folds/config, log software versions and seeds, and write stable artifacts. |
| Dependencies | Modify `pyproject.toml`. | Add model/training dependencies as an optional extra (PyTorch and scikit-learn); keep current data-access installation lightweight. Pin or record exact resolved versions in run metadata. |
| Tests | Add `tests/test_research_protocol.py`, `tests/test_splits.py`, `tests/test_models.py`, `tests/test_evaluation.py`, and `tests/test_leakage.py`. | Unit, integration, and leakage tests described below. Keep the existing data-access tests passing. |
| Existing documentation | Update `docs/SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md`, `docs/ML_EXECUTION_PLAN.md`, and `docs/README.md`. | Reconcile the new protocol with pending gates and explicitly retire obsolete/contradictory claims. This decision document is the single source for the selected research question. |

### Artifacts per experiment

Write to a run-specific directory under `artifacts/` (not into `dataset/`): frozen JSON config; source-data checksums; trial/participant/fold manifest; per-fold and per-seed logs; model state dictionaries; environment/package versions; training history; trial-level OOF probabilities; one-row-per-child OOF probabilities; metrics and bootstrap intervals; and a run summary. Do not persist a transformed full-dataset EEG tensor. Record failure status and traceback rather than emitting success-shaped partial results.

### Required tests

- **Unit:** manifest-to-label mapping; class order; tensor shape/dtype; folds have no participant intersection; each child is outer-test once; all trial indices appear once in OOF predictions.
- **Leakage:** inject sentinel data into an outer-test participant and verify train-fitted statistics, class weights, early stopping, checkpoint selection, and model parameters do not change; ensure metadata IDs/labels never enter model tensors.
- **Fold-contained scaling:** fit baseline scaling on a training fold, verify validation/test values do not affect stored means/scales, and test that each inner/outer fit gets a separate scaler.
- **Aggregation/metrics:** deterministic probability averaging, correct per-child class assignment, equal participant weights despite unequal trial counts, 1/3 chance reference, bootstrap reproducibility, and fixed class ordering.
- **Integration:** run a tiny synthetic three-class cohort through all folds and all model interfaces; assert saved trial and child predictions reconcile and no source EEG file is modified.
- **Reproducibility:** same config/seed produces identical split IDs and deterministic CPU predictions within declared numerical tolerance.

## 6. Publication case and rejection conditions

**New knowledge if successful:** a defensible estimate of participant-level three-class performance and its uncertainty under a transparent, strictly nested protocol, plus a controlled quantification of how the same cohort appears under a seen-participant trial split. This can clarify whether headline results describe unseen children or only additional trials from familiar children.

**Use to the community:** EEG researchers can compare reported metrics only when the independent unit, aggregation, fitting boundary, and uncertainty are explicit. Releasing per-child predictions and exact folds makes a small, repeatedly reused cohort more auditable and reduces the temptation to treat 33,902 trials as independent people.

**Results this work challenges or clarifies:** Wang's 98.23% trial-split versus 39.59% LOSO contrast; Vahid's 39% full-cohort estimate and above-chance-selected subset result; and He et al.'s high trial-based Transformer result whose participant independence is not established. This work does not declare any prior result invalid.

**Convincing evidence:** verified labels; deterministic participant IDs; nested outer/inner splits; identical model budgets and metrics; no outer-fold feedback; calibrated descriptions of uncertainty; complete OOF files and executable configuration; and a clear statement of which tensor/preprocessing version was used.

**Results that invalidate or narrow the contribution:**

1. Label provenance or participant-to-trial mapping cannot be established: the proposed diagnosis-label experiment is infeasible.
2. The 0.10 split-gap hypothesis is not supported: report the null/contrary result, reject the split-sensitivity claim, and do not rescue it with a new architecture.
3. A verified SCANet, GAFNet, or other primary paper already provides the same controlled participant-level benchmark and reproducible artifacts: the novelty claim is substantially narrowed or rejected.
4. Independent reproduction merely confirms the already reported estimates without resolving protocol/data contradictions or adding useful uncertainty/artifacts: publication case is weak.
5. Model scores are near chance or unstable: this is a valid negative result only if the data/label gates and analysis are sound; it is not evidence that EEG can clinically diagnose ADHD.

**Likely paper type:** reproducibility-oriented empirical evaluation / benchmark paper. It is not a methodological paper because the split, bootstrap, and CNN architectures are established methods.

**Strongest reviewer rejection reason:** “Vahid and Wang already performed participant-held-out evaluation, so this is an incremental rerun of one small, single-cohort dataset.” The response must be an evidence-backed reproducibility improvement, not a claim of first use. If no material protocol conflict is resolved and no genuinely useful new estimate is produced, accept that the novelty threshold is not met.

## 7. Unresolved blockers and frozen choices

### Hard blockers

1. **Clinical-label provenance:** confirm local block order maps to the clinical HC/ADD/ADHD groups, rather than only matching paper counts.
2. **Paper/data comparability:** document whether local 385x56 files match the reported 256-Hz/1.5-s model input; retain all stored positions and do not silently crop or reinterpret them.
3. **Prior-paper coverage:** obtain and protocol-audit SCANet and GAFNet full text/supplement/code before final novelty wording. Current evidence verifies their publication/abstract claims only, not Dresden participant-level validity.
4. **Preprocessing contract reconciliation:** current filter/ICA module contradicts the approved no-transformation contract. Keep it out of the selected pipeline and update the CLI/docs before ML execution.

### Frozen for the proposed study

- Research direction: conditional reproducibility benchmark; no Transformer contribution.
- Primary task: three dataset groups; child is the inference unit.
- Main model: compact EEGNet; comparison models are ShallowConvNet and a small CNN-LSTM.
- Primary metric: participant-level macro-balanced accuracy using mean trial probabilities.
- Outer/inner splits: 5x4 participant-grouped, deterministic; fixed seed 1729 for primary split.
- Signal handling: identity/no-op; preserve `(385,56)` and `float32`; no current preprocessing module.
- OOF predictions, participant-bootstrap uncertainty, and split-unit control are required outputs.

### Evidence-dependent / revisable

- Whether the labels may be called clinical diagnoses (blocked by source provenance).
- Whether the current data version is comparable enough to published inputs for a faithful reproduction.
- Whether a stronger prior SCANet/GAFNet result already addresses the proposed contribution.
- Whether the proposed evaluation has sufficient novelty after those checks.

## 8. Ordered implementation sequence

| # | Step and files | Expected output | Tests | Dependencies | Acceptance criteria |
|---:|---|---|---|---|---|
| 1 | **Add a fail-closed protocol gate.** Add `src/adhd_eeg/research_protocol.py` and a runner stub at `scripts/run_experiment.py`; add `tests/test_research_protocol.py`; document the current preprocessor as a separate, unapproved legacy path in `docs/SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md`. Do not modify the legacy signal-processing behavior. | A run cannot create labeled splits or train unless explicit label-provenance approval, identity preprocessing, and the frozen input contract are provided. | Gate unit tests: missing/false provenance fails; unexpected preprocessing fails; identity contract preserves shape/dtype. Integration test verifies the research runner never imports or calls `preprocessing.py`. | None; safe to implement before provenance is resolved. | No benchmark code can accidentally call the current 0.5–45 Hz / 50-Hz notch / ICA route. The existing CLI remains separate and unchanged. |
| 2 | **Resolve label provenance.** Evidence-only work; record source evidence in `docs/trial_subject_mapping_validation.md` and update `src/adhd_eeg/data_access.py` only if the verified group/label distinction requires it; update `tests/test_data_access.py`. | Approved label mapping record and versioned metadata contract, or a documented stop/reframing decision. | Tests verify all 144 subject keys, all trials have one group/subject, exact class counts, no class/subject metadata in EEG arrays. | Step 1. | An independent reviewer can trace every local label to a source; otherwise training remains blocked. |
| 3 | **Freeze folds and manifests.** Add `src/adhd_eeg/splits.py`, `src/adhd_eeg/experiment_manifest.py`, `tests/test_splits.py`, `tests/test_leakage.py`. | Saved outer/inner participant assignments and random-trial control assignments mapped to the trial/participant manifest. | No participant overlap in grouped folds; every subject tested once; exact OOF trial coverage; deterministic replay; trial split explicitly flagged non-independent. | Step 2. | Split artifacts are fixed and checksummed before model runs; all negative leakage tests pass. |
| 4 | **Freeze package/model contracts.** Add optional `ml` extra to `pyproject.toml`; add `src/adhd_eeg/models.py`, `src/adhd_eeg/fold_transforms.py`, and `tests/test_models.py`. | Probability-producing EEGNet, ShallowConvNet, CNN-LSTM, and dummy interfaces; identity raw-input path. | Shape/class-order tests; forward/backward smoke tests; model outputs finite 3-class probabilities; no access to IDs/labels in forward path. | Step 3. | All models accept identical stored trials and return the same declared class order; no Transformer dependency. |
| 5 | **Implement nested training and selection.** Add `src/adhd_eeg/training.py`; add `configs/dresden_reproducibility.json`; extend training tests. | Per-outer-fold selected configs, fit histories, saved checkpoints, seed/environment logs. | Inner folds use only outer-train subjects; fold-local weighting/scaling; stopping and selection never read outer labels; sentinel leakage tests pass. | Step 4. | Full configurations and the selection metric are frozen before outer prediction; reproducibility metadata is complete. |
| 6 | **Implement OOF storage, aggregation, and statistics.** Add `src/adhd_eeg/predictions.py`, `src/adhd_eeg/evaluation.py`, `tests/test_evaluation.py`. | Trial-OOF and child-OOF probability tables; primary/secondary metrics; bootstrap intervals and paired differences. | Synthetic known-label tests; equal-child weighting despite unequal trial counts; deterministic bootstrap; validate one and only one OOF prediction per trial/child. | Step 5. | The primary result can be recomputed from saved child probabilities alone; intervals state their conditional scope. |
| 7 | **Run integration and leakage suite.** Add `tests/test_pipeline_integration.py`; preserve existing tests. | Synthetic end-to-end run and validated artifact schema. | `python -m unittest discover -s tests`; focused failure tests for incomplete labels, overlapping subjects, test-informed selection, and modified raw data. | Steps 1–6. | Full local suite passes; real dataset is not used in synthetic integration tests. |
| 8 | **Run the locked experiments.** Add `scripts/run_experiment.py`; use the checked-in config; write to run-specific `artifacts/`. | E1–E5 results, OOF predictions, model checkpoints, logs, environment and source checksums. | Pre-run assertions, post-run coverage/manifest checks, independent metric recomputation, no source changes. | Steps 1–7 and all blockers passed. | Primary split and all prescribed robustness results complete; no post-hoc settings or best-seed reporting. |
| 9 | **Reconcile prior work and write results.** Update `docs/ML_EXECUTION_PLAN.md`, `docs/README.md`, and this decision record with exact run IDs/results; prepare a manuscript/report outside code changes as requested. | Reproducible results narrative with limitations, prior-study comparison, and released artifacts. | Re-run artifact-to-table reproduction; check claims against the OOF files and evidence citations. | Step 8; SCANet/GAFNet review completed. | Retain, revise, or reject the hypothesis using the criteria below; no unsupported clinical or novelty claim remains. |

### Final hypothesis decision rules

- **Retain:** participant-disjoint OOF evaluation is valid, the observed trial-random minus grouped child-level macro-balanced accuracy is at least 0.10, and the lower bound of the paired participant-bootstrap 95% interval is also above 0.10. State the estimate and limits, not a universal rule.
- **Revise:** the observed effect is directionally positive but below 0.10, or its interval does not establish a difference of at least 0.10, or results are strongly seed/aggregation sensitive. Report the estimate as inconclusive; do not claim a robust split penalty.
- **Reject:** the difference is zero/reversed, or confidence intervals and sensitivity tests do not support the stated direction. Report that the hypothesis was not supported; do not change the primary metric or architecture after seeing outcomes.
- **Stop the clinical-label study:** label provenance fails or a primary source contradicts the local mapping. No model result can repair an invalid target label.

## References and evidence trail

- Vahid et al. (2019), *Journal of Clinical Medicine*, 8(7), 1055. DOI: [10.3390/jcm8071055](https://doi.org/10.3390/jcm8071055); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679086/).
- Wang et al. (2022), *Journal of Neural Engineering*, 19(4), 046015. DOI: [10.1088/1741-2552/ac7f5d](https://doi.org/10.1088/1741-2552/ac7f5d).
- He et al. (2023), *Journal of Neural Engineering*, 20, 056013. DOI: [10.1088/1741-2552/acf7f5](https://doi.org/10.1088/1741-2552/acf7f5); repository PDF: [`papers/base-paper.pdf`](../papers/base-paper.pdf).
- Hu et al. (2025), *IEEE Sensors Journal*, 25, 20920–20932. DOI: [10.1109/JSEN.2025.3560349](https://doi.org/10.1109/JSEN.2025.3560349).
- Sagar & S (2026), *2026 6th International Conference on Intelligent Technologies (CONIT)*, 1–9. DOI: [10.1109/CONIT69683.2026.11620713](https://doi.org/10.1109/CONIT69683.2026.11620713).
- Project evidence: [`trial_subject_mapping_validation.md`](./trial_subject_mapping_validation.md), [`SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md`](./SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md), [`05_RESEARCH_DESIGN_AND_EVALUATION_AUDIT.md`](./architecture-preparation/05_RESEARCH_DESIGN_AND_EVALUATION_AUDIT.md), [`ML_EXECUTION_PLAN.md`](./ML_EXECUTION_PLAN.md), and [`ADHD_EEG_Literature_Review.md`](./literature-review/ADHD_EEG_Literature_Review.md).
