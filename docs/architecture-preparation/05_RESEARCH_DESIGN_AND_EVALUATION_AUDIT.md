# 05 — Research Design and Evaluation Audit

**Date:** 2026-10-06  
**Purpose:** Clarify what the project needs to measure to answer its research question, what would make that measurement untrustworthy, and which choices remain open.  
**Audience:** A Computer Science student new to machine learning, EEG, neuroscience, and research methods.  
**Scope:** Evidence and requirements audit only. This document does **not** select a model, architecture, preprocessing pipeline, split algorithm, or final experiment design.

## How to read this audit

- **Required** means an existing project objective, constraint, or gate says the issue must be addressed.
- **Decided** means the project has explicitly frozen the decision. A suggestion in an older audit, execution plan, or guide is not automatically a decision.
- **Open** means the exact choice has not been approved or finalized.
- **Paper-reported** describes what He et al. (2023) say they did. It is not an automatic requirement for our experiment.

The current project-state and Sprint 2 records are used to identify what is frozen. Where older documents make stronger or conflicting claims, this audit notes the conflict rather than silently adopting a proposal. In particular, Gate 2, Gate 3, Gate 4, and Gate 6 are recorded as not passed in the current project-state audit.

## 1. The research question and what a result would mean

The project asks whether EEG recordings from the available cohort can distinguish its three recorded groups—healthy controls (HC), ADD, and ADHD—for **subjects not used to fit or select the model**. It also intends to examine which cases succeed or fail and what information contributes to predictions.

This is a research classification question about one released dataset and its group labels. A good evaluation can provide evidence about performance under the stated data and partition protocol. It does not, by itself, show that a model diagnoses ADHD, is useful in clinical practice, or will work on data from other hospitals or recording systems.

| Concept | What it means | Why it matters here | Required by project? | Decided? | What remains open? |
|---|---|---|---|---|---|
| Research question | Ask whether EEG carries information that lets a system distinguish the dataset's HC, ADD, and ADHD groups for people absent from model development. | Makes the target claim testable and prevents confusing group classification with diagnosis or clinical usefulness. | **Yes.** The scope and project-state audits describe three-group classification on unseen subjects. | The broad question and three group labels are decided. | Exact model, input processing, primary metric, evaluation unit, and test design. |
| Evaluation population | The people and recordings to which the reported result directly applies. | The files represent 144 children from one dataset/cohort, not thousands of independent participants and not a collection of independent external sites. | **Yes.** The project scope says this dataset is for development/evaluation and is not an external clinical dataset. | The available cohort and its counts are documented; broader target populations are not established. | Any broader population claim would require evidence beyond this dataset. |
| Group label | The supplied HC/ADD/ADHD group assignment used as the classification target. The local block-to-group mapping is strongly supported by filename prefixes and exact count agreement with the paper; the data do not contain an explicit participant diagnosis field. | Evaluation can only test prediction of these assigned labels. It cannot independently verify the underlying diagnosis. | **Yes.** The three groups are the task labels. | Use block identity for the group label; do not use `y_stim` rows 1–3 directly as input labels. The individual diagnosis provenance remains limited. | How to describe the label provenance and its limitation in final reporting. |

### What “unseen subject” means

A subject is a person; a trial is one short EEG segment from that person. Here, one child can contribute many trials (46–295 according to the mapping report). An **unseen subject** is a person whose EEG trials have not been used to train the model, fit data-dependent preprocessing, select features or settings, choose a checkpoint, tune thresholds, or otherwise make development decisions.

“Unseen” does not mean a new trial from a person whose other trials were already used. It also does not automatically mean a person from a new hospital, a different recording setup, or a future clinical population.

| Concept | What it means | Why it matters here | Required by project? | Decided? | What remains open? |
|---|---|---|---|---|---|
| Subject-level separation | Keep every trial from a given subject in one partition only. | Trials from one person share person-specific signal characteristics. If some enter training and others enter testing, the model may recognize the person rather than learn patterns that transfer to new people. The trial count (33,902) must not be mistaken for the number of independent people (144). | **Yes. Explicitly required** for claims about unseen participants; trial-level splitting is prohibited by the project-state audit. | The grouping unit is frozen as the composite subject key (group block plus within-block index). | Exact partition assignments, number of folds, held-out test design, and random seed. |
| Subject versus filename identity | The composite key (`HC_01` … `ADHD_48`) groups trials. Its association with an original human filename is only strongly supported, not independently proven. | Safe subject grouping does not require a person's filename. Avoid treating the ordinal filename link as stronger evidence than it is. | **Yes**, preserve subject identity through partitioning. | Composite keys are available and validated for grouping; the filename caveat is documented. | Whether filename identity is needed for any future analysis. |
| Trial-level versus subject/session-level result | A trial-level score counts each EEG segment as an observation. A subject/session-level score first produces a result for a person/session, potentially from multiple trials, and then evaluates those person/session results. | The prototype objective calls for a subject/session-level output. Trial counts vary considerably, so pooling all trials gives more influence to people with more trials. | **Subject/session-level output is required in the prototype scope.** The research evaluation unit is explicitly listed as undecided in the project-state audit. | Output intent is decided; metric unit and any aggregation rule are not. | How predictions from a subject's trials become one result, and whether both trial- and subject-level views are needed for analysis. |

## 2. Training, validation, and testing

These names describe different roles in evaluation; they are not interchangeable.

- **Training data** are used to fit model parameters.
- **Validation data** are used during development to compare candidate settings, select a model/checkpoint, or decide when training should stop. They are not a final unbiased score if repeatedly consulted during development.
- **Test data** are reserved for final evaluation after the choices above have been made. If results from the test set drive further choices, it has become development data.

For this project, the unit of partitioning must be the **subject**, so a validation or test subject must not also contribute trials to training. Any data-derived preprocessing or feature-selection settings must also be learned without the held-out subjects, consistent with the project’s no-whole-dataset-normalization constraint.

| Concept | What it means | Why it matters here | Required by project? | Decided? | What remains open? |
|---|---|---|---|---|---|
| Training partition | Subjects whose allowed data are used to fit the model and training-only transformations. | This is the information available to the model during learning. | **Yes.** A training stage is necessary to answer the research question. | No model or training process exists yet. | Exact protocol and how many subjects are allocated to each role. |
| Validation partition | Separate development subjects used for model selection and training decisions. | Reusing final test performance to tune a model makes the reported test result optimistic. | **Yes in principle** for a defensible model-selection process; project documents require a validation strategy before training. | No. | Whether and how validation is separated or nested within grouped evaluation. |
| Test partition | Subjects kept untouched until the protocol and model-selection decisions are fixed, then used to estimate final performance. | It provides the clearest estimate available from this dataset of performance on subjects outside model development. | A final evaluation is required for a final performance claim; whether the design uses a separate held-out set or a cross-validation-only design is open. | No. | Final holdout versus cross-validation design, number of folds, and how test use is governed. |
| Fold / seed / assignments | A fold is one particular division of subjects into roles; a random seed helps reproduce randomized assignments. | Different subject divisions can yield different results, particularly with only 144 subjects. | Reproducible assignments are required by the execution plan; a specific fold scheme is not. | No. Five-fold stratified grouped splitting appears as a proposal, not a frozen decision. | Fold count, exact grouped splitting method, seed, and treatment of class proportions. |
| Model selection boundary | Restrict comparisons, tuning, threshold selection, and checkpoint selection to the development data allowed by the chosen protocol. | A test result used to choose the “best” configuration is no longer an independent final check. | **Yes.** Leakage controls require test data not to be used for selection. | The control is required; the detailed procedure is open. | Exact selection rules and the timing/frequency of final test evaluation. |

He et al. report a 6,000-trial test set and five-fold training/validation procedure over the remaining trials. The reviewed paper does **not** specify whether partitions were formed by subject or trial. This is uncertainty, not proof that the paper had leakage. Its reported scores therefore do not settle whether performance transfers to unseen children.

## 3. Leakage: what would make the evaluation unreliable

**Data leakage** happens when information that would not legitimately be available at prediction time influences model fitting or evaluation. Leakage can make scores look better than the model's actual ability to generalize.

| Risk or control | What it means | Why it matters here | Required by project? | Decided? | What remains open? |
|---|---|---|---|---|---|
| Same-subject trials cross partitions | The model trains on some trials from a person and is tested on other trials from that person. | EEG segments from the same person are related. The score may reward recognition of familiar people, not transfer to new subjects. | **Prohibited** for the unseen-subject evaluation. | Subject is the required grouping unit; exact partition implementation is open. | Verify zero subject overlap for every partition/fold. |
| Whole-dataset fitted preprocessing | A mean, scale, feature selection, learned transform, or other data-dependent setting is computed using training and held-out subjects together. | Held-out data influence the representation or settings before evaluation. This can bias results even if the labels are not directly passed to the model. | **Prohibited** for normalization; training-only fitting is required for data-dependent operations. | No global normalization; exact preprocessing is not approved and remains open. | For each future transformation, specify whether it is data-dependent and where it is fitted. |
| Test-driven decisions | The test score or test labels influence architecture, preprocessing, features, thresholds, early stopping, or which result is reported. | The test set then acts like another validation set; the final score no longer provides a clean check. | **Prohibited** by the split contract. | The principle is required; exact test protocol is open. | Define the final test-use rule before evaluation. |
| Label or subject metadata as model input | The target label, group block, subject key, trial order, or other metadata that directly reveals the answer is accidentally included as EEG input. | The model could learn the answer from metadata rather than EEG. `y_stim` rows 1–3 are group-block indicators, not stimulus categories. | **Required control.** Group label must be separated from EEG input and derived from block identity. | Label source and the ban on using rows 1–3 as classifier labels are decided. | Confirm the model input contains only the intended EEG-derived information. |
| Resampling before splitting | Duplicated or synthesized examples from one subject are created before the split, allowing closely related examples to land in different partitions. | This can recreate the same-subject overlap problem indirectly. | Follows from the subject-separation and leakage requirements. | No balancing method is selected. | If imbalance handling is chosen, define it inside the allowed training partition. |
| Adaptive preprocessing across held-out people | A procedure uses the held-out subjects' data to estimate shared parameters or to make decisions that are then evaluated as if fixed in advance. | The evaluation may no longer reflect a model and pipeline fixed without those subjects. | Required to be controlled and documented; training-only fitting is explicit for learned statistics. | Exact processing is undecided. | Distinguish fixed signal operations from operations that estimate parameters from a cohort. |

Leakage is not limited to a programming mistake. It can be introduced by an unclear research workflow—for example, repeatedly changing the protocol after examining final test results. The project documents do not establish that the base paper leaked; they establish that its split unit is unspecified.

## 4. Metrics, class-wise results, and confusion matrices

A **metric** is a numerical summary of model performance. Different metrics answer different questions. Because the groups have unequal numbers of subjects and unequal numbers of trials, the metric and the unit it counts must be stated together.

| Evaluation concept | What it means | Why it matters here | Required by project? | Decided? | What remains open? |
|---|---|---|---|---|---|
| Accuracy | Fraction of evaluated predictions that are correct. It can be calculated over trials or over subject-level results; these are different quantities. | A pooled trial accuracy gives greater weight to subjects with more trials. It also hides which group is often confused. | Accuracy is named as a candidate in project documents; the final metric set is not frozen. | **No.** Evaluation unit and primary metric remain open. | Whether and how to report accuracy, and whether it is trial- or subject-level. |
| Class-wise recall/sensitivity | For one class, the fraction of its true examples correctly identified. | Shows whether, for example, ADHD subjects are often missed even if overall accuracy looks high. | Per-class reporting appears in the baseline gate and evaluation proposals. | Not frozen as a complete reporting contract. | Which per-class measures to report and the evaluation unit. |
| Precision | Of predictions assigned to a class, the fraction that truly belongs to that class. | A class may have high recall but many false alarms; precision shows that trade-off. | Discussed in the paper and methodology audit; not frozen as a project metric. | No. | Whether to include it and how to average it. |
| F1 score | Combines precision and recall into one value. Macro-F1 first calculates the value per class and gives each class equal weight. | It can expose poor performance on a smaller class that a pooled score might obscure. It still needs a declared unit and averaging rule. | Macro-F1 is proposed in the execution plan and gate material. | **Not frozen.** The current project-state audit says the metric choice is undecided. | Whether to use macro-F1 and how predictions are aggregated. |
| AUC | Summarizes how well model scores rank examples from one class above the others across thresholds. For three classes, a one-versus-rest or other averaging convention must be stated. | It uses class scores rather than only the selected label, but it is not the same as accuracy and can be hard to interpret without an explicit multiclass convention. | AUC is discussed in the paper and proposed project metrics. | No. | The multiclass calculation, averaging rule, unit, and whether it is needed. |
| Confusion matrix | A table comparing true classes with predicted classes; each off-diagonal cell shows a particular type of confusion. | It can reveal whether ADD is mistaken for ADHD, or a clinical group for HC—information hidden by a single score. | Explicitly proposed in the baseline/evaluation gate; no final reporting standard is frozen. | No. | Whether it is reported in counts, normalized rates, or both, and whether it is trial- or subject-level. |
| Class and subject support | The number of subjects and predictions behind each reported class result. | There are 44 HC, 52 ADD, and 48 ADHD subjects, but 10,129, 13,031, and 10,742 trials respectively. Subjects contribute 46–295 trials. Reporting denominators makes weighting visible. | Required for interpreting results fairly; exact presentation is open. | Counts are known; reporting format is not decided. | How to show class support and unequal trial contribution alongside metrics. |
| Uncertainty across partitions | Variation in scores across folds or repeated subject assignments; an interval expresses uncertainty in an estimate. | A single split can be unusually easy or hard. The paper's “mean ±” values do not clearly identify the repetition unit or statistical meaning. | Transparent reporting is a stated concern in project audits; a specific uncertainty method is not frozen. | No. | Which variability/interval summary is appropriate for the selected design. |

### Why accuracy alone may not be enough

One accuracy number does not identify which groups are recognized or confused. It can also reflect the weighting induced by unequal trial counts. A complete interpretation therefore needs the evaluation unit, group-specific outcomes, and class supports to be clear. Project documents propose several complementary summaries, but have **not** selected a final set.

### Reading a three-class confusion matrix

In a confusion matrix, rows can represent the true group and columns the predicted group (or vice versa, if clearly labelled). Correct predictions lie on the diagonal; off-diagonal cells show mistakes. For this task, a reader should be able to see separately how often HC, ADD, and ADHD are correctly identified and which pairs are confused. The matrix should state whether its entries count trials or subjects; otherwise the varying number of trials per subject can change its interpretation.

The matrix is a way to inspect errors, not a replacement for a defined evaluation protocol or other metrics.

## 5. Failure cases and robustness

**Failure analysis** asks where the model makes mistakes, rather than only how many. **Robustness** asks whether performance remains dependable under specified changes or conditions. These ideas are related but not interchangeable: a held-out-subject test measures one kind of generalization; it does not automatically test every form of robustness.

| Concept | What it means | Why it matters here | Required by project? | Decided? | What remains open? |
|---|---|---|---|---|---|
| Failure-case analysis | Inspect misclassified subjects/trials and the true-to-predicted class patterns, with the chosen evaluation unit made explicit. | Helps identify systematic confusions (including ADD versus ADHD) and prevents a good aggregate score from hiding concentrated failures. The project objective explicitly says to examine what succeeds and fails. | **Yes**, failure cases are part of the research objective. | The objective is decided; the analysis protocol is not. | What error summaries or follow-up analyses are appropriate, and how to avoid selecting only illustrative examples. |
| Subject-level variability | Compare outcomes across held-out subjects, not just pooled trial counts. | Subjects have unequal trial counts, so a few high-trial subjects may dominate a trial-weighted summary. | Subject/session-level output is required by the prototype scope; subject-level research metric remains open. | Subject grouping is decided; result aggregation is not. | How to summarize variation across people and classes. |
| Unseen-subject generalization | Evaluate on entire subjects not used during development. | Directly tests the central project claim more fairly than holding out trials from already-seen people. | **Yes.** | Required invariant is decided; the exact protocol is not. | Fold/test design and final metric/reporting choices. |
| Robustness to input variation | Test reliability under specified variations that are relevant to the intended use. | The project scope explicitly expects evaluation of robustness to reasonable input variation, but “reasonable” needs a defensible, stated meaning. | **Yes, as a project requirement.** | No particular perturbation, variation, or robustness threshold is decided. | Which variations the available data can actually represent and how they would be assessed. |
| Confidence reliability | Check whether confidence/probability values correspond to how often predictions are correct, rather than merely displaying a number. | The prototype is intended to show per-class probabilities and confidence; a confident but often wrong prediction could mislead users. | Confidence display is required in the prototype scope; confidence calibration/evaluation is not explicitly frozen. | Display requirement decided; reliability measure and method open. | Whether and how confidence reliability will be evaluated before presenting it as meaningful. |
| Explanation stability | Check whether an explanation remains similar across subjects or folds, if explanations are produced. | The project wants to investigate information contributing to predictions, but a visualization alone does not validate an explanation or biomarker. | The scope requires experimentally validated explanations; Gate 7 requires bounded claims. | The boundary against clinical/causal claims is stated; explanation method and stability test are open. | Which explanation evidence is supportable given unresolved channel identities and the final model. |

The dataset is one cohort from one source. It does not provide documented independent sites or a prospective deployment sample. Therefore, broad claims about robustness to other hospitals, devices, populations, or acquisition protocols cannot be supported by this dataset alone. An internal held-out-subject result should not be described as external validation.

## 6. Reproducibility

**Reproducibility** means another person can understand and repeat the stated evaluation closely enough to check the result. It is more than obtaining the same score once: the data, labels, partitions, processing, selection rules, metric definitions, and software/run details need to be described.

| Concept | What it means | Why it matters here | Required by project? | Decided? | What remains open? |
|---|---|---|---|---|---|
| Data and label traceability | Record how each trial maps to its subject and group, and preserve source data. | The project has verified a composite key and block mapping; a reproducible result depends on using them consistently. | **Yes.** Mapping and raw-data preservation are frozen constraints. | Yes, with a documented filename-linkage caveat. | How final experiment outputs record the mapping and label provenance. |
| Partition reproducibility | Record the exact subject assignments to training, validation, and test/folds. | “Subject-wise split” is a requirement, but cannot be independently checked from a description that omits the actual assignments. | **Yes** for a verifiable evaluation. | No exact split is decided. | Split method, seed, assignments, and fold/test protocol. |
| Protocol and metric definitions | Record approved preprocessing, model-selection rules, metric unit, class averaging, and any aggregation. | Two reports can use the same metric name but measure different things. | Required by Gates 2, 3, and 6 before training/evaluation. | Only core constraints are frozen; full protocol is not. | Exact choices and explicit documentation. |
| Run and software records | Record relevant software versions, run settings, seeds, and the code/configuration used for each reported result. | Supports checking whether a result came from the stated procedure and whether reruns differ. | The execution plan proposes this; no ML run exists yet. | No final record format or run has been established. | Which run metadata to preserve and how variability will be reported. |
| Raw versus derived data | Keep source files immutable and document any separately created derived data. | Allows the path from released data to reported result to be audited. | **Yes.** Source files are read-only; preprocessing is not approved yet. | Yes for preserving raw data; derived-data policy is constrained by the pending preprocessing decision. | Future approved transformations and the associated records. |

The paper analysis finds that He et al.'s exact preprocessing, partition membership, evaluation unit, and repeated-run aggregation cannot be reconstructed from the available description. This limits exact reproduction of its reported numbers; it does not prove the numbers are false.

## 7. Baseline comparisons and ablations

These are two different comparison questions:

- A **baseline comparison** asks how the proposed approach performs relative to other established or simpler approaches under the same evaluation conditions.
- An **ablation** removes or changes one component of a chosen approach to investigate whether that component appears to matter under the tested setup.

| Concept | What it means | Why it matters here | Required by project? | Decided? | What remains open? |
|---|---|---|---|---|---|
| Baseline comparison | Evaluate the developed approach alongside comparison methods using the same subject partitions, label mapping, data access rules, and metric definitions. | A score alone cannot show whether the approach is useful relative to alternatives. The project has a Gate 4 requiring baselines before claiming a benefit from a Transformer. | **Yes, before making a claim of benefit over baselines or of Transformer-specific benefit.** | The gate requirement is recorded; baseline identities and complete comparison protocol are not frozen. | Which baseline(s), how many, and what constitutes a fair matched comparison. This audit selects none. |
| Paper-reported comparisons | He et al. report comparisons with EEGNet, ShallowConvNet, and DeepConvNet. | Provides context for what the paper compared, but the unclear split unit and metric details prevent treating the scores as directly comparable to a new subject-wise evaluation. | Not an automatic project requirement to reproduce every paper comparison. | No paper baseline is selected for this project by the current decisions. | Whether any paper comparison is selected later under the approved protocol. |
| Ablation experiment | Compare versions that differ in a defined component while keeping other conditions sufficiently controlled. | Can test a specific claim about a component or processing choice; it cannot establish that every component is necessary. | No specific project ablation is frozen. The project objective requires investigating contributing information, and an ablation may be one possible evidence type, not a mandated design. | No ablation list or component is decided. | Whether a stated research claim requires an ablation and what controlled contrast would answer it. |
| Paper's ablations | The base paper reports pooling/component removals and changes to block/head counts, with some conflicting table and prose results. | These are facts about the paper, not results for our implementation. Conflicts and missing details limit exact reproduction. | No. | Not a project decision. | Whether to discuss/reproduce any of them is open and should not be assumed. |

The older master guide and ML execution plan include candidate baseline and architecture ideas. Those lists are proposals, not approval to use particular models. This audit does not recommend or select a baseline architecture.

## 8. What the available dataset can and cannot support

### It can support, subject to a future approved protocol

- A three-group classification study on the released EEG trials and their HC/ADD/ADHD group labels.
- An internal test of generalization to held-out subjects from this cohort, **if** every subject is kept entirely outside training and all development decisions, and the label/data mapping is used as documented.
- Reporting class-specific successes and errors, with the evaluation unit and unequal subject trial counts made explicit.
- Controlled comparisons within this dataset if the comparison methods, protocol, and selection boundaries are decided in advance.
- A reproducibility record for the project’s own final protocol and results.

### It cannot establish by itself

- Clinical diagnosis, clinical utility, safety, or readiness for clinical deployment.
- Generalization to other hospitals, devices, recording protocols, age groups, or populations not represented and independently evaluated.
- That a model has found a causal brain mechanism or validated ADHD biomarker.
- That attention or other model visualizations are clinically valid explanations.
- That the base paper's reported score is invalid, or that leakage occurred in its experiment. Its participant split is undocumented in the reviewed evidence.
- That the project approach is better than another model unless a fair baseline comparison is completed under the same protocol.
- Reliable estimates for a broader population solely by treating the 33,902 trials as independent people.

The dataset contains 144 children from one cohort (44 HC, 52 ADD, 48 ADHD), with varying numbers of trials per person. Group labels are supported by block and filename/count evidence, not by a separate per-participant diagnosis field in the supplied files. The paper also has unresolved reporting inconsistencies. These are limits on interpretation, not evidence for an invented research gap.

## A. What we must evaluate

1. Whether the developed approach distinguishes the three dataset groups for subjects excluded from model development.
2. Whether subject identity is strictly separated across training, validation, and test roles; each subject's trials must remain together.
3. Which groups and subject/session cases are correctly classified or confused, not only one pooled score.
4. Whether results remain dependable under explicitly stated, dataset-supported conditions relevant to the project’s robustness requirement.
5. Whether any probability/confidence and explanation presented by the prototype have been evaluated and are described within their evidential limits.
6. Whether any claimed advantage over a baseline is supported by a fair comparison under the same partitions and evaluation rules.
7. Enough protocol, data mapping, partition, metric, and run information for another person to understand and check the result.

These are evaluation questions and controls, not an instruction to implement a particular experimental design.

## B. What would make an evaluation unreliable

- Allowing trials from the same person into both development and test partitions while claiming unseen-subject generalization.
- Fitting data-dependent transformations or selecting features using held-out subjects, or using test outcomes to tune or select the reported result.
- Accidentally including group/subject metadata that reveals the target in the model input.
- Reporting a trial-weighted metric as if every subject contributed equally, without disclosing the unequal trial counts and metric unit.
- Reporting accuracy alone without enough class-wise information to reveal group-specific failure.
- Leaving the partition, metric unit, averaging rule, test-use boundary, or prediction aggregation ambiguous.
- Claiming that within-cohort held-out subjects establish external or clinical validity.
- Treating an explanation visualization, confidence value, or baseline score as validated evidence without testing and appropriate qualification.
- Describing paper split uncertainty as proven leakage, or treating paper-reported results as directly comparable without a matched protocol.

## C. What is already decided

- The broad research task is three-group EEG classification for **unseen subjects**, with attention to strengths and failure cases.
- All trials from one subject must stay in the same partition for unseen-subject evaluation; the composite subject key is available for this purpose.
- The three group labels are HC, ADD, and ADHD. The target is derived from group-block identity; `y_stim` rows 1–3 are not to be used directly as labels.
- The dataset contains 144 subjects and 33,902 trials; trial counts vary by subject and group.
- The raw files must remain unchanged. The stored 385 time positions, 56 channel columns, and `float32` values are to be preserved unless a separately approved decision changes that handling.
- Full-dataset normalization is prohibited; any approved fitted statistics must use training data only.
- The base paper’s architecture is a reference, not an automatically adopted design.
- The prototype's intended primary result is subject/session-level, with prediction probabilities/confidence; explanations must be experimentally validated and must not be presented as clinical diagnosis.
- The available dataset is not equivalent to an independent external clinical dataset.

## D. What remains to be decided

- Exact subject-grouped split method, number of folds, subject assignments, random seed, and whether there is a separate held-out test set.
- Validation and model-selection rules, including how to keep final test subjects untouched.
- Whether the primary research metric is trial-level or subject/session-level, how predictions are aggregated, and the metric set/averaging conventions.
- How to present class-wise results, confusion matrices, per-subject variability, and uncertainty.
- Whether and how to account for class imbalance at the subject and trial levels.
- Which baseline comparisons are necessary, and whether any ablation is needed to support a specific claim. No specific baseline model is selected here.
- Which data variations can be evaluated credibly with this single cohort and what robustness claims those tests would support.
- Whether confidence calibration or explanation stability will be evaluated and how results will be bounded.
- The preprocessing and data-dependent transformations, which remain subject to the separate preprocessing decision gate.

## E. What claims we should avoid making

- “The model diagnoses ADHD,” “is clinically validated,” “is ready for deployment,” or “can replace clinical judgement.”
- “The model generalizes to all children/patients,” “other hospitals,” “other devices,” or populations not evaluated.
- “The model works on unseen people” if subjects or data-derived decisions crossed partition boundaries, or if the split unit is not reported.
- “The method is better” or “the Transformer is beneficial” without a fair baseline comparison on the same subject-grouped protocol.
- “The model found the cause/biomarker/brain region of ADHD” based only on model attribution or attention visualizations.
- “The paper leaked” or “the paper's result is invalid” based solely on its unspecified split unit.
- Any expected or promised performance percentage before the experiment is run.

## Sources reviewed

- [01_PROJECT_STATE_AUDIT.md](./01_PROJECT_STATE_AUDIT.md) — current objectives, frozen constraints, unresolved split and metric decisions, gate status.
- [02_DATASET_AND_DATA_CONTRACT_AUDIT.md](./02_DATASET_AND_DATA_CONTRACT_AUDIT.md) — dataset facts, labels, subject and trial counts, limitations.
- [03_BASE_PAPER_AND_RESEARCH_EVIDENCE_AUDIT.md](./03_BASE_PAPER_AND_RESEARCH_EVIDENCE_AUDIT.md) — paper's reported evaluation, split uncertainty, results, and reproduction boundary.
- [04_PREPROCESSING_DECISION_AUDIT.md](./04_PREPROCESSING_DECISION_AUDIT.md) and [SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md](../SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md) — leakage controls and unresolved data/split choices.
- [PROJECT_SCOPE_AND_SYSETEM_SPECIFICATION.md](../PROJECT_SCOPE_AND_SYSETEM_SPECIFICATION.md) — prototype-level subject/session output, robustness and explainability requirements, and non-clinical scope.
- [trial_subject_mapping_validation.md](../trial_subject_mapping_validation.md) — evidence and caveats for trial-to-subject/group mapping.
- [ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md](../ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md) and [ML_EXECUTION_PLAN.md](../ML_EXECUTION_PLAN.md) — candidate metrics, gates, baselines, and reproducibility ideas; proposals were not treated as frozen choices.
- [README.md](../../README.md) — repository-level research objective and record.

