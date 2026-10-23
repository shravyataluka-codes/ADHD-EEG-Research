# 03 — Base Paper and Research Evidence Audit

**Primary paper:** Yuchao He et al., “Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model,” *Journal of Neural Engineering*, 20 (2023), 056013. DOI: [10.1088/1741-2552/acf7f5](https://doi.org/10.1088/1741-2552/acf7f5). The repository copy is [`papers/base-paper.pdf`](../../papers/base-paper.pdf).

**Purpose:** Record what the base paper says, what follows directly from those statements, what remains unclear, what our project has independently checked, and what cannot be reproduced exactly from the available evidence. This document is an evidence summary, **not an architecture or implementation proposal**.

**Audience:** A Computer Science student who is new to EEG, signal processing, machine learning, and Transformers.

## How to read evidence labels

- **Paper states** — explicitly reported in He et al. (2023), as checked against the paper and summarized in [`base-paper-analysis.md`](../base-paper-analysis.md).
- **Direct reading** — a conservative interpretation of the paper's reported information; it is not an additional statement by the authors.
- **Project-verified** — reported as independently checked against our downloaded data, validation script, current code, or tests in the linked project documents. This does not mean the paper reported it.
- **Unspecified / unresolved** — the paper or project evidence does not settle the question. No missing details are filled in by assumption.
- **Candidate research question** — a topic previously raised in project documents, not a selected research direction or a claim of novelty.

## 1. Research problem

### Paper states

The authors investigate automatic classification of EEG data into three groups:

1. Healthy controls (**HC**)
2. Attention deficit disorder (**ADD**)
3. Attention-deficit/hyperactivity disorder (**ADHD**)

They motivate EEG-based automated classification as a possible aid to clinical assessment. The paper presents its EEG-Transformer as an auxiliary research/clinical-support tool, not a replacement for diagnosis.

The authors argue that EEG contains temporal and spatial information, and that their attention-based model may learn relationships in that data. Their paper compares it with EEGNet, ShallowConvNet, and DeepConvNet.

### Direct reading

The experiments address a **three-class classification task on trials from a cohort of children**. The participant count and trial count refer to different things: a participant is a child; a trial is a short EEG segment from that child. The 33,902 trials must not be read as 33,902 independent people.

The paper's reported performance is evidence about its own dataset and experimental setup. It is not, by itself, evidence of prospective clinical validity, generalization to a new hospital/dataset, or performance on unseen children.

## 2. Dataset, subjects, and data structure

### Paper states

| Item | Reported by He et al. |
|---|---|
| Source | Open EEG dataset associated with Technical University of Dresden; OSF project `6594x` |
| Participants | 144 children |
| Groups | 44 HC, 52 ADD, 48 ADHD |
| Number of trials | 33,902 total |
| Trials by group | HC 10,129; ADD 13,031; ADHD 10,742 |
| Trial/acquisition duration | 1.5 seconds |
| Sampling rate | 256 Hz |
| EEG channels | 56 |
| Broad representation | Two-dimensional data indexed by channel and sampling point |

The paper says that participants were classified/diagnosed by psychiatrists and psychologists using clinical indicators, interviews, questionnaires, and IQ/attention tests. It reports no group differences in age, IQ, or sex distribution, but does not provide a complete group-wise demographic table in the dataset-method description summarized in the paper audit.

### Project-verified

Our dataset audit reports that:

- The seven EEG arrays have shapes `(385, 56, 5000)` for `d1`–`d6` and `(385, 56, 3902)` for `d7`, for 33,902 trials in total.
- Each trial read by the current data-access code has shape `(385, 56)` and dtype `float32`.
- `y_stim.mat` has shape `(4, 33902)`. Its first row contains contiguous within-block subject indices; its remaining three rows are one-hot indicators that are constant within the three group blocks.
- The three metadata blocks contain 44, 52, and 48 subjects, and their group trial counts match the counts reported in the paper.
- `chan.mat` has 60 entries while the EEG arrays have 56 channel columns. The identities of the four unmatched entries have not been established.
- The subject-to-filename ordinal correspondence is supported by the metadata ordering but cannot be independently checked against the absent original individual subject files.

These findings and their limitations are documented in [`trial_subject_mapping_validation.md`](../trial_subject_mapping_validation.md) and [`01_PROJECT_STATE_AUDIT.md`](01_PROJECT_STATE_AUDIT.md). They are facts about our downloaded files and project checks, not extra statements made by the paper.

### Unspecified or unresolved

- The paper's reported duration and rate imply 384 sampling intervals (`1.5 × 256`), but its architecture table gives a length of 385. Neither source explains the extra sample or its sampling convention.
- The paper does not state a channel-name-to-column mapping. Our local `chan.mat`/EEG count mismatch leaves the exact 56-channel identities unresolved.
- Physical amplitude units and the electrode reference scheme are not firmly established in the available project evidence. Values that appear consistent with microvolts are not proof of units.
- The full participant-level inclusion/exclusion protocol and detailed demographics are not provided in the paper's dataset-method description.
- Earlier project notes sometimes describe rows 1–3 of `y_stim` as stimulus-category indicators. The later direct validation disproves that interpretation for these files: the rows encode the three group blocks. This is a project-data finding, not a paper statement.

## 3. Preprocessing reported by the paper

### Plain-language term

**Independent Component Analysis (ICA)** is a signal-separation method. In EEG work, it can be used to separate mixtures of activity into components that may include artifacts. Merely saying “ICA was used” does not tell a reader which components were removed or how they were chosen.

### Paper states

The authors state that the original EEG data were preprocessed using ICA to remove noise, then divided into 33,902 trials. The paper gives the 1.5-second duration, 56 channels, and 256 Hz sampling rate.

### What the paper does not specify clearly

The available paper description does not provide enough detail to reconstruct the preprocessing exactly, including:

- ICA algorithm/software, number of components, or component-removal criteria;
- which artifact types were identified or removed;
- filter type or frequency cutoffs, notch filtering, or re-referencing;
- baseline correction, normalization, bad-channel handling, or channel interpolation;
- trial rejection criteria, exact epoch boundaries, or whether trials overlap;
- whether preprocessing occurred before or after data splitting, or how any learned preprocessing settings were restricted to training data.

The paper does **not** establish that EOG or ECG components specifically were removed. Project literature notes that name particular artifact types or add specific filtering operations are not evidence that He et al. performed them.

### Project status

Our project documents that no preprocessing has been run and that the state of the downloaded OSF EEG with respect to the paper's ICA step remains unresolved. In particular, the paper's statement does not prove that the local files are already ICA-cleaned in a way we can identify or reproduce. Filtering, re-running ICA, normalization, and other transformations are not implemented or approved in the current project state; see [`01_PROJECT_STATE_AUDIT.md`](01_PROJECT_STATE_AUDIT.md) and [`ADHD_EEG_PREPROCESSING_AUDIT.md`](../ADHD_EEG_PREPROCESSING_AUDIT.md).

## 4. EEG representation and reported model

### Plain-language terms

- A **sequence** is an ordered list. In this setting the paper presents EEG as a sequence-like array over sampling positions, with channel values at each position.
- A **positional embedding** is a learned value added to each position so the model can distinguish where in the sequence a value occurs.
- **Attention** compares positions to estimate which ones are useful in representing another position. **Multi-head attention** performs several such comparisons in parallel.
- A **Transformer encoder block** is a repeated feature-processing unit containing attention, residual connections/normalization, and a feed-forward network.
- **Global max pooling** reduces a sequence of features to one summary by retaining the largest value for each feature across positions. In this pipeline, it turns the block's sequence output into a fixed-size vector for classification.
- **Softmax** converts the final class scores into three values that sum to one, commonly presented as class probabilities. This alone does not show that the values are calibrated confidence estimates.

### Paper states

The paper describes an encoder-style Transformer and explicitly says it removes the traditional Transformer decoder. Its reported main pipeline is:

```text
EEG input
→ learnable positional embedding
→ 6 Transformer blocks
   (multi-head attention, Add & Norm, feed-forward network, Add & Norm)
→ global max pooling
→ dropout
→ dense layer with 64 units and ReLU
→ dropout
→ 3-output softmax classifier
```

The base configuration is reported as six attention heads and six blocks. The architecture table reports a post-embedding shape of `(None, 385, 56)`, where `None` is conventionally the variable batch dimension. The paper does not fully explain the conversion from a raw trial into this tensor.

The paper evaluates a global-average-pooling alternative in an ablation study. **Global average pooling** would summarize each feature by its average over positions, rather than retaining its maximum. This describes the paper's comparison; it is not a recommendation for our project.

### Unspecified or conflicting model details

- The exact token/input construction and reconciliation of 384 implied samples versus 385 reported positions are not given.
- The paper reports conflicting total parameter counts: 506,883 in Table 1/Section 2.4 and Table 4 for six blocks, versus 528,423 in Table 2.
- Exact dropout rate and several low-level implementation choices are not available in the paper's report.
- The six-head parameter count is 201,759 in Table 4, but the prose says 20,179 for the two-head case.
- A reported parameter count does not uniquely specify a model implementation; the conflicting values prevent treating the count as an exact reproduction target.

The architecture above is a summary of what the paper reports, not a proposal to reuse it.

## 5. Training, splitting, and evaluation

### Paper states

The paper reports:

| Setting | Reported value |
|---|---|
| Maximum epochs | 300 |
| Batch size | 256 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Early stopping | Monitor validation loss; stop after 30 epochs without improvement |
| Learning-rate reduction | Halve the rate after 20 epochs without validation-loss improvement |
| Checkpointing | Model checkpointing is mentioned |
| Software/hardware | Python 3.8, TensorFlow-GPU 2.50 as printed, Windows 10 Enterprise, four NVIDIA TITAN Xp GPUs with 12 GB each |

The paper describes 6,000 trials as a test set and the remaining 27,902 trials as five parts, using four parts for training and one for validation in each fold. It also contains a sentence about the validation set after five training epochs that is unclear and does not define an exact schedule.

The paper reports accuracy, precision, recall, F1, ROC/AUC and ANOVA comparisons. It does not clearly specify the multiclass averaging details for the metrics or the exact ANOVA design, assumptions, degrees of freedom, or multiple-comparison procedure.

### Crucial split uncertainty

The paper does **not** state whether the test and cross-validation partitions were formed by participant or by trial. Thus:

- The counts and five-fold arrangement are reported.
- Participant-independent splitting is **not** reported.
- Trial-wise splitting is also **not proven** by the paper text available to this audit.
- Whether the same child appears in multiple partitions cannot be determined from the paper alone.

Some earlier project literature notes assert trial-wise/random splitting as fact. Because the paper does not document the split unit, that assertion is stronger than the available direct paper evidence and must remain unverified, rather than being repeated as a finding.

### Other unreported training details

The paper does not give random seeds, Adam beta settings, all checkpoint-selection details, dropout rate, exact repeated-run/fold aggregation rules, or a complete executable training configuration. The reported “mean ±” results are not sufficiently tied to a clearly described unit of repetition to reconstruct their statistical meaning exactly.

## 6. Reported results

### Main comparison

The paper reports the following model accuracies:

| Model | Reported accuracy |
|---|---:|
| EEG-Transformer | 95.58% ± 0.52% |
| EEGNet | 93.66% ± 0.79% |
| ShallowConvNet | 94.46% ± 2.16% |
| DeepConvNet | 75.08% ± 17.83% |

The authors report EEG-Transformer comparisons as statistically significant against EEGNet and DeepConvNet (`p < 0.05`) but not against ShallowConvNet (`p > 0.05`).

For EEG-Transformer, the paper also reports average AUC 0.9926, with class values HC 0.9932, ADD 0.9922, ADHD 0.9923. It describes approximate class-wise accuracies of 0.96, 0.97, and 0.96 and an average around 96.3% in its confusion-matrix discussion.

### Ablations and model-parameter experiments

The authors compare global max pooling with global average pooling, remove positional embedding, and remove attention, feed-forward, and Add & Norm components. The paper reports a large drop when replacing max pooling with average pooling (95.58% ± 0.52% versus 84.08% ± 3.31%), and reports little mean-accuracy change when positional embedding is removed (95.58% ± 0.52% versus 95.82% ± 0.71%). Other ablations are reported to reduce performance substantially, but the attention and Add & Norm values conflict between tables and prose.

The paper also varies block count and attention-head count. Its tables report increasing parameter counts and epoch times as these counts increase. The authors conclude that their six-block/six-head configuration provided a useful performance/cost balance in their experiment. That is the authors' conclusion for their setup, not a design decision or recommendation for our project.

### Results that should not be silently reconciled

The paper contains several numerical or reporting conflicts:

1. **Participant count:** Methods reports 144 children, but the Discussion later says the sample size is 300 without explaining the difference.
2. **Trial count/split wording:** A training-section sentence mentions 5,580 trials but is incomplete/unclear about their role.
3. **Sequence length:** 1.5 seconds at 256 Hz suggests 384 sampling intervals, while the model table reports length 385.
4. **Total parameters:** 506,883 and 528,423 are both reported for the EEG-Transformer in different places.
5. **Two-head parameters:** Table 4 says 201,759; prose says 20,179.
6. **Epoch time:** Table 2 reports 30 seconds/epoch for EEG-Transformer; Table 4 reports 45 seconds/epoch for the six-block setting.
7. **Attention-removal ablation:** Table reports 39.12% ± 14.88%; prose reports 39.20% and SD 0.55%.
8. **Add & Norm-removal ablation:** Table reports 16.23% ± 6.62%; prose reports 38.38% ± 0.81%.
9. **Six/eight-head F1:** Reported F1 values do not align obviously with nearby metrics; the paper does not explain the discrepancy.
10. **Alternative accuracy summaries:** Main comparison gives 95.58% ± 0.52%, while the abstract/optimization and confusion-matrix discussion contain different values or summaries, including 95.85% and approximately 96.3%.
11. **Block-optimization prose:** The discussion's approximate two-block result differs from the detailed table value.

These differences are recorded in [`base-paper-analysis.md`](../base-paper-analysis.md). They are unresolved in the source; this audit does not choose which number is “correct.”

## 7. Limitations and what the paper does not establish

### Author-stated limitation

The authors acknowledge the limited underlying participant sample and distinguish the 33,902 trial records from the number of people. They state that a larger sample is needed before clinical application and frame the current work as theoretical research.

### Evidence-based limits on interpretation

The paper does not provide enough detail to establish:

- whether results generalize to children excluded from training at the **participant** level;
- whether its reported scores are trial-level or subject-level summaries, or the exact population of predictions behind each score;
- whether any trial/subject overlap exists between training, validation, and test partitions;
- an externally validated or prospective clinical result;
- an exactly reproducible preprocessing and training pipeline;
- that attention weights provide a clinically valid explanation of a prediction.

The first three are uncertainties in the reported experimental protocol, not proof that leakage occurred. The high reported results should therefore be attributed to the paper's experiment, with its split/evaluation ambiguity kept visible.

## 8. Project evidence: what has been independently checked here

Based on the project's mapping report, data-access code, and state audit:

- **Dataset structure:** all 33,902 trials were checked across `d1`–`d7`; each has 385 samples and 56 channels, stored as `float32`. The validation report records no NaN/Inf values.
- **File order to metadata columns:** sequential `d1`–`d7` order was checked at all six boundaries. The same-subject spatial-variance correlations were reported from 0.9084 to 0.9977, exceeding the different-subject controls in those tests.
- **Subject runs and group blocks:** `y_stim` contains 144 contiguous subject runs and three group blocks; subject/trial totals match the paper's Methods counts.
- **Group-indicator rows:** rows 1–3 of `y_stim` are constant one-hot group-block indicators in the local data; they are not varying stimulus conditions.
- **Subject grouping key:** the project defines unique block/index IDs (`HC_01`…`HC_44`, `ADD_01`…`ADD_52`, `ADHD_01`…`ADHD_48`) for grouping trials by participant.
- **Filename association caveat:** the ordinal correspondence between each subject index and the original individual filename is not independently demonstrated because the individual source files are unavailable. The composite subject key is still suitable for keeping each contiguous subject run together; the original human name is not required for that purpose.
- **Current code:** the only implemented project module is data access (`TrialManifest` and `EEGTrialStore`). It reads metadata/trials and validates/accesses them; it does not preprocess, split, train, or classify.
- **Test scope:** the project state audit reports seven passing data-access tests, but says they use synthetic files. These tests check module behavior, not model performance or full processing on real data.
- **Work not done:** no preprocessing pipeline, model, training loop, or model evaluation has been implemented or run. The current project audit lists preprocessing Gate 2 and later ML gates as not passed.

For full evidence and scripts, see [`trial_subject_mapping_validation.md`](../trial_subject_mapping_validation.md), [`01_PROJECT_STATE_AUDIT.md`](01_PROJECT_STATE_AUDIT.md), and [`ML_EXECUTION_PLAN.md`](../ML_EXECUTION_PLAN.md).

## 9. Research gaps already raised in project documents

The following are **candidate questions already present in project notes**, not choices made by this audit, validated findings, or claims that no prior study has addressed them:

1. **Subject-wise generalization:** Does high reported performance persist when the test subjects are wholly absent from training? The paper's split unit is unspecified; our local subject mapping is now validated sufficiently for subject grouping.
2. **Preprocessing sensitivity:** How much do results depend on choices that the paper does not specify, such as artifact handling, filtering, normalization, or epoch construction?
3. **Fair baseline comparison:** How do model families compare under the same well-documented participant split and evaluation protocol? The original paper compares CNN baselines, but its split/evaluation details remain unclear.
4. **Subtype distinction:** How reliably can ADD and ADHD be distinguished from each other, rather than only from controls? Prior literature notes this as a challenge; this is a candidate question, not a confirmed gap unique to our project.
5. **Interpretability claims:** Which EEG positions/features influence predictions, and do any explanations remain stable across subjects? An attention mechanism alone does not establish clinical interpretability.
6. **Statistical reporting:** What do fold/run variability, participant-level outcomes, uncertainty intervals, and class-specific metrics show when the evaluation unit and aggregation are explicit?

These topics appear in [`ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md`](../ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md) and [`sprint_1_research_landscape.md`](../sprint_1_research_landscape.md). Some of those documents also propose methods or assert trial-wise splitting; those proposals are not adopted here, and the paper does not establish trial-wise splitting as fact.

## 10. Reproduction boundary

### Can currently be reproduced or checked from project materials

- The broad task definition, group names/counts, trial totals, sampling rate, duration, and channel count reported by the paper.
- The local EEG tensor shapes, file concatenation, subject runs, group-block structure, and a deterministic subject grouping key, according to the documented direct inspections.
- The paper's high-level encoder pipeline and the hyperparameters/results explicitly reported in its tables and text.

### Cannot currently be reproduced exactly

- The exact preprocessed EEG presented to the original model, including the ICA implementation and artifact decisions.
- The exact raw-trial-to-model-tensor conversion, especially the 384/385 discrepancy and token interpretation.
- The exact original participant/trial partition assignments and whether participants crossed partitions.
- The precise training, validation, test, and repeated-run aggregation behind every reported number.
- An exact model parameterization and implementation in light of omitted layer settings and conflicting parameter counts.
- The reported ablation/statistical results where text and tables conflict or the statistical design is incomplete.
- Clinical generalization claims beyond the reported cohort and evaluation.

“Not exactly reproducible” does not mean the paper's result is false. It means the available specification does not uniquely determine the original procedure or permit all its claims to be independently recreated.

## A. What the original paper definitely did

- It studied a three-group EEG classification problem on the Dresden/OSF cohort reported as 144 children: HC, ADD, ADHD.
- It reported 33,902 trials, 56 channels, 256 Hz, and 1.5-second trials.
- It stated that ICA was applied for noise removal before trial division.
- It described an encoder-style EEG-Transformer, with a learnable positional embedding, six attention heads and six Transformer blocks in the main configuration, global max pooling, a dense/ReLU classification stage, dropout, and a three-class softmax output.
- It compared EEG-Transformer with EEGNet, ShallowConvNet, and DeepConvNet; performed pooling/component ablations and block/head-count experiments; and reported high classification metrics.
- It acknowledged the limited underlying sample and the need for more data before clinical application.

## B. What the paper only mentions without enough detail

- ICA/noise removal and the trial-generation step, without an executable preprocessing recipe.
- A 385-position model input without reconciling it with 1.5 seconds at 256 Hz.
- A test set and five-fold training/validation procedure without naming the split unit or fully describing assignment/aggregation.
- Reported training settings and evaluation metrics without all implementation/statistical details needed for exact reruns.
- Component ablations and high performance alongside unreconciled numerical contradictions.

## C. What we have independently verified

- The local file shapes, trial count, channel count, metadata layout, group-block counts, and contiguous subject runs.
- The `d1`–`d7` sequential join to `y_stim` through reported boundary signal checks.
- The mapping from trials to unique project subject-group keys, with original individual filename identity remaining a separate caveat.
- The current implemented scope is data access only; the project has not yet implemented preprocessing, a model, training, or evaluation.

## D. What we cannot safely assume

- That the original paper used trial-wise splitting, or that it used subject-wise splitting.
- That the reported results are for unseen participants, or that trial overlap definitely occurred.
- That the OSF files' current preprocessing state exactly matches the paper's ICA statement.
- That channel names can be assigned to the 56 array columns from the 60-entry metadata alone.
- That the missing sample, amplitude units, reference, overlap policy, tokenization, dropout rate, or conflicting model/result values have an obvious intended meaning.
- That the paper's architecture, preprocessing, metrics, or reported performance should be copied into our project.

## E. Evidence useful when we later design our own system

The following are facts and constraints to keep available for a later, separately made design decision:

- The task is three-class classification over 144 subjects but 33,902 highly repeated trials; subjects, not trials, are the independent people represented in the cohort.
- The local project now has a validated trial-to-subject/group crosswalk, while the original participant filenames are not fully cross-validated.
- The local EEG shape is `(385, 56)`, yet the paper's duration/rate imply a 384-sample expectation; all 385 positions are currently preserved by project policy pending clarification.
- The channel-name mapping, amplitude units/reference, and exact preprocessing state remain unresolved.
- The paper's preprocessing description, split unit, evaluation aggregation, and several architecture/result values are insufficiently specified for exact replication.
- Paper-reported accuracy is not directly comparable to a future project result until the evaluation units and split rules are stated and differences in preprocessing/model setup are made explicit.
- Current project gates and frozen decisions are documented in [`01_PROJECT_STATE_AUDIT.md`](01_PROJECT_STATE_AUDIT.md). This audit makes no architecture choice and does not authorize any preprocessing choice.

## Source documents in this repository

- Primary source PDF: [`papers/base-paper.pdf`](../../papers/base-paper.pdf)
- Paper-specific extraction and inconsistency catalog: [`base-paper-analysis.md`](../base-paper-analysis.md)
- Project methodology/gap audit: [`ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md`](../ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md)
- Dataset/preprocessing audit: [`ADHD_EEG_PREPROCESSING_AUDIT.md`](../ADHD_EEG_PREPROCESSING_AUDIT.md)
- Trial/subject/group validation: [`trial_subject_mapping_validation.md`](../trial_subject_mapping_validation.md)
- Current project state: [`01_PROJECT_STATE_AUDIT.md`](01_PROJECT_STATE_AUDIT.md)
