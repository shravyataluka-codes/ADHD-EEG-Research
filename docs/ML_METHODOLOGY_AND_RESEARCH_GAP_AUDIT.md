# ADHD EEG — ML Methodology & Research Gap Audit

**Status: ML METHODOLOGY AUDIT IN PROGRESS**  
**Pre-coding research checkpoint**  
**No model implementation authorized**

- **Date of audit:** 2026-09-29
- **Base paper:** Yuchao He et al., "Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model", *Journal of Neural Engineering*, 20 (2023), 056013. DOI: 10.1088/1741-2552/acf7f5.
- **Dataset:** Open-source ADHD/ADD/HC EEG dataset associated with the Technical University of Dresden and OSF project 6594x; local repository inventory includes 7 MATLAB EEG files (`d1.mat`–`d7.mat`), `y_stim.mat`, `sub_name_stim.mat`, and `chan.mat`.
- **Audit purpose:** To reconstruct the original paper's ML pipeline, identify what was actually reported, determine what is reproducible, and clarify the methodological gap before any implementation.
- **Evidence classification:** Important statements are labeled using the following tags where appropriate: [VERIFIED FROM PAPER], [VERIFIED FROM CODE], [VERIFIED FROM DATA], [STRONGLY SUPPORTED], [INFERENCE], [UNKNOWN], [RESEARCHER DECISION].

---

## 1. Project Context

### 1.1 What our project is

This repository is a research project investigating ADHD EEG classification from a three-class clinical dataset. The immediate goal is not to produce a final model yet, but to audit the evidence, methodology, and dataset constraints before coding or training. [VERIFIED FROM CODE]

### 1.2 What the base paper is

The base paper is a 2023 study by Yuchao He et al. titled "Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model". The authors present an EEG-Transformer for three-class classification of ADHD, ADD, and healthy controls (HC). [VERIFIED FROM PAPER]

### 1.3 Why the base paper matters

The base paper is the central methodological reference for this project because it appears to define the intended task, clinical groups, and a candidate model architecture. It is also the paper our future implementation will either reproduce, criticize, or improve upon. [STRONGLY SUPPORTED]

### 1.4 What part of the project is the ML responsibility

The ML responsibility is the transformation of EEG trials into a supervised learning problem and the generation of a clinically relevant classification pipeline. That includes the data handling, preprocessing decision points, representation choice, model architecture, objective, evaluation, and reporting rules. [RESEARCHER DECISION]

### 1.5 Why this audit is being performed before implementation

This audit is required before implementation to ensure that the following are not silently assumed:

- the exact trial-to-subject mapping,
- the subject-to-label mapping,
- the preprocessing protocol,
- the model input shape,
- the evaluation unit,
- and the split strategy.

The project has already documented that the crosswalk from EEG trial to participant to diagnosis is not independently proven. [VERIFIED FROM DATA / VERIFIED FROM CODE]

### 1.6 Original paper versus our future implementation

- Original paper: a published, retrospective study with a reported pipeline and results.
- Our future implementation: a new, transparent, leakage-aware implementation that must be justified with explicit data provenance, subject-aware splits, and auditable preprocessing choices.

These are not the same thing. The paper is not evidence that our implementation is correct or that its reported performance is valid under a subject-wise evaluation protocol. [INFERENCE]

---

## 2. Original Paper ML Pipeline

The reconstructed pipeline reported by the paper is:

RAW EEG  
↓  
PREPROCESSING (ICA / noise removal)  
↓  
TRIAL / SEGMENT REPRESENTATION  
↓  
INPUT TRANSFORMATION (channel × time matrix; positional embedding)  
↓  
TRANSFORMER ENCODER BLOCKS  
↓  
GLOBAL MAX POOLING (GMP)  
↓  
CLASSIFICATION HEAD  
↓  
SOFTMAX OUTPUT  
↓  
EVALUATION

### 2.1 What happens at each stage

1. Raw EEG data are acquired as EEG recordings from children in three groups.
2. The paper states that ICA was applied to remove noise.
3. The data were divided into 33,902 trials.
4. Each trial is represented as a 2D EEG matrix with one dimension for channel and one for sampled points.
5. The model adds positional embedding before attention-based feature extraction.
6. A 6-block Transformer encoder is used.
7. The output is pooled by Global Max Pooling.
8. A dense layer with ReLU and dropout is applied.
9. A softmax layer predicts 3 classes.
10. Performance is reported with accuracy, AUC, and additional metrics. [VERIFIED FROM PAPER]

### 2.2 Exact parameters available from the paper

- 56 channels
- 1.5-second EEG trial
- 256 Hz sampling rate
- 33,902 trials in total
- 3 classes: HC, ADD, ADHD
- 6 Transformer blocks
- 6 attention heads
- positional embedding reported in architecture table as `(None, 385, 56)`
- GMP used in the main architecture
- Dense layer with 64 units
- ReLU activation
- Softmax output
- Early stopping, learning rate reduction, Adam optimizer, 300 epochs, batch size 256

The paper does not provide a complete end-to-end implementation description, so several exact low-level settings remain unreported. [VERIFIED FROM PAPER / UNKNOWN]

### 2.3 Missing information

The paper does not sufficiently explain:

- exact ICA implementation and rejection criteria,
- filtering details,
- baseline procedure,
- re-referencing method,
- channel ordering and physical labeling,
- how the 385-sample dimension is formed,
- exact label construction from subject metadata,
- exact train/validation/test partition strategy,
- whether subjects overlap across splits,
- exact loss function,
- random seed and repeated-run variance handling,
- and whether evaluation is trial-level or subject-level. [UNKNOWN]

### 2.4 Reproducibility

The paper is partially reproducible at a conceptual level but not fully reproducible at the implementation detail level. The main issue is that the project documentation and paper provide enough to understand the high-level task but not enough to reconstruct a faithfully equivalent model and evaluation protocol without assuming missing rules. [INFERENCE]

---

## 3. Exact Model Input

### 3.1 Original EEG shape

The paper reports:

- 56 channels,
- 1.5-second duration,
- 256 Hz sampling rate,
- and a 2D representation of channel × sampling-point measurements.

This implies approximately 384 time samples if 1.5 × 256 = 384. However, the model table reports a positional-embedding tensor of `(None, 385, 56)`, which adds one extra position. This discrepancy is unresolved. [VERIFIED FROM PAPER / UNKNOWN]

### 3.2 Number of channels, time points, sequence length

- Channels: 56 [VERIFIED FROM PAPER]
- Sampling rate: 256 Hz [VERIFIED FROM PAPER]
- Trial duration: 1.5 s [VERIFIED FROM PAPER]
- Expected time points: 384 [INFERENCE from 1.5 × 256]
- Reported model sequence length: 385 [VERIFIED FROM PAPER]

The paper does not explain the difference between 384 and 385. [UNKNOWN]

### 3.3 Token definition

The paper describes the EEG input as a 2D representation of channels and sampled time points. It does not explicitly define a token. The most plausible interpretation is either:

- one token = one time step with 56 channel values, or
- one token = one channel vector over time.

But the exact tokenization rule is not stated. [UNKNOWN]

### 3.4 Embedding mechanism

The paper states that the model adds learnable positional embedding before Transformer blocks, and the architecture table reports a shape of `(None,385,56)`. This indicates that positional embedding is learned and added to the sequence representation, rather than a fixed sinusoidal encoding. [VERIFIED FROM PAPER]

### 3.5 Positional encoding

The paper says the model uses positional embedding to preserve time/location information. It is described as trainable and added to the input. It is not a sinusoidal encoder in the commonly cited original Transformer formulation. [VERIFIED FROM PAPER]

### 3.6 Channel representation

The paper references 56 channels but does not provide a mapping from channel number to electrode names, reference scheme, or spatial ordering semantics. The local dataset includes a MATLAB `chan.mat` with 60 channel metadata entries, but the exact 56-channel array mapping is not yet independently proven. [VERIFIED FROM DATA / UNKNOWN]

### 3.7 Temporal representation

The paper reports that temporal structure is retained in the sequence dimension of the Transformer. However, the exact interpretation of time positions and the required sequence window is not fully specified. [STRONGLY SUPPORTED]

### 3.8 Input dimensionality

The architecture table appears to indicate an input of the form:

- sequence length: 385
- feature size per time step: 56
- batch dimension: variable (`None`)

The paper therefore effectively reports a tensor shape approximated as `(batch, 385, 56)` after positional embedding. [VERIFIED FROM PAPER]

### 3.9 How one raw EEG trial becomes the input to the Transformer

Conceptual flow:

RAW EEG TRIAL  
→ 1.5-second recording across 56 channels  
→ channel × time representation  
→ sequence of time steps / channel vectors  
→ learnable positional embedding  
→ Transformer encoder blocks  
→ classification head

The paper does not give the exact implementation formula for this conversion, so the exact tensor construction cannot be reproduced with certainty. [UNKNOWN]

### 3.10 Simple conceptual diagram

```text
Raw EEG trial (56 channels × 1.5 s)
        ↓
Trial matrix (channel × time)
        ↓
Tokenize / sequence representation
        ↓
Learnable positional embedding
        ↓
Transformer encoder blocks
        ↓
Global max pooling
        ↓
Classification head
```

This is the most faithful conceptual reconstruction from the paper. It should not be mistaken for a proven exact implementation. [INFERENCE]

---

## 4. Exact Model Output

### 4.1 Classification task

The output task is a three-class classification problem among:

- HC (healthy control),
- ADD,
- ADHD.

This is stated explicitly in the paper and reflected in the dataset description. [VERIFIED FROM PAPER]

### 4.2 Number of classes

The output layer contains 3 classes. [VERIFIED FROM PAPER]

### 4.3 Class definitions

The classes correspond to the three clinical groups: healthy control, ADD, and ADHD. [VERIFIED FROM PAPER]

### 4.4 Final layer

The paper describes a softmax output layer after a dense layer and pooling. The final layer is therefore interpreted as a softmax classifier over 3 classes. [VERIFIED FROM PAPER]

### 4.5 Activation

The final layer uses softmax. Earlier stages use ReLU. [VERIFIED FROM PAPER]

### 4.6 Loss

The paper does not explicitly report the loss function in the sections reviewed here. A standard setup for a three-class softmax classification task would be categorical cross-entropy, but that is not directly stated in the paper. [UNKNOWN]

### 4.7 Prediction unit

The paper appears to predict a class for one EEG trial, not one participant. The model input is an EEG trial and the output is class membership for that segment. Because the task involves many trials per subject, it is not safe to assume the prediction unit is a person unless the paper explicitly defines that. [STRONGLY SUPPORTED / INFERENCE]

### 4.8 Trial-level vs subject-level prediction

The original study reports the model on trial-level classification metrics. The paper does not establish a subject-level summary such as one prediction per child or a subject-aware aggregation of patient-level probabilities. [VERIFIED FROM PAPER / UNKNOWN]

### 4.9 How predictions become reported results

The paper reports accuracy, AUC, precision, recall, F1, and confusion-matrix-style class results, but the evaluation protocol is not fully transparent. The project audit indicates that repeated trials from the same subject can inflate performance if the split is not subject-wise. [STRONGLY SUPPORTED]

### 4.10 Explicit answer: What does one model prediction represent?

One model prediction represents the class label assigned to one EEG trial or segment under the paper's experimental design; it does not by itself prove a diagnosis for a person unless the evaluation is explicitly aggregated or the subject mapping is validated. [INFERENCE]

---

## 5. 144 Subjects vs 33,902 Trials

This section is dedicated to the exact relationship between subjects, trials, and labels.

### 5.1 What the project has verified

Local dataset inspection shows:

- 7 MATLAB EEG files (`d1.mat`–`d7.mat`),
- file sums giving 33,902 trials,
- 56 channels,
- and a candidate three-block subject structure in `y_stim.mat` and `sub_name_stim.mat`.

The exact trial-to-subject crosswalk is not independently proven. [VERIFIED FROM DATA / UNKNOWN]

### 5.2 How 144 subjects relate to 33,902 trials

The numbers are consistent with a repeated-trials-per-subject structure:

- 144 subjects,
- many EEG trials per subject,
- total trials > total subjects by a large margin.

This means 33,902 is not 33,902 unique children. [VERIFIED FROM DATA / STRONGLY SUPPORTED]

### 5.3 Are trial IDs available?

No independent, explicit trial IDs were found in the project audit. The raw EEG files appear to be arrays grouped by file blocks, not keyed to unique subject/trial identifiers. [VERIFIED FROM DATA / UNKNOWN]

### 5.4 Are subject IDs available?

The local metadata strongly suggests subject indices are encoded in `y_stim.mat` as row-0 sequences within each block, but these values are not an independently established, authoritatively cross-linked patient ID table. [STRONGLY SUPPORTED / UNKNOWN]

### 5.5 Can each trial be assigned to a subject?

A candidate mapping can be reconstructed from file totals and block ordering, but it has not been independently validated. There is no direct, immutable crosswalk from raw EEG trial position to patient ID across all files. [INFERENCE / UNKNOWN]

### 5.6 Number of trials per subject

The local metadata suggests highly variable numbers of trials per subject, with counts ranging across the candidate groups. This is important because unequal trials per subject can create a natural weighting effect and generate misleadingly strong performance if splits are trial-wise. [VERIFIED FROM DATA]

### 5.7 Does the paper document the trial-to-subject mapping?

The paper does not give a clean, explicit trial-to-subject crosswalk. It states the counts and group totals, but not the exact mapping from a trial index to a participant identity. [UNKNOWN]

### 5.8 Can the local dataset reconstruct the mapping?

The local dataset can reconstruct a deterministic candidate mapping using file ordering and row-0 sequence runs, but that is not the same as verified provenance. It remains a plausible reconstruction, not a proven truth. [INFERENCE]

### 5.9 Does the mapping require assumptions?

Yes. The reconstruction requires assumptions that:

- file order matches metadata block order,
- `y_stim` row 0 is the person index within a block,
- the `sub_name_stim` filename order matches index order,
- and the block decomposition corresponds to HC / ADD / ADHD labels.

These assumptions are strong but not independently demonstrated. [INFERENCE]

### 5.10 Explicit result classification

- Verified: repeated-trials-per-subject structure is present in the data.
- Reconstructable: a candidate deterministic mapping can be built from metadata and file counts.
- Assumed: file order to `y_stim` join and label block alignment are assumed unless proven.
- Unknown: exact person-level provenance for every EEG trial.

Therefore: "Which particular EEG trial represents which particular child?" — this is not yet verified. A subject-level crosswalk remains unknown or only partially reconstructable. [VERIFIED FROM DATA / UNKNOWN]

---

## 6. Clinical Label Construction

This section determines how the clinical labels were obtained and how they are linked to the EEG data.

### 6.1 What the original work claims

The paper states the dataset contains 3 clinical groups:

- HC / controls,
- ADD,
- ADHD.

It describes participants as diagnosed by clinical personnel using standard clinical indicators and interviews. [VERIFIED FROM PAPER]

### 6.2 Where the clinical labels originate

The labels are understood to come from clinical diagnosis at the subject level, not directly from the raw EEG files. The raw `.mat` archive does not present an explicit object like `subject_label` or `diagnosis` keyed to each trial. [VERIFIED FROM DATA / UNKNOWN]

### 6.3 Are labels subject-level or trial-level?

They are subject-level in principle, since diagnosis is attached to a child. However, the paper appears to use repeated EEG trials per subject and reports trial-level classification metrics. This creates a labeling ambiguity unless every trial is linked to the same subject and the partitioning respects subject boundaries. [STRONGLY SUPPORTED]

### 6.4 Are labels explicitly stored?

The local data do not show a clean, explicit subject-to-diagnosis table. The project audit has identified a candidate block-based arrangement in `sub_name_stim.mat` and group totals in `y_stim.mat`, but not a direct diagnosis table. [VERIFIED FROM DATA / UNKNOWN]

### 6.5 Are filenames used?

Filenames and metadata blocks are strongly suggestive of group assignment, but they are not the same as a verified diagnosis table. The local audit identifies that the three metadata blocks correspond to 44, 52, and 48 subject names and that filename strings include `controls`, `subtype1`, and `subtype2`. Those tokens are consistent with the paper's group counts. [VERIFIED FROM DATA / STRONGLY SUPPORTED]

### 6.6 Is `y_stim` a clinical label?

No. The project audit makes a critical distinction:

- `y_stim.mat` is not a subject diagnosis table,
- row 0 appears to encode a within-block subject index,
- rows 1–3 appear to encode stimulus/category indicators.

The data structure is consistent with task or stimulus coding rather than a direct clinical diagnosis label. [VERIFIED FROM DATA / STRONGLY SUPPORTED]

### 6.7 Is the mapping reproducible?

The grouping can be reconstructed at the block/subject level using metadata and counts, but the exact crosswalk between trial and patient is not fully validated. Therefore, the clinical label construction is strongly supported as a candidate interpretation, but not independently proven as a final ground-truth mapping. [STRONGLY SUPPORTED / UNKNOWN]

---

## 7. Original Preprocessing

### 7.1 What the paper says

The paper states that the original data were preprocessed by ICA to remove noise and then divided into 33,902 trials. [VERIFIED FROM PAPER]

### 7.2 Preprocessing table

| Operation | Reported? | Exact parameters | Reproducible? | Evidence | Our status |
|---|---|---|---|---|---|
| Filtering | Not clearly reported | None stated | No | [UNKNOWN] | Not approved |
| Notch filtering | Not clearly reported | None stated | No | [UNKNOWN] | Not approved |
| ICA | Yes | Exact algorithm and component criteria not reported | No | [VERIFIED FROM PAPER / UNKNOWN] | Not approved |
| Artifact removal | Implicitly yes via ICA, but without details | No thresholds or rejection criteria | No | [VERIFIED FROM PAPER / UNKNOWN] | Not approved |
| Baseline correction | Not reported | None stated | No | [UNKNOWN] | Not approved |
| Epoching | Yes, implicitly | 1.5 s trial duration reported; exact segmentation rule not supplied | No | [VERIFIED FROM PAPER / UNKNOWN] | Not approved |
| Cropping | Not explicitly described | Not specified | No | [UNKNOWN] | Not approved |
| Normalization | Not reported | None stated | No | [UNKNOWN] | Not approved |
| Referencing | Not reported | None stated | No | [UNKNOWN] | Not approved |
| Channel selection | 56 channels reported | Channel choice not explained in detail | No | [VERIFIED FROM PAPER / UNKNOWN] | Not approved |
| Resampling | Reported as 256 Hz | No resampling method or interpolation rule | No | [VERIFIED FROM PAPER] | Not approved |
| Amplitude scaling | Not reported | No scaling rule | No | [UNKNOWN] | Not approved |

### 7.3 Important distinction

The distinction we must keep clear is:

- “paper says this was done” is not the same as
- “we know exactly how it was done.”

The paper text provides high-level statements, but the exact implementation is not transparent. [INFERENCE]

---

## 8. Original Transformer Architecture

### 8.1 Reconstructed architecture

The paper describes an EEG-Transformer based on the standard Transformer encoder without the decoder. The high-level structure is:

```text
EEG input
  ↓
Learnable positional embedding
  ↓
Transformer blocks (×6)
  ├── Multi-Head Attention
  ├── Add & Norm
  ├── Feed-Forward Network
  └── Add & Norm
  ↓
Global Max Pooling
  ↓
Dropout
  ↓
Dense(64) + ReLU
  ↓
Dropout
  ↓
Softmax(3)
```

[VERIFIED FROM PAPER]

### 8.2 Reported architectural values

- Input sequence: approximately `(None, 385, 56)` after positional embedding
- 6 Transformer blocks
- 6 attention heads
- GMP pooling
- Dense classification layer with 64 units
- ReLU activation
- residual connections / Add & Norm
- final softmax output for 3 classes

### 8.3 Missing exact values

The paper does not clearly specify:

- exact `d_model`,
- feed-forward hidden dimension,
- exact dropout rate,
- layer normalization details,
- exact attention scaling,
- sequence tokenization rule,
- and whether the final head is a 3-way dense layer or a pooled representation followed by a classifier. [UNKNOWN]

### 8.4 Tensor-shape flow

The paper gives enough to reconstruct the conceptual flow:

```text
(batch, 56, time_points)
      ↓
(positional embedding)
(batch, 385, 56)
      ↓
Transformer blocks
(batch, 385, d_model)
      ↓
Global max pooling
(batch, d_model)
      ↓
Dense(64)
(batch, 64)
      ↓
Softmax(3)
(batch, 3)
```

This is a conceptual reconstruction, not a full verified implementation. [INFERENCE]

---

## 9. Original Training Procedure

The paper reports the following training configuration:

- Optimizer: Adam [VERIFIED FROM PAPER]
- Learning rate: 0.001 [VERIFIED FROM PAPER]
- Epochs: 300 [VERIFIED FROM PAPER]
- Batch size: 256 [VERIFIED FROM PAPER]
- Early stopping: Yes, monitored on validation loss [VERIFIED FROM PAPER]
- Learning-rate reduction: Yes, reduce by half if validation loss does not improve for 20 epochs [VERIFIED FROM PAPER]
- Weight decay: Not reported [UNKNOWN]
- Dropout: Mentioned in architecture but exact rates not reported [UNKNOWN]
- Initialization: Not reported [UNKNOWN]
- Class weighting: Not reported [UNKNOWN]
- Sampling: Not reported in a reproducible way [UNKNOWN]
- Random seed: Not reported [UNKNOWN]
- Hardware: Not reported [UNKNOWN]
- Hyperparameter tuning: Not reported as a formal search [UNKNOWN]

For each item not explicitly stated, the correct status is: NOT REPORTED / UNKNOWN. [RESEARCHER DECISION]

---

## 10. Data Splitting and Leakage

### 10.1 What the paper likely did

The project audit indicates that training, validation, and test splits may have been trial-wise rather than subject-wise, but this is not fully transparent in the paper. The paper does not clearly specify a subject-aware split. [INFERENCE / UNKNOWN]

### 10.2 Trial-level split?

The project indicates a strong risk that trials from the same child may appear in multiple partitions if the split is done by trial. This would inflate the apparent performance because many EEG trials are not independent observations. [STRONGLY SUPPORTED]

### 10.3 Subject-level split?

This is required for a valid unseen-participant claim. All trials from the same subject must stay together. [RESEARCHER DECISION]

### 10.4 Validation and test split?

The paper mentions validation loss with early stopping and learning-rate reduction, but the exact split strategy is not adequately described. [UNKNOWN]

### 10.5 Cross-validation?

Not clearly reported. [UNKNOWN]

### 10.6 Repeated experiments?

The project documents some repeated-run reporting, but the exact repeated-run protocol is not fully specified. [UNKNOWN]

### 10.7 Same participant across partitions?

This cannot be established from the paper alone. It is a critical concern because the dataset contains multiple trials from each subject. [UNKNOWN]

### 10.8 Why subject-level evaluation matters for our experiment

For our future project, evaluating at the subject level is critical because a participant's EEG trials share the same person-level diagnosis and recording/behavioral characteristics. Trial-wise evaluation can make a model look better than it is by accident when there is participant overlap between train and test data. [RESEARCHER DECISION]

---

## 11. Original Results Audit

### 11.1 Reported results

The paper reports the following main comparison results for the model families:

| Model | Accuracy reported |
|---|---:|
| EEG-Transformer | 95.58% ± 0.52% |
| EEGNet | 93.66% ± 0.79% |
| ShallowConvNet | 94.46% ± 2.16% |
| DeepConvNet | 75.08% ± 17.83% |

[VERIFIED FROM PAPER]

### 11.2 Additional metrics

The paper also reports:

- AUC for EEG-Transformer around 0.9926,
- class-wise proportions around 0.96–0.97,
- precision, recall, F1 and confusion-matrix-style results in optimization and ablation sections,
- ANOVA-based comparisons with p-values between major models.

[VERIFIED FROM PAPER]

### 11.3 Measurement unit

The reported results are not clearly established as subject-level metrics. The strongest reading is that they are trial-level metrics over segmented EEG trials. [STRONGLY SUPPORTED]

### 11.4 Important warning

33,902 trials are not equivalent to 33,902 independent participants. A model can appear highly accurate while using repeated trials from the same participant across splits. [VERIFIED FROM DATA]

### 11.5 What is actually demonstrated

The paper demonstrates that, under its own experimental protocol, the proposed EEG-Transformer achieved high reported trial-level accuracy and AUC in the reported setup. [VERIFIED FROM PAPER]

### 11.6 What it does not demonstrate

It does not fully demonstrate:

- generalization to unseen children,
- subject-level robustness,
- a leakage-safe trial-to-subject mapping,
- a fully reproducible preprocessing protocol,
- an exact, replicable architecture implementation,
- and a fully transparent evaluation unit. [UNKNOWN / INFERENCE]

---

## 12. What Is Missing or Underspecified?

### What Is Missing or Underspecified?

The following items are missing or underspecified in the paper and/or local implementation documentation:

- exact trial-to-subject mapping,
- exact subject-to-label mapping,
- exact trial ordering and file-to-metadata join,
- subject-aware train/test separation,
- exact preprocessing parameters,
- implementation-specific ICA settings,
- EEG-to-token transformation,
- channel semantics and ordering,
- handling of the 385-sample discrepancy,
- normalization scheme,
- random seed,
- hyperparameter search protocol,
- evaluation unit,
- repeated-run statistics,
- subject-level metrics,
- implementation details for attention and pooling,
- and interpretability/attribution methodology.

[UNKNOWN / INFERENCE]

These are not minor omissions. Several of them determine whether the reported performance is valid or simply a consequence of leakage or a dataset-order artifact. [STRONGLY SUPPORTED]

---

## 13. What Can We Improve?

### 13.1 Reproducibility

- What the paper does: reports a high-level architecture and some training settings.
- What is missing: exact data processing choices, full architecture parameters, and a clear split procedure.
- What we could do: document every preprocessing step, version, seed, split logic, and model hyperparameter before evaluation.
- Why it matters: this enables a fair comparison and protects against hidden leakage.
- Feasible with our dataset: yes, if the run is approved and the mapping is validated. [RESEARCHER DECISION]

### 13.2 Data handling

- What the paper does: uses many repeated trials from each subject.
- What is missing: exact subject IDs and the trial-to-subject mapping.
- What we could do: validate the block/subject crosswalk independently and enforce subject-wise splits.
- Why it matters: only subject-wise evaluation can answer the real clinical generalization question.
- Feasible with our dataset: yes, but only after the mapping is validated. [RESEARCHER DECISION]

### 13.3 Evaluation

- What the paper does: reports trial-level metrics and some class-wise performance.
- What is missing: subject-level results, confidence intervals, repeated-run statistics, and explicit split rules.
- What we could do: report patient-wise metrics, robust estimates, and clear train/validation/test separation.
- Why it matters: avoids overestimating generalization.
- Feasible with our dataset: yes, subject-wise evaluation is feasible if the subject mapping is valid. [RESEARCHER DECISION]

### 13.4 ML architecture

- What the paper does: uses a 6-block EEG-Transformer with GMP and softmax output.
- What is missing: full implementation details and exact hyperparameters.
- What we could do: build a smaller, more transparent architecture and compare against baselines under subject-wise evaluation.
- Why it matters: reduces opacity and supports fair ablation studies.
- Feasible with our dataset: yes, but not before the data contract is validated. [RESEARCHER DECISION]

### 13.5 Statistical reporting

- What the paper does: reports means and p-values, but some values appear inconsistent.
- What is missing: exact experimental design, confidence intervals, and reproducible repeated-run reporting.
- What we could do: report per-fold subject-level metrics, confidence intervals, variance, and effect sizes.
- Why it matters: it makes the result interpretable and comparable.
- Feasible with our dataset: yes. [RESEARCHER DECISION]

### 13.6 Interpretability/transparency

- What the paper does: uses attention as a core mechanism but does not establish that attention is a clinically valid explanation.
- What is missing: validated attribution methods and post-hoc explanation analysis.
- What we could do: add attention visualization, feature attribution, and ablation studies that relate model decisions to EEG structure.
- Why it matters: it helps distinguish a predictive signal from a valid clinical explanation.
- Feasible with our dataset: yes, but must be treated as secondary to subject-wise validation. [RESEARCHER DECISION]

---

## 14. Transformer Transparency

This section is critical: attention is not automatically an explanation.

### 14.1 Attention visualization

- What it explains: which time steps or positions the model attends to.
- What it does not explain: whether those positions are clinically meaningful or causally relevant.
- Suitability for EEG: moderate, because temporal and channel patterns can be inspected.
- Difficulty: low to moderate.
- Limitation: attention may reflect learned shortcuts or dataset regularities, not true clinical signal. [INFERENCE]

### 14.2 Attention rollout

- What it explains: approximate aggregation of attention across layers.
- What it does not explain: validated neurophysiological truth or a direct causal explanation.
- Suitability for EEG: moderate.
- Difficulty: moderate.
- Limitation: still not a validated clinical explanation. [INFERENCE]

### 14.3 Temporal attribution

- What it explains: which time windows contribute most to a prediction.
- What it does not explain: whether those windows are diagnostically meaningful or stable across subjects.
- Suitability for EEG: high.
- Difficulty: moderate.
- Limitation: temporal saliency can be sensitive to preprocessing and baseline choices. [INFERENCE]

### 14.4 Channel attribution

- What it explains: which EEG channels contribute most to the prediction.
- What it does not explain: whether the channels are biologically interpretable without a clear channel-reference map.
- Suitability for EEG: high if channel mapping is valid.
- Difficulty: moderate.
- Limitation: uncertain mapping and subject variability reduce interpretability. [INFERENCE]

### 14.5 Saliency

- What it explains: gradient-based sensitivity of the output to input features.
- What it does not explain: whether the highlighted features are causal or robust across subjects.
- Suitability for EEG: high.
- Difficulty: moderate.
- Limitation: can be noisy and sensitive to preprocessing choices. [INFERENCE]

### 14.6 Integrated gradients

- What it explains: average contribution of each input feature over a path from baseline to input.
- What it does not explain: direct disease mechanism or causal effect.
- Suitability for EEG: high.
- Difficulty: moderate.
- Limitation: depends on baseline choice and model stability. [INFERENCE]

### 14.7 Occlusion / perturbation

- What it explains: the effect of masking or perturbing parts of the input on the prediction.
- What it does not explain: whether the model has discovered a valid mechanistic biomarker.
- Suitability for EEG: high.
- Difficulty: moderate.
- Limitation: can be computationally expensive and hard to interpret biologically without domain priors. [INFERENCE]

### 14.8 Feature attribution

- What it explains: which features contribute to model prediction.
- What it does not explain: whether a feature is clinically validated or diagnostically meaningful.
- Suitability for EEG: high.
- Difficulty: moderate to high.
- Limitation: attribution is not the same as clinical explanation. [INFERENCE]

### 14.9 Mandatory caution

The following statement must remain explicit:

> MODEL ATTENTION is not equivalent to a validated or clinically meaningful explanation.

This distinction is essential. Attention maps are useful as descriptive diagnostics, but they cannot replace robust, domain-aware feature attribution, replication across subjects, or clinically grounded validation. [RESEARCHER DECISION]

---

## 15. Our Possible ML Pipeline

This is a preliminary, not-frozen pipeline for a future study.

```text
RAW EEG
  ↓
metadata validation
  ↓
preprocessing
  ↓
representation
  ↓
Transformer
  ↓
classification
  ↓
interpretability
  ↓
subject-level evaluation
```

| Component | Status | Rationale |
|---|---|---|
| RAW EEG | REQUIRED | Raw data must remain immutable and auditable. |
| metadata validation | REQUIRED | Trial-subject-label mapping must be validated before any split. |
| preprocessing | REQUIRED, but not frozen | Exact processing must be prespecified, documented, and fit within training folds only. |
| representation | REQUIRED | 2D channel × time representation or an explicit encoded variant must be defined. |
| Transformer | OPTIONAL for future experiments | Candidate architecture, not final. |
| classification | REQUIRED | Need clear three-class objective and output contract. |
| interpretability | OPTIONAL but recommended | Required for transparency and scientific credibility. |
| subject-level evaluation | REQUIRED | Crucial for valid generalization claims. |

This is a project decision framework, not an implementation plan. [RESEARCHER DECISION]

---

## 16. Possible Research Questions

The following are candidate questions that flow from the actual literature gap identified by this audit.

### 16.1 Research question 1

- Research question: Does the reported EEG-Transformer performance hold under subject-wise evaluation rather than trial-wise evaluation?
- Hypothesis: Performance will decrease substantially when repeated trials from the same subject are not shared across train and test partitions.
- Input: raw or minimally processed EEG trials with explicit subject IDs.
- Baseline: trial-wise pipeline and subject-wise baseline.
- Proposed method: subject-aware split using validated subject keys; compare identical model architecture under both splits.
- Evaluation: subject-level accuracy, macro-F1, AUC, and confidence intervals.
- Potential contribution: clarifies whether the original result is robust or partly inflated by leakage.
- Main limitation: requires a validated subject-trial crosswalk. [RESEARCHER DECISION]

### 16.2 Research question 2

- Research question: Which EEG time windows or channels drive ADHD vs ADD vs HC discrimination in the Transformer?
- Hypothesis: The model relies on a subset of temporal windows and channels rather than all EEG dynamics equally.
- Input: EEG trials with channel-level attribution and temporal saliency.
- Baseline: standard Transformer with no attribution analysis.
- Proposed method: use saliency, integrated gradients, or occlusion-based attribution.
- Evaluation: stable attribution across subjects and cross-validation folds.
- Potential contribution: improves trust, transparency, and neuroscientific plausibility.
- Main limitation: attribution is not equivalent to a validated biomarker. [RESEARCHER DECISION]

### 16.3 Research question 3

- Research question: How sensitive are the reported results to preprocessing choices such as filtering, ICA, normalization, and epoch handling?
- Hypothesis: Small but systematic changes in preprocessing can materially affect the model's apparent performance.
- Input: the same subject-wise dataset under multiple prespecified preprocessing pipelines.
- Baseline: no extra preprocessing beyond the fixed raw representation.
- Proposed method: controlled ablation across explicit preprocessing variants.
- Evaluation: performance differences under a frozen subject-wise protocol.
- Potential contribution: clarifies whether performance is driven by signal processing rather than the model itself.
- Main limitation: requires a well-validated data contract, and preprocessing choices can easily leak information. [RESEARCHER DECISION]

### 16.4 Research question 4

- Research question: Does an attention-based model provide information that baseline CNNs or classical EEG features do not?
- Hypothesis: Transformer attention captures temporal dependencies that are useful but not necessarily clinically interpretable.
- Input: same EEG trials, same subject-wise splits, compare Transformer vs CNN / conventional feature models.
- Baseline: EEGNet/ShallowConvNet/DeepConvNet or classical spectral features.
- Proposed method: matched architecture study under identical subject-wise data splits.
- Evaluation: accuracy, macro-F1, AUC, calibration, and stability over folds.
- Potential contribution: clarifies the value of attention in EEG classification.
- Main limitation: without validated subject mapping and controls, any improvement may be due to trial leakage or inadequate preprocessing. [RESEARCHER DECISION]

---

## 17. Final Audit Conclusion

### 17.1 What exactly did the original paper do?

The original paper presents a three-class EEG classification system based on EEG-Transformer for HC, ADD, and ADHD. It uses a 1.5-second trial representation, 56 channels, attention-based sequence feature extraction, pooling, and a softmax classifier. It reports very high trial-level performance. However, several implementation details remain underspecified and the exact trial-to-subject mapping is not independently established. [VERIFIED FROM PAPER / UNKNOWN]

### 17.2 What can we reproduce?

Conceptually, we can reproduce the high-level ML objective: a three-class EEG classification task based on EEG trials. We can also reproduce the broad architecture concept: attention-based feature extraction plus a classification head. [VERIFIED FROM PAPER]

### 17.3 What cannot we reproduce?

We cannot reproduce the original paper's exact implementation details without missing information about preprocessing, split strategy, trial-to-subject mapping, and architecture parameters. The paper does not provide enough to recreate the exact data pipeline or experimental conditions reliably. [UNKNOWN]

### 17.4 What is missing?

The main missing items are the exact subject-trial mapping, the subject-level split procedure, the precise preprocessing decisions, the exact architecture parameters, the exact training losses and seeds, and the clear evaluation unit. [UNKNOWN]

### 17.5 What do the reported results actually demonstrate?

The results demonstrate that the paper's own implementation, under its own trial-based setup, achieved high reported metrics. They do not yet establish a leakage-safe, subject-level generalization claim. [STRONGLY SUPPORTED]

### 17.6 What do they not demonstrate?

They do not establish that the model would generalize to new children under a valid subject-wise split, nor that the attention mechanism encodes valid clinical explanation. [INFERENCE]

### 17.7 What methodological improvements are possible?

The most important improvements are: validate the subject-level data contract, enforce subject-wise splits, document preprocessing and architecture fully, report stable subject-level metrics, and separate prediction performance from clinical interpretability. [RESEARCHER DECISION]

### 17.8 What transparency mechanisms are realistic?

Attention maps, temporal attribution, channel attribution, integrated gradients, saliency, perturbation/occlusion studies, and feature attribution are all realistic. However, they should be treated as evidence of model sensitivity, not as proof of clinical truth. [RESEARCHER DECISION]

### 17.9 What are our candidate research directions?

Candidate directions include leakage-controlled subject-wise benchmarking, preprocessing sensitivity analysis, attention and attribution analysis, channel/time window analysis, and comparison against classical EEG baselines under a rigorously frozen evaluation protocol. [RESEARCHER DECISION]

### 17.10 What should we do next?

1. Validate the exact trial-to-subject and subject-to-label mapping.
2. Freeze a subject-wise evaluation protocol.
3. Decide and document the EEG preprocessing contract before implementation.
4. Define the explicit model input, output, and evaluation unit.
5. Only then implement and benchmark the model.

This is the correct ordering for any future model implementation. [RESEARCHER DECISION]

---

## Current Status

- Dataset investigation: COMPLETED
- ML methodology audit: COMPLETED
- Research gap identification: COMPLETED
- Preprocessing protocol: NOT FROZEN
- Model architecture: NOT FROZEN
- Model implementation: NOT STARTED
- Model training: NOT STARTED
- Experimental results: NOT GENERATED

"NO MODEL CODE HAS BEEN IMPLEMENTED."  
"NO EEG PREPROCESSING HAS BEEN EXECUTED."  
"NO EXPERIMENTAL RESULTS HAVE BEEN GENERATED."
