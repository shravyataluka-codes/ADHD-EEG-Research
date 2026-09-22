# Base Paper Analysis

> **Primary source:** Yuchao He et al., *Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model*, Journal of Neural Engineering, 20 (2023) 056013.
>
> **Scope:** This document audits what the paper itself reports. It separates author statements, reported results, observations/questions arising from the paper, and information that cannot be determined from the paper. No external methodological assumptions are used to fill gaps.

## 1. Paper Information

| Item | Reported information |
|---|---|
| **Title** | *Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model* |
| **Authors** | Yuchao He, Xin Wang, Zijian Yang, Lingbin Xue, Yuming Chen, Junyu Ji, Feng Wan, Subhas Chandra Mukhopadhyay, Lina Men, Michael Chi Fai Tong, Guanglin Li, Shixiong Chen |
| **Journal** | *Journal of Neural Engineering* |
| **Year** | 2023 |
| **Volume / article** | 20, 056013 |
| **DOI** | 10.1088/1741-2552/acf7f5 |
| **Received / revised / accepted / published** | 24 April 2023 / 3 September 2023 / 8 September 2023 / 21 September 2023 |
| **Dataset/source** | Open-source EEG dataset from the Technical University of Dresden, Germany; data structure is linked by the authors to OSF: https://osf.io/6594x/ |
| **Task** | Three-class EEG classification: ADHD, ADD, and healthy control (HC) |

The paper states that the dataset was approved by the Ethics Committee of the Technical University of Dresden and that informed consent was obtained from participants and their guardians.

**Source evidence:** the paper identifies itself as a 2023 *Journal of Neural Engineering* paper and gives the DOI and author list; it identifies the Technical University of Dresden dataset and OSF location in the Methods section.

---

## 2. Research Problem

### What problem are the authors trying to solve?

The authors address automatic classification of ADHD-related EEG signals. More specifically, their experimental task distinguishes three groups:

1. ADHD
2. ADD
3. Healthy controls (HC)

The proposed system is intended to extract features from EEG signals and classify the resulting EEG data automatically.

The authors frame the clinical problem around the fact that ADHD diagnosis is primarily based on clinical assessment using DSM-5-related criteria and that assessment can be time-consuming and affected by a shortage of trained specialists. They therefore investigate whether EEG combined with deep learning can provide an objective method that can assist physicians.

### Why do the authors consider this important?

The paper states that ADHD can seriously affect attention, cognitive processes, working memory, learning, study and life. It also states that accurate diagnosis is needed for treatment and that EEG and MRI-based methods can assist clinical diagnosis.

### What limitations of existing approaches do the authors identify?

The paper distinguishes between several existing approaches:

- **Traditional machine learning:** requires manual feature extraction and feature selection, which the authors describe as difficult for high-dimensional data.
- **CNN:** the authors state that CNNs can extract spatial information, but their feedforward structure does not explicitly consider correlations between data in the way required for temporal information.
- **RNN:** the authors state that RNNs can capture time information through recurrent connections, but are complex, cannot perform parallel computation, and can have gradient disappearance/explosion problems.
- **Need for EEG-specific handling:** EEG contains both spatial and temporal information. The authors therefore motivate an attention-based architecture capable of processing sequence relationships in parallel.

### Explicit statement vs inference

**Explicitly stated by the authors:**
- EEG contains substantial spatial and temporal information.
- They consider CNN and RNN limitations relevant to EEG processing.
- Transformer attention can process relationships in parallel.
- They propose an EEG-Transformer for EEG feature extraction and classification.

**Reasonable inference from the paper:**
- The paper is specifically trying to replace or supplement convolution/recurrent feature extraction with attention-based feature extraction for this three-class EEG task.

The latter is an interpretation of the paper's reasoning rather than a new claim made independently.

---

## 3. Research Motivation

The authors' reasoning can be reconstructed as:

**Existing clinical problem**  
→ ADHD diagnosis relies substantially on clinical assessment and can be delayed because of time requirements and shortage of trained specialists.

**Existing computational approaches**  
→ Machine-learning approaches require manually engineered features; CNN and RNN approaches have limitations for the combination of spatial and temporal EEG information as described by the authors.

**Identified gap / motivation**  
→ EEG contains time information and correlations between signal elements, while Transformer attention can model relationships in parallel.

**Proposed approach**  
→ Adapt the Transformer architecture to EEG as an **EEG-Transformer**, retaining an encoder-style Transformer structure while removing the decoder and adding EEG-oriented processing and classification components.

**Expected benefit stated by the authors**  
→ Better extraction of EEG spatiotemporal information, high-performance classification, faster convergence than the compared CNN models, and a possible auxiliary tool for clinical diagnosis / basis for transferable EEG classification models.

This is the authors' chain of reasoning, not a justification for a future project.

---

## 4. Dataset

### Dataset source

The paper states that it uses an open-source dataset from the **Technical University of Dresden in Germany** and gives the OSF location:

`https://osf.io/6594x/`

The paper says the data structure of the dataset can be viewed at that location.

### Participants

The Methods section reports:

- **Total participants:** 144 children
- **ADHD:** 48
- **ADD:** 52
- **HC:** 44

The paper states that the groups had no differences in age, IQ, or sex distribution. It also states that patients had no other severe or acute psychiatric comorbidities such as autism, convulsions, or depressive episodes.

The participants were diagnosed/classified by psychiatrists and psychologists using standard clinical indicators, home and school interviews, questionnaires, and IQ and attention tests.

### Trials / samples

The paper states:

- Original data were preprocessed using independent component analysis (ICA) to remove noise.
- The data were divided into **33,902 trials**.
- ADHD: **10,742 trials**
- ADD: **13,031 trials**
- HC: **10,129 trials**

These group counts sum to 33,902.

### EEG acquisition

The paper reports:

- **Acquisition time per experiment/trial:** 1.5 s
- **Number of electrode channels:** 56
- **Sampling rate:** 256 Hz
- **Representation:** two-dimensional arrangement of channel number and sampling point.

From the stated duration and sampling rate, 1.5 × 256 corresponds to 384 sampling points. However, the model parameter table reports a positional-embedding tensor of `(None, 385, 56)`. The paper does not explain this discrepancy.

### Demographic information

The paper states only that there were no differences in age, IQ, or sex distribution between the groups. It does not provide, in the visible dataset-method description, a full table of group-wise age means, standard deviations, sex counts, or IQ statistics.

### Inclusion/exclusion information

The paper states that the children were diagnosed/classified by psychiatrists and psychologists and that patients had no other severe or acute psychiatric comorbidities. It does not provide a complete participant-level inclusion/exclusion protocol in the Methods section.

### Participant vs trial/sample

This distinction is essential.

- **Participant:** a child in the clinical dataset. The Methods section reports 144 children.
- **Trial/sample:** a 1.5-second EEG experiment/trial derived from the EEG data. The paper reports 33,902 trials.

Therefore, 33,902 trials **must not be interpreted as 33,902 independent children**.

### Reported Inconsistencies

The paper contains several internally inconsistent or unclear statements:

1. **144 participants vs 300 participants**
   - Section 2.1 explicitly reports **144 children**: 48 ADHD + 52 ADD + 44 HC.
   - Near the end of the Discussion, the paper states: **"The sample size of this study is 300"**.
   - No explanation is given for the change from 144 to 300.
   - The paper therefore does not establish whether 300 is a typo, a different sample definition, or refers to something else.

2. **33,902 trials vs later unclear test/training statement**
   - Section 2.1 clearly states 33,902 total trials and 6,000 test trials.
   - A later line in the training section is grammatically/visually incomplete: `"Were used as the training sets and 5580 trials were used as"`.
   - The surrounding text does not clearly define the role of the 5,580 figure.

3. **1.5 s × 256 Hz vs model input length 385**
   - The acquisition information implies 384 sampling points per trial.
   - Table 1 reports `(None,385,56)` after position embedding.
   - The paper does not explain the extra position/time dimension.

4. **Total parameter count**
   - Section 2.4 states **506,883 model training parameters**.
   - Table 2 reports **528,423** parameters for EEG-Transformer.
   - Table 4 reports **506,883** for the six-block configuration.
   - The paper does not reconcile these values.

5. **Attention-head parameter count**
   - Table 4 gives **201,759** parameters for 2 attention heads.
   - The prose says **20,179**.
   - These are not the same number; the paper does not clarify which is correct.

6. **Single-epoch training time**
   - Table 2 gives EEG-Transformer **30 s** per epoch.
   - Table 4 gives the six-Transformer-block configuration **45 s** per epoch.
   - Because both appear to describe EEG-Transformer but under different experimental sections, the paper does not clearly explain the configuration difference responsible for the discrepancy.

7. **Ablation accuracy for Multi-Head Attention**
   - Table 3 reports **39.12 ± 14.88%**.
   - The text states that removing Multi-Head Attention gives **39.20%**, with a standard deviation of **0.55%**.
   - These values conflict.

8. **Ablation interpretation of the Multi-Head Attention standard deviation**
   - The table and prose provide substantially different variation values for the same ablation.
   - This matters because the paper uses stability/variation to interpret the experiments.

9. **F1-score for six attention heads**
   - The prose reports **F1 = 96.05 ± 0.54%** for six heads, whereas Accuracy, Precision and Recall are around 95%.
   - The paper does not explain the unusually different F1 value or whether this is a transcription/calculation issue.

10. **Discussion wording for Transformer-block results**
    - The numerical results in Section 3.3 and later Discussion are not always identical. For example, the Discussion describes the two-block performance as approximately **85.42 ± 0.72%**, whereas the earlier detailed result gives Accuracy **84.98 ± 0.76%**.
    - The paper does not reconcile these figures.

These inconsistencies should be preserved rather than silently corrected.

---

## 5. Preprocessing

### Reported pipeline

The paper provides only a high-level preprocessing description:

**Raw/original EEG data**  
→ **Independent Component Analysis (ICA) for noise removal**  
→ **division into 33,902 trials**  
→ **1.5 s trials, 56 channels, 256 Hz**  
→ **2D channel × sampling-point representation**  
→ **EEG-Transformer input**

### What the authors did

The paper states that the original data were **preprocessed by independent component analysis to remove noise**.

It then states that the data were divided into 33,902 trials.

### Why they say it was necessary

The stated purpose of ICA preprocessing was **noise removal**.

The paper does not give a detailed justification of the exact ICA procedure.

### What changed as a result

The paper indicates that noise was removed and the data were divided into trials. It does not provide enough information to reconstruct exactly which EEG artifacts/components were removed.

### Missing preprocessing details

**Not specified in the paper:**

- Exact ICA algorithm/implementation
- Number of ICA components
- Criteria for identifying components as noise
- Whether eye/muscle/cardiac artifacts were separately identified
- Filtering frequencies
- Notch filtering
- Re-referencing method
- Baseline correction
- Normalization/scaling
- Channel interpolation
- Bad-channel rejection
- Trial rejection criteria
- Whether preprocessing was performed before or after train/test splitting
- Whether the same preprocessing parameters were learned using only training data
- Exact trial segmentation procedure
- Whether trials overlap
- Exact relationship between original continuous recordings and the 33,902 trials

These cannot be determined from the paper.

---

## 6. Methodology

### Complete reported flow

**EEG trial**  
→ 2D EEG representation  
→ learnable positional embedding  
→ Transformer blocks  
→ Global Max Pooling (GMP)  
→ Dropout  
→ fully connected Dense layer with 64 units + ReLU  
→ Dropout  
→ Softmax output with 3 classes.

The model is called **EEG-Transformer**.

### Input

The paper describes each EEG group as a two-dimensional arrangement of:

- channel number
- sampling point.

There are 56 channels and a stated 1.5-second acquisition at 256 Hz.

The architecture table reports the tensor shape `(None,385,56)` after position embedding, although the paper does not explain why the sequence dimension is 385 rather than the 384 sampling points implied by 1.5 s × 256 Hz.

### Representation

The Transformer receives a sequence-like representation of EEG data and adds a trainable positional embedding before attention.

### Model

The architecture is based on the Transformer encoder.

The paper explicitly says that the **Decoder structure is cancelled** from the traditional Transformer.

The remaining Transformer-style feature extraction is followed by GMP and a classification head.

### Training

The reported training configuration includes:

- Epochs: 300
- Batch size: 256
- Optimizer: Adam
- Learning rate: 0.001
- Model checkpointing
- Early stopping
- Reduce learning rate

Early stopping monitors validation loss and stops training if the loss does not improve within 30 epochs.

Reduce-learning-rate monitors validation loss and halves the learning rate if validation loss does not improve within 20 epochs.

### Prediction

The final Softmax layer has three outputs corresponding to the three classification groups:

- HC
- ADD
- ADHD

The paper does not explicitly provide the numerical mapping between Softmax index and class label.

---

## 7. Architecture

### Overall architecture

The paper's architecture can be reconstructed as:

```text
EEG input
   ↓
Position Embedding
   ↓
Transformer Block × 6
   ├── Multi-Head Attention
   ├── Add & Norm
   ├── Feed Forward Network
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

The model removes the traditional Transformer **Decoder**.

### 7.1 Input representation

The paper reports the EEG data in two dimensions: channel number × sampling point.

The architecture table gives the post-position-embedding tensor as:

`(None, 385, 56)`

The meaning of `None` is not explicitly discussed in the paper; it is the variable batch dimension in the architecture table.

The paper does not provide a complete explicit equation converting the raw 56-channel EEG trial into this exact tensor.

### 7.2 Position embedding

**What it is:** a trainable weight matrix added to the input with the same dimension as the input data.

**Where:** before the attention mechanism.

**Role:** the authors say EEG contains time-related/location information, and attention by itself may ignore location information. The learnable positional encoding therefore supplies positional information.

**Parameters reported:** 21,560.

**Reported shape:** `(None,385,56)`.

**Important observation:** the paper later reports that removing positional embedding slightly increased mean accuracy, while the error increased slightly. The authors nevertheless retain it for stability/location information.

### 7.3 Multi-Head Attention

**What it is:** the core attention-based feature extraction module.

The paper defines:

`Q = WQ X`

`K = WK X`

`V = WV X`

It then computes scaled dot-product attention:

`Attention(Q,K,V) = softmax(QK^T / √dk)V`

Multiple attention operations are performed in parallel and concatenated:

`MultiHead(Q,K,V) = Concat(head1,...,headh) WO`

The paper says that the parallel heads allow the model to focus on information in different subspaces.

**Where:** inside every Transformer block.

**Role:** EEG feature extraction using attention-based relationships.

**Reported six-head block parameter count:** 76,328 per Multi-Head Attention module in Table 1.

**Number of heads in the reported base architecture:** 6.

### 7.4 Feed-Forward Network

The FFN is a two-layer fully connected module.

The first layer uses ReLU:

`FFN(x) = max(0, xW1 + b1)W2 + b2`

**Where:** after Multi-Head Attention and the first Add & Norm.

**Role:** the authors describe it as part of the Transformer feature-extraction structure; in the Discussion they associate the nonlinear ReLU component with nonlinear fitting and training stability.

**Reported parameters:** 7,288 per Transformer block in Table 1.

### 7.5 Add & Norm

The model uses Add & Norm twice:

`Add&Norm1 = LayerNorm(X + MultiHeadAttention(X))`

`Add&Norm2 = LayerNorm(X + FeedForward(X))`

**Where:**
- first after Multi-Head Attention
- second after FFN

**Role:** combines residual connections with Layer Normalization. The authors state that the residual structure helps prevent network degradation and gradient disappearance/explosion, while normalization helps convergence and stability.

**Reported parameters per Add & Norm:** 112.

### 7.6 Number of Transformer blocks

The base architecture contains:

**6 Transformer blocks**

Each block contains:

1. Multi-Head Attention
2. Add & Norm
3. Feed Forward
4. Add & Norm

The optimization experiment tests 2, 4, 6 and 8 blocks.

### 7.7 Global Max Pooling

The paper uses **Global Max Pooling (GMP)** after the Transformer blocks.

**Role:** dimensionality reduction while retaining the most influential/salient feature information.

Reported output:

`(None,56)`

The ablation experiment compares GMP with Global Average Pooling (GAP).

### 7.8 Fully connected layer

After pooling:

- Dense layer: 64 units
- Activation: ReLU
- Reported parameters: 3,648

A Dropout layer follows the Dense layer.

### 7.9 Softmax/output

Final layer:

- Softmax
- 3 outputs
- 175 reported parameters

This produces the three-class classification output.

### 7.10 Components removed from traditional Transformer

The paper explicitly states that the **Decoder** was removed.

The resulting model is therefore encoder-style rather than the original encoder-decoder Seq2Seq Transformer.

### Architecture parameter inconsistency

The paper reports multiple total parameter counts:

- Table 1 / Section 2.4: **506,883**
- Table 2: **528,423**
- Table 4 for 6 blocks: **506,883**

This must be treated as an unresolved inconsistency.

---

## 8. Training and Experimental Setup

### Train/validation/test strategy

The paper states:

- Total: 33,902 trials.
- **6,000 trials** were divided into the test set.
- Remaining **27,902 trials** were divided equally into five parts.
- In each fold, four parts were used for training and one for validation.

Thus the reported structure is a five-fold cross-validation process applied to the 27,902 non-test trials, with a fixed 6,000-trial test set.

The paper also states that after five training epochs, the validation set goes through all training data. This sentence is unclear and is not sufficient to reconstruct an exact training/evaluation schedule.

### Participant independence

**Not specified in the paper.**

The paper does not state whether the 6,000 test trials and the five validation/training partitions were created at the participant level or at the trial level.

Therefore, from the paper alone, it cannot be determined whether trials from the same participant could occur in different partitions.

### Training hyperparameters

| Setting | Reported value |
|---|---|
| Epochs | 300 |
| Batch size | 256 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Early stopping | Yes |
| Early stopping patience | 30 epochs without validation-loss improvement |
| Reduce learning rate | Yes |
| LR reduction patience | 20 epochs without validation-loss improvement |
| LR reduction | Learning rate halved |
| Model checkpoint | Yes |
| Hardware | Windows 10 Enterprise; 4 × NVIDIA TITAN Xp GPUs, 12 GB each |
| Programming language | Python 3.8 |
| Framework | TensorFlow-GPU 2.50 as written in the paper |

The paper does not provide additional details such as Adam beta parameters, random seeds, exact checkpoint-selection rule beyond monitoring the training process, or exact ReduceLR implementation.

### Other hyperparameters

The paper provides the main architectural values:

- 6 Transformer blocks
- 6 attention heads
- Dense layer of 64 units
- ReLU in FFN and Dense layer
- GMP
- Dropout

It does not provide all low-level implementation settings needed for exact replication.

### Training-time observations

Table 2 reports one-epoch times:

| Model | Parameters | Time per epoch |
|---|---:|---:|
| EEGNet | 2,659 | 6 s |
| ShallowConvNet | 23,923 | 8 s |
| DeepConvNet | 180,278 | 9 s |
| EEG-Transformer | 528,423 | 30 s |

However, Table 4 reports 45 s per epoch for the six-block EEG-Transformer configuration. The paper does not reconcile the difference.

---

## 9. Baseline Models

The paper compares EEG-Transformer against three CNN-based EEG classification models.

### EEGNet

The paper describes EEGNet as having three convolutional layers:

1. 2D convolution for frequency filtering
2. Deep convolution for spatial filtering of specific frequencies
3. Separable convolution for temporal summarization and feature-map mixing

**Purpose of comparison:** provide a commonly used compact CNN EEG classifier.

### ShallowConvNet

The paper describes it as using a large convolution kernel and two convolutional layers for temporal convolution and spatial filtering, with square nonlinearity, average pooling, and logarithmic activation.

**Purpose of comparison:** represent a shallow convolutional EEG architecture designed to learn time/frequency-band-power structure.

### DeepConvNet

The paper describes it as structurally related to ShallowConvNet but using four convolutional layers and deeper feature extraction.

**Purpose of comparison:** represent a deeper CNN architecture for EEG classification.

The paper compares these models on parameters, training time, accuracy, confusion matrices, ROC/AUC and convergence behavior.

---

## 10. Evaluation Metrics

### Accuracy

The paper defines:

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

Accuracy measures the proportion of correctly classified observations.

The paper uses classification accuracy as one of its principal comparison measures.

**Audit note:** the paper's task is three-class classification, but the formula shown is the standard binary TP/TN/FP/FN form. The paper does not explicitly explain the multiclass averaging/calculation procedure used to produce the reported three-class accuracy.

### Precision

`Precision = TP / (TP + FP)`

Measures the proportion of predicted positive cases that are actually positive.

The paper includes precision in the Transformer optimization experiments.

### Recall

`Recall = TP / (TP + FN)`

Measures the proportion of actual positive cases correctly identified.

The paper includes recall in the Transformer optimization experiments.

### F1-score

`F1 = 2 × Precision × Recall / (Recall + Precision)`

Combines precision and recall.

The paper uses F1 in the Transformer architecture optimization experiments.

### ROC

The paper defines a Receiver Operating Characteristic curve using true-positive rate (TPR) and false-positive rate (FPR) as the threshold varies.

### AUC

The paper defines AUC as the area under the ROC curve, ranging from 0 to 1.

The authors use ROC/AUC to evaluate discriminative performance and state that they are useful where positive/negative sample distributions are unbalanced.

### Statistical test

The paper uses **ANOVA** to quantitatively compare the four model groups' accuracy results.

Reported comparisons include:

- EEG-Transformer vs EEGNet: p < 0.05
- EEG-Transformer vs DeepConvNet: p < 0.05
- EEG-Transformer vs ShallowConvNet: p > 0.05
- EEGNet vs ShallowConvNet: p > 0.05

The paper does not provide the exact ANOVA design, degrees of freedom, assumptions, or multiple-comparison correction details.

---

## 11. Main Results

### Model accuracy

The paper reports:

| Model | Accuracy reported |
|---|---:|
| EEG-Transformer | 95.58% ± 0.52% |
| EEGNet | 93.66% ± 0.79% |
| ShallowConvNet | 94.46% ± 2.16% |
| DeepConvNet | 75.08% ± 17.83% |

The paper reports statistically significant differences for EEG-Transformer vs EEGNet and EEG-Transformer vs DeepConvNet (p < 0.05), while the comparison with ShallowConvNet was reported as p > 0.05.

### Class-wise confusion-matrix results

For EEG-Transformer, the paper reports classification accuracy for the three groups as:

- HC: 0.96
- ADD: 0.97
- ADHD: 0.96

It describes an average of approximately **96.3%** in this confusion-matrix analysis.

The paper does not explain why this average differs from the 95.58% ± 0.52% result reported in the main model-comparison experiment.

### AUC

For EEG-Transformer:

- Overall average AUC: **0.9926**
- HC: **0.9932**
- ADD: **0.9922**
- ADHD: **0.9923**

For DeepConvNet:

- Average AUC: **0.6290**
- HC: **0.6990**
- ADD: **0.5788**
- ADHD: **0.6091**

### Convergence observations

According to the paper:

- EEGNet converged at approximately 100 epochs.
- DeepConvNet did not converge in the shown training behavior and had large validation fluctuations.
- ShallowConvNet had not fully converged and showed some instability.
- EEG-Transformer was reported to converge within approximately 30 epochs, with a steadily increasing curve and validation accuracy close to training accuracy.

These are observations from the paper's reported training curves.

### Training complexity

The paper reports EEG-Transformer as having more parameters and longer single-epoch training time than the three CNN baselines in Table 2.

The paper explicitly states that the Transformer is more complex and that its greater parameter count increases training time.

---

## 12. Ablation Study

The paper investigates:

- GMP vs GAP
- Position embedding
- Multi-Head Attention
- Feed Forward Network
- Add & Norm

The baseline/control row is the standard EEG-Transformer configuration with GMP.

### Ablation table

| Position embedding | Multi-Head attention | FFN | Add & Norm | GMP | GAP | Accuracy |
|---|---|---|---|---|---|---:|
| ✓ | ✓ | ✓ | ✓ | ✓ | × | 95.58 ± 0.52% |
| ✓ | ✓ | ✓ | ✓ | × | ✓ | 84.08 ± 3.31% |
| × | ✓ | ✓ | ✓ | ✓ | × | 95.82 ± 0.71% |
| ✓ | × | ✓ | ✓ | ✓ | × | 39.12 ± 14.88%* |
| ✓ | ✓ | × | ✓ | ✓ | × | 82.39 ± 32.47% |
| ✓ | ✓ | ✓ | × | ✓ | × | 16.23 ± 6.62%* |

\* The prose gives different numerical values for the Multi-Head Attention and Add & Norm removals; see the audit below.

### GMP vs GAP

- GMP: 95.58 ± 0.52%
- GAP: 84.08 ± 3.31%

The authors conclude that GMP performs better and is more stable in their experiment.

Their explanation is that GMP retains the most significant feature/fluctuation information, whereas GAP averages features and may lose detailed information.

### Position embedding removed

Accuracy increased slightly:

- With position embedding: 95.58 ± 0.52%
- Without position embedding: 95.82 ± 0.71%

The authors state that position embedding therefore did not significantly improve classification performance.

They nevertheless retain it because removing it loses location information and slightly increases error/instability.

### Multi-Head Attention removed

The paper reports a major accuracy reduction.

Table 3:
- 39.12 ± 14.88%

Prose:
- 39.20%
- standard deviation reported as 0.55%

The authors conclude that Multi-Head Attention is a core/indispensable feature-extraction module for their EEG-Transformer.

### Feed Forward removed

Accuracy:

- 82.39 ± 32.47%

The authors state that removing FFN causes approximately a 10% reduction relative to the control and substantial instability.

They associate the FFN's nonlinear ReLU transformation with nonlinear fitting and training stability.

### Add & Norm removed

Table 3:
- 16.23 ± 6.62%

But the prose states:
- 38.38 ± 0.81%

The authors describe Add & Norm as essential and say that without it the model loses high classification performance.

This is one of the strongest numerical inconsistencies in the paper's ablation section. The paper does not clarify whether the table or prose value is the intended result.

### What the ablation experiments actually tell us

Within the paper's own experimental setup:

- Replacing GMP with GAP produced a substantial decrease in reported accuracy.
- Removing positional embedding did not decrease mean accuracy; it slightly increased it.
- Removing Multi-Head Attention caused a very large decrease.
- Removing FFN reduced accuracy and greatly increased variation.
- Removing Add & Norm caused a very large reported decrease.

The authors interpret these findings as evidence that Multi-Head Attention, FFN, Add & Norm and GMP play important roles in their architecture, while positional embedding contributes more to location information/stability than to mean classification accuracy.

---

## 13. Architecture Optimization

The paper separately studies:

1. Number of Transformer blocks
2. Number of attention heads

The evaluation considers classification metrics together with parameter count and training time.

### Transformer blocks

Attention heads fixed at 6.

| Blocks | Parameters | Time/epoch | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 171,523 | 18 s | 84.98 ± 0.76% | 85.29 ± 0.76% | 86.47 ± 0.70% | 84.95 ± 0.81% |
| 4 | 339,203 | 30 s | 95.80 ± 0.89% | 95.60 ± 0.72% | 95.60 ± 0.89% | 95.59 ± 0.80% |
| 6 | 506,883 | 45 s | 95.58 ± 0.53% | 95.54 ± 0.53% | 95.17 ± 0.79% | 95.56 ± 0.54% |
| 8 | 674,563 | 58 s | 95.85 ± 0.70% | 95.82 ± 0.73% | 95.81 ± 0.70% | 95.83 ± 0.72% |

### Authors' interpretation

- 2 blocks produced lower performance.
- 4 blocks substantially increased performance.
- 6 blocks produced similar performance.
- 8 blocks produced only a small performance change while increasing parameters and computation.
- The authors discuss a trade-off between performance, error/stability and computational cost.

The paper states that 8-block training took substantial resources and reports 24.2 hours for 50 cross-validations at 300 epochs.

### Attention heads

Transformer blocks fixed at 6.

| Heads | Parameters* | Time/epoch | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 201,759** | 22 s | 93.28 ± 1.53% | 93.32 ± 1.47% | 93.44 ± 1.27% | 93.28 ± 1.51% |
| 4 | 354,339 | 31 s | 95.27 ± 0.28% | 95.25 ± 0.28% | 95.26 ± 0.24% | 95.25 ± 0.27% |
| 6 | 506,883 | 45 s | 95.58 ± 0.54% | 95.54 ± 0.53% | 95.17 ± 0.79% | 96.05 ± 0.54%*** |
| 8 | 659,427 | 54 s | 94.96 ± 2.42% | 94.75 ± 2.22% | 94.68 ± 2.31% | 96.05 ± 2.27%*** |

\* Table 4 is the source for the parameter counts.  
\** The prose says 20,179, while Table 4 gives 201,759.  
\*** The reported F1 values should be treated as written; the paper does not explain them.

### Authors' interpretation

The paper states that increasing the number of attention heads increases model complexity, parameter count and training cost.

It reports:

- 2 heads: approximately 93% performance.
- 4 heads: improved performance.
- 6 heads: slightly higher performance than 4 in the authors' interpretation.
- 8 heads: longer training and a decrease in performance relative to 6 heads.

The paper attributes the decline at 8 heads to overfitting and excess model complexity.

### Trade-offs reported

The authors describe:

**More Transformer blocks**
- More parameters
- More training time
- Potentially higher classification performance
- Diminishing performance gains at higher depth
- Greater computational cost

**More attention heads**
- More parameters
- More training time
- Initial performance improvement
- Performance reduction at 8 heads in the reported experiment

The authors ultimately state that, for their ADHD dataset, the 6-block/6-head architecture provides high classification accuracy and stability with moderate training time.

This is a report of the authors' conclusion, not an independent recommendation.

---

## 14. Authors' Claimed Contributions

The paper explicitly lists three main contributions.

### 1. Proposed methodology

The authors propose an **EEG-Transformer EEG signal classification model** using attention to extract EEG signal features and make use of EEG spatiotemporal information.

### 2. Experimental / ablation contribution

They perform ablation experiments to analyze the internal structure of EEG-Transformer and investigate the functions and effects of its modules.

### 3. Optimization contribution

They adjust Transformer block and attention-head structure and use the experiments to identify an architecture they regard as appropriate for the dataset.

The authors also state that the resulting structure provides a theoretical basis for adjustment/modification of the model for other EEG data and can serve as a basic model for EEG classification.

### What the paper does not claim as an explicit contribution

The paper does not present a new EEG dataset.

It does not claim that it has established a clinical diagnostic standard.

It presents the model as an **auxiliary tool** for clinical diagnosis rather than a replacement for clinical diagnosis.

---

## 15. Limitations

### Author-stated limitations

The paper explicitly acknowledges the limited participant/sample size near the end of the Discussion.

It states that:

- The sample size is small.
- Although dividing the data creates 33,902 experimental data records, the underlying sample size has not changed.
- The current data application is feasible for theoretical research.
- Further increases in sample size would be necessary before clinical application.
- The model would need continued training and greater visibility/validation for clinical diagnosis.

### Limitations We Notice From the Paper

The following are **observations/questions arising from the paper**, not claims that the paper's results are invalid.

1. **Participant-level independence is not established.**  
   The paper reports 144 children but 33,902 trials. It does not state whether train/validation/test splitting was performed by participant or by trial.

2. **Potential dependence between trials is unresolved.**  
   If trials from one child can occur in different partitions, the effective independence of the evaluation samples cannot be determined from the paper.

3. **Preprocessing is insufficiently specified for exact reproduction.**  
   ICA is mentioned, but the actual artifact-removal procedure is not described in sufficient detail.

4. **The input dimension is not fully reconciled.**  
   The paper reports 1.5 s at 256 Hz and 56 channels, but Table 1 uses a sequence dimension of 385.

5. **Parameter-count inconsistencies remain unresolved.**  
   Several parameter totals differ across sections/tables.

6. **Several reported ablation values conflict.**  
   In particular, Multi-Head Attention and Add & Norm have conflicting values between Table 3 and the explanatory text.

7. **Statistical-analysis details are incomplete.**  
   ANOVA is reported, but the exact design, assumptions and post-hoc/multiple-comparison procedures are not given.

8. **The paper does not fully specify class-wise metric calculation.**  
   This is relevant because the task has three classes while some metric equations are written in binary form.

9. **Exact randomization/reproducibility information is not provided.**  
   The paper does not state random seeds or all deterministic/non-deterministic settings.

10. **The reported training-time measurements are not fully standardized across sections.**  
    Table 2 gives 30 s/epoch for EEG-Transformer, whereas Table 4 gives 45 s/epoch for the six-block configuration.

These observations identify areas requiring clarification or replication; they do not by themselves establish that the reported results are wrong.

---

## 16. Reproducibility Audit

| Item | Provided? | Details / Missing Information |
|---|---|---|
| Dataset availability | **Yes** | Openly available according to the paper; OSF link `https://osf.io/6594x/`. |
| Dataset identification | **Partly** | Technical University of Dresden dataset is identified, but the paper does not provide a detailed dataset version/file manifest. |
| Participant counts | **Yes, but inconsistent later** | 144 participants in Methods; 300 is stated later without explanation. |
| Trial counts | **Yes** | 33,902 total; 10,742 ADHD; 13,031 ADD; 10,129 HC. |
| EEG channels | **Yes** | 56. |
| Sampling frequency | **Yes** | 256 Hz. |
| Trial duration | **Yes** | 1.5 s. |
| Raw input shape | **Partly** | 2D channel × sampling-point representation; exact model tensor is reported as 385 × 56, which is not reconciled with 384 points implied by duration × sampling rate. |
| ICA preprocessing | **Partly** | ICA/noise removal stated; exact implementation and component-selection procedure not stated. |
| Filtering | **No** | Not specified. |
| Re-referencing | **No** | Not specified. |
| Artifact rejection details | **No** | Not specified beyond ICA/noise removal. |
| Trial segmentation | **Partly** | 1.5 s acquisition/trial reported; exact segmentation process and overlap not specified. |
| Data splitting | **Partly** | 6,000 test trials; 27,902 remaining trials in five-fold train/validation procedure. Participant-level independence is not stated. |
| Participant-independent split | **No** | Cannot be determined from the paper. |
| Architecture | **Mostly** | Main components and several dimensions/parameter counts are given, but some dimensions and parameter totals conflict. |
| Transformer blocks | **Yes** | Base model: 6. Optimization tests 2/4/6/8. |
| Attention heads | **Yes** | Base model: 6. Optimization tests 2/4/6/8. |
| Pooling | **Yes** | GMP in base model; GAP tested in ablation. |
| Dense layer | **Yes** | 64 units, ReLU. |
| Dropout rate | **No** | Dropout is reported, but the rate is not stated in the architecture table. |
| Epochs | **Yes** | 300. |
| Batch size | **Yes** | 256. |
| Optimizer | **Yes** | Adam. |
| Learning rate | **Yes** | 0.001 initially. |
| LR scheduling | **Yes** | Halve LR after 20 epochs without validation-loss improvement. |
| Early stopping | **Yes** | Stop after 30 epochs without validation-loss improvement. |
| Checkpointing | **Yes** | Model checkpointing used. |
| Random seed | **No** | Not stated. |
| Software | **Partly** | Python 3.8 and TensorFlow-GPU 2.50 are stated. Exact environment/dependencies are not provided. |
| Hardware | **Yes** | Windows 10 Enterprise; 4 × NVIDIA TITAN Xp, 12 GB each. |
| Evaluation procedure | **Partly** | Accuracy/precision/recall/F1/ROC/AUC and ANOVA reported, but detailed aggregation/statistical procedures are incomplete. |
| Statistical test details | **Partly** | ANOVA and p-value comparisons are reported; exact statistical design and corrections are not. |
| Exact implementation/code | **No** | No code is provided in the paper itself. |

### Reproducibility conclusion

The paper provides enough information to understand and approximately reconstruct the proposed architecture and experimental concept, but **not enough information to guarantee an exact reproduction**.

The main unresolved reproduction issues are:

- participant-vs-trial splitting;
- exact preprocessing;
- exact input shape;
- conflicting parameter counts;
- conflicting ablation values;
- unclear statistical procedure;
- unspecified dropout rate;
- missing random seed;
- incomplete trial segmentation details.

---

## 17. Unresolved Questions

The following questions genuinely remain after reading the paper:

1. **Was the train/test/validation split participant-independent?**
2. **Could trials from the same participant occur in different partitions?**
3. **Exactly how were the 33,902 trials generated?**
4. **Were trials overlapping or non-overlapping?**
5. **Why does 1.5 s at 256 Hz correspond to an architecture dimension of 385 rather than 384?**
6. **What exactly does one model input represent at the tensor level?**
7. **What is the exact preprocessing pipeline after ICA?**
8. **Which ICA components were considered noise and removed?**
9. **Were any filters, re-referencing, normalization or bad-channel procedures used?**
10. **Was preprocessing performed before or after splitting the dataset?**
11. **How were the five cross-validation folds generated?**
12. **What exactly does the sentence about the validation set going through all training data after five epochs mean?**
13. **What is the correct total parameter count: 506,883 or 528,423?**
14. **Why does Table 4 report 201,759 parameters for two attention heads while the prose says 20,179?**
15. **Why does Table 2 report 30 s/epoch for EEG-Transformer while Table 4 reports 45 s/epoch for the six-block configuration?**
16. **Which Add & Norm ablation result is correct: 16.23 ± 6.62% or 38.38 ± 0.81%?**
17. **Which Multi-Head Attention ablation variation is correct?**
18. **Why does the six-head experiment report F1 = 96.05 ± 0.54% while the other reported metrics are around 95%?**
19. **What exact multiclass averaging method was used for precision, recall and F1?**
20. **How exactly was ANOVA applied to the model results?**
21. **Were post-hoc comparisons or multiple-comparison corrections performed?**
22. **What random seeds were used?**
23. **What is the exact dropout rate?**
24. **Why does the Discussion state a sample size of 300 when the Methods section reports 144 children?**
25. **What does the incomplete 5,580-trial statement in the training section mean?**
26. **Which reported accuracy should be treated as the main final EEG-Transformer result: 95.58 ± 0.52%, 95.85% in the abstract, or the approximately 96.3% class-wise confusion-matrix average?**
27. **Are the reported performance measures calculated on the fixed 6,000-trial test set, validation folds, or another aggregation?**
28. **Are the reported means and standard deviations across folds, repeated experiments, participants, or something else?**
29. **Does the claimed clinical-assistance conclusion extend beyond this dataset and experimental setup? The paper does not provide a clinical prospective validation.**
30. **Does the paper actually demonstrate transfer learning?** The paper describes the model as a basis for transferable learning, but the reported experiments do not themselves constitute a transfer-learning experiment.

---

## 18. Research Starting Point

### What This Paper Establishes

Based strictly on the reported experiments, the paper establishes that:

- The authors constructed an **EEG-Transformer** based on the Transformer encoder concept for three-class EEG classification of ADHD, ADD and healthy controls.
- Their model removes the traditional Transformer decoder and uses positional embedding, Transformer blocks, GMP and a classification head.
- In the reported experiments, the EEG-Transformer obtained high classification metrics, including a reported average accuracy of **95.58 ± 0.52%** in the main model comparison and an average AUC of **0.9926**.
- The paper reports comparisons with EEGNet, ShallowConvNet and DeepConvNet.
- The paper reports ablation experiments examining GMP/GAP, positional embedding, Multi-Head Attention, FFN and Add & Norm.
- The paper reports architecture experiments varying Transformer-block count and attention-head count.
- The authors report that their six-block/six-head configuration provided high performance and a balance they regarded as appropriate for this dataset.
- The authors explicitly acknowledge that the participant/sample size is limited for clinical application and state that larger samples are needed.

### What This Paper Does Not Establish

The experiments do **not** establish, from the information reported in the paper alone:

- that the model has been validated prospectively in clinical practice;
- that the model generalizes to an independent external EEG dataset;
- that the model is clinically diagnostic rather than an auxiliary classification tool;
- that the reported performance is participant-independent;
- that no participant's trials are shared between training and evaluation partitions;
- that the reported results are fully reproducible from the published methodological description;
- that the reported performance is free from effects caused by preprocessing or splitting choices;
- that the EEG-Transformer will retain the same performance on a new population, acquisition system or institution;
- that the architecture is universally optimal for EEG classification;
- that the proposed model actually performs transfer learning, since no transfer-learning experiment is reported;
- that reproducing the reported accuracy alone constitutes a research contribution.

### What We Need to Investigate Next

Before reproducing, modifying or extending the work, the following questions need to be resolved from the paper/dataset/implementation evidence:

1. Establish exactly how the 144 participants relate to the 33,902 trials.
2. Determine whether the original data and trial partitions permit participant-independent evaluation.
3. Reconstruct the exact EEG input shape and explain the 384-versus-385 discrepancy.
4. Reconstruct the preprocessing pipeline as far as the source dataset permits, without inventing steps absent from the paper.
5. Resolve the paper's conflicting parameter counts and ablation results.
6. Determine precisely how the five-fold procedure and fixed test set were implemented.
7. Determine how multiclass metrics were aggregated.
8. Determine what the reported mean ± standard deviation represents.
9. Determine the exact statistical testing procedure behind the ANOVA results.
10. Reproduce the reported baseline and EEG-Transformer experiments before interpreting differences between them.
11. Treat the authors' clinical and transfer-learning statements as claims requiring evidence beyond the reported experiment rather than as already-established outcomes.

---

## Audit Summary

The paper provides a clear high-level concept and enough architectural information to understand the proposed EEG-Transformer. It also provides substantial experimental results, including baseline comparisons, ablations and architecture-optimization experiments.

However, several details necessary for an exact scientific reconstruction are either missing or internally inconsistent. The most consequential unresolved issue is the relationship between **participants and trials**, because the paper reports 144 children but evaluates tens of thousands of trials without specifying whether partitioning was participant-independent. Other important unresolved issues include preprocessing details, input dimensionality, parameter counts, several ablation values, the statistical-analysis procedure and the meaning of reported means/standard deviations.

Accordingly, the paper is a useful **base paper and methodological reference**, but its published description alone is not sufficient to establish a fully reproducible implementation or to determine exactly how independent the reported evaluation samples are.

