# Sprint 1 — Research Landscape & Gap Discovery Report

## 1. Scope
The scope of this investigation is to map the research landscape concerning ADHD classification using EEG signals, with a specific focus on the Dresden/TU Dresden dataset (144 children, hosted on OSF). The objective is to identify existing methodologies, understand the lineage of the dataset, expose methodological patterns (and flaws), and discover technically meaningful open questions without prematurely deciding on a final research direction. 

## 2. Search Strategy
The literature search strategy prioritized:
1.  **Base Paper Investigation**: Analyzed the base paper (*He et al., 2023, "Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model"*) to extract methodologies, limitations, and dataset references.
2.  **Dataset Lineage**: Investigated the OSF dataset (`osf.io/6594x`) and its original source paper, identifying *Vahid et al. (2019)* as a seminal work using this dataset.
3.  **Literature Expansion**: Searched for additional papers applying Deep Learning (CNNs, Transformers) to this specific cohort (144 children; 44 HC, 52 ADD, 48 ADHD) and broader EEG-ADHD classification works to identify recurring methodological patterns and limitations.

## 3. Relevant Literature
*   **Vahid et al. (2019)**: *Deep Learning Based on Event-Related EEG Differentiates Children with ADHD from Healthy Controls*. Applied EEGNet to the dataset during a time-estimation task, achieving ~83% accuracy in distinguishing ADHD from Healthy Controls, but noted difficulty separating ADHD subtypes.
*   **He et al. (2023) [Base Paper]**: *Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model*. Proposed an EEG-Transformer model with GMP and claimed 95.58% accuracy. Used 1.5s epochs from the 144 subjects.
*   **Other Applications**: Various studies have utilized this OSF dataset to test architectures like CNN-LSTM models, multiscale selective channel attention networks (SCANet), and graph-theoretical approaches.

## 4. Dataset Lineage
*   **Origin**: Collected at the Department of Child and Adolescent Psychiatry, Technical University of Dresden (often associated with Prof. Beste).
*   **Cohort**: 144 children (44 Healthy Controls, 52 ADD/ADHD-Inattentive, 48 ADHD-Combined).
*   **Task**: Data recorded during a time-estimation task (pressing a button after 1200 ms).
*   **Availability**: Publicly hosted on the Open Science Framework (OSF) at `https://osf.io/6594x/`.
*   **Reuse Pattern**: Frequently used as a benchmark for EEG deep learning models due to its balanced group sizes and clinical relevance. 

## 5. Comparison Matrix

| Paper | Year | Dataset | Subjects | Groups | Channels | Trial/Epoch | Preprocessing | Target | Representation | Model | Split Strategy | Metrics | Main Result | Limitations | Code/Data | Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Vahid et al. | 2019 | Dresden (OSF) | 144 | HC (44), ADD (52), ADHD-C (48) | Not reported | Event-related (Time-estimation) | Not explicitly detailed | ADHD vs HC vs ADD | Raw/Processed EEG | EEGNet | Subject-wise (assumed, typical for original works) | Accuracy | ~83% acc for ADHD vs HC | Struggles separating ADD vs ADHD-C | Data (OSF) | Seminal paper for the dataset |
| He et al. | 2023 | Dresden (OSF) | 144 | HC (44), ADD (52), ADHD-C (48) | 56 | 1.5s epochs (33,902 trials) | ICA (EOG/ECG removal) | 3-class (HC, ADD, ADHD) | Raw Tensor (385x56) | EEG-Transformer (6 blocks, GMP) | **Trial-wise (Random)** | Acc, AUC | 95.58% Acc | Trial-level leakage; Black-box | Data (OSF) | Base reference |

*(Note: "Not reported" is used where exact parameters require full paper access not available in the current search.)*

## 6. Methodological Patterns
*   **Architecture Trend**: Shift from standard ML to CNNs (like EEGNet) and recently to Transformers (capturing long-range dependencies).
*   **Data Representation**: Most deep learning models use raw or minimally processed time-series channels rather than extracting clinical biomarkers (e.g., Theta/Beta Ratio) before modeling.
*   **Spatial Neglect**: Pure sequence models often treat the 56 EEG channels as unordered sequences, neglecting 3D spatial topography.
*   **Evaluation Leakage**: A pervasive pattern in high-accuracy papers (like the base paper) is using trial-wise splitting instead of subject-wise splitting, artificially inflating accuracy by learning subject-specific identities rather than disease-generalizable features.

## 7. Known Limitations
**A. Explicitly Stated by Authors**
*   Vahid et al.: Difficulty in distinguishing between ADHD subtypes (Inattentive vs. Combined).
*   He et al.: Models often act as black boxes lacking clinical explainability.

**B. Identified by Later Research / Methodological Critique**
*   **Trial-Level Leakage**: Papers splitting by trial across the dataset allow the model to memorize subject brain signatures across train and test sets.
*   **Lack of Spatial Inductive Bias**: Transformers used on raw EEG discard physical electrode locations.

**C. Methodological Concerns (Supported by Evidence)**
*   Extracting micro-spikes via self-attention is computationally expensive ($O(T^2)$) compared to local convolutions.
*   Ignoring traditional clinical biomarkers (e.g., spectral bands) removes valuable inductive biases.

## 8. Candidate Open Questions
1.  **Leakage-Free Transformer Generalization**: How does the performance of Transformer-based architectures degrade when transitioning from trial-wise splitting to strict subject-wise (`GroupKFold`) cross-validation on the Dresden dataset? 
    *   *Why it matters*: Validates if Transformers actually learn ADHD biomarkers or just patient identities.
2.  **Hybrid Spatial-Spectral Integration**: Does explicitly combining structural spatial biases (Convolutions/Topography) with spectral clinical features (Theta/Beta ratio) outperform pure sequence-based Transformers?
    *   *Why it matters*: Addresses the spatial neglect of pure Transformers and reintroduces clinical domain knowledge.
3.  **Subtype Differentiation**: Can advanced attention mechanisms or spatial-temporal models successfully differentiate ADHD-Inattentive (ADD) from ADHD-Combined, a task where traditional models (Vahid et al.) struggle?

## 9. Dataset Compatibility
*   **Leakage-Free Generalization**: *Compatible*. The dataset has subject identifiers, allowing for `GroupKFold` splitting.
*   **Hybrid Spatial-Spectral Integration**: *Compatible*. The dataset provides 56 channels, allowing for spatial mapping, and raw/preprocessed signals to extract spectral bands (Theta/Beta).
*   **Subtype Differentiation**: *Compatible*. The dataset explicitly contains 3 distinct groups (HC, ADD, ADHD-C).

## 10. Unresolved Questions
*   What is the true baseline accuracy of state-of-the-art models on this dataset when evaluated with strict subject-wise splitting?
*   Are there specific channel clusters (e.g., frontal, parietal) that drive the attention mechanism's decisions in true generalizing models?
*   How does the event-related nature of the dataset (time-estimation task) affect resting-state assumptions typically made by models applied to it?

## 11. Recommended Next Investigations
1.  **Re-evaluate Baselines**: Re-run the baseline models (EEGNet, ShallowConvNet) and the base paper's EEG-Transformer using strict Subject-Wise splitting (`StratifiedGroupKFold`) to establish a true benchmark devoid of leakage.
2.  **Spatial Topography Analysis**: Investigate the physical layout of the 56 channels in the dataset to determine how a spatial convolution layer or graph network could be integrated.
3.  **Clinical Biomarker Extraction**: Analyze the dataset to extract the Theta/Beta Ratio and spectral bands to see how they correlate with the provided labels before building complex models.
