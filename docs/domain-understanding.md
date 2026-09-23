# Domain Understanding and Terminology Audit

> **Purpose:** Precise, project-specific terminology for the ADHD-EEG research project. This is NOT a generic EEG glossary. Every definition here is grounded in (1) the base paper, (2) the dataset investigation, and (3) directly observed dataset findings. General domain knowledge is used only to help explain a term — and is clearly marked when used.
>
> **Primary sources (in priority order):**
> 1. Yuchao He et al., *Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model*, Journal of Neural Engineering, 20 (2023) 056013. (referred to as "the paper" throughout)
> 2. `docs/dataset-investigation.md` (referred to as "dataset investigation")
> 3. `docs/base-paper-analysis.md` (referred to as "paper analysis")
> 4. Directly observed dataset findings from investigation scripts
>
> **Evidence labels used throughout this document:**
> - **PAPER-STATED** — the paper directly says this
> - **DIRECTLY ESTABLISHED** — confirmed by direct observation of dataset files
> - **STRONGLY SUPPORTED** — well-supported by convergent evidence but not directly documented
> - **GENERAL DOMAIN KNOWLEDGE** — standard domain meaning, not necessarily unique to this project
> - **UNRESOLVED** — cannot be established from the available sources

---

## 1. Big-Picture Domain Map

### Simple meaning

Here is the chain of what is happening in this research, explained step by step:

```
A child sits in a clinical setting.
↓
Their brain produces tiny electrical signals as they think and respond to stimuli.
↓
Sensors (electrodes) placed on their scalp pick up those signals.
↓
The device records those signals through 56 measurement points (channels).
↓
The recording is made at high speed: 256 measurements per second.
↓
The recording is divided into short segments (1.5 seconds each).
   These short segments are called trials in this paper.
↓
Each segment becomes one input to the model: a matrix of channels × time measurements.
↓
This is done for 144 children: 44 healthy, 52 with ADD, 48 with ADHD.
↓
This produces a total of 33,902 such segments across all children.
↓
Each segment is associated with metadata: which child, which group, which stimulus category.
↓
A machine-learning model reads these segments and tries to predict the group (HC / ADD / ADHD).
```

That is the complete research loop: brain → electricity → sensors → data → segments → labels → model → prediction.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** Electroencephalography (EEG) measures the summed electrical potential changes at the scalp surface caused by synchronised postsynaptic activity of large populations of cortical neurons. These potentials are on the order of microvolts. Electrodes placed at standardised scalp positions record voltage as a function of time. The resulting multi-channel time-series is called an EEG recording.

In clinical and research settings, EEG is used to measure brain responses to events or stimuli. Recordings are often segmented into short fixed-length windows (epochs/trials) time-locked to specific events.

### Meaning in this research

**PAPER-STATED:** The paper uses EEG signals from children in three clinical groups (ADHD, ADD, HC) to build a three-class classification model called EEG-Transformer. The model takes a single 1.5-second EEG segment (trial) as input and predicts which group the child belongs to.

**DIRECTLY ESTABLISHED:** The dataset contains exactly 33,902 trials from 144 participants across three groups, stored in seven `.mat` files (`d1.mat` through `d7.mat`), with metadata in `y_stim.mat` and `Sub_name_stim.mat`.

---

## 2. ADHD / ADD / Healthy Control

### 2.1 ADHD

#### Simple meaning

ADHD stands for Attention Deficit Hyperactivity Disorder. It is a neurodevelopmental condition in children (and adults) that affects attention, impulse control, and activity levels.

#### Technical meaning

**PAPER-STATED:** The paper uses the term ADHD to refer to one of the three clinical groups in the dataset. Children in this group were diagnosed by psychiatrists and psychologists using standard clinical indicators, home and school interviews, questionnaires, and IQ and attention tests. The paper states that participants had no other severe or acute psychiatric comorbidities (such as autism, convulsions, or depressive episodes).

**PAPER-STATED:** The paper frames ADHD as a condition that "can seriously affect attention, cognitive processes, working memory, learning, study and life."

#### Meaning in this research

**DIRECTLY ESTABLISHED:** The ADHD group consists of 48 children. In `Sub_name_stim.mat`, this group corresponds to the third block (48 entries), with file paths containing the string `subtype2`. **PAPER-STATED:** This group contributed 10,742 trials to the dataset.

---

### 2.2 ADD

#### Simple meaning

ADD (Attention Deficit Disorder) is an older term, now generally considered a subtype of ADHD, referring to cases where inattention is the primary symptom without significant hyperactivity. In this project, ADD is used **as the paper uses it** — as a distinct clinical group separate from ADHD.

#### Technical meaning

**PAPER-STATED:** The paper treats ADD as a separate third class (alongside ADHD and HC) for classification purposes. The paper does not provide a detailed clinical definition separating ADD from ADHD beyond assigning them to different diagnostic groups.

> Do not reinterpret or modernise the paper's terminology. The paper uses ADD and ADHD as two separate categories. Whether this reflects current DSM-5 usage is not relevant for this project. The paper's grouping is preserved as-is.

#### Meaning in this research

**DIRECTLY ESTABLISHED:** The ADD group consists of 52 children. In `Sub_name_stim.mat`, this corresponds to the second block (52 entries), with file paths containing `subtype1`. **PAPER-STATED:** This group contributed 13,031 trials to the dataset — the largest group trial count.

---

### 2.3 Healthy Control (HC)

#### Simple meaning

A Healthy Control (HC) is a person included in the study who does NOT have the condition being studied. They are used as a reference/comparison group to contrast against the clinical groups.

#### Technical meaning

**PAPER-STATED:** The paper refers to the third group as "healthy controls" or "HC." These are children with no ADHD or ADD diagnosis.

#### Meaning in this research

**DIRECTLY ESTABLISHED:** The HC group consists of 44 children. In `Sub_name_stim.mat`, this corresponds to the first block (44 entries), with file paths containing `controls`. **PAPER-STATED:** This group contributed 10,129 trials.

---

### 2.4 Subject

#### Simple meaning

A subject is one individual person in the study. If you are one of the 144 children, you are one subject.

#### Technical meaning

**PAPER-STATED:** The paper reports 144 children total as participants. Each child belongs to one clinical group (HC, ADD, or ADHD).

#### Meaning in this research

**DIRECTLY ESTABLISHED:** `Sub_name_stim.mat` contains exactly 144 entries (44 + 52 + 48), one per subject. `y_stim` Row 0 contains subject index values (1 through 52 maximum) that **reset** when crossing from one group to the next — they are indices **within** each group, not globally unique subject IDs.

---

### 2.5 Participant

**PAPER-STATED:** The paper uses "participant" interchangeably with the concept of a child in the study. In this project, "subject" and "participant" refer to the same thing: one individual child.

---

### 2.6 Group

#### Simple meaning

A group is a set of subjects that share a clinical classification. In this study, there are three groups: HC, ADD, ADHD.

#### Meaning in this research

**DIRECTLY ESTABLISHED:** The three groups are physically encoded in `Sub_name_stim.mat` as three distinct blocks of 44, 52, and 48 entries. The row index in `y_stim` (Row 0) indexes a subject **within** their group, not across all groups.

**WARNING — Group (HC/ADD/ADHD) describes the clinical classification of a participant. This is NOT the same as the three stimulus categories in `y_stim` Rows 1–3. These are two completely different sets of three categories. Confusing them is a critical error. See Section 12 for details.**

---

### 2.7 Cohort

**GENERAL DOMAIN KNOWLEDGE:** In clinical research, a cohort is a group of participants sharing a defined characteristic. In this project, the three diagnostic groups (HC, ADD, ADHD) can be called cohorts in the general sense. However, the paper itself primarily uses "group" rather than "cohort."

---

## 3. EEG

### Simple meaning

EEG (Electroencephalography) is a method for measuring electrical activity in the brain from the outside of the head. Imagine placing many small sensors on your scalp — each sensor picks up the tiny electrical signals that your brain cells produce when they communicate with each other. The device records all these signals over time, producing a long stream of numbers for each sensor location.

EEG does **not** directly measure thoughts, intelligence, or emotions. It records electrical potential changes at the scalp surface. The interpretation of those signals is what researchers do.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** EEG records the summation of postsynaptic electrical potentials from large groups of neurons, primarily from the cortex. Signals are measured in microvolts (µV) and vary with frequency (typically 0.5–100 Hz in clinical settings). Multiple electrodes placed according to standardised systems (such as the 10–20 system) record simultaneously.

### Meaning in this research

**PAPER-STATED:** The paper uses EEG to measure brain activity in children with ADHD, ADD, and no diagnosis. The recorded EEG signals are the raw material for the classification task.

**PAPER-STATED:** The acquisition setup used 56 channels at 256 Hz with 1.5-second trial durations.

**DIRECTLY ESTABLISHED:** The actual EEG measurement values are stored in `d1.mat` through `d7.mat` as floating-point numbers in arrays of shape `(385, 56, 5000)` per file (for `d1`, confirmed directly).

---

## 4. Channel

### Simple meaning

An EEG channel is one measurement location on the scalp. Think of it as one sensor picking up the signal from one small area of the brain. If you have 56 channels, you have 56 sensors, each recording a separate signal over time. Each channel is named to indicate where on the scalp it sits — for example, `Fp1` is near the front-left of the scalp.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** In EEG, each channel corresponds to one electrode (or sometimes a combination) referenced to a common reference point. The electrode placement follows standardised naming conventions. For example, in the 10–20 system, letters indicate region (F = frontal, C = central, O = occipital, P = parietal, T = temporal) and numbers or letters indicate hemisphere (odd = left, even = right, z = midline).

#### Channel names from this dataset

**DIRECTLY ESTABLISHED:** `chan.mat` contains 60 channel entries. Example channel names observed:

| Name | Location meaning (general domain knowledge) |
|---|---|
| `Fp1` | Frontopolar, left |
| `Fp2` | Frontopolar, right |
| `Cz` | Central, midline |
| `FCz` | Fronto-central, midline |
| `Oz` | Occipital, midline |

### Meaning in this research

**DIRECTLY ESTABLISHED:**
- `chan.mat` lists **60** channels.
- `d1.mat` (and the other data files) contains **56** channels in the data array (second dimension of the array).

**UNRESOLVED:** The exact four channels present in `chan.mat` but absent from the data arrays are unknown. They could be reference electrodes, EOG channels, or channels removed during preprocessing — but this **cannot be established** from the available sources. Do not assume which four channels were removed.

---

## 5. Sample

### Simple meaning

A sample is one single measurement taken at one moment in time by one channel. If a device records 256 times per second, then in one second it produces 256 samples per channel. After 1.5 seconds, you would expect 384 samples per channel.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** Sampling is the process of measuring a continuous signal at discrete points in time. The sampling frequency (in Hz) determines how many such measurements are taken per second. According to the Nyquist theorem, the sampling rate must be at least twice the highest frequency of interest to avoid aliasing.

#### Sampling frequency in this research

**PAPER-STATED:** Sampling rate = **256 Hz**. This means 256 measurements (samples) are recorded per channel per second.

**PAPER-STATED:** Trial duration = **1.5 seconds**.

Expected from paper parameters: 256 samples/second × 1.5 seconds = **384 samples** per trial per channel.

**DIRECTLY ESTABLISHED:** The first dimension of `d1.mat` is **385**, not 384.

**UNRESOLVED:** The exact physical meaning of the 385th sample position is unknown. It could be an inclusive endpoint convention, a pre-stimulus baseline sample, a post-processing artefact, or something else. The paper does not explain this discrepancy, and the dataset files provide no documentation that resolves it. Do not discard or relabel the 385th position without evidence.

---

## 6. Time

### Simple meaning

EEG signals change over time. If you record someone's brain for 1.5 seconds, you get a sequence of numbers from each sensor — one number per sensor per moment in time. This sequence is called a time series. The first number is the signal at the start of the recording, the second number is the signal 1/256th of a second later, and so on.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** In EEG, the time dimension represents the sequence of sampling points within a recording or epoch. For a 1.5-second epoch at 256 Hz, the time axis has 384 points (or 385 as observed in this dataset). The spacing between consecutive time points is 1/256 seconds ≈ 3.9 milliseconds.

### Meaning in this research

**STRONGLY SUPPORTED:** In `d1.mat` with shape `(385, 56, 5000)`:
- The **first dimension (385)** corresponds to the time/sample dimension — 385 measurement positions within each trial.
- The **second dimension (56)** corresponds to channels.
- The **third dimension (5000)** corresponds to the number of trials/epochs in this file.

Note: This dimension interpretation is **STRONGLY SUPPORTED** (not directly documented). The evidence is: the model input shape `(None, 385, 56)` from the paper's architecture table, the paper's reported 56-channel count, and the total trial count consistent with `y_stim`. However, no OSF file directly states "axis 0 = time, axis 1 = channel, axis 2 = trial."

---

## 7. Trial

### Simple meaning

A trial is one short EEG recording segment. In an experiment, the child is shown a stimulus (something to look at, hear, or respond to), and while they respond, the EEG is recorded for 1.5 seconds. That 1.5-second block of data is one trial.

### Technical meaning

**PAPER-STATED:** The paper explicitly uses the word **"trial"** as its primary term for these 1.5-second EEG segments. Specifically:

- "The data were divided into **33,902 trials**."
- "**6,000 trials** were divided into the test set."
- "ADHD: **10,742 trials**; ADD: **13,031 trials**; HC: **10,129 trials**."
- "Acquisition time per experiment/trial: **1.5 s**."

The paper also uses the phrase **"experiment/trial"** together, suggesting the terms are being treated as overlapping in this context. The paper does not provide a separate technical definition distinguishing "trial" from "experiment."

### Meaning in this research

**PAPER-STATED + DIRECTLY ESTABLISHED:** A trial is one 1.5-second EEG segment, consisting of 56 channels × 385 time samples (as observed). There are 33,902 trials in total.

**PAPER-STATED + STRONGLY SUPPORTED:** The 33,902 trials are indexed by the third dimension of the data arrays (`d1.mat` shape: `(385, 56, 5000)` where 5000 is the trial count per file).

Note: The paper does not clearly distinguish "trial," "experiment," and "epoch" as entirely separate technical concepts. When the paper says "trial," it means one 1.5-second EEG data segment used as a model input. This project preserves that usage.

---

## 8. Epoch

### Simple meaning

"Epoch" is a common EEG analysis term that also refers to a short fixed-length segment of EEG data, usually time-locked to an event. In most EEG software, an epoch and a trial are essentially the same thing: a window of EEG data cut from a longer continuous recording.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** In EEG preprocessing, epoching (also called segmentation) is the process of extracting fixed-length windows from a continuous EEG recording, usually around events of interest. Each extracted window is called an epoch. Epochs typically extend from some time before the event to some time after it.

### Paper's use of "epoch"

**PAPER-STATED:** The paper does NOT use the word "epoch" to describe the 1.5-second EEG data segments. The paper's term is **"trial"** for these segments.

The word "epoch" does appear in the paper, but **exclusively to mean a training epoch** — one complete pass through the training data during model training (e.g., "training for 300 epochs").

**CRITICAL DISAMBIGUATION:**
- **In this paper:** "epoch" = one pass through the training data (a machine learning concept).
- **In general EEG research:** "epoch" = a short EEG data segment.
- **In this dataset investigation documentation:** "trial/epoch" is used as a combined term to refer to the 33,902 EEG data segments.

When reading this project's documentation, "epoch" in the context of the EEG data means the same as "trial." When "epoch" appears in the context of model training, it means a training iteration. Do not mix these two meanings.

### Meaning in this research

The dataset investigation document uses "trial/epoch" as a combined term for the 33,902 EEG segments. Going forward, **"trial"** is the paper's preferred term for EEG data segments and should be favoured when referring to the paper's terminology. The combined notation "trial/epoch" is used in dataset documentation to acknowledge both usages.

---

## 9. Recording

### Simple meaning

A recording, in the most general sense, is the act of capturing EEG data from one person in one session. It produces a long continuous stream of EEG data before any segmentation.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** An EEG recording is the continuous time-series data collected from one participant in one session. It can last from minutes to hours. From this continuous recording, individual trials/epochs are extracted by segmentation.

### Meaning in this research

**PAPER-STATED:** The paper states that "original data were preprocessed using independent component analysis (ICA) to remove noise" and then "divided into 33,902 trials." This implies a continuous-recording → segmentation pipeline.

**NOT ESTABLISHED:** The paper does not describe how many separate recording sessions each participant had, how long each continuous recording lasted, or whether the trials from one participant come from one session or multiple sessions. The term "recording" as a distinct unit is not explicitly used in the paper to describe something separable from "trial."

**UNRESOLVED:** The exact relationship between the concept of a "recording session" and the 33,902 trials cannot be established from the available sources. Do not assume that one recording = one trial, or that one recording = all of a participant's trials.

---

## 10. Subject / Participant

### Simple meaning

A subject or participant is one individual child in the study. The key thing to understand is:

**One subject contributes many trials.**

For example, if a child had 235 trials recorded from them, that is one subject but 235 data segments in the dataset.

### Technical meaning

**PAPER-STATED:** The paper reports 144 children total as participants. Each child belongs to one clinical group (HC, ADD, or ADHD).

### Meaning in this research

**DIRECTLY ESTABLISHED:** `Sub_name_stim.mat` contains 144 entries (44 + 52 + 48), one per subject. `y_stim` Row 0 encodes the subject index **within** each group (values from 1 up to 52 maximum), and these indices reset at group boundaries — they are not globally unique participant IDs.

#### Why one subject contributes many trials

**PAPER-STATED:** Each 1.5-second segment is one trial. A single participant's EEG session of several minutes could produce many such segments. The paper reports 33,902 total trials from 144 participants, meaning an average of approximately 235 trials per participant.

This is why **144 subjects ≠ 33,902 trials**. The 33,902 is the count of data segments, not the count of people.

**CAUTION — Critical for machine learning:** If a participant's trials appear in both training and test sets, the model may appear to perform well because it has effectively "seen" data from the same person during training. The paper does **not** establish whether its train/test split was performed at the participant level or the trial level. This is an unresolved concern documented in the paper analysis.

---

## 11. Stimulus

### Simple meaning

A stimulus is something that is shown to or experienced by the participant during the EEG experiment — for example, a sound, an image, or a task instruction. Different stimulus types may produce different brain responses, and the experiment records what the brain does in response to each.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** In EEG research, stimulus refers to the event that triggers or is presented during a trial. Common EEG paradigms include visual stimuli, auditory stimuli, and cognitive tasks. The type of stimulus is often an important variable that categorises trials.

### Meaning in this research

**DIRECTLY ESTABLISHED:** `y_stim.mat` shape is `(4, 33902)`. Rows 1, 2, and 3 contain binary (0/1) values. For every trial, exactly one of Rows 1–3 is 1, and the other two are 0. This encodes three mutually exclusive stimulus categories.

**DIRECTLY ESTABLISHED:** The counts of each stimulus category across all 33,902 trials:
- Row 1: **10,129** trials
- Row 2: **13,031** trials
- Row 3: **10,742** trials

**UNRESOLVED:** The physical meaning of these three stimulus categories is **unknown from the available sources**. We do not know what was shown or done to participants in each category, and we cannot assign labels like "target," "non-target," or "distractor" without explicit documentation from the paper or dataset.

**WARNING — The stimulus categories (Rows 1–3 in `y_stim`) are NOT the same as the participant groups (HC/ADD/ADHD). These are completely different classification axes. One describes what kind of stimulus a trial used; the other describes which clinical group the participant belongs to. See Section 12 for the critical distinction.**

---

## 12. Label

### Simple meaning

A label is the answer we want the model to predict. In a classification task, the label tells us the correct class for each example. This project has two different kinds of labels that must not be confused:

1. **Participant clinical group** — Is this person HC, ADD, or ADHD? (3 categories)
2. **Stimulus category** — Which of the three stimulus types was used in this trial? (3 categories)

### Technical meaning

**PAPER-STATED:** The paper's classification task is to predict the **clinical group** (HC, ADD, or ADHD) from the EEG trial data. This is the **target label** for model training and evaluation.

### The critical distinction

| | Participant Group | Stimulus Category |
|---|---|---|
| **What it describes** | The clinical diagnosis of the person | The type of stimulus used in the trial |
| **Values** | HC / ADD / ADHD | Category 1 / 2 / 3 (names unknown) |
| **Stored in** | `Sub_name_stim.mat` + `y_stim` Row 0 (indirectly) | `y_stim` Rows 1–3 |
| **Number of values** | 3 groups, 144 people total | 3 categories, 33,902 trials total |
| **Used as model target** | Yes — paper's classification task | Not established |

**Why confusing these two would be a serious machine-learning error:** If you accidentally use the stimulus category as the training label instead of the participant's clinical group, you would train the model to predict which stimulus type was used — not which group the person belongs to. The model would appear to learn something, but it would be learning the wrong thing entirely. This is a silent error that would corrupt all model evaluation results.

---

## 13. Feature vs Label

### Simple meaning

- A **feature** is the input data the model uses to make a prediction — in this case, the raw EEG measurements.
- A **label** is what the model is trying to predict — in this case, the participant's clinical group.

You feed features in; you get predicted labels out.

### Technical meaning

**GENERAL DOMAIN KNOWLEDGE:** In supervised machine learning, features (also called inputs or predictors) are the measurable variables used by the model. Labels (also called targets or ground truth) are the correct outputs that the model is trained to predict.

### Meaning in this research

**PAPER-STATED:** The model input is the EEG trial data — a 2D matrix of channel × sampling-point measurements. This is the **feature** input to the EEG-Transformer. The model is trained to predict which of the three groups (HC, ADD, ADHD) each trial belongs to — that is the **label**.

**PAPER-STATED:** The model architecture takes an EEG input of shape `(None, 385, 56)` after position embedding, where:
- `None` = batch dimension (variable batch size)
- `385` = time/sample positions
- `56` = channels

The output is a Softmax layer with **3 outputs** corresponding to the three classification groups.

Note: The stimulus categories (`y_stim` Rows 1–3) are not the labels the model predicts. They are metadata about each trial. Whether and how to use them has not been established in the current research plan.

---

## 14. Epoch × Channel × Time Concept

### Simple meaning

Think of the data as a three-level hierarchy:

```
One subject (person)
    └── Many trials (1.5-second EEG segments)
            └── Each trial contains 56 channels (measurement locations)
                    └── Each channel contains 385 measurements over time
```

If you pick one trial from one person, you have a 56 × 385 table of numbers. Each row is one channel. Each column is one time point. Each cell contains the electrical measurement at that channel at that moment.

### Technical meaning

**PAPER-STATED:** The paper describes the EEG data as a "two-dimensional arrangement of channel number and sampling point," with 56 channels and 1.5-second acquisition.

### Mapping to the observed dataset

**STRONGLY SUPPORTED:** The `d1.mat` array with shape `(385, 56, 5000)` maps to:

| Dimension | Size | Interpretation |
|---|---|---|
| First (axis 0) | 385 | Time positions (samples within each trial) |
| Second (axis 1) | 56 | EEG channels |
| Third (axis 2) | 5000 | Trials stored in this file |

So for trial number `k`, the EEG data is `d1[:, :, k]` — a 385 × 56 matrix of measurements.

This dimension mapping is **STRONGLY SUPPORTED** (not directly documented). The evidence is: the model input shape `(None, 385, 56)` from the paper's architecture table, the paper's reported 56-channel count, and the total trial count consistent with `y_stim`. However, no OSF file directly states "axis 0 = time, axis 1 = channel, axis 2 = trial."

---

## 15. Dataset File Terminology

### d1.mat through d7.mat

**Simple meaning:** The actual EEG measurement values, split into seven files because the complete dataset is too large for one file.

**Technical description:** Each file contains a 3D numeric array of float32 values. Based on **DIRECTLY ESTABLISHED** observations of `d1.mat`:
- Shape: `(385, 56, 5000)`
- Data type: `float32` (MATLAB `single`)
- Interpretation (**STRONGLY SUPPORTED**): `(time samples × channels × trials)`

**STRONGLY SUPPORTED** file contents:
- `d1.mat` through `d6.mat`: approximately 5,000 trials each (approximately 402 MB each)
- `d7.mat`: approximately 3,902 trials (approximately 314 MB)
- Total: 30,000 + 3,902 = 33,902 trials

**DIRECTLY ESTABLISHED (OSF README):** The OSF readme confirms that the sample data is split across seven files because it is too large for one file.

---

### y_stim.mat

**Simple meaning:** Information describing each of the 33,902 trials — which subject it belongs to and which stimulus category it was.

**Technical description:**
- Shape: `(4, 33902)` — four rows, one column per trial
- **DIRECTLY ESTABLISHED:** Row 0 = subject index within group (1 to max 52, resets per group)
- **DIRECTLY ESTABLISHED:** Rows 1–3 = three mutually exclusive binary stimulus category indicators (exactly one is 1 per trial)
- **UNRESOLVED:** The physical names of the three stimulus categories

---

### Sub_name_stim.mat

**Simple meaning:** A list of participant identifiers, organised into three groups corresponding to HC, ADD, and ADHD.

**Technical description:**
- **DIRECTLY ESTABLISHED:** Contains three blocks of 44, 52, and 48 entries
- **STRONGLY SUPPORTED:** Blocks correspond to HC (controls), ADD (subtype1), ADHD (subtype2) based on path strings
- Total: 144 participant entries
- Contains file path strings that link each participant to their original source data

---

### chan.mat

**Simple meaning:** Descriptions of the EEG channel positions and names — essentially a map of where each sensor was placed on the scalp.

**Technical description:**
- **DIRECTLY ESTABLISHED:** Contains 60 channel entries
- Contains standard EEG channel names (e.g., Fp1, Fp2, Cz, FCz, Oz)
- **UNRESOLVED:** Which 4 of these 60 channels are absent from the data arrays (56 channels in data vs 60 in chan.mat)

---

## 16. Important Numbers

| Number | Meaning | Source | Certainty |
|---|---|---|---|
| **144** | Total participants in the study | PAPER-STATED + DIRECTLY ESTABLISHED (Sub_name_stim) | High |
| **44** | Number of HC participants | PAPER-STATED + DIRECTLY ESTABLISHED (Sub_name_stim block 1) | High |
| **52** | Number of ADD participants | PAPER-STATED + DIRECTLY ESTABLISHED (Sub_name_stim block 2) | High |
| **48** | Number of ADHD participants | PAPER-STATED + DIRECTLY ESTABLISHED (Sub_name_stim block 3) | High |
| **33,902** | Total trials across all participants and groups | PAPER-STATED + DIRECTLY ESTABLISHED (y_stim columns) | High |
| **5,000** | Number of trials per file for d1 (directly) and d2–d6 (strongly supported) | DIRECTLY ESTABLISHED (d1) / STRONGLY SUPPORTED (d2–d6) | High for d1; Supported for d2–d6 |
| **56** | Number of EEG channels in data arrays | PAPER-STATED + DIRECTLY ESTABLISHED (d1 shape) | High |
| **60** | Number of channel entries in chan.mat | DIRECTLY ESTABLISHED | High |
| **385** | Time/sample dimension in d1 | DIRECTLY ESTABLISHED (d1 shape); interpretation UNRESOLVED | High for count; UNRESOLVED for meaning of 385th |
| **256 Hz** | EEG sampling frequency | PAPER-STATED | Paper-reported; not independently verified from files |
| **1.5 s** | Duration of each trial | PAPER-STATED | Paper-reported; not independently verified from files |
| **10,129** | Trials from HC participants (paper) / y_stim Row 1 count (dataset) | PAPER-STATED (group) + DIRECTLY ESTABLISHED (Row 1 count) | Counts established; connection between them is STRONGLY SUPPORTED |
| **13,031** | Trials from ADD participants (paper) / y_stim Row 2 count (dataset) | PAPER-STATED (group) + DIRECTLY ESTABLISHED (Row 2 count) | Counts established; connection is STRONGLY SUPPORTED |
| **10,742** | Trials from ADHD participants (paper) / y_stim Row 3 count (dataset) | PAPER-STATED (group) + DIRECTLY ESTABLISHED (Row 3 count) | Counts established; connection is STRONGLY SUPPORTED |

Note: The counts 10,129 / 13,031 / 10,742 appear in the paper as group trial counts (HC, ADD, ADHD). The same three numbers appear in the `y_stim` Row 1–3 stimulus-category counts. This coincidence is noted — but the physical interpretation of the stimulus rows and whether they directly encode group membership is **UNRESOLVED**. The paper does not explicitly state this connection.

---

## 17. Terms That Must Not Be Confused

### Subject vs Trial

| | Subject | Trial |
|---|---|---|
| **What it is** | One person in the study | One 1.5-second EEG data segment |
| **Count** | 144 | 33,902 |
| **Relationship** | One subject contributes many trials | Many trials belong to one subject |

**Why confusing these matters:** If you treat 33,902 as the number of independent people, every statistical calculation (sample size, test power, participant-level generalisation) will be wrong.

---

### Trial vs Epoch

| | Trial (paper's term) | Epoch |
|---|---|---|
| **Paper's use** | One 1.5-second EEG data segment | One training iteration (machine learning) |
| **General EEG use** | Roughly synonymous with epoch | A segmented EEG window |

**Why confusing these matters:** When you read "epoch" in the paper, it means a training iteration. When you read "epoch" in EEG literature or our dataset documentation, it means an EEG segment. Using the wrong meaning produces confusion about data structure vs training loop.

---

### Recording vs Trial/Epoch

| | Recording | Trial/Epoch |
|---|---|---|
| **What it is** | The complete continuous EEG data collection from one participant in one session | A short segment (1.5 s) cut from a recording |
| **Established in paper** | Implied but not explicitly defined | Explicitly defined as 1.5-second segments |

**Why confusing these matters:** One recording produces many trials. If you equate "recording" with "trial," you misrepresent how the dataset was built and may incorrectly estimate how many independent participants contributed.

---

### Channel vs Sample

| | Channel | Sample |
|---|---|---|
| **What it is** | One spatial measurement location on the scalp | One measurement at one moment in time at one channel |
| **Count** | 56 (in data arrays) | 385 (per channel per trial) |
| **Dimension** | Second axis in d1 | First axis in d1 |

**Why confusing these matters:** Channels are spatial; samples are temporal. Mixing them swaps the axes in the data array, producing completely wrong input tensors for the model.

---

### Channel vs Electrode

**GENERAL DOMAIN KNOWLEDGE:** In hardware, an electrode is the physical sensor placed on the scalp. In data, a channel is the recorded signal from that electrode (or a combination of electrodes). In most EEG setups, one electrode corresponds to one channel.

**In this research:** The paper uses "electrode channels" and "channels" interchangeably. The dataset investigation refers to them as "channels." This project uses "channel" as the primary term for the data dimension.

**Why confusing these matters:** Minor in practice for this dataset. The important practical distinction is that `chan.mat` lists 60 electrode/channel descriptions, while the data has 56 channels — a four-channel discrepancy that cannot be resolved without additional documentation.

---

### Stimulus vs Diagnosis/Group

| | Stimulus category | Participant group / diagnosis |
|---|---|---|
| **What it is** | The type of event/task in one trial | The clinical classification of the participant |
| **Values** | 3 unknown categories (Rows 1–3 in y_stim) | HC / ADD / ADHD |
| **Count** | 33,902 trials (one category per trial) | 144 participants (one group per person) |
| **Used as model target** | Not established | Yes — this is the classification target |

**Why confusing these matters:** Using stimulus category as the model's target variable trains the model to solve the wrong problem. This is an invisible error that would produce meaningless results.

---

### Label vs Feature

| | Label (target) | Feature (input) |
|---|---|---|
| **What it is** | The correct answer the model predicts | The data the model uses to predict |
| **In this research** | Participant group (HC/ADD/ADHD) | EEG trial data (385 × 56 measurement matrix) |

**Why confusing these matters:** If EEG data is treated as the label and group labels are treated as features, the model is inverted — it would attempt to predict EEG signals from group membership, which is the opposite of the research task.

---

### Participant vs Recording

These are distinct concepts (see Section 9 — Recording) that must not be treated as interchangeable. One participant can have one or more recording sessions, and each session produces many trials.

---

### Group vs Class

| | Group | Class |
|---|---|---|
| **In clinical context** | HC, ADD, ADHD — clinical classifications of participants | Not typically used |
| **In ML context** | Less common | HC, ADD, ADHD — output classes of the classifier |

These refer to the same three categories (HC, ADD, ADHD) but use language from different domains. In this project, both are used. "Group" follows the paper's participant-level language. "Class" follows standard ML language for the output categories.

---

## 18. What We Know vs What We Don't Know

### Established terminology and relationships

The following are **supported by the paper and/or directly observed dataset findings:**

1. **Total participants:** 144 (44 HC, 52 ADD, 48 ADHD) — PAPER-STATED + DIRECTLY ESTABLISHED
2. **Total trials:** 33,902 — PAPER-STATED + DIRECTLY ESTABLISHED
3. **Trial duration:** 1.5 seconds — PAPER-STATED
4. **Sampling rate:** 256 Hz — PAPER-STATED
5. **Channels in data:** 56 — PAPER-STATED + DIRECTLY ESTABLISHED
6. **Channels in chan.mat:** 60 — DIRECTLY ESTABLISHED
7. **d1.mat shape:** (385, 56, 5000) — DIRECTLY ESTABLISHED
8. **d1.mat data type:** float32 — DIRECTLY ESTABLISHED
9. **y_stim shape:** (4, 33902) — DIRECTLY ESTABLISHED
10. **y_stim Row 0:** subject index within group — DIRECTLY ESTABLISHED
11. **y_stim Rows 1–3:** three mutually exclusive binary stimulus categories — DIRECTLY ESTABLISHED
12. **Stimulus category counts:** 10,129 / 13,031 / 10,742 — DIRECTLY ESTABLISHED
13. **Sub_name_stim.mat blocks:** 44, 52, 48 entries — DIRECTLY ESTABLISHED
14. **Group-to-file mapping:** controls=HC, subtype1=ADD, subtype2=ADHD — STRONGLY SUPPORTED
15. **d1–d6 trial counts:** approximately 5,000 each — STRONGLY SUPPORTED
16. **d7 trial count:** approximately 3,902 — STRONGLY SUPPORTED
17. **d1 dimension order:** (time × channels × trials) — STRONGLY SUPPORTED
18. **d1 ↔ y_stim[:, 0:5000] correspondence** — STRONGLY SUPPORTED
19. **Classification target:** HC / ADD / ADHD participant group — PAPER-STATED
20. **Model input shape after position embedding:** (None, 385, 56) — PAPER-STATED
21. **Paper's primary term for EEG data segments:** "trial" — PAPER-STATED
22. **Paper's use of "epoch":** training iteration, not EEG segment — PAPER-STATED

---

### Still unresolved

The following **cannot be established from the available sources:**

1. **Physical meaning of the three stimulus categories** (y_stim Rows 1–3). We know they are mutually exclusive and have counts 10,129 / 13,031 / 10,742, but we do not know what stimulus types they represent.

2. **Exact d1 ↔ y_stim mapping.** It is strongly supported that `d1` corresponds to the first 5,000 columns of `y_stim`, but no OSF file directly documents this mapping.

3. **Exact meaning of the 385th sample.** Why there are 385 sample positions when 256 Hz × 1.5 s = 384 is unexplained by the paper and unresolvable from available files.

4. **Exact four-channel discrepancy.** Why `chan.mat` lists 60 channels while the data arrays contain 56 is unknown. Which specific channels were removed cannot be determined.

5. **Whether the train/test split was participant-independent.** The paper does not state this.

6. **Exact preprocessing pipeline beyond ICA noise removal.** Filtering, referencing, normalisation, artifact rejection criteria — all unspecified.

7. **Whether trials from one participant overlap.** The paper does not state whether trial segmentation was with or without overlap.

8. **The meaning of the "300" sample size figure** mentioned in the Discussion (vs 144 in the Methods). The paper does not reconcile this.

9. **What the incomplete "5,580 trials were used as..." sentence** in the paper refers to.

10. **The exact dimension ordering of d1.** It is strongly supported as (time × channels × trials), but not directly documented.

---

## 19. Simple End-to-End Explanation

Here is what is actually happening in this research, written for a CSE student encountering this topic for the first time:

---

**A group of 144 children participated in EEG experiments at the Technical University of Dresden, Germany.**

The children were divided into three groups: 44 who are considered neurotypically healthy (HC), 52 diagnosed with ADD, and 48 diagnosed with ADHD. The groups were matched so they had no significant differences in age, IQ, or sex distribution.

---

**EEG measurements were collected from each child.**

During the experiment, small sensors were placed on each child's scalp. These sensors measured tiny electrical signals produced by the child's brain. The recording device sampled these signals 256 times per second through 56 different sensor locations (channels).

---

**The measurements were organised into short trials.**

The continuous EEG recording was divided into segments of 1.5 seconds each. Each 1.5-second segment is called a **trial** in this paper. A single trial contains data from 56 channels, each with 385 measurement values spanning the 1.5-second window (the paper implies 384 from 256 Hz × 1.5 s, but 385 are observed in the dataset — this is unresolved).

Each trial is stored as a matrix: rows are time points (385), columns are channels (56). This is the direct input to the machine learning model.

---

**There are many trials per participant.**

The 144 children together produced 33,902 trials — an average of about 235 trials per child. This is important: **33,902 is not 33,902 people**. It is 33,902 data segments from 144 people.

---

**Metadata tells us what we know about each trial.**

For each of the 33,902 trials, metadata in `y_stim.mat` tells us:
- Which participant (by index within group) the trial came from
- Which of three stimulus categories the trial belongs to (the exact physical meaning of these three categories is unresolved)

We know from `Sub_name_stim.mat` which clinical group each participant belongs to (HC, ADD, or ADHD).

---

**The research task: train a model to classify the group from the EEG data.**

**PAPER-STATED:** The paper's goal is to build a model (EEG-Transformer) that reads one 1.5-second EEG trial and outputs a prediction of whether the trial came from an HC child, an ADD child, or an ADHD child. This is a three-class classification problem.

The model's input is the EEG measurements. The model's output is a prediction of the clinical group. This is what the model is trained and evaluated on.

**PAPER-STATED:** The paper reports that the EEG-Transformer achieved approximately 95.58% accuracy on this task in their experimental setup.

---

**What this research is and is not.**

**PAPER-STATED:** The authors frame the model as a potential auxiliary tool to assist physicians in clinical diagnosis — not as a replacement for clinical diagnosis. They acknowledge that the participant count is limited for clinical application and that further validation would be needed.

---

## 20. Research Terminology Principle

For this project, terminology must always be interpreted **in context**.

A word such as **"trial," "epoch," "recording,"** or **"label"** must not be assigned a meaning merely because that meaning is common elsewhere.

When the paper, dataset, and general domain terminology differ:

1. **Preserve the source terminology.** The paper's own words take priority.
2. **Document the difference.** If the paper's usage differs from general domain usage, both must be recorded.
3. **State the uncertainty.** If neither source clearly establishes a meaning, say so explicitly rather than filling the gap silently.
4. **Avoid silently redefining terms.** A term used in our dataset investigation documentation (e.g., "epoch" for EEG segments) may differ from the paper's use of the same word (training iterations). Both usages exist in this project's documents and must be distinguished by context.

This principle protects the integrity of the research. In machine learning applied to clinical data, a single terminological confusion — such as using stimulus categories as the label, or treating participant counts as trial counts — can silently corrupt an entire experimental pipeline.

The foundation of sound research is knowing precisely what each word means, what each number represents, and what remains genuinely unknown.
