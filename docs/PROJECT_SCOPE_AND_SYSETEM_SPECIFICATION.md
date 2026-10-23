# ADHD EEG Research & Prototype
# Project Scope and System Specification

## 1. Project Objective

Define what the final system is expected to do.

The system will take an EEG recording as input and determine whether
the recording belongs to one of the three target groups:

- HC
- ADD
- ADHD

The system must not be limited to producing a class label. The final
system should also provide information that allows us to examine why
the model produced that prediction and how reliable the prediction is.

The exact input representation, preprocessing, model architecture,
training procedure, evaluation procedure, and explanation method will
be defined in later sections of this specification.

## 2. Research Objective

The research focuses on studying the use of EEG recordings for distinguishing
between HC, ADD, and ADHD.

The study will examine the performance of the developed approach on unseen
subjects and investigate what information in the EEG contributes to its
predictions.

The research will also examine the strengths, limitations, and failure cases
of the developed approach rather than evaluating it only through a final
classification score.
## 3. Prototype Objective

The prototype is the usable form of the final research system.

It will take an EEG recording as input and present the resulting prediction
for HC, ADD, or ADHD together with the relevant information produced by the
model to help understand that prediction.

The prototype is intended to demonstrate the complete developed approach,
from EEG input to final result, through a simple and useful interface.

## 4. Intended User

The prototype is intended for authorized users who work with EEG recordings,
such as clinicians, EEG technicians, and researchers.

The user operates the system and provides the EEG recording to be analysed.
The person represented by the recording is the subject of the analysis.

The prototype is intended to support analysis of EEG data and is not intended
to replace clinical judgement or independently establish a medical diagnosis.

## 5. Prototype Input

The prototype is intended to accept an EEG recording belonging to a single subject/session.

The user of the prototype is assumed to be an authorized EEG technician, clinician, or researcher who has access to the subject's EEG recording. The prototype does not perform EEG acquisition; EEG acquisition is assumed to have already been completed using an appropriate EEG recording system.

### Input Definition

The prototype input is:

- One EEG recording corresponding to one subject/session.
- Initial target file format: `.mat`.
- The exact internal `.mat` structure will be defined and frozen after the model input representation and preprocessing pipeline are finalized.
- The recording must contain the EEG signal information required by the trained model.
- The prototype must not require the research dataset files (`d1.mat`–`d7.mat`, `y_stim.mat`, `sub_name_stim.mat`, etc.) from the end user.
- The research dataset is used for model development, training, validation, and evaluation; it is not the end-user input.

### Input Principle

The prototype must accept previously unseen EEG data from a subject/session that was not used during model training.

The inference pipeline applied to the uploaded EEG must be consistent with the preprocessing and representation pipeline used during model training.

The exact signal dimensions, channel configuration, sampling requirements, and other input constraints will be frozen later as part of the preprocessing and model-input specification.

---

## 6. Prototype Output

The prototype must provide more than a simple three-class prediction.

Its primary purpose is to produce an AI-assisted subject/session-level classification together with evidence that helps explain and assess the prediction.

### Primary Classification Output

The model will classify the EEG recording into one of three research classes:

- Healthy Control (HC)
- Attention Deficit Disorder (ADD)
- Attention Deficit Hyperactivity Disorder (ADHD)

The prototype will display:

- Predicted class
- Probability for each class
- Overall prediction confidence

Example:

    Predicted Class: ADHD

    HC:    0.08
    ADD:   0.17
    ADHD:  0.75

    Confidence: 75%

### Evidence / Explainability Output

The prototype must also provide interpretable evidence supporting the model's prediction.

The evidence should help answer:

> "Why did the model make this prediction?"

Depending on the final model and validated explainability methodology, the prototype may provide information such as:

- Important temporal regions of the EEG
- Important EEG channels or learned spatial representations
- Important frequency-domain characteristics
- Feature or representation importance
- Model attribution/activation visualizations
- Prediction stability across EEG windows/trials
- Uncertainty or confidence information

These explainability outputs must be based on methods that are actually implemented and experimentally validated. The prototype must not present an explanation merely because a visualization can be generated.

### Subject/Session-Level Result

Although the model may internally process multiple EEG windows or trials, the primary user-facing result should represent the subject/session rather than an isolated individual trial.

Conceptually:

    EEG recording
          ↓
    Multiple windows/trials
          ↓
    Model predictions
          ↓
    Prediction aggregation / analysis
          ↓
    Subject/session-level result

### Robustness Requirement

The prototype must be designed to operate on previously unseen EEG recordings and should be evaluated for robustness to reasonable variations in input data.

The research must therefore evaluate whether the model's prediction remains reliable when presented with data that was not observed during training.

Robustness claims must be supported by experiments and must not be assumed from the model architecture alone.

### Prototype Output Principle

The final prototype should communicate:

1. **What the model predicted**
2. **How confident the model is**
3. **What evidence contributed to the prediction**
4. **How reliable/stable the prediction is under the evaluated conditions**

The system is intended as an AI-assisted research/decision-support prototype and is not intended to independently establish a medical diagnosis.



## 7. Evidence and Explainability Requirement

## 8. Robustness / Unseen Data Requirement

## 9. Research Dataset

The research will use the EEG dataset associated with the base research paper:

He et al., "Classification of attention deficit/hyperactivity disorder
based on EEG signals using a EEG-Transformer model", Journal of Neural
Engineering, 2023.

The dataset contains EEG recordings from 144 subjects belonging to three
groups:

- HC
- ADD
- ADHD

The dataset contains 33,902 EEG trials recorded using 56 EEG channels.

The dataset will be treated as the research dataset used for development,
training, validation, and evaluation of the proposed approach.

The original dataset files will be preserved as the raw source data.
Any preprocessing, transformed data, extracted representations, or other
derived data will be kept separately from the original files.

The mapping between EEG trials, subjects, and the three groups will be
maintained explicitly so that subject identity is preserved throughout
the research pipeline.

The research dataset is used for model development and evaluation. It is
not considered equivalent to an independent external clinical dataset,
and performance on this dataset alone will not be interpreted as evidence
of clinical deployment readiness.

## 10. End-to-End System Flow

## 11. Preprocessing — TO BE FROZEN

## 12. Feature / Representation — TO BE FROZEN

## 13. Data Splitting — TO BE FROZEN

## 14. Baselines — TO BE FROZEN

## 15. Proposed Model — TO BE FROZEN

## 16. Evaluation — TO BE FROZEN

## 17. Explainability — TO BE FROZEN

## 18. Prototype Integration

## 19. Research Paper Contributions

## 20. Decisions / Constraints / Non-Goals