# 06 — Explainability and Output Audit

**Date:** 2026-10-06  
**Purpose:** Audit what the research system and prototype should report beyond a three-class label, what existing documents already require, and what remains undecided.  
**Audience:** A Computer Science student new to EEG, machine learning, neuroscience, and explainability.  
**Scope:** Requirements and evidence audit only. This document does **not** select an explainability technique, model, architecture, interface, or final output format.

## How to read this audit

- **Required** means an existing project objective or formal scope requires the output or the underlying analysis.
- **Possible option** means the project mentions it as a candidate, example, or useful analysis, but it is not a frozen output requirement.
- **Open** means the project has not settled the precise meaning, implementation, validation, or presentation.
- A prediction is a model's assignment to one of the dataset's labels. An explanation or evidence display is an account of model behavior; it is not automatically a scientific or clinical explanation.

This audit is for manually deciding the system design before implementation. It intentionally does not recommend or rank explainability methods.

## 1. Project context and important boundaries

The research task is to classify EEG data into the dataset's three recorded groups: **HC**, **ADD**, and **ADHD**. The central research question concerns distinguishing these groups for subjects excluded from model development. The intended prototype input is one EEG recording for one subject/session, and its primary user-facing result should represent that subject/session rather than only an isolated trial.

The project is still before preprocessing and model implementation. The project-state audit records that no preprocessing, model, or training has been completed, and Gate 7 (bounded explainability claims) has not passed. The base paper's Transformer is a reference, not an adopted architecture.

The relevant evidence has limitations:

- The dataset comprises 144 children from one cohort and contains 33,902 trials. A trial is a short EEG segment; many trials may come from the same subject. Trials are not 33,902 independent people.
- All trials from one subject must stay in one partition for an unseen-subject evaluation. A good-looking trial score does not establish subject-level generalization if the split leaks people across partitions.
- The EEG arrays have 56 channel columns, while `chan.mat` lists 60 channel entries. The mapping of the 56 columns to physical electrode names is unresolved.
- The physical interpretation of the 385th sample is unresolved. A time-index highlight can be reported as a position in the stored input; converting it to a precise physical time requires care.
- Dataset group labels are the classification targets. Predictions and evidence concern those supplied labels; they do not independently establish a person's diagnosis.

These facts matter to output design: a chart can look precise even where the underlying evidence does not support that precision.

## 2. Output categories: status and interpretation

| Output category | In simple language | Why it may matter | Status in this project | What it could actually tell us | What it would not prove |
|---|---|---|---|---|---|
| **Predicted class** | The model's chosen label: HC, ADD, or ADHD. | It is the basic answer to the classification task and the starting point for reviewing a case. | **Required.** The prototype scope explicitly requires a predicted class. | Which of the three dataset group labels the model selected for this subject/session result. | That the subject has that diagnosis, that the model is clinically correct, or that the prediction generalizes beyond the evaluated cohort. |
| **Class probabilities** | A score for each of the three classes, usually displayed as values that add to 100%. For example, HC 8%, ADD 17%, ADHD 75%. | Shows how the model's scores are distributed across the alternatives, rather than hiding them behind the winning label. | **Required for display** by the prototype scope. | Which classes the model scores relatively higher or lower for this input, according to its output layer and chosen aggregation procedure. | That a displayed 75% means there is a 75% real-world chance the person has ADHD. That interpretation depends on how the scores were produced, aggregated, and validated. |
| **Overall confidence** | A summary of how sure the system appears about its selected result. The project scope gives a percentage example. | A user should be able to notice when the system's own output is less decisive. | **Required for display** in the formal prototype scope; the exact meaning and calculation are **open**. | Whatever carefully defined score the final system chooses to call confidence. | That the prediction is correct, calibrated, safe, or medically meaningful. Confidence can be high on a wrong prediction. A top-class probability may not be a separately meaningful confidence measure. |
| **Uncertainty** | A description of how much doubt or variation surrounds a prediction. It is related to, but not automatically the same thing as, confidence. | Helps avoid presenting one precise-looking output when plausible inputs, windows, or model runs give different answers. | **Possible option.** The scope mentions uncertainty/confidence information; the research audit says how confidence reliability will be evaluated is open. No uncertainty quantity or display is frozen. | An explicitly defined source of doubt (for example, ambiguity among class scores or variation under a declared evaluation condition), if measured. | A guarantee that all relevant sources of uncertainty have been captured. A score alone does not reveal whether uncertainty comes from limited data, model instability, noisy input, or class overlap. |
| **Evidence supporting a prediction** | Information intended to help a user inspect why the model produced its result. | The project is not meant to stop at a label; it aims to investigate what information in EEG contributes to predictions. | **Required at a broad level.** The scope requires interpretable evidence and says it must be implemented and experimentally validated. The evidence representation and validation are **open**. | At most, evidence about the model's behavior on the supplied input, under a stated analysis and validation procedure. | The biological cause of the group difference, a causal mechanism, an ADHD biomarker, or proof that the model's reasoning matches a clinician's reasoning. |
| **Important EEG time regions** | Positions or intervals in the signal that appear relevant to the model's output. | Makes evidence easier to relate to a timeline instead of giving only a class score. | **Possible option**, named in the formal scope; not a required specific evidence type. | Which stored time positions or segments appear influential under a defined evidence analysis. | That the highlighted period contains a known brain event, is clinically important, or caused the class distinction. The 385th sample's physical timing remains unresolved. |
| **Important channels / spatial representations** | EEG columns or spatial patterns that appear relevant to the model's output. | Could help examine whether the model relies on all inputs or mainly on some channels. | **Possible option**, named in the formal scope; no channel-level output is frozen. | Which input columns or learned spatial representations appear influential, if the input and evidence analysis support that statement. | A named scalp location, brain region, or neural source while the 56-column-to-60-channel mapping is unresolved. Even a correct electrode attribution would not establish a clinical mechanism. |
| **Frequency-domain characteristics** | Information about how signal variation is distributed across oscillation rates (frequencies). | It is one candidate view of EEG information that may help describe a model's decision. | **Possible option**, listed in the formal scope. No frequency features or representation have been approved. | Which defined frequency values or bands correlate with model outputs under the chosen representation and analysis. | That a band is a validated ADHD marker, that it is the cause of the prediction, or that the result applies across recording systems and preprocessing choices. |
| **Feature / representation importance** | A summary of which input measurements or learned internal representations the model seems to use. | May give an evidence view even when the model does not directly use named EEG features. | **Possible option**, listed in the formal scope; exact meaning and validation are open. | Relative model sensitivity or contribution according to a specified analysis. | A causal explanation, a stable scientific feature, or evidence that a learned internal representation has a known neurological meaning. |
| **Attribution / activation visualization** | A picture that visualizes an evidence analysis or internal model activity. | A visual summary can make model behavior easier to inspect. | **Possible option.** The scope mentions visualizations as examples, but explicitly says a visualization alone is not sufficient justification. | A representation of whatever quantity the visualization actually encodes, if its construction and limits are stated. | That visually prominent areas are truly important, faithful, stable, or clinically meaningful merely because they look interpretable. **Transformer attention is not automatically an explanation.** |
| **Stability across windows/trials** | Whether predictions or evidence stay similar when a recording is examined in separate segments, or when multiple trials from the subject are considered. | A subject-level result may hide disagreement among the smaller pieces from which it was made. | **Subject/session-level output is required.** The scope also requires communicating reliability/stability under evaluated conditions and lists window/trial stability as a possible output. The exact aggregation and stability summary are **open**. | Whether segment-level results agree, vary, or change the subject/session aggregate under a defined procedure. | That the output is stable under every possible recording change, or that consistent model behavior is medically valid. |
| **Failure / low-confidence indication** | A clear warning that the system made an uncertain result, encountered an unsupported input, or could not produce a valid result. | Users need to know when not to treat the displayed prediction as dependable. Failure-case analysis is also important for understanding model limitations. | **Failure analysis is required in the research objective.** A user-facing low-confidence flag, abstention, or invalid-input state is **not explicitly specified** and remains open. | A warning can communicate a pre-defined condition the system has actually checked, such as an agreed confidence threshold or input-validation failure. | That no warning means a prediction is correct or safe. A warning threshold does not itself prove the cases above it are reliable. |

## 3. Distinguishing similar-looking outputs

These terms should not be silently treated as synonyms in the final design:

- **Class probability** is a model-produced class score that may be normalized to sum to one. The word “probability” must be supported by the model's output semantics and appropriate validation.
- **Confidence** is the project's required user-facing idea, but the project has not yet defined whether it means the highest class score, a separate metric, or something else.
- **Uncertainty** is doubt about the prediction. It can have multiple sources. A single confidence number need not capture them all.
- **Reliability** is whether output quality holds under a specified evaluation. For example, confidence reliability asks whether cases assigned similar confidence are correct at corresponding frequencies. The research audit explicitly says this validation is not yet frozen.
- **Stability** is consistency under a specified change, such as using different windows/trials, folds, or repeated runs. Stable output can still be wrong or biased.
- **Evidence / explanation** describes some aspect of model behavior. It is not the same thing as confidence, and it is not a diagnosis.

Consequently, the example in the project scope (`ADHD: 0.75`, `Confidence: 75%`) is an illustrative display, not yet a complete definition of two distinct quantities. The architecture discussion must decide whether those are the same number and how either should be validated.

There is also a wording difference between project documents: the formal scope asks for a probability for each class plus an overall confidence value, while the summary in the project-state audit describes a confidence score for each class and visualizations of what the model paid attention to. Those are not necessarily equivalent requirements. This audit uses the formal scope as the clearest current output contract, but the team should reconcile the wording before implementation; it does not assume that a class probability is a separate confidence measure or that attention visualization is mandatory.

## 4. Possible evidence views and their limits

The scope lists evidence examples rather than a settled menu of required displays. These views can be considered independently at design time; listing them here is not a recommendation to implement any one of them.

| Evidence view | A question it may help inspect | Possible useful observation | Main interpretation limit for this project |
|---|---|---|---|
| **Temporal** | Which stored positions or portions of a trial mattered to the model's score? | Whether the model's score changes or an evidence summary concentrates around particular parts of the input. | Time positions are not automatically interpretable as a particular event; the physical meaning of sample 385 is unresolved. Sensitivity to processing and trial alignment would need investigation. |
| **Channel / spatial** | Which input columns or spatial patterns mattered to the model? | Whether evidence is distributed across columns or dominated by a subset. | The electrode identity of each of 56 columns is not established against 60 metadata entries. Until resolved, report column indices rather than named brain locations; do not infer a brain region. |
| **Frequency-domain** | Which frequency characteristics were associated with the model's score? | Whether the model uses a particular frequency representation when distinguishing labels. | The project has not approved spectral features or frequency bands. A frequency association is not a biomarker or proof of mechanism. |
| **Input feature / learned representation** | Which measurements or internal patterns appear to influence the output? | Which inputs have relatively larger influence under a defined evidence analysis. | Learned units may not have clear human meaning. Attribution is method-dependent and does not establish causality or validity. |
| **Window/trial comparison** | Do smaller segments or trials agree with the subject/session result? | Whether segment-level predictions are consistent or whether a few segments dominate the aggregate. | Trial counts vary greatly by person (46–295). The aggregation rule and whether segments count equally are undecided. Correlated trials do not equal independent confirmations. |
| **Class-wise error view** | Which groups are confused, and what failures are hidden by an overall score? | Whether (for example) ADD and ADHD are confused more often than either is confused with HC. | This is an error analysis, not an explanation of the internal reason for each error. Its unit (trial or subject) and summary must be declared. |

For any evidence view, the eventual report should identify what was actually measured, the level it refers to (trial, window, subject/session, or group summary), and the conditions under which it was checked. A visually plausible pattern should not be promoted to a neuroscience claim without independent evidence.

## 5. Reliability, uncertainty, and failure communication

The existing documents require more than displaying a confidence-looking number:

1. **Show the required result:** predicted class, class probabilities, and an overall confidence display are specified in the prototype scope.
2. **Define the unit:** the user-facing result represents a subject/session, while the model may process multiple windows or trials. The mapping from component predictions to the one final result is not decided.
3. **Test the claim:** if the interface calls a number confidence or probability, the system should evaluate whether that value means what users could reasonably infer from its label. The research-design audit calls confidence reliability a needed check but leaves the measure and method open.
4. **Inspect variation:** the project aims to communicate how reliable/stable the prediction is under evaluated conditions. Which variations are evaluated, and how the outcome is shown, are not specified.
5. **Inspect failures:** the research objective requires examining successes and failures. The project has not decided whether a failed, invalid, or low-confidence case should still show a class, show a warning, abstain, or follow some combination of these.

One held-out-subject evaluation on this cohort cannot establish reliability for other hospitals, devices, recording protocols, ages, or populations. The existing research audit specifically rules out treating this dataset as an external clinical validation.

## A. Output requirements already established

1. The target is a **three-class research classification** over the supplied HC, ADD, and ADHD group labels.
2. The prototype should accept EEG for a **single subject/session** and return a subject/session-level result, not only an isolated trial result.
3. The prototype must display a **predicted class**, **probability for each class**, and **overall prediction confidence**.
4. The system should provide **evidence that helps examine the prediction**. That evidence must be implemented and experimentally validated; a generated visualization alone is not enough.
5. The research must examine which EEG information contributes to predictions and report strengths, limitations, and failure cases, rather than only an aggregate score.
6. The prototype should communicate confidence and reliability/stability **under conditions that are actually evaluated**. This does not imply that a specific uncertainty or calibration technique has already been selected.
7. Results must be bounded to the supplied dataset and evaluation. The system is not intended to independently diagnose a person, replace clinical judgment, or claim a validated biomarker or cause.

## B. Possible evidence types

The formal scope names temporal regions, channels or learned spatial representations, frequency characteristics, feature/representation importance, attribution/activation visualizations, stability across windows/trials, and uncertainty/confidence information. Error analysis, class-wise results, and subject-level variability also appear in the research-evaluation requirements.

These are **categories of output or analysis**, not selections of a final method. No specific one is approved by this audit.

## C. What each evidence type can and cannot tell us

- **Predicted class:** tells which dataset label the model selected; not a diagnosis.
- **Class probabilities:** show the relative class scores produced for an input; not automatically real-world diagnostic probabilities.
- **Overall confidence:** provides a requested summary only after its meaning is defined and checked; a high value does not guarantee correctness.
- **Uncertainty:** may describe a stated source of doubt; does not cover all uncertainty unless the design establishes that it does.
- **Evidence/attribution:** may describe what aspects of the input or model relate to its output; does not prove a cause, mechanism, biomarker, or clinical rationale.
- **Temporal evidence:** may identify influential stored positions; does not prove an event or exact physical time meaning.
- **Channel evidence:** may identify influential input columns; does not establish electrode location or brain region while channel mapping is unresolved.
- **Frequency evidence:** may describe dependence on defined frequency characteristics; does not establish a clinical marker.
- **Feature/representation importance:** may summarize model sensitivity; does not guarantee human interpretability or causality.
- **Visualizations:** can show a measured quantity; appearance alone does not validate that quantity as an explanation.
- **Window/trial stability:** may show consistency or disagreement among parts of a subject/session; does not turn correlated trials into independent subjects or guarantee correctness.
- **Class-wise error and failure analysis:** may reveal where the system confuses labels; does not, by itself, reveal why a particular mistake occurred.
- **Low-confidence/failure flag:** may identify a pre-defined condition for caution; absence of a flag does not guarantee a reliable answer.

## D. What remains undecided

- What “overall confidence” means, and whether it is separate from the highest class probability.
- How to reconcile the formal scope's per-class probabilities plus overall confidence with the project-state summary's per-class confidence scores and attention visualizations.
- Whether class scores will be described as probabilities, and what evidence is needed before using that term.
- How predictions from multiple windows/trials are combined into a subject/session result; whether all segments contribute equally; and whether users or researchers also see segment-level outputs.
- Which kinds of reliability and uncertainty the project will measure, and which ones the interface will communicate.
- What counts as an input where the model should warn, abstain, or report that it cannot provide a dependable result.
- Which evidence categories, if any, will be shown to prototype users versus included only in research analysis.
- How each evidence display will be validated for faithfulness, stability across held-out subjects/folds, and sensitivity to reasonable input variations.
- Whether time is shown as sample index or physical time, given the unresolved 385th sample.
- Whether any spatial evidence can be given physical electrode labels, given the unresolved 56-to-60 channel mapping.
- How class-wise performance, class supports, confusion patterns, subject-level variability, and failures are presented together.
- Which robustness variations are supported by the available single-cohort data and what claims each test permits.
- Final explainability technique, model, preprocessing, split protocol, aggregation rule, and interface details. These remain decisions for the researcher/team; this audit chooses none.

## E. Questions to answer before drawing the final architecture

1. **What exactly is one inference request?** Is it always one complete subject/session recording, and what input conditions can the prototype accept?
2. **What is the prediction unit?** Does the model classify trials/windows internally, and how will those become exactly one subject/session result?
3. **What does “confidence” mean to the user?** Is it the winning class probability, another score, or a distinct estimate? How will it be tested?
4. **Will displayed class values be called probabilities?** If so, what validation supports that interpretation, especially for held-out subjects?
5. **What should happen when segments disagree or the result is weak?** Should the system display the aggregate with a caution, abstain from selecting a class, or report a processing failure? What evidence and threshold justify that behavior?
6. **Which evidence is for users, and which is only for research?** A technically available analysis need not be a safe or useful prototype display.
7. **At what level will evidence be summarized?** A single subject/session, each window/trial, or a comparison across subjects/folds? What does that level let users conclude?
8. **Which properties of the evidence need validation?** For example: whether it reflects model behavior, whether it is stable across held-out subjects, and whether results survive explicitly defined input variations.
9. **Can time and channels be named physically?** Until sample timing and channel mapping uncertainties are addressed, should results be restricted to stored sample positions and anonymous input columns?
10. **How will failures be reported alongside successes?** Which class-wise and subject-level views make error patterns visible without implying that explanations establish diagnoses?
11. **What claims will the prototype explicitly avoid?** The current project boundary excludes independent diagnosis, clinical readiness, causal mechanism, and biomarker claims.
12. **What results must exist before an output is exposed in the interface?** Confidence reliability, evidence validation, and robustness evaluations are all unfinished decisions.

These decisions should be recorded before implementation. They need not force a particular explanation method; the final method can be selected later against the agreed output needs and validation criteria.

## Sources reviewed

- [PROJECT_SCOPE_AND_SYSETEM_SPECIFICATION.md](../PROJECT_SCOPE_AND_SYSETEM_SPECIFICATION.md) — formal research/prototype scope and output examples.
- [01_PROJECT_STATE_AUDIT.md](01_PROJECT_STATE_AUDIT.md) — current implementation state, unresolved dataset limitations, and gate status.
- [05_RESEARCH_DESIGN_AND_EVALUATION_AUDIT.md](05_RESEARCH_DESIGN_AND_EVALUATION_AUDIT.md) — evaluation unit, confidence reliability, failure analysis, and claims boundaries.
- [03_BASE_PAPER_AND_RESEARCH_EVIDENCE_AUDIT.md](03_BASE_PAPER_AND_RESEARCH_EVIDENCE_AUDIT.md) — what the reference paper reports and what it does not establish.
- [ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md](../ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md) — cautions on attention, attribution, and interpretability claims.
- [ML_EXECUTION_PLAN.md](../ML_EXECUTION_PLAN.md) — planning-only candidate explainability analyses and their unresolved status.
- [SPRINT_2_DECISION_ANALYSIS.md](../SPRINT_2_DECISION_ANALYSIS.md) — open evaluation-unit and data-contract choices, including the channel-mapping caveat.
