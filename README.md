# ADHD-EEG Research

This repository investigates whether EEG recordings from the Dresden ADHD/ADD cohort can support reproducible, participant-level classification research. It began with an interest in reproducing the EEG-Transformer study by He et al. (2023), but the project has since shifted toward checking the dataset evidence, prior work, and evaluation validity before choosing or running a study.

This is a research workspace, not a diagnostic product. No clinical validity or generalization to external cohorts has been established.

## Research motivation

The base paper reports high trial-level classification results, but its available description does not establish whether the test data were separated by participant. This leaves the paper's participant-level generalization unclear; it does **not** prove that leakage occurred. Prior studies have already reported participant-held-out results on this cohort, so simply repeating a subject-independent evaluation is not, by itself, a confirmed novel contribution.

The project's original broad aim was three-class classification of ADHD, ADD, and healthy controls (HC), with a possible research-support prototype. The specific scientific question, target-label provenance, evaluation protocol, and prototype design have not been approved. The intended audience and claims therefore remain research-oriented, not clinical.

## Current status and research decision

**Status: evidence review and implementation groundwork; the proposed benchmark has not been approved for execution.**

- The tracked Sprint 2 contract has not passed its decision gate. It allows preserving the supplied samples but does not approve signal-changing preprocessing, model training, or an evaluation split.
- A current **untracked** decision draft in this workspace proposes a conditional, reproducible evaluation of the Dresden cohort, potentially using compact EEGNet/CNN baselines and trial-random versus participant-held-out comparisons. It describes the novelty case as weak unless the work resolves meaningful provenance/comparability problems and reports an auditable participant-level estimate. This is a working proposal—not team approval, a frozen protocol, or a result—and the draft is not part of the tracked repository.
- Earlier plans and the root-level master guide propose a Transformer/Conformer, filtering, ICA, theta/beta features, or topographic explanations. These are not approved study decisions and should not be read as implemented functionality.
- The literature review finds that participant-held-out evaluation on this cohort is not new: Vahid et al. (2019) and Wang et al. (2022) report such evaluations. Whether a carefully specified reproduction could make a useful contribution remains conditional, not a confirmed novelty claim.

Numbers reported in papers are **not results reproduced by this repository**. In particular, the accuracy figures reported by He et al. are not results of this codebase.

## What has been completed

- The local dataset structure and metadata have been investigated without changing the source arrays.
- A trial/subject mapping report and later audits document structural evidence, candidate joins, and unresolved label provenance.
- The project has a detailed audit of the base paper, a critical ADHD-EEG literature review, an 82-paper dataset-evaluation catalog, and a focused review of reproducibility claims on the Dresden cohort.
- A Python data-access layer reads individual trials and bounded batches from HDF5 files and can write a metadata-only trial manifest.
- A configurable preprocessing implementation and synthetic-data tests exist. The preprocessing path has **not** been run on the cohort, and code availability does not authorize its use in the proposed study.
- There is no classifier, model architecture implementation, training loop, fold generator, evaluation pipeline, trained model, or project result artifact.

## Dataset and provenance

The dataset is associated with OSF project [6594x](https://osf.io/6594x/) and the Dresden cohort used by the base paper. The local `dataset/` directory contains the following files, but `.gitignore` excludes it from Git; dataset binaries are not included in a normal repository clone.

| Files | Documented contents |
|---|---|
| `d1.mat`–`d6.mat` | Each has 5,000 trials |
| `d7.mat` | 3,902 trials |
| `y_stim.mat` | Metadata array with shape `(4, 33902)` |
| `sub_name_stim.mat` | Three filename blocks containing 44, 52, and 48 entries |
| `chan.mat` | Channel metadata with 60 entries |

The EEG arrays are documented as `(385, 56, trials)`, `float32`; across the seven files they contain 33,902 trials. The paper reports 144 children (44 HC, 52 ADD, 48 ADHD), 56 channels, 256 Hz, and nominal 1.5-second trials. These are paper-reported cohort facts, not independent proof that every local trial is joined to a diagnosis correctly.

Important caveats:

- `y_stim` row 0 is interpreted as a within-block subject index. Rows 1–3 are one-hot block indicators in the inspected data; they are not independent per-trial stimulus labels and must not be treated as diagnostic labels. A task or stimulus label is not a diagnosis.
- Filename prefixes (`controls`, `subtype1`, `subtype2`), block sizes, and paper-reported group counts support a candidate HC/ADD/ADHD mapping. The files do not provide an independently keyed clinical diagnosis for every trial; independent confirmation of the diagnosis-label provenance remains a hard blocker.
- The audit reports disagree about the strength of the file-order join. A Sprint 1 report cites boundary signal correlations as validating sequential ordering, while the stricter preprocessing audit notes that the EEG arrays have no embedded trial IDs or independent join key. Filename ordinal-to-subject matching is also not independently keyed. Treat the crosswalk as provisional until this evidence is reconciled.
- The 385th sample position has no established physical interpretation. Do not crop it or infer a precise duration from the tensor shape.
- `chan.mat` has 60 metadata entries while the signals have 56 columns. The channel-column mapping and four unmatched entries are unknown; electrode-specific claims and topomaps are not justified.
- Units, reference, and the local files' upstream signal-processing history are not established. The paper reports ICA, but that does not verify how these local arrays were processed.

See the [preprocessing audit](docs/ADHD_EEG_PREPROCESSING_AUDIT.md), [trial/subject mapping validation](docs/trial_subject_mapping_validation.md), and [Sprint 2 data contract](docs/SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md) for evidence and status details.

## Research questions and novelty assessment

The project has considered reproducing He et al., evaluating performance on unseen participants, and proposing alternative architectures or EEG features. None is a frozen research protocol. Current evidence changes the framing:

- Vahid et al. (2019) already report participant-held-out evaluation on this cohort.
- Wang et al. (2022) report both random-trial and leave-one-subject-out results, so the contrast between those split types is also present in prior work.
- He et al. (2023) report EEG-Transformer results, but the reviewed paper description does not establish its split unit. The repository therefore treats participant overlap as **unknown**, not proven.
- Later high-accuracy claims in the literature need cohort and split verification before they can be used as evidence for or against a specific novelty claim.

The defensible open question is whether a transparent, comparable re-evaluation can resolve material data/provenance or reporting issues and provide an auditable participant-level estimate with uncertainty. Whether that is sufficiently novel and valuable remains undecided. The project has not established clinical usefulness, diagnostic validity, or external generalization.

## Repository contents

| Path | Purpose and status |
|---|---|
| `src/adhd_eeg/data_access.py` | Bounded, read-only HDF5 trial access and candidate trial metadata. Its default crosswalk encodes documented file/group ranges; it does not independently verify diagnosis labels. |
| `src/adhd_eeg/preprocessing.py` | Configurable band-pass (0.5–45 Hz), 50 Hz notch, and per-subject ICA path with externally reviewed component exclusions. This is implemented software, but it is not approved or run for the proposed study. |
| `src/adhd_eeg/preprocess_cli.py` | CLI requiring explicit JSON configuration and per-subject ICA decisions. No cohort configuration/review files are supplied. |
| `tests/` | Unit tests using synthetic HDF5/EEG data; they do not establish that the real-data crosswalk or scientific decisions are correct. |
| `scripts/` | Dataset inspection, metadata investigation, and mapping-validation utilities. Some are legacy exploratory scripts with machine-specific paths. The mapping-validation script makes stronger ordering claims than the strict audit accepts. |
| `docs/` | Research evidence, audits, contracts, literature reviews, methodology proposals, and sprint records. See the [documentation index](docs/README.md). |
| `docs/architecture-preparation/` | Six audits covering project state, data contract, paper evidence, preprocessing, evaluation, and explainability. |
| `docs/literature-review/` | Critical review, dataset-evaluation catalog, and Dresden-specific reproducibility memo. |
| `papers/` | The base paper PDF and a short folder README. |
| `dataset/` | Local source MAT files, ignored by Git and not shipped with the repository. |
| `adhd-eeg-transformer-master-guide.md` | Historical architecture and preprocessing proposal; not an approved protocol. |
| `pyproject.toml` | Python package metadata; Python `>=3.10`, with `h5py` and `numpy` required and MNE available as a preprocessing extra. |

There are no notebooks, checked-in model/training/evaluation packages, experiment outputs, or configuration directory in the current repository.

## Existing implementation and tests

Install the base package and run the data-access tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_data_access.py"
```

The preprocessing tests require the optional MNE dependency:

```bash
python -m pip install -e ".[preprocessing]"
python -m unittest discover -s tests -p "test_preprocessing.py"
```

The preprocessing CLI is an interface to the implemented gated path, not an end-to-end study runner. It requires a complete filter/ICA JSON configuration and reviewed ICA exclusions for every subject, neither of which is supplied here. Even with those inputs, study approval, label provenance, and the data-contract gate must be resolved before processing cohort EEG. The optional preprocessing code also performs ICA before any evaluation split and does not enforce fold-local fitting, which is an additional limitation for any future benchmark.

## Blockers and limitations

1. **Clinical-label provenance:** independently confirm the relationship between local trial indices, participant identifiers, filename blocks, and clinical diagnosis.
2. **Dataset provenance:** reconcile the file-order evidence and establish the upstream signal-processing state, channel mapping, units/reference, and meaning of the 385th position where possible.
3. **Research approval and novelty:** decide whether a reproducible cohort evaluation addresses a meaningful gap beyond existing participant-held-out studies. Do not treat the current untracked draft as approval.
4. **Evaluation contract:** approve the estimand, participant-level split, validation strategy, metrics, uncertainty reporting, and any trial-random control before implementation.
5. **Preprocessing:** decide whether any signal transformation is scientifically justified. The tracked Sprint 2 contract currently approves no signal-changing transformation; the existing ICA/filter code is not an approved substitute for that decision.
6. **Scope and claims:** no clinical validation, independent external-cohort validation, or diagnostic deployment is available.

## Next justified steps

Proceed in this order; do not begin model training or cohort preprocessing first:

1. Resolve the clinical-label and EEG-to-metadata provenance blockers, and reconcile the conflicting file-order assessments.
2. Decide whether the conditional Dresden-cohort evaluation has a sufficiently distinct contribution in light of existing participant-held-out results.
3. If approved, freeze and document the research question, split unit, estimand, model comparisons, metrics, uncertainty analysis, and data contract.
4. Separately decide whether preprocessing is needed and approve exact parameters, artifact-review procedures, and train/test isolation. Preserve the 385 samples unless evidence justifies another treatment.
5. Only after those gates pass, implement and test the approved evaluation workflow, then run it and report results with the evidence and limitations.

## Key papers and evidence

- He et al. (2023), “Classification of attention deficit/hyperactivity disorder based on EEG signals using a EEG-Transformer model,” *Journal of Neural Engineering* 20 (2023) 056013: [local PDF](papers/base-paper.pdf), [repository paper analysis](docs/base-paper-analysis.md), and [DOI](https://doi.org/10.1088/1741-2552/acf7f5). Its reported metrics are not reproduced here.
- Vahid et al. (2019) and Wang et al. (2022): prior Dresden-cohort work central to the participant-level evaluation and novelty assessment; see the [critical literature review](docs/literature-review/ADHD_EEG_Literature_Review.md).
- The [Dresden reproducibility evidence memo](<docs/literature-review/Elicit - Reproducible ADHDADD classification on the Dresden EEG cohort evidence and research gaps.md>) and [82-paper dataset-evaluation catalog](docs/literature-review/ADHD_EEG_Dataset_Evaluation.md) summarize the repository's literature checks and their evidence limits.

## Further documentation

- [Project scope and system specification](docs/PROJECT_SCOPE_AND_SYSETEM_SPECIFICATION.md)
- [Sprint map](docs/PROJECT-SPRINT-MAP.md)
- [Sprint 2 decision analysis](docs/SPRINT_2_DECISION_ANALYSIS.md)
- [Sprint 2 preprocessing and data contract](docs/SPRINT_2_PREPROCESSING_AND_DATA_CONTRACT.md)
- [Preprocessing audit](docs/ADHD_EEG_PREPROCESSING_AUDIT.md) and [earlier preprocessing audit](docs/dataset-preprocessing-audit.md)
- [Dataset investigation](docs/dataset-investigation.md) and [trial/subject mapping validation](docs/trial_subject_mapping_validation.md)
- [ML methodology and research-gap audit](docs/ML_METHODOLOGY_AND_RESEARCH_GAP_AUDIT.md)
- [ML execution plan](docs/ML_EXECUTION_PLAN.md) — planning only; not evidence that planned work has run
- Architecture-preparation audits: [project state](docs/architecture-preparation/01_PROJECT_STATE_AUDIT.md), [dataset and data contract](docs/architecture-preparation/02_DATASET_AND_DATA_CONTRACT_AUDIT.md), [base paper and research evidence](docs/architecture-preparation/03_BASE_PAPER_AND_RESEARCH_EVIDENCE_AUDIT.md), [preprocessing decisions](docs/architecture-preparation/04_PREPROCESSING_DECISION_AUDIT.md), [research design and evaluation](docs/architecture-preparation/05_RESEARCH_DESIGN_AND_EVALUATION_AUDIT.md), and [explainability and outputs](docs/architecture-preparation/06_EXPLAINABILITY_AND_OUTPUT_AUDIT.md)
- [Literature review](docs/literature-review/ADHD_EEG_Literature_Review.md)
