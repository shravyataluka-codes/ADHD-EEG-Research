# Sprint 2 — External Primary Provenance Recovery

**Scope:** Read-only search of OSF, publication records, and public code sources
**Retrieved:** 2026-11-05
**Status:** Primary-source investigation; no participant or trial mapping is promoted to verified

## 1. Executive findings

The public OSF project was accessible, and its paginated storage listing and original `readme.txt` were retrieved. The README says the sample data was split into seven files because a single file has a 5 GB limit. The current OSF inventory contains `d1.mat`–`d7.mat`, `y_stim.mat`, `sub_name_stim.mat`, `chan.mat`, and `readme.txt`; the API exposes file sizes, version numbers, and SHA-256 metadata. This identifies the public artifacts and explains the file split, but says nothing about the order in which trials were written, metadata columns were aligned, or participants were indexed.

No original MATLAB/Octave construction script, trial-order key, filename-to-participant crosswalk, participant-level diagnosis record, field dictionary for `y_stim`, or 56-to-60 channel map was found in the OSF inventory or targeted public-code searches. One public downstream MATLAB script was found; it applies the same assumed sequential file/metadata join and verifies the total count, so it is not independent provenance.

The publisher landing page for the associated EEG-Transformer paper was accessed, but only its abstract and access notice were available. The project’s local paper analysis summarizes relevant methods and dimensions; those details are therefore not independently re-read from the full primary article during this investigation. The 385-position meaning remains unexplained by the accessible external sources.

**Conclusion:** OSF resolves the source location, the existence of the seven data chunks, and the stated reason for splitting. It does not resolve any identity-bearing join or semantic blocker. No evidence found here disproves the candidate sequential mapping; it remains unverified because the required independent evidence was unavailable.

**Terminology boundary:** `(source_filename, within_file_trial_index)` is a source-local trial locator. A `candidate_participant_id` or `candidate_diagnostic_group` is an unverified mapping, not a verified identity or clinical label. Verified diagnosis requires an authoritative participant-level diagnosis source linked to the local trial locators. `y_stim` rows 1–3 do not establish clinical targets.

## 2. Sources investigated

| Source | Type and access | What was actually accessed |
|---|---|---|
| [OSF project `6594x`](https://api.osf.io/v2/nodes/6594x/) | Primary project record | API response identifies the public project as “Deep learning EEG”; its description is empty. The node reports a 2020 modification date. |
| [OSF storage providers](https://api.osf.io/v2/nodes/6594x/files/) and [data-folder listing](https://api.osf.io/v2/nodes/6594x/files/osfstorage/5c8b82b380f5d2001681e01b/) | Primary file metadata | Followed API pagination to enumerate the data-folder contents. Metadata includes names, paths, sizes, `current_version`, timestamps, and hashes. The API exposes one storage provider and the listed folder/files. |
| [OSF `readme.txt` metadata](https://api.osf.io/v2/files/5c8cfa7718184e0019c06da4/) and [download](https://osf.io/download/mvq92/) | Primary project documentation | Retrieved the 104-byte README text; OSF metadata gives version 1 and SHA-256 `13b77ab258f77b291e45c4735ab6a3769f44a286cf76b09027b2b36b614711f3`. |
| [OSF contributors API](https://api.osf.io/v2/nodes/6594x/contributors/) and [wiki API](https://api.osf.io/v2/nodes/6594x/wikis/) | Primary project records | Contributor response lists one project contributor with `admin` permission. The wiki list is empty. Admin permission does not establish that person as the data custodian. |
| [IOP publisher landing page](https://iopscience.iop.org/article/10.1088/1741-2552/acf7f5) | Primary publication page | The page provides the article citation, author listing, abstract, and an access notice. Full article text was not accessible. A direct PDF request was blocked by the publisher’s bot-verification page. |
| [Crossref DOI record](https://api.crossref.org/works/10.1088/1741-2552/acf7f5), [OpenAlex record](https://api.openalex.org/works/https://doi.org/10.1088/1741-2552/acf7f5), and [PubMed record](https://pubmed.ncbi.nlm.nih.gov/37683665/) | Secondary bibliographic records / access attempts | Crossref metadata and abstract were accessible; OpenAlex reports no open-access full text or repository copy. PubMed’s page required cookies and did not load. These records do not supply dataset joins. |
| [GitHub code search for `sub_name_stim.mat`](https://github.com/search?q=%22sub_name_stim.mat%22&type=code) and [repository search](https://github.com/search?q=Dresden+EEG+ADHD&type=repositories) | Public code/repository search | Searches did not identify original dataset-generation code. One downstream implementation was inspected; it is not presented as an original source. |
| [Downstream MATLAB implementation](https://github.com/sibghatece/adhd-eeg-leakage-benchmark/blob/25583f8f2e9b2dfd03e8b7bf7a595d4772c9e983/stage1d_osf_vahid_epochs.m) | Secondary, non-authoritative code | Inspected the file’s comments and logic that interpret `y_stim`, sort `d*.mat`, concatenate trial positions, and compare the total count. It provides no independent identity key or original generation procedure. |

The OSF API was queried live on the retrieval date. The listed hashes and versions identify the current API records; local dataset hashes were **not** compared in this investigation.

## 3. Primary artifacts found

### OSF file inventory

The paginated OSF storage API lists these 11 items in `/data/`:

- EEG arrays: `d1.mat` through `d7.mat`
- Metadata: `y_stim.mat`, `sub_name_stim.mat`, `chan.mat`
- Documentation: `readme.txt`

Each listed file has `current_version: 1`; the API supplies file sizes and SHA-256 values. The EEG files are each listed as distinct OSF objects. Those file identifiers and hashes are useful for later source-version verification, but do not encode trial identity or order. The API’s listing order is not evidence of acquisition, concatenation, or trial order.

### Original OSF README

The downloaded 104-byte README states that the sample data was “split into several files (7)” because one file can be no larger than 5 GB. This is **primary evidence for the stated reason and number of chunks**. It does not identify a source array, split axis, per-file offsets, the order of samples within files, or whether metadata was split/reordered in lockstep.

### OSF project records

The project API identifies a public node titled “Deep learning EEG”; the description is empty. Its wiki endpoint returns no pages. The contributor endpoint lists one contributor with project-admin permission. These records identify an available project contact route, but do not establish authorship of the data, custody of participant records, or responsibility for the original preprocessing.

### Publication and public code

The IOP page confirms the article identity and exposes its abstract, but its full text was inaccessible. The local paper analysis reports that the paper describes 144 children, clinical grouping, 33,902 trials, 56 channels, 256 Hz, 1.5-second experiments, and an architecture input of `(None, 385, 56)`. Those are **secondary repository summaries of the primary article**, not details independently verified against the full external article in this session. The accessible publisher abstract does not explain the sample dimension or dataset-file joins.

The only relevant public code-search hit inspected was a downstream preprocessing script. It reads `y_stim`, interprets its columns as group indicators, sorts `d*.mat`, iterates in presumed global order, and asserts the resulting row count equals the number of metadata rows. A matching total checks cardinality only; the script has no independent trial key. Its comments calling interpretations “verified” do not provide source evidence.

## 4. Evidence extracted and blocker status

| Provenance question | Exact source and evidence | Source status | Blocker result |
|---|---|---|---|
| **Why are there seven EEG files?** | OSF [`readme.txt`](https://osf.io/download/mvq92/) states seven files were used because of a 5 GB per-file limit. | **Primary; direct for stated chunking rationale.** | **Partially resolved.** The reason and count of chunks are documented. The procedure, data order, offsets, and metadata coupling are not. |
| **What are the public source files and versions?** | OSF [data-folder API](https://api.osf.io/v2/nodes/6594x/files/osfstorage/5c8b82b380f5d2001681e01b/) enumerates the seven EEG files and three MAT metadata files, with version/size/hash metadata; OSF README metadata identifies its current version and hash. | **Primary; direct for current inventory/metadata.** | **Partially resolved.** Current OSF artifacts can be identified for a future hash comparison. Local files were not checked against those hashes, and the metadata does not establish trial identity. |
| **What is the EEG-file/trial order relative to `y_stim` columns?** | The README only describes splitting files. Neither the OSF file metadata nor the accessible publication abstract supplies offsets, trial IDs, or a full crosswalk. The downstream [MATLAB script](https://github.com/sibghatece/adhd-eeg-leakage-benchmark/blob/25583f8f2e9b2dfd03e8b7bf7a595d4772c9e983/stage1d_osf_vahid_epochs.m) imposes an order and asserts only total row-count equality. | README/API are **primary but insufficient**; downstream script is **secondary and candidate-dependent**. | **Unresolved.** No independent evidence establishes even one EEG locator’s corresponding `y_stim` column. The candidate numeric-file-order join is neither confirmed nor disproved. |
| **What are participant identities and filename ordering?** | OSF provides `sub_name_stim.mat`; no participant key, filename-order algorithm, or filename-to-row-0 crosswalk appears in the README or file metadata. Existing local inspection reports filename blocks and row-0 runs, but those structural observations do not key a person. | OSF artifact is **primary binary data**, but its listing is **not semantic documentation**; local structural reports are secondary to this external search. | **Unresolved.** No authoritative source links a filename position or row-0 run to a stable participant identifier. |
| **Are participant-level clinical diagnoses available?** | OSF inventory lists no separate diagnosis table or clinical-record file. The local [paper analysis](base-paper-analysis.md#4-dataset) summarizes aggregate cohort counts and the reported clinical assessment process, but contains no participant-keyed diagnosis crosswalk. | Inventory is **primary for listed files**; paper summary is **secondary** for article statements. | **Unresolved.** Aggregate cohort descriptions do not assign a diagnosis to any local participant or trial. Absence of a separate listed file does not prove that records do not exist elsewhere. |
| **What do `y_stim` rows 1–3 mean?** | The OSF README gives only the file-splitting rationale; the API gives only file metadata. No external data dictionary or original construction code was found. The downstream script assigns group meanings but supplies no independent supporting source. | **Unresolved**; downstream interpretation is **secondary/candidate**. | **Unresolved.** Do not treat the one-hot patterns as clinical labels or assume their intended field semantics. |
| **What does the 385th EEG position represent?** | The external publisher abstract and OSF README do not define it. The local [paper analysis](base-paper-analysis.md#4-dataset) reports 1.5 seconds at 256 Hz and a model input shape with 385 positions, while noting that the paper does not explain the extra position. | External primary evidence **not obtained** for this detail; local source is **secondary**. | **Unresolved.** No endpoint, padding, baseline, or sample-index interpretation is established. |
| **How do 56 EEG columns map to 60 `chan.mat` entries?** | OSF lists `chan.mat` but provides no channel-order dictionary in the README/API. The local [preprocessing audit](ADHD_EEG_PREPROCESSING_AUDIT.md#3-raw-dataset-inventory) records 56 signal columns and 60 channel entries and says their correspondence is unknown. | OSF artifact inventory is **primary**; local counts are **secondary reports of binary inspection**. | **Unresolved.** No primary source maps signal columns to named entries or explains the four-entry discrepancy. |
| **What original scripts or supplementary materials exist?** | The current OSF storage inventory contains only the listed MAT files and README; its wiki is empty. Targeted GitHub repository/code searches found no original generator or supplementary source. The one public MATLAB hit is a downstream implementation, not a generator. | **Primary for current OSF inventory; bounded search result elsewhere.** | **Unresolved.** No original MATLAB/Octave construction script, full data dictionary, source archive, trial log, or authoritative crosswalk was found. Search failure is not proof that none exists outside the searched sources. |

## 5. Evidence limitations

1. **Current OSF snapshot:** The API describes the project as currently served on 2026-11-05. It is not a preserved historical snapshot of what was originally downloaded by this repository.
2. **Hash scope:** OSF returns per-file hashes, but this task did not download or hash-compare the local MAT files. Matching a local file hash would confirm file identity, not trial order or labels.
3. **MAT contents:** The OSF metadata listing was inspected, but the large EEG files and metadata MAT binaries were not downloaded for a fresh content audit. The prior repository reports remain secondary evidence about their local structures.
4. **Publication access:** IOP exposed the publisher landing page and abstract but gated the full text; the PDF request reached a bot-verification page. The publisher abstract has no trial-level data mapping. The full-text details attributed above to the article are explicitly attributed to the repository’s paper summary.
5. **Search coverage:** GitHub code/repository searches and the cited OSF, publisher, and bibliographic endpoints are not an exhaustive search of all institutional archives or private repositories.
6. **Unavailable is not disproven:** No evidence obtained here disproves the candidate mappings. Equally, none supplies the independent evidence required to verify them.

## 6. Remaining blockers and contact recommendation

| Blocker | Status |
|---|---|
| Public OSF project identity and current artifact inventory | **Resolved** for the live listing and listed file metadata |
| Reason the EEG data are split into seven chunks | **Partially resolved** by the original README |
| Local raw-file identity versus current OSF versions | **Unresolved**; compare hashes before relying on local provenance |
| Original EEG construction/splitting procedure and trial order | **Unresolved** |
| Complete `(source file, within-file trial index) → y_stim column` mapping | **Unresolved** |
| Stable participant ID ↔ metadata run ↔ source filename ordering | **Unresolved** |
| Participant-level clinical diagnosis/group crosswalk | **Unresolved** |
| `y_stim` rows 1–3 semantics | **Unresolved** |
| Meaning of the 385th sample | **Unresolved** |
| Mapping of 56 signal columns to 60 channel entries | **Unresolved** |

**Contact is necessary.** The public materials found do not provide the keys needed for participant-independent evaluation. The OSF contributor API lists a project administrator, which is a reasonable first route to request the data custodian’s identity and the source-generation materials; it does not prove that the administrator is the custodian. The original dataset team and the paper’s corresponding authors should be approached if the OSF contact cannot provide authoritative answers.

Request the original generator or a signed/dated custodian description of its exact behavior; a complete trial-locator-to-metadata-column crosswalk; the original participant/filename ordering key; an identity-keyed diagnostic group table or approved pseudonymous equivalent; a `y_stim` data dictionary; the 385-position definition; the 56-to-60 channel mapping; and hashes/version identifiers for the source files used. Avoid requesting or publishing direct identifiers or unnecessary sensitive clinical details.

## 7. Recommendation for the next Sprint 2 action

Request the missing provenance package from the OSF project administrator and original research team. Preserve any response and artifacts with retrieval date, source/version, and hashes, then independently reconcile coverage and uniqueness for every EEG trial, metadata column, participant key, and diagnostic label. First compare OSF hashes with the local raw files without changing them. If authoritative trial and participant keys cannot be obtained, record the dataset as blocked for real participant-independent evaluation and do not manufacture a crosswalk from counts, filenames, correlations, or row values.

No split generator, model, preprocessing, or data transformation is authorized by this recommendation.

## 8. Repository files inspected

- [docs/SPRINT_2_EVIDENCE_RECONCILIATION_AUDIT.md](SPRINT_2_EVIDENCE_RECONCILIATION_AUDIT.md)
- [docs/SPRINT_2_PROVENANCE_EVIDENCE_INVESTIGATION.md](SPRINT_2_PROVENANCE_EVIDENCE_INVESTIGATION.md)
- [docs/base-paper-analysis.md](base-paper-analysis.md)
- [docs/ADHD_EEG_PREPROCESSING_AUDIT.md](ADHD_EEG_PREPROCESSING_AUDIT.md)
- [README.md](../README.md)
- [papers/README.md](../papers/README.md)
- [src/adhd_eeg/data_access.py](../src/adhd_eeg/data_access.py)
- [scripts/validate_trial_subject_mapping.py](../scripts/validate_trial_subject_mapping.py)

No source code or dataset file was modified. No local dataset hash comparison, preprocessing, split generation, model training, staging, or commit was performed.
