# Phase 4.4: Persistent analysis and explainability

## Architecture and workflow

Authenticated routes in `api/app/persistent_analysis_routes.py` orchestrate the Phase 4 persistence boundary. The existing `get_authenticated_user` dependency verifies the bearer token and resolves the internal application user ID. The route first loads a recording with `RecordingRepository.get_owned`; only after that succeeds does it retrieve the opaque file reference via `FileStorageProvider`. CSV and MAT payloads are rehydrated by the existing Phase 2 `read_csv` and `read_mat` readers. The existing Phase 3 `AnalysisService` then performs compatibility validation, calls the already initialized startup predictor, and optionally calls the existing Integrated Gradients implementation. No checkpoint is loaded on a request.

The analysis service remains independent of MongoDB and file storage. Routes coordinate application services and repository contracts; MongoDB adapters implement those contracts. Owner IDs always come from the authenticated identity.

## Repository operations

`AnalysisRepository` and `ExplanationRepository` inherit generic create/get/update/delete and provide owner scoped retrieval/listing. Analyses support `get_owned`, `list_owned`, and `list_for_recording`; explanations support `get_owned`, `list_owned`, and `list_for_analysis`. MongoDB uses the existing async client and connection lifecycle. Indexes cover owner/creation time and recording/owner for analyses, plus owner/creation time and analysis relationship for explanations. MongoDB's built in unique `_id` index covers entity IDs.

## Stored schemas

Analysis documents use the existing domain `Analysis` entity: `_id` (analysis ID), `owner_id`, `recording_id`, `created_at`, `completed_at`, `status`, optional `error`, and `result`. `result` preserves the Phase 3 structured prediction, probabilities when produced, threshold, model/checkpoint artifact type and processing metadata, and scientific limitations. The analysis result stores explanation availability metadata but omits attribution arrays to avoid duplication.

Explanation documents use `Explanation`: `_id`, `owner_id`, `analysis_id`, `created_at`, and `content`. Content contains recording ID, method, integration steps, availability status, metadata, and trial attribution data. Current attribution payloads are stored in MongoDB; if they exceed document limits, a future provider can store them via the existing storage abstraction and retain a reference. MongoDB and file writes are not atomic.

## API endpoints

- `POST /api/v2/analyses/persistent/recordings/{recording_id}` with `{"include_explanation": true|false}` creates an analysis from an owned persistent recording.
- `GET /api/v2/analyses/persistent` lists the authenticated user's analyses. Optional `recording_id`, `limit`, and `skip` filters are supported.
- `GET /api/v2/analyses/persistent/{analysis_id}` returns a historical result and associated explanation trials when available. It never runs inference or loads a checkpoint.
- `GET /api/v2/analyses/persistent/{analysis_id}/explanation` retrieves stored explanation metadata and attribution; missing analysis/explanation returns 404.

Existing `/api/v2/analyses/{recording_id}` in-memory Phase 3 routes and `/api/v1/analyze` are unchanged. All new operations require Phase 4.2 authentication. Cross-owner recording, analysis, and explanation lookups return 404.

## Scientific provenance and limitations

The reused Phase 3 pipeline preserves 256 Hz sampling, two channel slots (physical electrode identities remain unverified), four-second epochs, one-second edge trimming, zero-phase order-4 Butterworth 0.5–40 Hz filtering, disabled notch filtering, 40 handcrafted features, Control=0/ADHD=1, and threshold 0.5. Checkpoint artifact type is `final_cell1_transformer_handcrafted_fusion`; training used 553 Cell 1 epochs. Participant independent generalization and independent test evaluation have not been established. This is not a clinical diagnostic tool, probabilities are not calibrated diagnostic confidence, and Integrated Gradients are exploratory attribution rather than causal, biological, or diagnostic evidence.

## Error handling and consistency

Missing or foreign recording, missing file, unsupported or incompatible recording, unavailable predictor, inference failure, persistence failure, and missing historical records are surfaced with appropriate HTTP errors. Phase 3 already treats explanation generation failure as unavailable while retaining prediction. If explanation persistence fails after prediction persistence, the route retains the analysis and attempts to update its explanation status; this is compensating behavior, not a transaction. Analysis persistence failure returns 503. Historical reads only access stored repositories and remain available without a predictor when persistence dependencies are connected.

## Tests

Run focused Phase 4.4 tests with `pytest api/tests/test_phase4_4_analysis.py`. Repository tests use fake collections and require neither MongoDB nor Firebase credentials. Existing API and model adapter tests can be run with `pytest api/tests api/tests/model_adapter`; Phase 1–3 tests are under `tests/`. Real MongoDB, real Firebase, and real-checkpoint execution require configured credentials/services and are not represented by mocked tests.

## Known limitations and setup

Persistent analysis requires the Phase 4 MongoDB configuration and the configured recording file storage provider. Current storage configuration supports local files. No MongoDB/file atomic transaction is claimed. Attribution payload size may require an externalized reference for unusually large results. Phase 3 scientific limitations remain in force.
