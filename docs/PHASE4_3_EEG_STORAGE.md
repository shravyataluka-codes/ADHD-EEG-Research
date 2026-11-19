# Phase 4.3: Persistent EEG recording storage

## Architecture

The Phase 2 CSV and MAT readers remain the only EEG parsing pipeline and continue to produce `EEGRecording`. `FileStorageProvider` defines save, retrieve, delete, exists, and reference operations. `LocalFileStorageProvider` implements that contract; recording management depends on the contract through `StorageService`, so another provider can be added without changing these routes or the Mongo repository contract.

The original EEG file is stored as a generated UUID key with its validated extension. MongoDB's existing `recordings` collection stores a `Recording` entity with its owner and metadata. Large EEG arrays are not put in Mongo documents. The existing `RecordingRepository` contract and `MongoRecordingRepository` adapter are used.

## Configuration

```env
STORAGE_PROVIDER=local
LOCAL_STORAGE_PATH=./data/eeg_recordings
INGESTION_MAX_UPLOAD_SIZE_BYTES=104857600
```

The storage root is created and checked during app startup. Local storage files are excluded from Git. `STORAGE_PROVIDER` currently accepts `local`; cloud credentials and S3 support are not part of Phase 4.3.

## Upload workflow and metadata

`POST /api/v2/recordings/persistent/upload` requires the Phase 4.2 Firebase Bearer authentication dependency and an internal application user mapping. It validates `.csv`/`.mat`, enforces the configured limit, calls `read_csv` or `read_mat`, then saves the original file and inserts owner-scoped metadata. A response is returned only after both writes succeed. If Mongo persistence fails after file save, the service attempts to delete the file and logs cleanup failures for recovery. Filesystem and Mongo writes are not atomic.

Metadata includes recording ID, owner ID in Mongo, original filename, generated storage reference in Mongo, format and size, sampling rate, channel/sample/trial dimensions, channel names, duration, import/validation status, provenance, and timestamps. The storage reference is omitted from API responses.

## API

- `POST /api/v2/recordings/persistent/upload` uploads a CSV or MAT recording.
- `GET /api/v2/recordings/persistent` lists the authenticated user's recordings.
- `GET /api/v2/recordings/persistent/{recording_id}` returns owned metadata.
- `GET /api/v2/recordings/persistent/{recording_id}/download` streams the owned original file.
- `POST /api/v2/recordings/persistent/{recording_id}/load` reloads an owned file through the existing Phase 2 reader into the canonical in-process `EEGRecording` representation.
- `DELETE /api/v2/recordings/persistent/{recording_id}` deletes the owned file and metadata.

Each operation resolves the internal user from the verified bearer identity. Client-supplied owner IDs are not accepted. Repository queries include the owner ID, and a missing or foreign recording returns the same 404 response. File operations happen only after the ownership query.

## Failure handling and future providers

Storage keys must match a generated-key pattern before filesystem access, and path resolution is constrained to the configured root. Files are written to a temporary file and atomically published without overwriting an existing key. Retrieval reports missing files as 404. Cleanup failures are logged with the recording ID and must be handled during operational recovery.

An S3-compatible implementation can implement `FileStorageProvider` and return an opaque reference. Configuration and provider selection can then be extended without changing recording metadata workflows. No cloud credentials or S3 integration are included here.
