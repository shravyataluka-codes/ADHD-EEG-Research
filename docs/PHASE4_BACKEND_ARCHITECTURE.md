# Phase 4 Backend Architecture (Stage 4.1)

## Scope

Stage 4.1 establishes persistence boundaries and MongoDB infrastructure. The existing Phase 1 inference API, Phase 2 in-memory ingestion routes, and Phase 3 in-memory analyses remain unchanged in behavior. MongoDB persistence is not wired into those workflows yet.

## Boundaries and dependency direction

```text
FastAPI/routes and existing Phase 1–3 services
                 ↓ (future use)
       application persistence services
                 ↓
          repository protocols
                 ↑
       MongoDB repository adapters
                 ↓
        Async MongoDB database
```

Domain entities in `api/app/domain/` contain Python data only. Repository contracts in `api/app/repositories/` describe asynchronous create, retrieve, update, delete, and ownership-scoped operations. `PersistenceService` depends on the generic repository contract and knows no MongoDB details. The adapter in `api/app/infrastructure/mongodb/` maps domain entities to MongoDB documents and implements those contracts.

## Directory structure

- `api/app/domain/entities.py`: user, recording, analysis, and explanation records.
- `api/app/repositories/contracts.py`: typed repository protocols, including owner-aware lookups and listings.
- `api/app/application/persistence.py`: generic application-level persistence service.
- `api/app/infrastructure/mongodb/database.py`: async client lifecycle, ping, index creation, health check, and shutdown.
- `api/app/infrastructure/mongodb/repositories.py`: MongoDB CRUD and ownership-aware adapters.
- `api/app/settings.py`: environment-based MongoDB settings and validation.

## Database design

The database defaults to `drushti_detect` and prepares four collections:

- `users`: unique `email` index.
- `recordings`: owner and creation-time index.
- `analyses`: owner and creation-time index; recording lookup index.
- `explanations`: owner and creation-time index; analysis lookup index.

Identifiers are represented in the domain as strings and stored as MongoDB `_id`. Recording and analysis documents include `owner_id`; owner-aware adapter methods include that owner in every retrieval/list query. EEG signal bytes and large arrays are intentionally not persisted in MongoDB at this stage.

## Configuration and lifecycle

Copy `.env.example` to a local `.env` and provide a MongoDB Atlas URI. `.env` is ignored by Git. Set `MONGODB_REQUIRED=true` to make MongoDB configuration and startup connectivity mandatory. When a URI is configured, the FastAPI lifespan creates one async PyMongo client, pings the server, initializes indexes, and closes the client on shutdown. If required initialization fails, startup stops with a clear configuration/network message. Without a URI and with MongoDB not required, the existing app starts without MongoDB.

`MONGODB_URI` accepts `mongodb://` or `mongodb+srv://`; `MONGODB_DATABASE` defaults to `drushti_detect`. The driver is PyMongo's async API with SRV support (`pymongo[srv]`), avoiding a second/deprecated driver dependency.

## Contracts

`UserRepository` supports create/get/update/delete and lookup by email. `RecordingRepository`, `AnalysisRepository`, and `ExplanationRepository` support CRUD plus owner-scoped retrieval/listing. Analyses can be listed by owned recording, and explanations by owned analysis. These contracts are ready for dependency injection; no endpoint or complete persistence workflow uses them in Stage 4.1.

## Future integration

Stage 4.2 can introduce authentication and supply a trusted user ID to owner-aware services. A later recording-storage stage should store large EEG payloads in object storage and keep only metadata and stable storage references in MongoDB. Analysis and explanation persistence can then be integrated through the application services without making domain or route code depend on MongoDB.
