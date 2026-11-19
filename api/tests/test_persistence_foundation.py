"""Focused, credential-free tests for the Stage 4.1 persistence foundation."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import anyio

from api.app.domain.entities import Recording
from api.app.infrastructure.mongodb.database import DatabaseUnavailableError, MongoDatabase
from api.app.infrastructure.mongodb.repositories import MongoRecordingRepository
from api.app.repositories.contracts import RecordingRepository
from api.app.settings import Settings


def test_mongodb_settings_validate_uri_and_required_mode(monkeypatch):
    with pytest.raises(ValueError, match="MONGODB_URI"):
        Settings(mongodb_uri="https://example.com")
    with pytest.raises(ValueError, match="required"):
        Settings(mongodb_required=True)
    monkeypatch.setenv("MONGODB_URI", "mongodb+srv://example.invalid")
    monkeypatch.setenv("MONGODB_DATABASE", "research")
    configured = Settings.from_env()
    assert configured.mongodb_uri == "mongodb+srv://example.invalid"
    assert configured.mongodb_database == "research"


def test_repository_contract_is_runtime_checkable_by_shape():
    class Stub:
        async def create(self, entity): ...
        async def get(self, entity_id): ...
        async def update(self, entity_id, changes): ...
        async def delete(self, entity_id): ...
        async def get_owned(self, recording_id, owner_id): ...
        async def list_owned(self, owner_id, *, limit=100, skip=0): ...

    # Protocol contracts remain usable for structural typing without inheritance.
    assert isinstance(Stub(), RecordingRepository)
    assert all(hasattr(Stub, method) for method in ("get_owned", "list_owned", "create"))


def test_database_connect_initializes_once_indexes_and_closes():
    async def scenario():
        class FakeCollection:
            async def create_index(self, *args, **kwargs):
                return "index"

        class FakeDatabase:
            def __getitem__(self, _name):
                return FakeCollection()

            users = recordings = analyses = explanations = FakeCollection()

        class FakeAdmin:
            calls = 0

            async def command(self, _command):
                self.calls += 1
                return {"ok": 1}

        class FakeClient:
            def __init__(self):
                self.admin = FakeAdmin()
                self.closed = 0

            def __getitem__(self, _name):
                return database

            async def close(self):
                self.closed += 1

        database = FakeDatabase()
        client = FakeClient()
        db = MongoDatabase("mongodb://localhost", "drushti_detect", client_factory=lambda *a, **k: client)
        first = await db.connect()
        second = await db.connect()
        assert first is second is database
        assert client.admin.calls == 1
        assert await db.health_check()
        assert client.admin.calls == 2
        await db.close()
        assert client.closed == 1
        assert db.client is db.database is None

    anyio.run(scenario, backend="asyncio")


def test_database_connection_failure_closes_and_has_clear_error():
    async def scenario():
        client = MagicMock()
        client.admin.command = AsyncMock(side_effect=OSError("offline"))
        client.close = AsyncMock()
        db = MongoDatabase("mongodb://localhost", "drushti_detect", client_factory=lambda *a, **k: client)
        with pytest.raises(DatabaseUnavailableError, match="Atlas IP access rules"):
            await db.connect()
        client.close.assert_awaited_once()
        assert db.client is None

    anyio.run(scenario, backend="asyncio")


def test_mongo_recording_repository_crud_and_owner_scoping():
    async def scenario():
        collection = MagicMock()
        collection.insert_one = AsyncMock()
        collection.find_one = AsyncMock(return_value={
            "_id": "r1", "owner_id": "u1", "created_at": datetime.now(timezone.utc), "metadata": {}
        })
        collection.update_one = AsyncMock()
        collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
        database = MagicMock()
        database.__getitem__.return_value = collection
        repository = MongoRecordingRepository(database)
        entity = Recording("r1", "u1", datetime.now(timezone.utc))
        assert isinstance(repository, RecordingRepository)
        await repository.create(entity)
        collection.insert_one.assert_awaited_once()
        assert await repository.get_owned("r1", "u1") == entity
        assert collection.find_one.await_args.args[0] == {"_id": "r1", "owner_id": "u1"}
        await repository.update("r1", {"metadata": {"label": "x"}})
        collection.update_one.assert_awaited_once()
        assert await repository.delete("r1")

    anyio.run(scenario, backend="asyncio")


def test_index_initialization_covers_all_collections_and_relationships():
    async def scenario():
        class Collection:
            def __init__(self):
                self.calls = []

            async def create_index(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        class Database:
            pass

        database = Database()
        for name in ("users", "recordings", "analyses", "explanations"):
            setattr(database, name, Collection())
        db = MongoDatabase("mongodb://localhost", "db")
        db.database = database
        await db.ensure_indexes()
        assert database.users.calls[0][0][0] == "email"
        assert len(database.recordings.calls) == 1
        assert len(database.analyses.calls) == 2
        assert len(database.explanations.calls) == 2

    anyio.run(scenario, backend="asyncio")
