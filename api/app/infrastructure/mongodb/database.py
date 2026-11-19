"""Async MongoDB client lifecycle and database readiness."""

from pymongo import AsyncMongoClient


class DatabaseUnavailableError(RuntimeError):
    """MongoDB could not be initialized or reached."""


class MongoDatabase:
    def __init__(self, uri: str, database_name: str, *, client_factory=None):
        self.uri = uri
        self.database_name = database_name
        self._client_factory = client_factory or AsyncMongoClient
        self.client = None
        self.database = None

    async def connect(self):
        if self.client is not None:
            return self.database
        try:
            self.client = self._client_factory(self.uri, serverSelectionTimeoutMS=5000)
            await self.client.admin.command("ping")
            self.database = self.client[self.database_name]
            await self.ensure_indexes()
            return self.database
        except Exception as exc:
            await self.close()
            raise DatabaseUnavailableError(
                "MongoDB initialization failed. Check MONGODB_URI, network access, "
                "and Atlas IP access rules."
            ) from exc

    async def ensure_indexes(self):
        if self.database is None:
            raise DatabaseUnavailableError("MongoDB is not connected.")
        await self.database.users.create_index(
            "email", unique=True, sparse=True, name="uq_users_email"
        )
        await self.database.users.create_index(
            "attributes.firebase_uid", unique=True, sparse=True, name="uq_users_firebase_uid"
        )
        await self.database.recordings.create_index(
            [("owner_id", 1), ("created_at", -1)], name="ix_recordings_owner_created"
        )
        await self.database.analyses.create_index(
            [("owner_id", 1), ("created_at", -1)], name="ix_analyses_owner_created"
        )
        await self.database.analyses.create_index([("recording_id", 1), ("owner_id", 1)], name="ix_analyses_recording_owner")
        await self.database.analyses.create_index("owner_id", name="ix_analyses_owner")
        await self.database.explanations.create_index(
            [("owner_id", 1), ("created_at", -1)], name="ix_explanations_owner_created"
        )
        await self.database.explanations.create_index("analysis_id", name="ix_explanations_analysis")
        await self.database.explanations.create_index("owner_id", name="ix_explanations_owner")

    async def health_check(self) -> bool:
        if self.client is None:
            return False
        try:
            await self.client.admin.command("ping")
            return True
        except Exception:
            return False

    async def close(self):
        client, self.client = self.client, None
        self.database = None
        if client is not None:
            await client.close()
