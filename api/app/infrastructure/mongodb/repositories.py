"""MongoDB implementations of the storage-independent repository contracts."""

from dataclasses import asdict, is_dataclass
from typing import Any, Generic, TypeVar

from api.app.domain.entities import Analysis, Explanation, Recording, User

T = TypeVar("T")


class MongoRepository(Generic[T]):
    collection_name = ""
    entity_type: type[T]

    def __init__(self, database):
        self.collection = database[self.collection_name]

    @staticmethod
    def _document(entity):
        data = asdict(entity) if is_dataclass(entity) else dict(entity)
        data["_id"] = data.pop("id")
        return data

    def _entity(self, document):
        if document is None:
            return None
        data = dict(document)
        data["id"] = str(data.pop("_id"))
        return self.entity_type(**data)

    async def create(self, entity: T) -> T:
        document = self._document(entity)
        await self.collection.insert_one(document)
        return entity

    async def get(self, entity_id: str) -> T | None:
        return self._entity(await self.collection.find_one({"_id": entity_id}))

    async def update(self, entity_id: str, changes: dict[str, Any]) -> T | None:
        changes = dict(changes)
        changes.pop("id", None)
        changes.pop("_id", None)
        if not changes:
            return await self.get(entity_id)
        await self.collection.update_one({"_id": entity_id}, {"$set": changes})
        return await self.get(entity_id)

    async def delete(self, entity_id: str) -> bool:
        result = await self.collection.delete_one({"_id": entity_id})
        return result.deleted_count > 0

    async def _find_many(self, query, *, limit=100, skip=0):
        cursor = self.collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        return [self._entity(doc) async for doc in cursor]


class MongoUserRepository(MongoRepository[User]):
    collection_name, entity_type = "users", User

    async def get_by_email(self, email: str):
        return self._entity(await self.collection.find_one({"email": email}))

    async def get_by_firebase_uid(self, firebase_uid: str):
        return self._entity(await self.collection.find_one({"attributes.firebase_uid": firebase_uid}))


class MongoRecordingRepository(MongoRepository[Recording]):
    collection_name, entity_type = "recordings", Recording

    async def get_owned(self, recording_id: str, owner_id: str):
        return self._entity(await self.collection.find_one({"_id": recording_id, "owner_id": owner_id}))

    async def list_owned(self, owner_id: str, *, limit=100, skip=0):
        return await self._find_many({"owner_id": owner_id}, limit=limit, skip=skip)


class MongoAnalysisRepository(MongoRepository[Analysis]):
    collection_name, entity_type = "analyses", Analysis

    async def get_owned(self, analysis_id: str, owner_id: str):
        return self._entity(await self.collection.find_one({"_id": analysis_id, "owner_id": owner_id}))

    async def list_owned(self, owner_id: str, *, limit=100, skip=0):
        return await self._find_many({"owner_id": owner_id}, limit=limit, skip=skip)

    async def list_for_recording(self, recording_id: str, owner_id: str, *, limit=100, skip=0):
        return await self._find_many({"recording_id": recording_id, "owner_id": owner_id}, limit=limit, skip=skip)


class MongoExplanationRepository(MongoRepository[Explanation]):
    collection_name, entity_type = "explanations", Explanation

    async def get_owned(self, explanation_id: str, owner_id: str):
        return self._entity(await self.collection.find_one({"_id": explanation_id, "owner_id": owner_id}))

    async def list_owned(self, owner_id: str, *, limit=100, skip=0):
        return await self._find_many({"owner_id": owner_id}, limit=limit, skip=skip)

    async def list_for_analysis(self, analysis_id: str, owner_id: str, *, limit=100, skip=0):
        return await self._find_many({"analysis_id": analysis_id, "owner_id": owner_id}, limit=limit, skip=skip)
