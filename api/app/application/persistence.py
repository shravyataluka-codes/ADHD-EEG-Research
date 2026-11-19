"""Persistence services depend only on repository contracts."""

from typing import Any, Generic, TypeVar

from api.app.repositories.contracts import CrudRepository

T = TypeVar("T")


class PersistenceService(Generic[T]):
    def __init__(self, repository: CrudRepository[T]):
        self.repository = repository

    async def create(self, entity: T) -> T:
        return await self.repository.create(entity)

    async def get(self, entity_id: str) -> T | None:
        return await self.repository.get(entity_id)

    async def update(self, entity_id: str, changes: dict[str, Any]) -> T | None:
        return await self.repository.update(entity_id, changes)

    async def delete(self, entity_id: str) -> bool:
        return await self.repository.delete(entity_id)
