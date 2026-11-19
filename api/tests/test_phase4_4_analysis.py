from datetime import datetime, timezone

import pytest

from api.app.domain.entities import Analysis, Explanation
from api.app.infrastructure.mongodb.repositories import MongoAnalysisRepository, MongoExplanationRepository


class Cursor:
    def __init__(self, docs): self.docs = docs
    def sort(self, *_): return self
    def skip(self, _): return self
    def limit(self, _): return self
    def __aiter__(self):
        async def iterate():
            for doc in self.docs:
                yield doc
        return iterate()


class Collection:
    def __init__(self): self.docs = {}
    async def insert_one(self, doc): self.docs[doc["_id"]] = doc
    async def find_one(self, query):
        doc = self.docs.get(query.get("_id"))
        return doc if doc and all(doc.get(k) == v for k, v in query.items() if k != "_id") else None
    def find(self, query): return Cursor([d for d in self.docs.values() if all(d.get(k) == v for k, v in query.items())])


class Database:
    def __init__(self): self.collections = {}
    def __getitem__(self, key): return self.collections.setdefault(key, Collection())


def test_analysis_and_explanation_repositories_are_owner_scoped():
    # Fake collection methods are immediately completing coroutines, so drive
    # this repository contract check without creating a Windows asyncio loop.
    coroutine = _repository_scenario()
    try:
        coroutine.send(None)
    except StopIteration:
        return
    raise AssertionError("Repository fake unexpectedly suspended")


async def _repository_scenario():
    db = Database()
    analyses, explanations = MongoAnalysisRepository(db), MongoExplanationRepository(db)
    now = datetime.now(timezone.utc)
    record = Analysis("a1", "u1", "r1", now, {"predicted_class": "ADHD"})
    await analyses.create(record)
    await analyses.create(Analysis("a2", "u2", "r2", now))
    assert (await analyses.get_owned("a1", "u1")).recording_id == "r1"
    assert await analyses.get_owned("a1", "u2") is None
    assert [x.id for x in await analyses.list_owned("u1")] == ["a1"]
    assert [x.id for x in await analyses.list_for_recording("r1", "u1")] == ["a1"]
    exp = Explanation("e1", "u1", "a1", now, {"method": "joint_integrated_gradients"})
    await explanations.create(exp)
    assert (await explanations.get_owned("e1", "u1")).content["method"] == "joint_integrated_gradients"
    assert await explanations.get_owned("e1", "u2") is None
    assert [x.id for x in await explanations.list_for_analysis("a1", "u1")] == ["e1"]
