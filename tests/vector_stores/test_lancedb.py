from pathlib import Path

import pytest

from mem0.utils.factory import VectorStoreFactory
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.vector_stores.lancedb import LanceDB


@pytest.fixture
def lancedb_uri(tmp_path: Path) -> str:
    return str(tmp_path / "lancedb")


@pytest.fixture
def lancedb_store(lancedb_uri: str) -> LanceDB:
    return LanceDB(uri=lancedb_uri, collection_name="memories", embedding_model_dims=3)


def test_lancedb_config_and_factory(lancedb_uri: str):
    config = VectorStoreConfig(
        provider="lancedb",
        config={
            "uri": lancedb_uri,
            "collection_name": "factory_memories",
            "embedding_model_dims": 3,
        },
    )

    store = VectorStoreFactory.create("lancedb", config.config)

    assert isinstance(store, LanceDB)
    assert store.table_name == "factory_memories"
    assert "factory_memories" in store.list_cols()


def test_lancedb_crud_and_filtering(lancedb_store: LanceDB):
    lancedb_store.insert(
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        payloads=[
            {
                "data": "likes sci-fi movies",
                "user_id": "alice",
                "category": "preferences",
                "count": 1,
            },
            {
                "data": "prefers travel planning",
                "user_id": "alice",
                "agent_id": "travel-bot",
                "category": "travel",
                "count": 3,
            },
        ],
        ids=["mem-1", "mem-2"],
    )

    hits = lancedb_store.search(
        query="movies",
        vectors=[1.0, 0.0, 0.0],
        limit=5,
        filters={"user_id": "alice", "count": {"lte": 1}},
    )

    assert [hit.id for hit in hits] == ["mem-1"]
    assert hits[0].payload["category"] == "preferences"
    assert pytest.approx(hits[0].score, rel=1e-6) == 1.0

    memories = lancedb_store.list(
        filters={
            "$or": [
                {"category": {"icontains": "pref"}},
                {"agent_id": "travel-bot"},
            ]
        },
        limit=10,
    )[0]

    assert {memory.id for memory in memories} == {"mem-1", "mem-2"}

    fetched = lancedb_store.get("mem-2")
    assert fetched is not None
    assert fetched.payload["agent_id"] == "travel-bot"

    lancedb_store.update(
        "mem-2",
        vector=[1.0, 0.0, 0.0],
        payload={
            "data": "prefers city breaks",
            "user_id": "alice",
            "priority": 5,
        },
    )

    updated = lancedb_store.get("mem-2")
    assert updated is not None
    assert updated.payload == {
        "data": "prefers city breaks",
        "priority": 5,
        "user_id": "alice",
    }

    priority_matches = lancedb_store.list(filters={"priority": {"gte": 5}}, limit=10)[0]
    assert [memory.id for memory in priority_matches] == ["mem-2"]

    excluded = lancedb_store.list(filters={"$not": [{"priority": {"gte": 5}}]}, limit=10)[0]
    assert [memory.id for memory in excluded] == ["mem-1"]

    lancedb_store.delete("mem-1")
    assert lancedb_store.get("mem-1") is None


def test_lancedb_supports_keyword_search_and_batch_delete(lancedb_store: LanceDB):
    lancedb_store.insert(
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        payloads=[
            {
                "data": "TypeScript strict mode rollout checklist",
                "user_id": "alice",
                "tags": ["typescript", "strict"],
                "topic": "frontend",
            },
            {
                "data": "Chrome DevTools MCP timeout handling SOP",
                "user_id": "alice",
                "tags": ["mcp", "timeout"],
                "topic": "ops",
            },
            {
                "data": "Rust service deployment notes",
                "user_id": "bob",
                "tags": ["rust", "deploy"],
                "topic": "backend",
            },
        ],
        ids=["kw-1", "kw-2", "kw-3"],
    )

    keyword_hits = lancedb_store.search_keyword(
        "TypeScript strict",
        limit=5,
        filters={"user_id": "alice"},
    )

    assert keyword_hits
    assert keyword_hits[0].id == "kw-1"
    assert "kw-1" in [hit.id for hit in keyword_hits]
    assert keyword_hits[0].source == "keyword"
    assert keyword_hits[0].score is not None

    candidate_sets = lancedb_store.search_hybrid_candidates(
        query="timeout",
        vectors=[0.0, 1.0, 0.0],
        semantic_limit=3,
        keyword_limit=3,
        filters={"user_id": "alice"},
    )

    index_names = {str(getattr(index, "name", "") or "") for index in lancedb_store.table.list_indices()}
    assert "user_id_idx" in index_names
    assert "agent_id_idx" in index_names
    assert "run_id_idx" in index_names
    assert "search_text_idx" in index_names

    assert [hit.id for hit in candidate_sets["semantic"]] == ["kw-2", "kw-1"]
    assert candidate_sets["keyword"]
    assert candidate_sets["keyword"][0].id == "kw-2"
    assert "kw-2" in [hit.id for hit in candidate_sets["keyword"]]
    assert lancedb_store.count({"user_id": "alice"}) == 2

    deleted = lancedb_store.delete_many({"user_id": "alice"})
    assert deleted == 2
    assert lancedb_store.count({"user_id": "alice"}) == 0
    assert lancedb_store.count({"user_id": "bob"}) == 1


def test_lancedb_adds_dynamic_columns_across_batches(lancedb_store: LanceDB):
    lancedb_store.insert(
        vectors=[[1.0, 0.0, 0.0]],
        payloads=[{"data": "first memory", "topic": "science", "is_active": True}],
        ids=["first"],
    )
    lancedb_store.insert(
        vectors=[[0.0, 1.0, 0.0]],
        payloads=[{"data": "second memory", "topic": "travel", "visits": 2}],
        ids=["second"],
    )

    assert "topic" in lancedb_store.table.schema.names
    assert "is_active" in lancedb_store.table.schema.names
    assert "visits" in lancedb_store.table.schema.names

    active_matches = lancedb_store.list(filters={"is_active": True}, limit=10)[0]
    assert [memory.id for memory in active_matches] == ["first"]

    visit_matches = lancedb_store.list(filters={"visits": {"gte": 2}}, limit=10)[0]
    assert [memory.id for memory in visit_matches] == ["second"]


def test_lancedb_reset_recreates_collection(lancedb_store: LanceDB):
    lancedb_store.insert(
        vectors=[[1.0, 0.0, 0.0]],
        payloads=[{"data": "memory before reset", "user_id": "alice"}],
        ids=["before-reset"],
    )

    lancedb_store.reset()

    assert "memories" in lancedb_store.list_cols()
    assert lancedb_store.list(limit=10) == [[]]


def test_lancedb_validates_existing_dimension(lancedb_uri: str):
    LanceDB(uri=lancedb_uri, collection_name="memories", embedding_model_dims=3)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        LanceDB(uri=lancedb_uri, collection_name="memories", embedding_model_dims=4)
