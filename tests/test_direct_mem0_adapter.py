from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

from headroom.memory.backends.direct_mem0 import DirectMem0Adapter, Mem0Config


class _FakeMem0Client:
    def __init__(self) -> None:
        self.search_kwargs: dict[str, object] | None = None

    def search(self, **kwargs):  # noqa: ANN001, ANN201
        self.search_kwargs = kwargs
        return [
            {
                "id": "memory-1",
                "memory": "opsx runtime memory marker",
                "score": 0.99,
                "metadata": {"user_id": kwargs["filters"]["user_id"]},
            }
        ]


class _EmptyMem0Client:
    def search(self, **kwargs):  # noqa: ANN001, ANN201
        del kwargs
        return {"results": []}


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.query_kwargs: dict[str, object] | None = None

    def query_points(self, **kwargs):  # noqa: ANN001, ANN201
        self.query_kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="qdrant-memory-1",
                    score=0.87,
                    payload={
                        "memory": "opsx runtime memory marker",
                        "user_id": "opsx-runtime-20260701",
                        "importance": 0.8,
                        "entity_refs": ["headroom"],
                    },
                )
            ]
        )


def test_direct_mem0_search_uses_filters_for_user_scope() -> None:
    adapter = DirectMem0Adapter(Mem0Config())
    fake_client = _FakeMem0Client()
    adapter._initialized = True
    adapter._mem0_client = fake_client

    results = asyncio.run(
        adapter.search_memories(
            "opsx runtime memory marker",
            user_id="opsx-runtime-20260701",
            top_k=2,
        )
    )

    assert fake_client.search_kwargs == {
        "query": "opsx runtime memory marker",
        "filters": {"user_id": "opsx-runtime-20260701"},
        "top_k": 2,
    }
    assert results[0].memory.user_id == "opsx-runtime-20260701"


def test_direct_mem0_search_falls_back_to_qdrant_for_direct_writes(monkeypatch) -> None:
    class _FieldCondition:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

    class _Filter:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

    class _MatchValue:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

    models_module = ModuleType("qdrant_client.models")
    models_module.FieldCondition = _FieldCondition
    models_module.Filter = _Filter
    models_module.MatchValue = _MatchValue

    qdrant_module = ModuleType("qdrant_client")
    qdrant_module.models = models_module

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models_module)

    adapter = DirectMem0Adapter(Mem0Config(collection_name="headroom_memories"))
    fake_qdrant = _FakeQdrantClient()
    adapter._initialized = True
    adapter._mem0_client = _EmptyMem0Client()
    adapter._qdrant_client = fake_qdrant
    adapter._embed = lambda text: [0.1, 0.2, 0.3]

    results = asyncio.run(
        adapter.search_memories(
            "opsx runtime memory marker",
            user_id="opsx-runtime-20260701",
            top_k=2,
        )
    )

    assert fake_qdrant.query_kwargs is not None
    assert fake_qdrant.query_kwargs["collection_name"] == "headroom_memories"
    assert fake_qdrant.query_kwargs["query"] == [0.1, 0.2, 0.3]
    assert fake_qdrant.query_kwargs["limit"] == 2
    assert results[0].memory.content == "opsx runtime memory marker"
    assert results[0].memory.user_id == "opsx-runtime-20260701"
