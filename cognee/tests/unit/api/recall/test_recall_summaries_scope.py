"""scope=session + query_type=SUMMARIES runs the semantic summaries lane.

The hooks' default injection lane asks for the datasets' pre-generated
TextSummary nodes. All tests stub ``authorized_search`` (and, for the
fallback, the session manager) the same way the code-scope tests do: no
databases, vector engines, or LLMs involved.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.api.v1.recall.recall import recall
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

search_mod = importlib.import_module("cognee.modules.search.methods.search")
recall_mod = importlib.import_module("cognee.api.v1.recall.recall")
session_manager_mod = importlib.import_module(
    "cognee.infrastructure.session.get_session_manager"
)


def summaries_payload(text="采购链路闭环已实测走通"):
    return SearchResultPayload(
        dataset_id=uuid4(),
        context=text,
        only_context=True,
        search_type=SearchType.SUMMARIES,
    )


@pytest.fixture
def captured_search(monkeypatch):
    calls = []
    results = {"value": []}

    async def fake_authorized_search(**kwargs):
        calls.append(kwargs)
        value = results["value"]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(search_mod, "authorized_search", fake_authorized_search)
    return {"calls": calls, "results": results}


@pytest.mark.asyncio
async def test_session_scope_with_summaries_runs_vector_lane(captured_search):
    user = SimpleNamespace(id=uuid4())
    dataset_id = uuid4()
    session_id = "zcode_sess_x"
    captured_search["results"]["value"] = [summaries_payload()]

    results = await recall(
        "供应链决策工作台和Foundry什么关系",
        scope=["session"],
        session_id=session_id,
        query_type=SearchType.SUMMARIES,
        dataset_ids=[dataset_id],
        only_context=True,
        top_k=5,
        user=user,
    )

    assert len(captured_search["calls"]) == 1
    call = captured_search["calls"][0]
    assert call["query_type"] is SearchType.SUMMARIES
    assert call["query_text"] == "供应链决策工作台和Foundry什么关系"
    assert call["dataset_ids"] == [dataset_id]
    assert call["session_id"] == session_id
    assert call["top_k"] == 5

    assert results
    assert all(entry.source == "summaries" for entry in results)


@pytest.mark.asyncio
async def test_summaries_miss_falls_back_to_session_qa_keyword_match(
    captured_search, monkeypatch
):
    """No summaries yet -> the lane degrades to the upstream keyword match."""
    user = SimpleNamespace(id=uuid4())
    captured_search["results"]["value"] = []

    class _FakeManager:
        is_available = False

    def fake_get_session_manager():
        return _FakeManager()

    monkeypatch.setattr(session_manager_mod, "get_session_manager", fake_get_session_manager)

    results = await recall(
        "anything",
        scope=["session"],
        session_id="zcode_sess_x",
        query_type=SearchType.SUMMARIES,
        dataset_ids=[uuid4()],
        only_context=True,
        user=user,
    )

    assert results == []


@pytest.mark.asyncio
async def test_session_scope_without_summaries_keeps_keyword_lane(captured_search, monkeypatch):
    """No query_type -> the upstream keyword QA path is untouched."""

    class _FakeManager:
        is_available = False

    def fake_get_session_manager():
        return _FakeManager()

    monkeypatch.setattr(session_manager_mod, "get_session_manager", fake_get_session_manager)

    results = await recall(
        "anything",
        scope=["session"],
        session_id="zcode_sess_x",
        dataset_ids=[uuid4()],
        user=SimpleNamespace(id=uuid4()),
    )

    assert not captured_search["calls"]
    assert results == []
