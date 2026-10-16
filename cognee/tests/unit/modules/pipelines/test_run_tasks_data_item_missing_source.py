"""A data item whose source file is gone must be terminal, not retried forever.

Before the fix, an item whose stored file disappeared kept NO pipeline_status
entry: ``run_tasks_data_item_incremental`` caught the FileNotFoundError, yielded
a PipelineRunErrored, and re-raised (RAISE_INCREMENTAL_LOADING_ERRORS defaults to
true). Every later cognify re-read the same missing file and failed the whole
build again — telemetry showed one deployment failing 880 times on the same
documents. The fix marks the item DATA_ITEM_PROCESSING_ERRORED so later runs
skip it, and lets the current run continue with a PipelineRunSkipped event.

These tests drive the real incremental wrapper with a stubbed relational engine
(session-counting fake, same pattern as test_run_tasks_data_item_sessions.py).
"""

import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks_data_item as item_module
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunErrored,
    PipelineRunSkipped,
)


class _FakeSessionFactory:
    """Counts sessions, merges, and commits; every select returns the row."""

    def __init__(self, row):
        self.row = row
        self.opened = 0
        self.merged = []
        self.committed = 0

    def __call__(self):
        self.opened += 1
        session = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = self.row
        session.execute = AsyncMock(return_value=result)

        async def _merge(obj):
            self.merged.append(obj)

        async def _commit():
            self.committed += 1

        session.merge = AsyncMock(side_effect=_merge)
        session.commit = AsyncMock(side_effect=_commit)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx


def _wire(monkeypatch, tmp_path, *, stored_file: pathlib.Path, pipeline_status=None):
    """Wire the module for one Data item whose stored file lives at stored_file."""
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    data_item = Data(
        id=uuid4(),
        name="gone.txt",
        raw_data_location=str(stored_file),
        pipeline_status=pipeline_status,
    )

    factory = _FakeSessionFactory(data_item)
    engine = MagicMock()
    engine.get_async_session.side_effect = factory
    monkeypatch.setattr(item_module, "get_relational_engine", lambda: engine)

    telemetry_calls = []

    def _set_pipeline(behavior):
        async def _pipeline(**kwargs):
            telemetry_calls.append(kwargs.get("data"))
            await behavior()
            yield  # pragma: no cover - never reached when behavior raises

        monkeypatch.setattr(item_module, "run_tasks_with_telemetry", _pipeline)

    async def _run():
        events = []
        async for event in item_module.run_tasks_data_item_incremental(
            data_item=data_item,
            dataset=dataset,
            tasks=[],
            pipeline_name="cognify_pipeline",
            pipeline_id=uuid4(),
            pipeline_run_id=uuid4(),
            ctx=None,
            user=user,
        ):
            events.append(event)
        return [e["run_info"] for e in events if isinstance(e, dict) and "run_info" in e]

    return factory, _set_pipeline, _run, dataset, data_item, telemetry_calls, user


@pytest.mark.asyncio
async def test_missing_source_file_is_marked_errored_and_does_not_fail_the_run(
    monkeypatch, tmp_path
):
    # First encounter: the stored file is gone. Even with the default
    # RAISE_INCREMENTAL_LOADING_ERRORS=true the item must not raise out of the
    # wrapper — the run keeps going, and the item is marked terminal.
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "true")

    missing = tmp_path / "data-root" / "gone.txt"
    factory, set_pipeline, run, dataset, data_item, calls, _user = _wire(
        monkeypatch, tmp_path, stored_file=missing
    )

    async def _raise_missing():
        raise FileNotFoundError(f"[Errno 2] No such file or directory: '{missing}'")

    set_pipeline(_raise_missing)

    run_infos = await run()

    assert len(run_infos) == 1
    skipped = run_infos[0]
    assert isinstance(skipped, PipelineRunSkipped)
    assert not isinstance(skipped, PipelineRunErrored)
    assert skipped.error_class == "FileNotFoundError"
    assert "No such file" in (skipped.error_message or "")
    # The terminal status is persisted for this pipeline + dataset slot.
    assert (
        data_item.pipeline_status["cognify_pipeline"][str(dataset.id)]
        == DataItemStatus.DATA_ITEM_PROCESSING_ERRORED
    )
    assert factory.committed == 1
    # The task actually ran once (and failed) — this is the first encounter.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_item_marked_errored_is_skipped_without_rerunning_tasks(monkeypatch, tmp_path):
    # A later run must skip the item before any task executes — re-reading the
    # missing file is exactly the failure loop this fix removes.
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "true")

    missing = tmp_path / "data-root" / "gone.txt"

    factory, set_pipeline, run, dataset, data_item, calls, _user = _wire(
        monkeypatch,
        tmp_path,
        stored_file=missing,
        pipeline_status={"cognify_pipeline": {}},
    )
    data_item.pipeline_status["cognify_pipeline"][str(dataset.id)] = (
        DataItemStatus.DATA_ITEM_PROCESSING_ERRORED
    )

    async def _must_not_run():
        raise AssertionError("tasks must not run for a terminal-errored item")

    set_pipeline(_must_not_run)

    run_infos = await run()

    assert len(run_infos) == 1
    assert isinstance(run_infos[0], PipelineRunSkipped)
    assert len(calls) == 0
    # Skips are cheap: pre-check only, no status-write session.
    assert factory.opened == 1
    assert factory.committed == 0


@pytest.mark.asyncio
async def test_file_not_found_for_another_path_stays_retryable(monkeypatch, tmp_path):
    # A FileNotFoundError that is NOT this item's missing source file (e.g. a
    # prompt template) must keep the old behavior: PipelineRunErrored + re-raise
    # at the default, and no status recorded — the item stays retryable.
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "true")

    existing = tmp_path / "data-root" / "still-there.txt"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("content")

    _factory, set_pipeline, run, _dataset, data_item, _calls, _user = _wire(
        monkeypatch, tmp_path, stored_file=existing
    )

    async def _raise_unrelated():
        raise FileNotFoundError("[Errno 2] No such file or directory: '/opt/app/missing.yaml'")

    set_pipeline(_raise_unrelated)

    with pytest.raises(FileNotFoundError):
        await run()

    assert data_item.pipeline_status in (None, {})
