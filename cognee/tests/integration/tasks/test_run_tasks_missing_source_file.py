"""A document whose stored file disappeared must not fail every later cognify.

Telemetry (2026-10-16 insight): FileNotFoundError ended 25.8% of failed runs on
10-15 (680/2,639) and one 1.6.4 deployment failed 880 times — every run re-read
the same missing file because the item kept no pipeline_status entry.

This drives the REAL pipeline (add() staging, sqlite relational engine, tmp
roots) with the item's stored file deleted from disk:

1. Run 1 (RAISE_INCREMENTAL_LOADING_ERRORS at its default "true") must COMPLETE,
   not raise: the item is flagged with a PipelineRunSkipped event and the data
   row records DATA_ITEM_PROCESSING_ERRORED.
2. Run 2 must skip the item before any task executes — the reader task is never
   invoked, and the run completes again.

Lives in the integration tree because it exercises the real add() pipeline and
relational engine; the environment fixture scopes every write to a tmp root.
"""

import pathlib
from urllib.parse import urlparse

import pytest
import pytest_asyncio
from sqlalchemy import select

import cognee
from cognee.infrastructure.databases.relational import create_db_and_tables, get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunSkipped,
)
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user


@pytest_asyncio.fixture
async def clean_test_environment(request, tmp_path, monkeypatch):
    """Scope every write to a per-test tmp root; never touch ambient data."""
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    # The telemetry deployment that failed 880 times runs the default (raise):
    # the fix must keep the run alive under exactly that setting.
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "true")

    root = pathlib.Path(tmp_path) / request.node.name

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    create_relational_engine.cache_clear()

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))

    await create_db_and_tables()

    yield

    create_relational_engine.cache_clear()


async def _stage_dataset(dataset_name):
    """add() creates the dataset + data rows (with real stored files) with no LLM call."""
    user = await get_default_user()
    await cognee.add("missing source probe text", dataset_name=dataset_name, user=user)

    from cognee.modules.data.methods import get_authorized_existing_datasets, get_dataset_data

    [dataset] = await get_authorized_existing_datasets([dataset_name], "read", user)
    data_items = await get_dataset_data(dataset.id)
    assert data_items, "add() should have created a data row"
    return user, dataset, data_items


async def _delete_stored_files(data_items):
    """Delete every staged item's stored source file from disk; return the ids."""
    from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path

    async with get_relational_engine().get_async_session() as session:
        rows = (
            (await session.execute(select(Data).where(Data.id.in_([d.id for d in data_items]))))
            .scalars()
            .all()
        )
        locations = [(row.id, row.raw_data_location) for row in rows]

    deleted = 0
    for _data_id, location in locations:
        resolved = get_data_file_path(location)
        if urlparse(resolved).scheme:  # pragma: no cover - test storage is local
            continue
        path = pathlib.Path(resolved)
        if path.exists():
            path.unlink()
            deleted += 1
    assert deleted == len(data_items), (
        f"test setup: expected to delete {len(data_items)} stored files, deleted {deleted}"
    )
    return [data_id for data_id, _ in locations]


def _reader_task(calls):
    """A cognify-shaped task: opens each item's stored file and reads it.

    Tasks receive the batch (the item wrapper passes a one-element list), like
    the real ``classify_documents(data_documents: list[Data])``.
    """

    async def read_source_file(data_items):
        from cognee.infrastructure.files.utils.open_data_file import open_data_file

        for data_item in data_items:
            calls.append(data_item.id)
            async with open_data_file(
                data_item.raw_data_location, mode="r", encoding="utf-8"
            ) as file:
                file.read()
        return data_items

    return read_source_file


async def _run_cognify(dataset, data_items, user, read_source_file):
    run_infos = []
    async for run_info in run_tasks(
        [Task(read_source_file)],
        dataset_id=dataset.id,
        data=data_items,
        user=user,
        pipeline_name="cognify_pipeline",
        incremental_loading=True,
    ):
        run_infos.append(run_info)
    return run_infos


@pytest.mark.asyncio
async def test_missing_source_file_completes_run_and_is_skipped_after(
    clean_test_environment,
):
    user, dataset, data_items = await _stage_dataset("missing_source")
    await _delete_stored_files(data_items)

    reader_calls = []

    # Run 1: first encounter with the gone file. The run must complete — one
    # unrecoverable document must not fail the whole build — and the item must
    # come out flagged as skipped, not silently swallowed and not errored.
    run_infos = await _run_cognify(dataset, data_items, user, _reader_task(reader_calls))

    completed = [info for info in run_infos if isinstance(info, PipelineRunCompleted)]
    errored = [info for info in run_infos if isinstance(info, PipelineRunErrored)]
    assert completed, f"run must complete despite the missing file, got {run_infos}"
    assert not errored, f"the run must not report an error for a skipped item, got {run_infos}"

    entries = completed[-1].data_ingestion_info or []
    assert entries, "the completed run must carry per-item results"
    assert all(isinstance(entry["run_info"], PipelineRunSkipped) for entry in entries)
    assert all(entry["run_info"].error_class == "FileNotFoundError" for entry in entries)
    # The task genuinely ran (and hit the gone file) on this first pass.
    assert len(reader_calls) == len(data_items)

    # The terminal status is persisted per item, per pipeline + dataset slot.
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (await session.execute(select(Data).where(Data.id.in_([d.id for d in data_items]))))
            .scalars()
            .all()
        )
        assert rows, "data rows must still exist"
        for row in rows:
            slot = (row.pipeline_status or {}).get("cognify_pipeline", {}).get(str(dataset.id))
            assert slot == DataItemStatus.DATA_ITEM_PROCESSING_ERRORED, (
                f"expected a terminal errored marker, got {slot!r}"
            )

    # Run 2: the item must be skipped before any task executes — this is the
    # exact re-read loop that failed the telemetry deployment 880 times.
    reader_calls.clear()
    run_infos_2 = await _run_cognify(dataset, data_items, user, _reader_task(reader_calls))

    completed_2 = [info for info in run_infos_2 if isinstance(info, PipelineRunCompleted)]
    assert completed_2, f"later run must complete, got {run_infos_2}"
    entries_2 = completed_2[-1].data_ingestion_info or []
    assert all(isinstance(entry["run_info"], PipelineRunSkipped) for entry in entries_2)
    assert reader_calls == [], (
        "the reader task must never run again for an item whose source file is gone"
    )
