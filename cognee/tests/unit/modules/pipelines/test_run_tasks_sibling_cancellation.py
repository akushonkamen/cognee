"""Sibling cancellation is its own telemetry outcome, not a task error.

When one data item fails, ``run_tasks`` cancels every still-running sibling
item (run_tasks.py's gather handler). Those siblings unwind with
``asyncio.CancelledError`` through the same per-item telemetry chain, which
used to report them as ``Pipeline Item Errored`` / ``{task} Task Errored`` —
on 1.6.3 that made CancelledError 85.6% of task errors (11,906/13,915) while
real failure causes were 200x rarer (telemetry insight #5807). The failing
item itself must still read as Errored; only the cancelled siblings move to
the Cancelled outcome.
"""

import asyncio
import importlib
from contextlib import aclosing, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.pipelines.operations import run_tasks_base as base_module
from cognee.modules.pipelines.operations import run_tasks_with_telemetry as telemetry_module
from cognee.modules.pipelines.tasks.task import Task

run_tasks_module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks")


class _FakeSession:
    def __init__(self, dataset):
        self._dataset = dataset

    async def get(self, _model, _dataset_id):
        return self._dataset

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_async_session(self):
        return _FakeSession(self._dataset)


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


@pytest.mark.asyncio
async def test_a_failed_item_reports_errored_while_its_cancelled_siblings_report_cancelled(
    monkeypatch,
):
    dataset = SimpleNamespace(id=uuid4(), name="dataset", owner_id=uuid4())
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    run_id = uuid4()
    events = []

    def capture(name, _user, additional_properties):
        events.append((name, dict(additional_properties or {})))

    slow_reached = asyncio.Event()
    hang = asyncio.Event()

    async def slow_task(data):
        slow_reached.set()
        await hang.wait()  # cancelled here when the sibling fails
        return data

    async def failing_task(data):
        await slow_reached.wait()  # only fail once the sibling is mid-flight
        raise ValueError("upstream boom")

    async def stub_data_item(
        data_item,
        dataset,
        item_tasks,
        pipeline_name,
        pipeline_id,
        pipeline_run_id,
        ctx,
        user,
        incremental_loading,
        data_cache,
        progress_state=None,
    ):
        # The real per-item telemetry chain (run_tasks_with_telemetry +
        # run_tasks_base), with only the DB-backed wrapper stubbed away.
        async with aclosing(
            telemetry_module.run_tasks_with_telemetry(
                tasks=item_tasks,
                data=[data_item],
                user=user,
                pipeline_name=pipeline_id,
                ctx=ctx,
            )
        ) as stream:
            async for _ in stream:
                pass
        return {
            "run_info": PipelineRunCompleted(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
            )
        }

    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(
        run_tasks_module, "get_graph_engine", AsyncMock(return_value=SimpleNamespace())
    )
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(
        run_tasks_module,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=run_id)),
    )
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_complete", AsyncMock())
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", AsyncMock())
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", AsyncMock())
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", stub_data_item)
    monkeypatch.setattr(run_tasks_module, "send_telemetry", capture)
    monkeypatch.setattr(telemetry_module, "send_telemetry", capture)
    monkeypatch.setattr(base_module, "send_telemetry", capture)
    monkeypatch.setattr(telemetry_module, "get_current_settings", lambda **_: {})

    def resolver(item):
        return [Task(failing_task)] if item == "bad" else [Task(slow_task)]

    with pytest.raises(ValueError, match="upstream boom"):
        async for _ in run_tasks_module.run_tasks(
            resolver,
            dataset.id,
            ["slow-1", "slow-2", "bad"],
            user,
            pipeline_name="cognify_pipeline",
        ):
            pass

    item_names = [name for name, _ in events if name.startswith("Pipeline Item")]
    # One Started per item (3), the failing item Errored, both cancelled
    # siblings Cancelled, nothing else.
    assert item_names.count("Pipeline Item Started") == 3
    assert item_names.count("Pipeline Item Errored") == 1
    assert item_names.count("Pipeline Item Cancelled") == 2
    assert item_names.count("Pipeline Item Completed") == 0

    errored = [props for name, props in events if name == "Pipeline Item Errored"]
    assert errored[0]["exception_type"] == "ValueError"
    cancelled = [props for name, props in events if name == "Pipeline Item Cancelled"]
    assert len(cancelled) == 2
    assert all(props.get("exception_type") is None for props in cancelled)
    assert all(props["pipeline_run_id"] == str(run_id) for props in cancelled + errored)

    task_names = [name for name, _ in events if name.endswith("Task Started")]
    assert task_names.count("Coroutine Task Started") == 3
    assert [name for name, _ in events if name.endswith("Task Errored")] == [
        "Coroutine Task Errored"
    ]
    assert [name for name, _ in events if name.endswith("Task Cancelled")] == [
        "Coroutine Task Cancelled",
        "Coroutine Task Cancelled",
    ]

    # The run itself stays Errored — the run-level outcome is unchanged.
    run_names = [name for name, _ in events if name.startswith("Pipeline Run")]
    assert run_names == ["Pipeline Run Started", "Pipeline Run Errored"]
    run_errored = [props for name, props in events if name == "Pipeline Run Errored"]
    assert run_errored[0]["exception_type"] == "ValueError"
