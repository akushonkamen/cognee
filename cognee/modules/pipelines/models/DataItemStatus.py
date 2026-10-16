import enum
from typing import Any


class DataItemStatus(str, enum.Enum):
    DATA_ITEM_PROCESSING_COMPLETED = "DATA_ITEM_PROCESSING_COMPLETED"
    # Terminal per-item failure: the item can never succeed on a retry, so a
    # later run must skip it instead of re-running it. Written today only when
    # an item's own source file is gone (FileNotFoundError on a missing
    # raw_data_location) — a transient failure (LLM outage, network blip) must
    # stay retryable and keeps no status.
    DATA_ITEM_PROCESSING_ERRORED = "DATA_ITEM_PROCESSING_ERRORED"


def _slot_status(status_value: Any) -> Any:
    """Unwrap one per-item ``pipeline_status`` slot to its bare status value.

    The slot (``Data.pipeline_status[pipeline_name][str(dataset_id)]``) has two
    accepted representations: the legacy bare string/enum value, and a dict
    carrying the status under its ``"status"`` key (the shape planned for the
    hash-aware rollout, so it can grow fields like ``content_hash`` without
    breaking readers).
    """
    if isinstance(status_value, dict):
        return status_value.get("status")
    return status_value


def is_data_item_completed(status_value: Any) -> bool:
    """Whether a per-item ``pipeline_status`` slot records a completed run.

    Anything else — a missing slot, ``None``, an unrelated string, or a dict
    without a completed status — is not completed.
    """
    return _slot_status(status_value) == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED


def is_data_item_errored(status_value: Any) -> bool:
    """Whether a per-item ``pipeline_status`` slot records a terminal failure.

    Such an item is skipped by later runs (see ``run_tasks_data_item``): retrying
    it can only repeat the same failure. A missing slot, ``None``, or any other
    value is not errored.
    """
    return _slot_status(status_value) == DataItemStatus.DATA_ITEM_PROCESSING_ERRORED
