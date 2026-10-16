"""Per-item status predicates must accept both slot representations (SDK-23).

The per-item slot ``Data.pipeline_status[pipeline][str(dataset_id)]`` is today
the bare string ``DATA_ITEM_PROCESSING_COMPLETED``; the hash-aware rollout plans a
dict carrying that string under ``"status"``. Every reader that counts or skips
items should go through these predicates so the two shapes cannot drift apart.

``is_data_item_errored`` gates the same skip logic for the terminal
``DATA_ITEM_PROCESSING_ERRORED`` value (source file gone): a later run must skip
the item, and it must never mistake it for completed.
"""

import pytest

from cognee.modules.pipelines.models import (
    DataItemStatus,
    is_data_item_completed,
    is_data_item_errored,
)

COMPLETED = "DATA_ITEM_PROCESSING_COMPLETED"
ERRORED = "DATA_ITEM_PROCESSING_ERRORED"


@pytest.mark.parametrize(
    "value",
    [
        DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED,
        COMPLETED,
        {"status": COMPLETED},
        {"status": DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED},
        {"status": COMPLETED, "content_hash": "abc123", "token_count": 42},
    ],
)
def test_completed_representations(value):
    assert is_data_item_completed(value) is True


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "DATA_ITEM_PROCESSING_STARTED",
        "completed",
        {},
        {"status": None},
        {"status": "DATA_ITEM_PROCESSING_STARTED"},
        {"content_hash": "abc123"},
        ["DATA_ITEM_PROCESSING_COMPLETED"],
        0,
    ],
)
def test_not_completed_representations(value):
    assert is_data_item_completed(value) is False


@pytest.mark.parametrize(
    "value",
    [
        DataItemStatus.DATA_ITEM_PROCESSING_ERRORED,
        ERRORED,
        {"status": ERRORED},
        {"status": DataItemStatus.DATA_ITEM_PROCESSING_ERRORED},
    ],
)
def test_errored_representations(value):
    assert is_data_item_errored(value) is True


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED,
        COMPLETED,
        {"status": COMPLETED},
        "DATA_ITEM_PROCESSING_STARTED",
        {},
        {"content_hash": "abc123"},
        0,
    ],
)
def test_not_errored_representations(value):
    assert is_data_item_errored(value) is False


def test_errored_slot_is_never_completed_and_vice_versa():
    """The skip predicate must not let an errored item read as completed (or the
    reverse): the completed check runs first in the pre-check, so an errored item
    misread as completed would be reported as processed."""
    assert is_data_item_completed(DataItemStatus.DATA_ITEM_PROCESSING_ERRORED) is False
    assert is_data_item_errored(DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED) is False
    assert is_data_item_completed({"status": ERRORED}) is False
    assert is_data_item_errored({"status": COMPLETED}) is False
