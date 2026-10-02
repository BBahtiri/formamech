"""Durable holdout consumption before reveal (``G14.holdout-order``).

``consume_holdout`` writes a consumption record exclusively and fsyncs the
file and its directory before returning. ``reveal_after_consumption`` reads the
private reveal only when a matching record already exists on disk. A crash at
any point after the record was written leaves the holdout consumed: a second
``consume_holdout`` for the same record path refuses, and nothing in this
module deletes or rewrites a record.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_T = TypeVar("_T")


class HoldoutAlreadyConsumedError(RuntimeError):
    """The holdout of this record path was consumed before."""


@dataclass(frozen=True, slots=True)
class ConsumptionRecord:
    """What was authorised to consume the holdout, and when."""

    campaign_id: str
    candidate_run_hash: str
    freeze_record_sha256: str
    consumed_utc: str
    purpose: str

    def __post_init__(self) -> None:
        for name in ("campaign_id", "consumed_utc", "purpose"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        for name in ("candidate_run_hash", "freeze_record_sha256"):
            if _DIGEST.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase 64-character hex hash")

    def to_bytes(self) -> bytes:
        return (json.dumps(asdict(self), sort_keys=True, indent=1) + "\n").encode()


def _fsync_directory(directory: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(directory, flags)
    except OSError:  # pragma: no cover - platforms without directory handles
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platforms that refuse directory fsync
        pass
    finally:
        os.close(descriptor)


def consume_holdout(record_path: Path, record: ConsumptionRecord) -> ConsumptionRecord:
    """Persist the consumption record exactly once, durably."""

    if not isinstance(record_path, Path):
        raise TypeError("record_path must be a pathlib.Path")
    if type(record) is not ConsumptionRecord:
        raise TypeError("record must be a ConsumptionRecord")
    if not record_path.parent.is_dir():
        raise FileNotFoundError("the record directory must exist")
    try:
        descriptor = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise HoldoutAlreadyConsumedError(
            f"holdout already consumed: {record_path}"
        ) from exc
    try:
        os.write(descriptor, record.to_bytes())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(record_path.parent)
    return record


def read_consumption(record_path: Path) -> ConsumptionRecord:
    """Parse an existing consumption record."""

    document = json.loads(record_path.read_text("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("consumption record must be a JSON object")
    return ConsumptionRecord(**document)


def reveal_after_consumption(
    record_path: Path,
    *,
    candidate_run_hash: str,
    reader: Callable[[], _T],
) -> _T:
    """Call ``reader`` only after a matching durable consumption record exists."""

    if not record_path.is_file():
        raise PermissionError("the holdout must be consumed before it is revealed")
    record = read_consumption(record_path)
    if record.candidate_run_hash != candidate_run_hash:
        raise PermissionError("the holdout was consumed for a different candidate")
    return reader()


__all__ = (
    "ConsumptionRecord",
    "HoldoutAlreadyConsumedError",
    "consume_holdout",
    "read_consumption",
    "reveal_after_consumption",
)
