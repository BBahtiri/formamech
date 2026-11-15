"""Locked, atomic publication of an exactly bound compiler plan."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from piginot.compiler.planning import (
    ExecutionPlan,
    PlanError,
    _implementation,
    check_output_path,
)
from piginot.contracts import canonical_json_bytes


@contextmanager
def run_lock(output: Path) -> Iterator[None]:
    path = output / ".compiler.lock"
    if path.is_symlink():
        raise PlanError("output", "run lock cannot be a symlink")
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise PlanError(
                "locked", "another process owns this compiler run"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise PlanError("output", "published records cannot be symlinks")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".partial"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _verify_inputs(plan: ExecutionPlan) -> None:
    expected = dict(plan.input_sha256)
    if len(expected) != len(plan.input_sha256) or len(expected) != len(
        plan.input_paths
    ):
        raise PlanError("input", "bound input roster has inconsistent paths or labels")
    for label, path in zip(
        (label for label, _ in plan.input_sha256), plan.input_paths, strict=True
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[label]:
            raise PlanError("input", f"{label} changed after plan validation")
    if _implementation() != (plan.package_version, plan.package_sha256):
        raise PlanError(
            "input", "compiler implementation changed after plan validation"
        )


def write_plan(plan: ExecutionPlan, *, resume: bool = False) -> Path:
    """Publish a new plan or verify an existing binding without resetting it."""

    _verify_inputs(plan)
    output = check_output_path(plan.output_path, plan.input_paths)
    path = output / "plan.json"
    if path.is_symlink():
        raise PlanError("output", "plan cannot be a symlink")
    if resume and not path.is_file():
        raise PlanError("resume", "resume requires an existing bound plan")
    if not resume and output.exists() and any(output.iterdir()):
        raise PlanError(
            "resume", "output is not empty; use resume for the same bound run"
        )
    output.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(plan.document()) + b"\n"
    with run_lock(output):
        if resume:
            if path.read_bytes() != payload:
                raise PlanError(
                    "binding", "saved plan differs from the requested run binding"
                )
        else:
            if any(item.name != ".compiler.lock" for item in output.iterdir()):
                raise PlanError("output", "output changed before plan publication")
            atomic_write(path, payload)
    return path
