"""Publish job-scoped evidence from the child that produced the workbench output."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

EVIDENCE_ENV = "FORMAMECH_JOB_EVIDENCE_DIR"
OUTPUT_ENV = "FORMAMECH_JOB_OUTPUT_DIR"
OWNER_ENV = "FORMAMECH_JOB_PROCESS_ID"


def bind_job_process() -> None:
    if EVIDENCE_ENV in os.environ:
        os.environ.setdefault(OWNER_ENV, str(os.getpid()))


def _capture_dir(output: Path | None = None) -> Path | None:
    archive = os.environ.get(EVIDENCE_ENV)
    if not archive or os.environ.get(OWNER_ENV) != str(os.getpid()):
        return None
    if output is not None and str(output.resolve()) != os.environ.get(OUTPUT_ENV):
        return None
    return Path(archive)


def _publish(directory: Path, files: dict[str, bytes]) -> None:
    from piginot.compiler.records import atomic_write

    try:
        for name, payload in files.items():
            atomic_write(directory / name, payload)
    except (OSError, ValueError) as exc:
        warnings.warn(f"Unable to retain workbench job evidence: {exc}", stacklevel=2)


def archive_report(output: Path, contents: bytes) -> None:
    if directory := _capture_dir(output):
        _publish(
            directory,
            {
                "report.md": contents,
                "report.sha256": hashlib.sha256(contents).hexdigest().encode("ascii"),
            },
        )


def archive_checkpoint(output: Path, checkpoint: Path, step: int) -> None:
    if directory := _capture_dir(output):
        try:
            digest = hashlib.sha256(
                (checkpoint / "checkpoint.json").read_bytes()
            ).hexdigest()
        except OSError as exc:
            warnings.warn(
                f"Unable to retain workbench checkpoint identity: {exc}", stacklevel=2
            )
            return
        _publish(
            directory,
            {"checkpoint.json": json.dumps({"step": step, "sha256": digest}).encode()},
        )


def record_job_exit(code: int) -> None:
    if directory := _capture_dir():
        _publish(directory, {"exit.txt": str(code).encode("ascii")})
