"""Atomic bound stage evidence and terminal, pending-only request outcomes."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.compiler.records import atomic_write
from piginot.contracts import canonical_json_bytes, strict_json_loads

Document = dict[str, Any]


def digest(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def save_bound(path: Path, binding: str, payload: Mapping[str, Any]) -> None:
    """Publish one independently verifiable evidence object atomically."""
    body = {"binding_sha256": binding, "payload": dict(payload)}
    document = {**body, "content_sha256": digest(body)}
    atomic_write(path, canonical_json_bytes(document))


def load_bound(path: Path, binding: str) -> Document | None:
    if path.is_symlink():
        raise ValueError("evidence cannot be a symlink")
    if not path.exists():
        return None
    document = strict_json_loads(path.read_bytes())
    if not isinstance(document, dict) or set(document) != {
        "binding_sha256",
        "payload",
        "content_sha256",
    }:
        raise ValueError("invalid evidence envelope")
    body = {key: document[key] for key in ("binding_sha256", "payload")}
    if document["content_sha256"] != digest(body):
        raise ValueError("evidence content digest differs")
    if document["binding_sha256"] != binding:
        raise ValueError("evidence binding differs")
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise ValueError("evidence payload must be an object")
    return payload


@dataclass(frozen=True, slots=True)
class CaseJob:
    request_id: str
    request_sha256: str
    run: Callable[[], Document]


class OutcomeStore:
    """Classify the complete roster before submitting any unfinished request.

    Each terminal envelope is an atomic commit, including numerical failures
    and exceptions. Interruptions leave no envelope for the interrupted job.
    """

    def __init__(self, directory: Path, binding: str) -> None:
        self.directory = directory
        self.binding = binding

    def _path(self, job: CaseJob) -> Path:
        name = hashlib.sha256(job.request_id.encode()).hexdigest()
        return self.directory / f"outcome-{name}.json"

    def _binding(self, job: CaseJob) -> str:
        return digest(
            {"run": self.binding, "id": job.request_id, "request": job.request_sha256}
        )

    def cached(self, job: CaseJob) -> Document | None:
        """Read a dependency outcome under its exact immutable request binding."""
        return load_bound(self._path(job), self._binding(job))

    def validate(self, jobs: tuple[CaseJob, ...]) -> tuple[Document | None, ...]:
        """Verify the fixed roster and every cached outcome without scheduling."""
        if self.directory.is_symlink():
            raise ValueError("outcome directory cannot be a symlink")
        expected = {
            "requests": [
                {"id": job.request_id, "sha256": job.request_sha256} for job in jobs
            ]
        }
        if load_bound(self.directory / "roster.json", self.binding) != expected:
            raise ValueError("request roster differs from its binding")
        return tuple(self.cached(job) for job in jobs)

    def run(self, jobs: tuple[CaseJob, ...]) -> tuple[Document, ...]:
        if self.directory.is_symlink():
            raise ValueError("outcome directory cannot be a symlink")
        ids = tuple(job.request_id for job in jobs)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("request roster must be nonempty and unique")
        self.directory.mkdir(parents=True, exist_ok=True)
        roster = {
            "requests": [
                {"id": job.request_id, "sha256": job.request_sha256} for job in jobs
            ]
        }
        roster_path = self.directory / "roster.json"
        previous = load_bound(roster_path, self.binding)
        if previous is None:
            save_bound(roster_path, self.binding, roster)
        elif previous != roster:
            raise ValueError("request roster differs from its binding")
        cached = self.validate(jobs)
        results = []
        for job, result in zip(jobs, cached, strict=True):
            if result is None:
                try:
                    result = job.run()
                    if result.get("status") not in {
                        "PASS",
                        "FAIL",
                        "INCOMPLETE",
                        "ERROR",
                    }:
                        raise ValueError(
                            "request must return a terminal scientific outcome"
                        )
                    canonical_json_bytes(result)
                except Exception as error:
                    result = {
                        "status": "ERROR",
                        "error": {
                            "category": type(error).__name__,
                            "message": str(error),
                        },
                    }
                result = {
                    **result,
                    "request_id": job.request_id,
                    "request_sha256": job.request_sha256,
                }
                save_bound(self._path(job), self._binding(job), result)
            results.append(result)
        return tuple(results)


__all__ = ("CaseJob", "Document", "OutcomeStore", "digest", "load_bound", "save_bound")
