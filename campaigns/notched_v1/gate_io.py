"""Gate-artifact helpers shared by the notched campaign gates."""

from __future__ import annotations

import enum
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from piginot.git_environment import isolated_git_environment


def plain(value: object) -> Any:
    """JSON-ready form of gate measurements (dataclasses, enums, tuples)."""

    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        items = [plain(item) for item in value]
        return sorted(items) if isinstance(value, (frozenset, set)) else items
    return value


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        env=isolated_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_pytest_with_coverage(
    root: Path, tests: Sequence[str], packages: Sequence[str]
) -> tuple[str, float, bool]:
    """Run the named tests with branch coverage of ``packages``.

    Returns the pytest summary line, the branch coverage percentage of the
    packages, and whether the run passed.
    """

    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "coverage.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--cov-branch",
            f"--cov-report=json:{report}",
            *(f"--cov={package}" for package in packages),
            *tests,
        ]
        completed = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        summary = ""
        for line in completed.stdout.splitlines()[::-1]:
            if re.search(r"\d+ (passed|failed)", line):
                summary = line.strip("= ").strip()
                break
        percent = 0.0
        if report.is_file():
            percent = float(json.loads(report.read_text())["totals"]["percent_covered"])
    return summary, round(percent, 2), completed.returncode == 0


def check_document(
    check_id: str,
    passed: bool,
    measured: object,
    threshold: object,
    comparison: str,
    **evidence: object,
) -> dict[str, Any]:
    document = {
        "id": check_id,
        "status": "PASS" if passed else "FAIL",
        "measured": plain(measured),
        "threshold": plain(threshold),
        "comparison": comparison,
    }
    if evidence:
        document["evidence"] = plain(evidence)
    return document


def gate_document(
    gate_id: str,
    step: str,
    root: Path,
    checks: Sequence[Mapping[str, Any]],
    **extra: object,
) -> dict[str, Any]:
    blockers = [check["id"] for check in checks if check["status"] != "PASS"]
    return {
        "gate_id": gate_id,
        "step": step,
        "campaign_id": "notched-v1",
        "commit": git_head(root),
        "timestamp_utc": utc_now(),
        "checks": list(checks),
        **{key: plain(value) for key, value in extra.items()},
        "verdict": "FAIL" if blockers else "PASS",
        "verdict_reason": (
            f"{len(blockers)} required check(s) not PASS: {', '.join(blockers)}"
            if blockers
            else "every required check PASS"
        ),
    }


def write_gate_document(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


__all__ = (
    "check_document",
    "gate_document",
    "git_head",
    "plain",
    "run_pytest_with_coverage",
    "utc_now",
    "write_gate_document",
)
