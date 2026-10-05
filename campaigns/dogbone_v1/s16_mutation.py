"""Run the S16 mutation suite, record signatures and compare them per release.

::

    python -m campaigns.dogbone_v1.s16_mutation run --report OUT.json \
        [--signatures SIGNATURES.json] [--compare BASELINE.json] \
        [--mutant NAME ...] [--without-detector CHECK_ID ...]
    python -m campaigns.dogbone_v1.s16_mutation gate --report RUN.json \
        --regression-report WEAKENED.json --revision REV ...

``run`` executes the control, every mutant and the control again. With
``--compare`` it exits with the comparison's code: 0 when every signature
holds, 2 when a mutant's edits changed, 3 for a coverage regression (a lost
secondary detection) and 4 when a primary gate no longer fires. A failed
control exits with 5.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from campaigns.dogbone_v1.s16_detectors import CACHES, DETECTORS, cleanup
from campaigns.dogbone_v1.s16_mutants import MUTANTS
from piginot.mutation import (
    SuiteReport,
    compare_signatures,
    run_suite,
    signature_document,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "artifacts" / "mutation" / "S16-SIGNATURES.json"
EXIT_CONTROL_FAILED = 5


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")


def _progress(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def run(
    names: list[str] | None = None, without: list[str] | None = None
) -> SuiteReport:
    mutants = MUTANTS if not names else tuple(m for m in MUTANTS if m.name in names)
    if names and len(mutants) != len(set(names)):
        raise SystemExit(f"unknown mutant in {names}")
    removed = set(without or ())
    detectors = tuple(d for d in DETECTORS if d.check_id not in removed)
    if len(detectors) != len(DETECTORS) - len(removed):
        raise SystemExit(f"unknown detector in {sorted(removed)}")
    try:
        return run_suite(mutants, detectors, caches=CACHES, progress=_progress)
    finally:
        cleanup()


def runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run")
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--signatures", type=Path, default=None)
    command.add_argument("--compare", type=Path, default=None)
    command.add_argument("--mutant", action="append", default=None)
    command.add_argument("--without-detector", action="append", default=None)
    gate = commands.add_parser("gate")
    gate.add_argument("--report", type=Path, required=True)
    gate.add_argument("--regression-report", type=Path, required=True)
    gate.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    gate.add_argument("--revision", required=True)
    gate.add_argument("--coverage-percent", type=float, required=True)
    gate.add_argument("--coverage-command", required=True)
    gate.add_argument("--focused-result", required=True)
    gate.add_argument("--full-result", required=True)
    args = parser.parse_args(argv)
    if args.command == "gate":
        from campaigns.dogbone_v1.s16_gate import write_gate

        return write_gate(args)

    report = run(args.mutant, args.without_detector)
    document = report.document() | {"runtime": runtime()}
    if not report.control_passed:
        _write(args.report, document)
        _progress("control run fired a detector; no signature is recorded")
        return EXIT_CONTROL_FAILED
    signatures = signature_document(report)
    if args.signatures is not None:
        _write(args.signatures, signatures)
    exit_code = 0
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        comparison = compare_signatures(baseline, signatures)
        document["comparison"] = comparison.document()
        for item in comparison.mutants:
            if item.status.value != "PASS" or item.new_detections:
                _progress(
                    f"{item.name}: {item.status.value} lost={list(item.lost_secondary)} "
                    f"new={list(item.new_detections)} {item.detail}"
                )
        _progress(f"comparison {comparison.status.value}")
        exit_code = comparison.exit_code
    _write(args.report, document)
    _progress(
        f"primary fired {report.primary_count}/{len(report.runs)} "
        f"in {report.seconds:.1f} s"
    )
    if report.primary_count != len(report.runs) and exit_code == 0:
        exit_code = 4
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = ("BASELINE_PATH", "main", "run", "runtime")
