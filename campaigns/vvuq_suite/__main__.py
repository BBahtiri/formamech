"""CLI entry point: ``python -m campaigns.vvuq_suite``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaigns.vvuq_suite.suite import run_vvuq_suite
from piginot.fem.registry import resolve_backend


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run FormaMech Level 1-3 ASME V&V 10 / NAFEMS nonlinear benchmarks."
    )
    parser.add_argument("--backend", type=str, default="scikit-fem")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    backend = resolve_backend(args.backend)
    report = run_vvuq_suite(backend)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
