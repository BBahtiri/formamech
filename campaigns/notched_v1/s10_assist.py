"""A second FEM worker for the notched S10 run (store-only, reverse order).

The production runner solves one case at a time and writes the ledger. This
worker solves cases from the end of the request plan into the same
content-addressed object store and writes nothing else: when the runner
reaches such a case it finds the result under the request's recipe hash and
records it in the ledger as ``RECOVERED_FROM_STORE``. The worker stops at the
first case the ledger already holds. A failed solve leaves nothing in the
store, so the runner solves that case itself and records the failure.

Usage (from a checkout whose FEM, morphing, store and request code equals the
run revision; ``--expected-revision`` names it)::

    python -m campaigns.notched_v1.s10_assist --run-root RUN --log LOG
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from campaigns.dogbone_v1.s10_closeout import _load_run_execution_profile
from campaigns.dogbone_v1.s10_production import ProductionFemRunLedger
from campaigns.notched_v1.s10 import public_requests, request_plan
from piginot.datasets import PublicFemRequestPlan
from piginot.fem.backend import ScikitFemBackend
from piginot.store import ObjectStore


def _completed(run_root: Path) -> set[str]:
    ledger = ProductionFemRunLedger.from_bytes((run_root / "ledger.json").read_bytes())
    return {record.case_id for record in ledger.records}


def assist(run_root: Path, log: Path) -> int:
    profile = _load_run_execution_profile(run_root)
    plan = request_plan(profile)
    persisted = PublicFemRequestPlan.from_bytes(
        (run_root / "request-plan.json").read_bytes()
    )
    if persisted.to_bytes() != plan.to_bytes():
        raise RuntimeError("this checkout builds a different request plan")
    requests = {request.request_id: request for request in public_requests(profile)}
    store = ObjectStore.open(run_root / "object-store")
    backend = ScikitFemBackend(store=store)
    solved = 0
    for entry in reversed(plan.entries):
        if entry.case_id in _completed(run_root):
            break
        request = requests[entry.case_id]
        if store.recipe_results(request.recipe_hash):
            continue
        start = time.perf_counter()
        (attempt,) = backend.solve_many((request,))
        record = {
            "case_id": entry.case_id,
            "recipe_hash": request.recipe_hash,
            "success": attempt.success,
            "store_content_hash": attempt.store_content_hash,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        solved += 1
    return solved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.s10_assist")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    arguments = parser.parse_args(argv)
    count = assist(arguments.run_root.resolve(), arguments.log.resolve())
    print(f"assisted {count} cases", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
