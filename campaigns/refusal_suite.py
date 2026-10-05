"""The S17 refusal suite over both families (``G17.refusal``).

Every labelled fixture of each family goes through its envelope checker
(240 DogBone and 216 notched-plate fixtures), and every fixture that has a
request form goes through ``piginot.bundle.predict`` on that family's bundle.
A case passes only when the request is refused with its expected reason code
(and, through ``predict``, with claim level ``NOT_VERIFIED``).

Usage::

    python -m campaigns.refusal_suite --dogbone-bundle B1 --notched-bundle B2 \
        --output artifacts/benchmarks/REFUSAL-SUITE.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from campaigns.dogbone_v1 import s14_envelope as dogbone
from campaigns.notched_v1 import s14_envelope as notched
from piginot.bundle import Verdict, load_bundle, predict

FAMILIES: dict[str, Any] = {"dogbone-v1": dogbone, "notched-v1": notched}


def _checker_run(module: Any) -> dict[str, Any]:
    fixtures = module.refusal_fixtures()
    misses = []
    by_category: dict[str, list[int]] = {}
    for fixture in fixtures:
        decision = module.check_fixture(fixture)
        ok = not decision.accepted and fixture.expected_code in decision.reason_codes
        counts = by_category.setdefault(fixture.category, [0, 0])
        counts[0] += 1
        counts[1] += int(ok)
        if not ok:
            misses.append(fixture.fixture_id)
    return {
        "fixtures": len(fixtures),
        "refused_with_expected_code": len(fixtures) - len(misses),
        "by_category": {
            k: {"tested": v[0], "passed": v[1]} for k, v in by_category.items()
        },
        "missed": misses,
        "fixture_set_hash": module.fixture_set_hash(fixtures),
    }


def _bundle_run(bundle_path: Path, fixtures: Sequence[Any]) -> dict[str, Any]:
    bundle = load_bundle(bundle_path)
    tested = [fixture for fixture in fixtures if fixture.mesh_mutation is None]
    misses = []
    for fixture in tested:
        prediction = predict(
            bundle, fixture.request_theta(), fixture.queries, confidence=False
        )
        if not (
            prediction.verdict is Verdict.REFUSED
            and fixture.expected_code in prediction.reason_codes
            and prediction.claim_level.value == "NOT_VERIFIED"
        ):
            misses.append(fixture.fixture_id)
    return {
        "bundle": bundle.manifest.name,
        "bundle_manifest_sha256": bundle.manifest.manifest_sha256,
        "request_form_fixtures": len(tested),
        "refused_with_expected_code": len(tested) - len(misses),
        "missed": misses,
        "mesh_mutation_fixtures_without_request_form": len(fixtures) - len(tested),
    }


def run(
    bundles: dict[str, Path | None], progress: Callable[[str], None] = print
) -> dict[str, Any]:
    report: dict[str, Any] = {"families": {}}
    for name, module in FAMILIES.items():
        progress(f"{name}: envelope checker")
        entry: dict[str, Any] = {"checker": _checker_run(module)}
        path = bundles.get(name)
        if path is not None:
            progress(f"{name}: bundle predict")
            entry["bundle"] = _bundle_run(path, module.refusal_fixtures())
        report["families"][name] = entry
    report["cases"] = sum(
        entry["checker"]["fixtures"]
        + entry.get("bundle", {}).get("request_form_fixtures", 0)
        for entry in report["families"].values()
    )
    report["missed"] = sum(
        len(entry["checker"]["missed"]) + len(entry.get("bundle", {}).get("missed", []))
        for entry in report["families"].values()
    )
    report["bundles_run"] = sorted(
        name for name, path in bundles.items() if path is not None
    )
    report["passed"] = report["missed"] == 0 and set(report["bundles_run"]) == set(
        FAMILIES
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.refusal_suite")
    parser.add_argument("--dogbone-bundle", type=Path)
    parser.add_argument("--notched-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = run(
        {"dogbone-v1": arguments.dogbone_bundle, "notched-v1": arguments.notched_bundle}
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("cases", "missed", "passed")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
