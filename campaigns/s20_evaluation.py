"""S20 records: harness reproduction of every S14 budget, and burned HOLDOUTs.

Writes ``artifacts/evaluation/S20-HARNESS.json`` (``G20.harness``) and one
burned-case registry per campaign (``G20.burned``, ADR 0101 decision 4)::

    python -m campaigns.s20_evaluation --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from piginot.evaluation import BurnedCaseRegistry, compare_budget_record

REPOSITORY = Path(__file__).resolve().parents[1]
CAMPAIGNS = ("dogbone-v1", "notched-v1")
SPLITS = ("DEVELOPMENT", "HOLDOUT")
OUTPUT = Path("artifacts/evaluation")


def _record_path(campaign: str, split: str) -> Path:
    return Path("artifacts/campaigns") / campaign / f"S14-{split}.json"


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def harness_report(root: Path) -> dict[str, Any]:
    records = []
    for campaign in CAMPAIGNS:
        for split in SPLITS:
            path = _record_path(campaign, split)
            record = json.loads((root / path).read_text("utf-8"))
            mismatches = compare_budget_record(record)
            records.append(
                {
                    "campaign_id": campaign,
                    "split": split,
                    "record": path.as_posix(),
                    "record_sha256_lf": _sha256_lf(root / path),
                    "gated_rows": len(record["budget"]["gated"]),
                    "reported_rows": len(record["budget"].get("reported", [])),
                    "mismatches": len(mismatches),
                    "first_mismatches": [
                        {
                            "kind": m.kind,
                            "slice": m.slice_id,
                            "quantity": m.quantity,
                            "field": m.field,
                        }
                        for m in mismatches[:5]
                    ],
                }
            )
    return {
        "schema_version": "1.0",
        "step": "S20",
        "comparison": "exact (cases, bit-identical statistic, tolerance, status)",
        "records": records,
        "reproduced": all(r["mismatches"] == 0 for r in records),
    }


def burned_registries(root: Path) -> dict[str, dict[str, Any]]:
    registries = {}
    for campaign in CAMPAIGNS:
        path = _record_path(campaign, "HOLDOUT")
        record = json.loads((root / path).read_text("utf-8"))
        registry = BurnedCaseRegistry.from_holdout_records(
            campaign, {path.as_posix(): record}
        )
        registries[campaign] = registry.to_document()
    return registries


def write(root: Path) -> list[Path]:
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "S20-HARNESS.json"]
    written[0].write_text(
        json.dumps(harness_report(root), indent=2, sort_keys=True) + "\n", "utf-8"
    )
    for campaign, document in burned_registries(root).items():
        path = out / f"BURNED-HOLDOUT-{campaign}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", "utf-8")
        written.append(path)
    return written


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S20 harness and burned records.")
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    report = harness_report(REPOSITORY)
    for entry in report["records"]:
        print(
            f"{entry['campaign_id']} {entry['split']}: {entry['mismatches']} mismatches"
        )
    if options.write:
        for path in write(REPOSITORY):
            print(f"wrote {path.relative_to(REPOSITORY).as_posix()}")
    return 0 if report["reproduced"] else 1


if __name__ == "__main__":
    sys.exit(main())
