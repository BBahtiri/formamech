"""H1 confirmation sidecars from the committed S23 reveals (ADR 0104).

A bound is confirmed when its H1 coverage count reached the preregistered
acceptance count. The sidecar is derived only from committed files — the
reveal and the calibration — so it rebuilds byte for byte::

    python -m campaigns.s23_confirmation --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from campaigns import s23_calibration as s23
from campaigns import s23_reveal as reveal
from piginot.bounds import BOUND_KINDS, CalibrationRecord, H1Confirmation
from piginot.contracts import canonical_json_bytes

REPOSITORY = s23.REPOSITORY


def confirmation_sidecar(bundle: str) -> Path:
    return Path("bundles/h1-results") / f"{bundle}.json"


def build(root: Path, campaign: str) -> H1Confirmation:
    """The H1 confirmation of one family's calibration."""

    protocol, _ = s23.accepted_protocol(root)
    bundle = protocol["families"][campaign]["bundle"]
    calibration = CalibrationRecord.from_document(
        json.loads((root / s23.sidecar_path(bundle)).read_text("utf-8"))
    )
    path = root / reveal.reveal_path(campaign)
    document = json.loads(path.read_text("utf-8"))
    if document["calibration_sha256"] != calibration.sha256():
        raise ValueError(f"the {campaign} reveal tested another calibration")
    counts = reveal.count_coverage(
        protocol, calibration, document["cases"], document["coverage"]["case_status"]
    )["counts"]
    if counts != document["coverage"]["counts"]:
        raise ValueError(f"the {campaign} reveal's counts do not recount")
    confirmed = {
        kind: {
            quantity: bool(counts[f"{kind}:{quantity}"]["pass"])
            for quantity in protocol["quantities"]
        }
        for kind in BOUND_KINDS
    }
    return H1Confirmation(
        bundle=bundle,
        bundle_manifest_sha256=calibration.bundle_manifest_sha256,
        calibration_sha256=calibration.sha256(),
        reveal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        confirmed=confirmed,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S23 H1 confirmation sidecars.")
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    for campaign, bundle in s23.FAMILIES.items():
        confirmation = build(REPOSITORY, campaign)
        payload = canonical_json_bytes(confirmation.document()) + b"\n"
        target = REPOSITORY / confirmation_sidecar(bundle)
        if options.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        status = (
            "written"
            if options.write
            else (
                "matches"
                if target.is_file() and target.read_bytes() == payload
                else "differs"
            )
        )
        withheld = {k: confirmation.unconfirmed(k) for k in BOUND_KINDS}
        print(f"{campaign}: {status}; not confirmed: {withheld}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
