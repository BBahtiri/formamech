"""The committed H1 confirmation sidecars rebuild from the S23 reveals."""

from __future__ import annotations

from pathlib import Path

from campaigns import s23_calibration as s23
from campaigns import s23_confirmation as confirmation
from piginot.contracts import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]


def test_the_sidecars_rebuild_byte_for_byte() -> None:
    for campaign, bundle in s23.FAMILIES.items():
        built = confirmation.build(ROOT, campaign)
        committed = (ROOT / confirmation.confirmation_sidecar(bundle)).read_bytes()
        assert canonical_json_bytes(built.document()) + b"\n" == committed


def test_only_the_dogbone_displacement_budget_bound_is_withheld() -> None:
    dogbone = confirmation.build(ROOT, "dogbone-v1")
    notched = confirmation.build(ROOT, "notched-v1")
    assert dogbone.unconfirmed("budget_total") == ("u_relative_l2",)
    assert dogbone.unconfirmed("reference_terms") == ()
    assert notched.unconfirmed("budget_total") == ()
    assert notched.unconfirmed("reference_terms") == ()
