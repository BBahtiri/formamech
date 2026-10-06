"""G17.non-dogbone passes a lower claim only through an exact signed acceptance."""

from __future__ import annotations

from typing import Any

from campaigns.g17_gate import acceptance_covers

GATE = "a" * 64
MANIFEST = "b" * 64


def _measured(**changes: Any) -> dict[str, Any]:
    measured = {
        "notched_g15_verdict": "PASS",
        "notched_bundle_claim_level": "DEVELOPMENT_VERIFIED",
        "notched_bundle_manifest_sha256": MANIFEST,
        "undeclared_framework_changes": [],
    }
    measured.update(changes)
    return measured


def _acceptance(**changes: Any) -> dict[str, Any]:
    acceptance = {
        "check": "G17.non-dogbone",
        "signed_off_by": "Betim Bahtiri",
        "accepted_claim_level": "DEVELOPMENT_VERIFIED",
        "notched_g15_sha256": GATE,
        "bundle_manifest_sha256": MANIFEST,
        "decision_record": "ADR 0031",
    }
    acceptance.update(changes)
    return acceptance


def test_exact_signed_acceptance_covers_the_computed_shortfall() -> None:
    assert acceptance_covers(_acceptance(), _measured(), GATE)


def test_no_acceptance_or_no_gate_is_never_covered() -> None:
    assert not acceptance_covers(None, _measured(), GATE)
    assert not acceptance_covers(_acceptance(), _measured(), None)


def test_acceptance_must_name_this_gate_bundle_claim_and_signatory() -> None:
    assert not acceptance_covers(_acceptance(), _measured(), "c" * 64)
    assert not acceptance_covers(
        _acceptance(bundle_manifest_sha256="d" * 64), _measured(), GATE
    )
    assert not acceptance_covers(
        _acceptance(accepted_claim_level="NOT_VERIFIED"), _measured(), GATE
    )
    assert not acceptance_covers(
        _acceptance(signed_off_by="someone"), _measured(), GATE
    )
    assert not acceptance_covers(_acceptance(check="G17.refusal"), _measured(), GATE)


def test_failing_g15_and_undeclared_framework_changes_are_never_accepted() -> None:
    assert not acceptance_covers(
        _acceptance(), _measured(notched_g15_verdict="FAIL"), GATE
    )
    assert not acceptance_covers(
        _acceptance(),
        _measured(undeclared_framework_changes=["src/piginot/x.py"]),
        GATE,
    )
