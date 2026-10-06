"""G18.claim-ceiling: nothing above HOLDOUT_VERIFIED is representable or loadable."""

from __future__ import annotations

from pathlib import Path

import pytest

from piginot.bundle import BundleIntegrityError, load_bundle
from piginot.verification import ClaimLevel
from tests.bundle import helpers

SOURCE = Path(__file__).resolve().parents[1] / "src" / "piginot"
FORBIDDEN = ("SEALED_RELEASE_VERIFIED", "SEALED_RELEASE")


def test_claim_levels_stop_at_holdout_verified() -> None:
    assert [level.value for level in ClaimLevel] == [
        "NOT_VERIFIED",
        "DEVELOPMENT_VERIFIED",
        "HOLDOUT_VERIFIED",
    ]
    with pytest.raises(ValueError):
        ClaimLevel("SEALED_RELEASE_VERIFIED")


def test_no_source_file_can_emit_a_sealed_release_claim() -> None:
    offenders = [
        path.relative_to(SOURCE).as_posix()
        for path in sorted(SOURCE.rglob("*.py"))
        if any(token in path.read_text(encoding="utf-8") for token in FORBIDDEN)
    ]
    assert offenders == []


@pytest.mark.parametrize("claim", ["SEALED_RELEASE_VERIFIED", "CERTIFIED"])
def test_a_bundle_claiming_more_than_holdout_is_refused(
    tmp_path: Path, claim: str
) -> None:
    directory = tmp_path / "bundle"
    helpers.write_raw(directory, {}, claim_level=claim)
    with pytest.raises(BundleIntegrityError) as error:
        load_bundle(directory)
    assert "BUNDLE_REPORT_INVALID" in [code.value for code in error.value.reason_codes]
