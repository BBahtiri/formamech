from __future__ import annotations

import json
from pathlib import Path

import pytest

from piginot.verification import (
    REGISTERED_FAMILIES,
    CheckStatus,
    ClaimLevel,
    GateFamily,
    inject_failures,
    read_family,
    read_registered,
    reduce_registered,
)
from piginot.verification.reducer import Reduction, VerificationFamily

_ROOT = Path(__file__).resolve().parents[2]


def test_twelve_families_cover_the_upstream_gates_in_order() -> None:
    assert len(REGISTERED_FAMILIES) == 12
    assert [f.gate_id for f in REGISTERED_FAMILIES] == [f"G{n}" for n in range(2, 14)]
    assert len({f.family_id for f in REGISTERED_FAMILIES}) == 12
    with pytest.raises(ValueError):
        GateFamily("F99", "", "G1", "x.json")


def test_recorded_families_all_pass_and_reach_the_evidence_level() -> None:
    evidence = read_registered(_ROOT)
    assert all(item.problem is None for item in evidence)
    reduction = reduce_registered(evidence, evidence_level=ClaimLevel.HOLDOUT_VERIFIED)
    assert reduction.claim_level is ClaimLevel.HOLDOUT_VERIFIED
    assert all(f.status is CheckStatus.PASS for f in reduction.families)


def test_every_injected_failure_gives_not_verified() -> None:
    evidence = read_registered(_ROOT)
    result = inject_failures(evidence, evidence_level=ClaimLevel.HOLDOUT_VERIFIED)

    assert result.cases == 48
    assert result.not_verified == 48
    assert result.control_level is ClaimLevel.HOLDOUT_VERIFIED
    assert result.passed


def test_the_pass_on_not_run_mutant_is_detected() -> None:
    from piginot.verification.reducer import reduce_families

    def pass_on_not_run(
        families: tuple[VerificationFamily, ...],
        statuses: dict[str, CheckStatus],
        *,
        evidence_level: ClaimLevel,
    ) -> Reduction:
        promoted = {
            check: CheckStatus.PASS if status is CheckStatus.NOT_RUN else status
            for check, status in statuses.items()
        }
        for family in families:
            for check in family.check_ids:
                promoted.setdefault(check, CheckStatus.PASS)
        return reduce_families(families, promoted, evidence_level=evidence_level)

    evidence = read_registered(_ROOT)
    result = inject_failures(
        evidence,
        evidence_level=ClaimLevel.HOLDOUT_VERIFIED,
        reducer=pass_on_not_run,
    )
    assert not result.passed
    assert result.not_verified == 24


def _artifact(tmp_path: Path, **document: object) -> GateFamily:
    path = tmp_path / "S2.json"
    path.write_text(json.dumps(document))
    return GateFamily("F01", "contracts-identity", "G2", "S2.json")


def test_missing_and_inconsistent_artifacts(tmp_path: Path) -> None:
    family = GateFamily("F01", "contracts-identity", "G2", "missing.json")
    missing = read_family(family, tmp_path)
    assert missing.statuses == {"F01:artifact": CheckStatus.NOT_RUN}

    passing = [{"id": "G2.a", "status": "PASS"}, {"id": "G2.b", "status": "PASS"}]
    good = read_family(
        _artifact(tmp_path, gate_id="G2", verdict="PASS", checks=passing), tmp_path
    )
    assert good.problem is None and good.check_ids == ("F01:G2.a", "F01:G2.b")

    failing = [{"id": "G2.a", "status": "FAIL"}]
    honest = read_family(
        _artifact(tmp_path, gate_id="G2", verdict="FAIL", checks=failing), tmp_path
    )
    assert honest.statuses == {"F01:G2.a": CheckStatus.FAIL}

    for document in (
        {"gate_id": "G3", "verdict": "PASS", "checks": passing},
        {"gate_id": "G2", "verdict": "PASS", "checks": failing},
        {"gate_id": "G2", "verdict": "PASS", "checks": passing + passing},
        {"gate_id": "G2", "verdict": "PASS", "checks": []},
        {"gate_id": "G2", "verdict": "PASS", "checks": [{"status": "PASS"}]},
        {"gate_id": "G2", "verdict": "PASS", "checks": [{"id": "x", "status": "?"}]},
    ):
        bad = read_family(_artifact(tmp_path, **document), tmp_path)
        assert bad.statuses == {"F01:artifact": CheckStatus.ERROR}, document
    (tmp_path / "S2.json").write_text("{not json")
    broken = read_family(GateFamily("F01", "x", "G2", "S2.json"), tmp_path)
    assert broken.statuses == {"F01:artifact": CheckStatus.ERROR}
