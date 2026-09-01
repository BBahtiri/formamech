from __future__ import annotations

import json
from pathlib import Path

from piginot.doe import (
    DesignCoverageDiagnostics,
    GateStatus,
    ReplacementEvidence,
    S9GateMeasurements,
    evaluate_s9_gate,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GATE_PATH = _REPOSITORY_ROOT / "artifacts" / "gates" / "S9.json"
_REQUIRED_CHECK_IDS = (
    "G9.immutable",
    "G9.one-role",
    "G9.leakage",
    "G9.visibility",
    "G9.no-node-split",
    "G9.commitment-order",
    "G9.replacement",
    "G9.coverage",
)
_FORBIDDEN_PRIVATE_FIELDS = (
    b"allocation_seed_hex",
    b"doe_plan_preimage",
    b"doe_seed_hex",
    b"holdout_coordinates",
    b"holdout_salt_hex",
    b"replacement_policy_preimage",
    b"role_ledger_preimage",
)


def _gate() -> dict[str, object]:
    return json.loads(_GATE_PATH.read_bytes())


def test_s9_gate_reduces_all_eight_required_checks_to_pass() -> None:
    gate = _gate()
    recorded_checks = gate["checks"]
    assert isinstance(recorded_checks, list)
    checks = {check["id"]: check for check in recorded_checks}
    assert tuple(checks) == _REQUIRED_CHECK_IDS

    design_policy = gate["production_design_policy"]
    assert isinstance(design_policy, dict)
    diagnostics = design_policy["measured"]
    assert isinstance(diagnostics, dict)
    replacement = checks["G9.replacement"]["measured"]
    assert isinstance(replacement, dict)
    commitment = checks["G9.commitment-order"]["measured"]
    assert isinstance(commitment, dict)

    evaluation = evaluate_s9_gate(
        S9GateMeasurements(
            immutable_rewrite_raised=checks["G9.immutable"]["measured"],
            one_role_trials=checks["G9.one-role"]["measured"]["trials"],
            one_role_violations=checks["G9.one-role"]["measured"]["violations"],
            leakage_trials=checks["G9.leakage"]["measured"]["trials"],
            leakage_violations=checks["G9.leakage"]["measured"]["violations"],
            holdout_visibility_violations=checks["G9.visibility"]["measured"],
            node_split_mutant_fired=checks["G9.no-node-split"]["measured"],
            commitment_file_order_verified=commitment["file_mtime_precedes"],
            commitment_git_order_verified=commitment["git_commit_precedes"],
            reveal_verified=commitment["reveal_verifies"],
            replacement=ReplacementEvidence(
                failures_injected=replacement["failures_injected"],
                failures_recorded=replacement["failures_recorded"],
                deterministic_rule_matches=replacement["deterministic_rule_matches"],
            ),
            code_coverage_percent=checks["G9.coverage"]["measured"],
            design_diagnostics=DesignCoverageDiagnostics(
                empirical_fill_distance=diagnostics["empirical_fill_distance"],
                minimum_stratum_count=diagnostics["minimum_stratum_count"],
                boundary_coverage=diagnostics["boundary_coverage"],
                stratum_coverage=diagnostics["stratum_coverage"],
                interaction_coverage=diagnostics["interaction_coverage"],
                difficult_region_coverage=diagnostics["difficult_region_coverage"],
                uncovered_region_ids=tuple(diagnostics["uncovered_region_ids"]),
                unresolved_region_ids=tuple(diagnostics["unresolved_region_ids"]),
            ),
        )
    )

    assert evaluation.verdict is GateStatus.PASS
    assert all(check["status"] == "PASS" for check in recorded_checks)
    assert gate["verdict"] == "PASS"


def test_s9_gate_fails_closed_on_the_ratified_production_design_vector() -> None:
    policy = _gate()["production_design_policy"]
    assert isinstance(policy, dict)
    measured = policy["measured"]
    thresholds = policy["thresholds"]
    assert isinstance(measured, dict)
    assert isinstance(thresholds, dict)

    expected_checks = {
        "empirical_fill_distance": measured["empirical_fill_distance"]
        <= thresholds["maximum_empirical_fill_distance"],
        "minimum_stratum_count": measured["minimum_stratum_count"]
        >= thresholds["minimum_stratum_count"],
        "stratum_coverage": measured["stratum_coverage"]
        >= thresholds["minimum_coverage_fraction"],
        "boundary_coverage": measured["boundary_coverage"]
        >= thresholds["minimum_coverage_fraction"],
        "interaction_coverage": measured["interaction_coverage"]
        >= thresholds["minimum_coverage_fraction"],
        "difficult_region_coverage": measured["difficult_region_coverage"]
        >= thresholds["minimum_coverage_fraction"],
        "uncovered_region_ids": len(measured["uncovered_region_ids"])
        <= thresholds["maximum_uncovered_regions"],
        "unresolved_region_ids": len(measured["unresolved_region_ids"])
        <= thresholds["maximum_unresolved_regions"],
    }

    assert policy["plan_hash_verified"] is True
    assert policy["checks"] == expected_checks
    assert all(expected_checks.values())
    assert policy["status"] == "PASS"


def test_s9_gate_records_nonvacuous_visibility_and_honest_order_scope() -> None:
    gate = _gate()
    checks = {check["id"]: check for check in gate["checks"]}
    visibility = checks["G9.visibility"]["evidence"]
    order = checks["G9.commitment-order"]["evidence"]

    assert visibility == {
        "case_id_overlaps": 0,
        "claim_scope": (
            "S9 public collocation-eligible population; "
            "the S10 sampler is not implemented"
        ),
        "exact_coordinate_overlaps": 0,
        "holdout_cases_checked": 64,
        "leakage_group_overlaps": 0,
        "public_ledger_mismatches": 0,
        "public_manifest_holdout_entries": 0,
        "public_points_checked": 376,
        "sealed_leakage_groups_checked": 64,
    }
    assert order["pre_training_snapshot"] is True
    assert order["witnessed_nonlegacy_artifact_count"] == 0
    assert any(
        "future training artifact" in limitation for limitation in gate["limitations"]
    )


def test_s9_gate_contains_no_private_reveal_material_or_external_path() -> None:
    payload = _GATE_PATH.read_bytes()
    gate = json.loads(payload)

    assert all(field not in payload for field in _FORBIDDEN_PRIVATE_FIELDS)
    assert gate["private_custody"] == {
        "backup_operator_only_acl": True,
        "backup_present": True,
        "copies_byte_identical": True,
        "external_path_disclosed": False,
        "primary_operator_only_acl": True,
        "primary_present": True,
        "status": "PASS",
    }
