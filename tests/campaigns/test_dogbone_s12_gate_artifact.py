from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import rfc8785

from campaigns.dogbone_v1.s12_closeout import s12_gate_payload
from piginot.contracts import strict_json_loads
from piginot.objectives import (
    GateStatus,
    S12GateMeasurements,
    evaluate_s12_gate,
    run_s12_verification,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GATE_PATH = _REPOSITORY_ROOT / "artifacts" / "gates" / "S12.json"
_IMPLEMENTATION_REVISION = "0bc851413e13fb13843ebb2b016c2ca2b87f69cc"
_EXPECTED_CONTENT_HASH = (
    "621c48a0903d33ed1af7e32fe2b4bc09a5d0315c96f8a1d19720e4b0836e2dd0"
)
_REQUIRED_CHECK_IDS = (
    "G12.exact",
    "G12.perturbed",
    "G12.fd",
    "G12.reachability",
    "G12.anisotropy",
    "G12.weak-strong",
    "G12.constitutive-shape",
    "G12.role-isolation",
    "G12.coverage",
)
_FORBIDDEN_PAYLOAD_TERMS = (
    b"checkpoint_state",
    b"holdout_coordinates",
    b"holdout_salt",
    b"optimizer_state",
    b"private_reveal",
    b"training_state",
)


def _document() -> dict[str, object]:
    document = strict_json_loads(_GATE_PATH.read_bytes())
    assert isinstance(document, dict)
    return cast(dict[str, object], document)


def _content_hash(document: dict[str, object]) -> str:
    framed = rfc8785.dumps(cast(Any, {"domain": "content", "payload": document}))
    return hashlib.blake2b(framed, digest_size=32).hexdigest()


def _measurements(gate: dict[str, object]) -> S12GateMeasurements:
    values = gate["measurements"]
    assert isinstance(values, dict)
    typed = cast(dict[str, object], values)
    return S12GateMeasurements(
        exact_max_residuals=cast(dict[str, float], typed["exact_max_residuals"]),
        perturbation_fired_terms={
            name: tuple(fired)
            for name, fired in cast(
                dict[str, list[str]], typed["perturbation_fired_terms"]
            ).items()
        },
        perturbation_recomputed_term_counts=cast(
            dict[str, int], typed["perturbation_recomputed_term_counts"]
        ),
        finite_difference_relative_errors=cast(
            dict[str, float], typed["finite_difference_relative_errors"]
        ),
        detached_terms=tuple(cast(list[str], typed["detached_terms"])),
        reachable_parameter_names=tuple(
            cast(list[str], typed["reachable_parameter_names"])
        ),
        required_parameter_names=tuple(
            cast(list[str], typed["required_parameter_names"])
        ),
        reachability_loss_paths=cast(dict[str, str], typed["reachability_loss_paths"]),
        anisotropy_relative_change=cast(float, typed["anisotropy_relative_change"]),
        coordinate_anisotropy_mutant_fired=cast(
            bool, typed["coordinate_anisotropy_mutant_fired"]
        ),
        fem_solve_success=cast(bool, typed["fem_solve_success"]),
        weak_strong_difference=cast(float, typed["weak_strong_difference"]),
        weak_residual_max=cast(float, typed["weak_residual_max"]),
        projected_strong_residual_max=cast(
            float, typed["projected_strong_residual_max"]
        ),
        reconstructed_stress_variation=cast(
            float, typed["reconstructed_stress_variation"]
        ),
        weak_strong_fem_request_id=cast(str, typed["weak_strong_fem_request_id"]),
        weak_strong_quadrature_point_count=cast(
            int, typed["weak_strong_quadrature_point_count"]
        ),
        s8_discretization_error_bound=cast(
            float, typed["s8_discretization_error_bound"]
        ),
        constitutive_residual_shape=tuple(
            cast(list[int], typed["constitutive_residual_shape"])
        ),  # type: ignore[arg-type]
        constitutive_rejects_out_of_plane=cast(
            bool, typed["constitutive_rejects_out_of_plane"]
        ),
        holdout_refused=cast(bool, typed["holdout_refused"]),
        holdout_access_mutant_fired=cast(bool, typed["holdout_access_mutant_fired"]),
        code_coverage_percent=cast(float, typed["code_coverage_percent"]),
    )


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def test_s12_gate_artifact_reduces_all_required_checks_to_pass() -> None:
    gate = _document()
    raw_checks = gate["checks"]
    assert isinstance(raw_checks, list)
    evaluation = evaluate_s12_gate(_measurements(gate))

    assert rfc8785.dumps(cast(Any, gate)) == _GATE_PATH.read_bytes()
    assert tuple(check["id"] for check in raw_checks) == _REQUIRED_CHECK_IDS
    assert gate["schema_version"] == "1.0"
    assert gate["step"] == "S12"
    assert gate["gate_id"] == "G12"
    assert gate["implementation_source_revision"] == _IMPLEMENTATION_REVISION
    assert gate["verdict"] == evaluation.verdict.value == GateStatus.PASS.value
    for recorded, reduced in zip(raw_checks, evaluation.checks, strict=True):
        assert recorded["status"] == reduced.status.value
        assert recorded["measured"] == _json_value(reduced.measured)
        assert recorded["threshold"] == _json_value(reduced.threshold)
        assert recorded["comparison"] == reduced.comparison
        assert recorded["reason"] == reduced.reason


def test_s12_gate_artifact_replay_preserves_policy_and_passes() -> None:
    gate = _document()
    evidence = gate["implementation_evidence"]
    assert isinstance(evidence, dict)
    expected = s12_gate_payload(
        run_s12_verification(),
        code_coverage_percent=cast(float, evidence["scoped_branch_coverage_percent"]),
        implementation_source_revision=_IMPLEMENTATION_REVISION,
        branch=cast(str, gate["branch"]),
        timestamp_utc=cast(str, gate["timestamp_utc"]),
        focused_test_result=cast(str, evidence["focused_test_result"]),
        full_test_result=cast(str, evidence["full_test_result"]),
        repository_root=_REPOSITORY_ROOT,
    )

    assert expected["verdict"] == gate["verdict"] == GateStatus.PASS.value
    assert expected["objective_policy"] == gate["objective_policy"]
    assert expected["parameter_reachability"] == gate["parameter_reachability"]
    assert expected["upstream_evidence"] == gate["upstream_evidence"]
    assert expected["scope"] == gate["scope"]
    expected_checks = cast(list[dict[str, object]], expected["checks"])
    recorded_checks = cast(list[dict[str, object]], gate["checks"])
    assert [(item["id"], item["status"]) for item in expected_checks] == [
        (item["id"], item["status"]) for item in recorded_checks
    ]


def test_s12_gate_artifact_binds_upstream_evidence_and_measured_commands() -> None:
    gate = _document()
    upstream = gate["upstream_evidence"]
    evidence = gate["implementation_evidence"]
    assert isinstance(upstream, dict)
    assert isinstance(evidence, dict)

    assert upstream["s8_gate"]["content_hash"] == (
        "6b72f5bb0cf55bbd0ea8c7e8b149b5c37b162069ed450d80712452246705cbf4"
    )
    assert upstream["s11_gate"]["content_hash"] == (
        "8f5396827f2dea671c50ad6c1a8a1b147ba4b9f51a44c74b845131d394c68a24"
    )
    assert upstream["s11_contingency"]["content_hash"] == (
        "aae56fcb3ab6b6b2c693c5b3dbdbce89a1716fd90086dd1786217b189957ec18"
    )
    assert evidence["focused_test_result"] == (
        "89 passed; total branch coverage 92.99%"
    )
    assert evidence["full_test_result"] == "1928 passed, 2 skipped"
    assert evidence["focused_test_pass_count"] == 89
    assert evidence["full_test_pass_count"] == 1928
    assert evidence["full_test_skipped_count"] == 2


def test_s12_gate_artifact_has_expected_identity_and_no_private_payload() -> None:
    gate = _document()
    scope = gate["scope"]
    assert isinstance(scope, dict)
    payload = _GATE_PATH.read_bytes()

    assert _content_hash(gate) == _EXPECTED_CONTENT_HASH
    assert scope["training_run"] is False
    assert scope["production_checkpoint_created"] is False
    assert scope["holdout_accessed"] is False
    assert scope["development_accessed"] is False
    assert scope["legacy_experiments_accessed"] is False
    assert scope["s13_unblocked"] is True
    assert all(term not in payload for term in _FORBIDDEN_PAYLOAD_TERMS)
