from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import rfc8785

from campaigns.dogbone_v1.s12_closeout import (
    s12_gate_payload,
    write_s12_gate_artifact,
)
from piginot.objectives import (
    GateStatus,
    S12GateMeasurements,
    evaluate_s12_gate,
    run_s12_verification,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REVISION = "1" * 40
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


def _payload() -> dict[str, object]:
    return s12_gate_payload(run_s12_verification(), **_payload_kwargs())


def _payload_kwargs() -> dict[str, object]:
    return {
        "code_coverage_percent": 93.83,
        "implementation_source_revision": _REVISION,
        "branch": "codex/s12-objectives",
        "timestamp_utc": "2026-09-10T00:00:00Z",
        "focused_test_result": "77 passed; total branch coverage 93.83%",
        "full_test_result": "1903 passed, 2 skipped",
        "repository_root": _REPOSITORY_ROOT,
    }


def _measurements(payload: dict[str, object]) -> S12GateMeasurements:
    values = payload["measurements"]
    assert isinstance(values, dict)
    return S12GateMeasurements(
        exact_max_residuals=values["exact_max_residuals"],  # type: ignore[arg-type]
        perturbation_fired_terms={
            name: tuple(fired)
            for name, fired in cast(
                dict[str, list[str]], values["perturbation_fired_terms"]
            ).items()
        },
        perturbation_recomputed_term_counts=values[
            "perturbation_recomputed_term_counts"
        ],  # type: ignore[arg-type]
        finite_difference_relative_errors=values["finite_difference_relative_errors"],  # type: ignore[arg-type]
        detached_terms=tuple(cast(list[str], values["detached_terms"])),
        reachable_parameter_names=tuple(
            cast(list[str], values["reachable_parameter_names"])
        ),
        required_parameter_names=tuple(
            cast(list[str], values["required_parameter_names"])
        ),
        reachability_loss_paths=values["reachability_loss_paths"],  # type: ignore[arg-type]
        anisotropy_relative_change=cast(float, values["anisotropy_relative_change"]),
        coordinate_anisotropy_mutant_fired=cast(
            bool, values["coordinate_anisotropy_mutant_fired"]
        ),
        fem_solve_success=cast(bool, values["fem_solve_success"]),
        weak_strong_difference=cast(float, values["weak_strong_difference"]),
        weak_residual_max=cast(float, values["weak_residual_max"]),
        projected_strong_residual_max=cast(
            float, values["projected_strong_residual_max"]
        ),
        reconstructed_stress_variation=cast(
            float, values["reconstructed_stress_variation"]
        ),
        weak_strong_fem_request_id=cast(str, values["weak_strong_fem_request_id"]),
        weak_strong_quadrature_point_count=cast(
            int, values["weak_strong_quadrature_point_count"]
        ),
        s8_discretization_error_bound=cast(
            float, values["s8_discretization_error_bound"]
        ),
        constitutive_residual_shape=tuple(
            cast(list[int], values["constitutive_residual_shape"])
        ),  # type: ignore[arg-type]
        constitutive_rejects_out_of_plane=cast(
            bool, values["constitutive_rejects_out_of_plane"]
        ),
        holdout_refused=cast(bool, values["holdout_refused"]),
        holdout_access_mutant_fired=cast(bool, values["holdout_access_mutant_fired"]),
        code_coverage_percent=cast(float, values["code_coverage_percent"]),
    )


def test_s12_gate_payload_reduces_all_nine_checks_to_pass() -> None:
    payload = _payload()
    checks = payload["checks"]
    assert isinstance(checks, list)
    evaluation = evaluate_s12_gate(_measurements(payload))

    assert payload["schema_version"] == "1.0"
    assert payload["step"] == "S12"
    assert payload["gate_id"] == "G12"
    assert payload["implementation_source_revision"] == _REVISION
    assert tuple(check["id"] for check in checks) == _REQUIRED_CHECK_IDS
    assert payload["verdict"] == evaluation.verdict.value == GateStatus.PASS.value
    for recorded, reduced in zip(checks, evaluation.checks, strict=True):
        assert recorded["status"] == reduced.status.value
        assert recorded["measured"] == _json_value(reduced.measured)
        assert recorded["threshold"] == _json_value(reduced.threshold)
        assert recorded["comparison"] == reduced.comparison
        assert recorded["reason"] == reduced.reason


def test_s12_gate_payload_binds_s8_s11_and_contingency_evidence() -> None:
    payload = _payload()
    upstream = cast(dict[str, Any], payload["upstream_evidence"])

    assert upstream["s8_gate"] == {
        "path": "artifacts/gates/S8.json",
        "content_hash": "6b72f5bb0cf55bbd0ea8c7e8b149b5c37b162069ed450d80712452246705cbf4",
        "patch_check_id": "G8.patch",
        "patch_measured": 3.2722362831057246e-13,
        "discretization_error_bound": 1.0e-12,
        "verdict": "PASS",
    }
    assert upstream["s11_gate"] == {
        "path": "artifacts/gates/S11.json",
        "content_hash": "8f5396827f2dea671c50ad6c1a8a1b147ba4b9f51a44c74b845131d394c68a24",
        "architecture_id": "pi-ginot-s11-boundary-cross-attention-v1",
        "module_digest": "blake2b256:ac064556f53e573b4f9de63a2610077dfd38141b6eef695731302cea3a560f47",
        "verdict": "PASS",
    }
    assert upstream["s11_contingency"] == {
        "path": "artifacts/campaigns/dogbone-v1/S11-CONTINGENCY-EVIDENCE.json",
        "content_hash": "aae56fcb3ab6b6b2c693c5b3dbdbce89a1716fd90086dd1786217b189957ec18",
        "protocol_content_hash": "f16a4f63c3a8cfc3ee321bc4e15162802cbc6f106ed807f5aa4bc59eceac81df",
        "selected_candidate_id": "A-s11-baseline",
        "fallback_to_baseline": True,
        "development_opened": False,
    }


def test_s12_gate_payload_maps_family_neutral_reachability_to_campaign_inputs() -> None:
    payload = _payload()
    reachability = cast(dict[str, Any], payload["parameter_reachability"])
    bindings = cast(list[dict[str, Any]], reachability["campaign_bindings"])

    assert reachability["all_s11_conditioning_parameters_covered"] is True
    assert {source for item in bindings for source in item["source_parameters"]} == {
        "L_total",
        "W_grip",
        "W_gauge",
        "R_fillet",
        "E",
        "nu",
        "nominal_axial_strain",
    }
    prescribed = next(
        item
        for item in bindings
        if item["harness_parameter"] == "prescribed_displacement"
    )
    assert prescribed["source_parameters"] == ["L_total", "nominal_axial_strain"]
    assert prescribed["binding_kind"] == "derived_boundary_quantity"


def test_s12_gate_payload_preserves_scope_and_fixture_caveats() -> None:
    payload = _payload()
    scope = cast(dict[str, Any], payload["scope"])
    evidence = cast(dict[str, Any], payload["implementation_evidence"])

    assert scope["training_run"] is False
    assert scope["production_checkpoint_created"] is False
    assert scope["holdout_accessed"] is False
    assert scope["development_accessed"] is False
    assert scope["legacy_experiments_accessed"] is False
    assert scope["selected_architecture_candidate"] == "A-s11-baseline"
    assert scope["s13_unblocked"] is True
    assert evidence["perturbation_fixture_scope"] == (
        "channel-isolated analytic residual fixtures; not one coupled physical state"
    )
    assert evidence["focused_test_pass_count"] == 77
    assert evidence["full_test_pass_count"] == 1903
    assert evidence["full_test_skipped_count"] == 2
    assert evidence["focused_coverage_command"].startswith(
        "uv run --frozen --all-extras python -m pytest"
    )
    assert evidence["full_test_command"] == (
        "uv run --frozen --all-extras python -m pytest -q"
    )
    assert evidence["weak_strong_fixture"]["element_order"] == 2
    assert evidence["weak_strong_fixture"]["comparison"] == (
        "weak residual versus strong residual projected onto the same test function"
    )
    assert evidence["weak_strong_fixture"]["claim_scope"].startswith(
        "same-field integration-by-parts"
    )
    assert (
        "not a claim that FEM discretization error is 1e-12"
        in evidence["weak_strong_fixture"]["comparison_bound_scope"]
    )


def test_s12_gate_writer_is_canonical_and_refuses_divergent_rewrites(
    tmp_path: Path,
) -> None:
    payload = _payload()
    target = tmp_path / "S12.json"

    write_s12_gate_artifact(payload, target)
    write_s12_gate_artifact(payload, target)

    assert target.read_bytes() == rfc8785.dumps(cast(Any, payload))
    changed = dict(payload)
    changed["verdict"] = "FAIL"
    with pytest.raises(ValueError, match="refusing to replace"):
        write_s12_gate_artifact(changed, target)


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"implementation_source_revision": "short"}, "revision"),
        ({"code_coverage_percent": float("nan")}, "coverage"),
        ({"code_coverage_percent": 93.82}, "focused coverage"),
        ({"branch": " branch"}, "branch"),
        ({"timestamp_utc": "not-a-date"}, "timestamp_utc"),
        ({"focused_test_result": "FAILED"}, "focused_test_result"),
        ({"full_test_result": "FAILED"}, "full_test_result"),
    ),
)
def test_s12_gate_payload_rejects_malformed_closeout_inputs(
    changes: dict[str, object], match: str
) -> None:
    kwargs = _payload_kwargs()
    kwargs.update(changes)

    with pytest.raises(ValueError, match=match):
        s12_gate_payload(run_s12_verification(), **kwargs)  # type: ignore[arg-type]


def test_s12_closeout_refuses_to_publish_a_nonpassing_gate() -> None:
    failing = replace(run_s12_verification(), weak_residual_max=0.0)

    with pytest.raises(RuntimeError, match="G12 gate is not PASS"):
        s12_gate_payload(failing, **_payload_kwargs())  # type: ignore[arg-type]


def _json_value(value: object) -> object:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}  # type: ignore[union-attr]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
