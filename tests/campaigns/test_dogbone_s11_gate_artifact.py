from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, cast

import rfc8785

from campaigns.dogbone_v1.s11_closeout import s11_gate_payload
from campaigns.dogbone_v1.s11_evidence import run_dogbone_s11_evidence
from piginot.contracts import strict_json_loads
from piginot.operators import (
    ARCHITECTURE_ID,
    GateStatus,
    S11GateMeasurements,
    evaluate_s11_gate,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GATE_PATH = _REPOSITORY_ROOT / "artifacts" / "gates" / "S11.json"
_IMPLEMENTATION_REVISION = "8b3db083fa2fb8f6a861305ae89ecc3f7c0a0a23"
_EXPECTED_CONTENT_HASH = (
    "8f5396827f2dea671c50ad6c1a8a1b147ba4b9f51a44c74b845131d394c68a24"
)
_REQUIRED_CHECK_IDS = (
    "G11.permutation",
    "G11.resampling",
    "G11.node-order",
    "G11.query-batch",
    "G11.reachability",
    "G11.mask",
    "G11.safetensors",
    "G11.heads",
    "G11.coverage",
)
_FORBIDDEN_SCOPE_TERMS = (
    b"checkpoint_state",
    b"optimizer_state",
    b"private_reveal",
    b"holdout_coordinates",
    b"holdout_salt",
    b"torch.load",
    b"pickle.load",
)
_KERNEL_ZERO_TOLERANCE = 1.0e-12
_KERNEL_RELATIVE_TOLERANCE = 1.0e-6
_KERNEL_ZERO_METRICS = (
    "permutation_max_relative_change",
    "node_order_relative_change",
    "query_batch_relative_change",
    "mask_control_relative_change",
)
_KERNEL_NONZERO_METRICS = (
    "resampling_relative_change",
    "mask_mutant_relative_change",
)


def _document(path: Path) -> dict[str, object]:
    document = strict_json_loads(path.read_bytes())
    assert isinstance(document, dict)
    return cast(dict[str, object], document)


def _content_hash(document: dict[str, object]) -> str:
    framed = rfc8785.dumps(cast(Any, {"domain": "content", "payload": document}))
    return hashlib.blake2b(framed, digest_size=32).hexdigest()


def _measurements(gate: dict[str, object]) -> S11GateMeasurements:
    measurements = gate["measurements"]
    assert isinstance(measurements, dict)
    return S11GateMeasurements(**measurements)


def _kernel_metrics(payload: dict[str, object]) -> dict[str, float]:
    measurements = cast(dict[str, object], payload["measurements"])
    diagnostics = cast(dict[str, object], payload["diagnostics"])
    return {
        "permutation_max_relative_change": float(
            measurements["permutation_max_relative_change"]
        ),
        "resampling_relative_change": float(measurements["resampling_relative_change"]),
        "node_order_relative_change": float(measurements["node_order_relative_change"]),
        "query_batch_relative_change": float(
            measurements["query_batch_relative_change"]
        ),
        "mask_control_relative_change": float(
            diagnostics["mask_control_relative_change"]
        ),
        "mask_mutant_relative_change": float(
            diagnostics["mask_mutant_relative_change"]
        ),
    }


def _assert_check_metrics_match_measurements(payload: dict[str, object]) -> None:
    measurements = cast(dict[str, object], payload["measurements"])
    checks = cast(list[dict[str, object]], payload["checks"])
    by_id = {str(check["id"]): check for check in checks}
    permutation = cast(dict[str, object], by_id["G11.permutation"]["measured"])
    resampling = cast(dict[str, object], by_id["G11.resampling"]["measured"])

    assert (
        permutation["max_relative_change"]
        == measurements["permutation_max_relative_change"]
    )
    assert resampling["relative_change"] == measurements["resampling_relative_change"]
    assert (
        by_id["G11.node-order"]["measured"]
        == measurements["node_order_relative_change"]
    )
    assert (
        by_id["G11.query-batch"]["measured"]
        == measurements["query_batch_relative_change"]
    )


def _without_kernel_metrics(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    measurements = cast(dict[str, object], normalized["measurements"])
    diagnostics = cast(dict[str, object], normalized["diagnostics"])
    for key in (
        "permutation_max_relative_change",
        "resampling_relative_change",
        "node_order_relative_change",
        "query_batch_relative_change",
    ):
        measurements[key] = 0.0
    diagnostics["mask_control_relative_change"] = 0.0
    diagnostics["mask_mutant_relative_change"] = 0.0

    checks = cast(list[dict[str, object]], normalized["checks"])
    by_id = {str(check["id"]): check for check in checks}
    permutation = cast(dict[str, object], by_id["G11.permutation"]["measured"])
    permutation["max_relative_change"] = 0.0
    resampling = cast(dict[str, object], by_id["G11.resampling"]["measured"])
    resampling["relative_change"] = 0.0
    by_id["G11.node-order"]["measured"] = 0.0
    by_id["G11.query-batch"]["measured"] = 0.0
    return normalized


def test_s11_gate_artifact_reduces_all_required_checks_to_pass() -> None:
    gate = _document(_GATE_PATH)
    raw_checks = gate["checks"]
    assert isinstance(raw_checks, list)

    evaluation = evaluate_s11_gate(_measurements(gate))

    assert rfc8785.dumps(cast(Any, gate)) == _GATE_PATH.read_bytes()
    assert tuple(check["id"] for check in raw_checks) == _REQUIRED_CHECK_IDS
    assert gate["schema_version"] == "1.0"
    assert gate["step"] == "S11"
    assert gate["gate_id"] == "G11"
    assert gate["implementation_source_revision"] == _IMPLEMENTATION_REVISION
    assert gate["verdict"] == evaluation.verdict.value == GateStatus.PASS.value
    for recorded, reduced in zip(raw_checks, evaluation.checks, strict=True):
        assert recorded["status"] == reduced.status.value
        assert recorded["measured"] == reduced.measured
        assert recorded["threshold"] == reduced.threshold
        assert recorded["comparison"] == reduced.comparison
        assert recorded["reason"] == reduced.reason


def test_s11_gate_artifact_binds_the_architecture_and_upstream_scope() -> None:
    gate = _document(_GATE_PATH)
    architecture = gate["architecture"]
    scope = gate["scope"]
    upstream = gate["upstream_reference"]
    implementation_evidence = gate["implementation_evidence"]
    assert isinstance(architecture, dict)
    assert isinstance(scope, dict)
    assert isinstance(upstream, dict)
    assert isinstance(implementation_evidence, dict)

    assert architecture["architecture_id"] == ARCHITECTURE_ID
    assert architecture["module_digest"] == (
        "blake2b256:ac064556f53e573b4f9de63a2610077dfd38141b6eef695731302cea3a560f47"
    )
    assert architecture["boundary_features"] == [
        "xy",
        "semantic_tag_one_hot",
        "normal",
    ]
    assert architecture["output_heads"] == {
        "P": {"shape": "(batch, query_points, 2, 2)"},
        "u": {"shape": "(batch, query_points, 2)"},
    }
    assert upstream["repository"] == "https://github.com/BBahtiri/PI-GINOT-DogBone"
    assert upstream["revision"] == "9f6794141beca9d3e66433465c3b07039a7e41d2"
    assert upstream["files"] == [
        "models/pi_ginot.py",
        "models/geometry_encoder.py",
        "models/physics_decoder.py",
    ]
    assert scope["training_run"] is False
    assert scope["production_checkpoint_created"] is False
    assert scope["post_hoc_field_scaling"] is False
    assert scope["http_or_ui_surface"] is False
    assert scope["served_stress_source"] == (
        "P_served = P_predicted from the independent P head"
    )
    assert "s12_unblocked" not in scope
    assert scope["next_step"] == (
        "run the ADR 0001 S11-S12 architecture iteration before S12"
    )
    assert "ADR 0001" in str(scope["s12_contingency_requirement"])
    assert implementation_evidence["scoped_coverage_percent"] == 89.9
    assert implementation_evidence["focused_test_result"] == (
        "92 passed; total coverage 89.90%"
    )
    assert "exact byte layout" in str(implementation_evidence["safe_loader"])
    limitations = gate["limitations"]
    assert isinstance(limitations, list)
    assert any("one normalized parameter probe" in item for item in limitations)


def test_s11_gate_artifact_matches_the_generator_across_numeric_kernels() -> None:
    gate = _document(_GATE_PATH)
    implementation_evidence = gate["implementation_evidence"]
    assert isinstance(implementation_evidence, dict)
    expected = s11_gate_payload(
        run_dogbone_s11_evidence(code_coverage_percent=89.9),
        implementation_source_revision=_IMPLEMENTATION_REVISION,
        branch=str(gate["branch"]),
        timestamp_utc=str(gate["timestamp_utc"]),
        focused_test_result=str(implementation_evidence["focused_test_result"]),
    )

    recorded_metrics = _kernel_metrics(gate)
    generated_metrics = _kernel_metrics(expected)
    _assert_check_metrics_match_measurements(gate)
    _assert_check_metrics_match_measurements(expected)
    for metric in _KERNEL_ZERO_METRICS:
        assert abs(recorded_metrics[metric]) <= _KERNEL_ZERO_TOLERANCE
        assert abs(generated_metrics[metric]) <= _KERNEL_ZERO_TOLERANCE
    for metric in _KERNEL_NONZERO_METRICS:
        assert math.isclose(
            recorded_metrics[metric],
            generated_metrics[metric],
            rel_tol=_KERNEL_RELATIVE_TOLERANCE,
            abs_tol=_KERNEL_ZERO_TOLERANCE,
        )

    assert rfc8785.dumps(cast(Any, _without_kernel_metrics(expected))) == rfc8785.dumps(
        cast(Any, _without_kernel_metrics(gate))
    )


def test_s11_gate_artifact_contains_no_private_or_training_payloads() -> None:
    payload = _GATE_PATH.read_bytes()

    assert _content_hash(_document(_GATE_PATH)) == _EXPECTED_CONTENT_HASH
    assert all(term not in payload for term in _FORBIDDEN_SCOPE_TERMS)
