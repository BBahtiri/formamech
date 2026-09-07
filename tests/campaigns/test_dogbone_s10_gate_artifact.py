from __future__ import annotations

import hashlib
from pathlib import Path

import rfc8785

from piginot.contracts import strict_json_loads
from piginot.datasets import (
    GateStatus,
    S10GateMeasurements,
    evaluate_s10_gate,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GATE_PATH = _REPOSITORY_ROOT / "artifacts" / "gates" / "S10.json"
_REPORT_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "s10-production-dataset.json"
)
_REPORT_CONTENT_HASH = (
    "ed9a7758a87ce9fddab1ec365fc3529372b49c181406a09723573a91a96c9ef2"
)
_REQUIRED_CHECK_IDS = (
    "G10.traceability",
    "G10.projection-error",
    "G10.cross-mesh",
    "G10.normalization",
    "G10.failed-visible",
    "G10.reproducible",
    "G10.coverage",
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


def _document(path: Path) -> dict[str, object]:
    document = strict_json_loads(path.read_bytes())
    assert isinstance(document, dict)
    return document


def _content_hash(document: dict[str, object]) -> str:
    framed = rfc8785.dumps({"domain": "content", "payload": document})
    return hashlib.blake2b(framed, digest_size=32).hexdigest()


def _measurements(gate: dict[str, object]) -> S10GateMeasurements:
    raw_checks = gate["checks"]
    assert isinstance(raw_checks, list)
    checks = {check["id"]: check for check in raw_checks}
    traceability = checks["G10.traceability"]["measured"]
    projection = checks["G10.projection-error"]["measured"]
    cross_mesh = checks["G10.cross-mesh"]["measured"]
    normalization = checks["G10.normalization"]["measured"]
    failed_visible = checks["G10.failed-visible"]["measured"]
    reproducible = checks["G10.reproducible"]["measured"]
    coverage = checks["G10.coverage"]["measured"]
    assert isinstance(traceability, dict)
    assert isinstance(projection, dict)
    assert isinstance(cross_mesh, dict)
    assert isinstance(normalization, dict)
    assert isinstance(failed_visible, dict)
    assert isinstance(reproducible, dict)

    return S10GateMeasurements(
        sample_count=traceability["samples"],
        fem_result_trace_count=traceability["fem_result_hash"],
        sampler_stream_trace_count=traceability["sampler_stream"],
        ambiguous_sample_trace_count=traceability["ambiguous"],
        cp_manifest_binding_verified=traceability["cp_manifest_binding_verified"],
        request_plan_binding_verified=traceability["request_plan_binding_verified"],
        fem_result_store_resolution_verified=traceability[
            "fem_result_store_resolution_verified"
        ],
        projection_field_count=projection["projected_fields"],
        projection_error_artifact_count=projection["error_artifacts"],
        source_verified_projection_count=projection["source_verified"],
        drop_projection_error_mutant_fired=projection[
            "drop_projection_error_mutant_fired"
        ],
        physical_integral_comparison_count=cross_mesh["physical_integral_comparisons"],
        source_verified_integral_comparison_count=cross_mesh["source_verified"],
        stress_node_pairing_mutant_fired=cross_mesh["stress_node_pairing_mutant_fired"],
        normalization_statistic_count=normalization["statistics"],
        normalization_non_train_source_count=normalization["non_train_sources"],
        normalization_non_active_source_count=normalization["non_active_sources"],
        normalization_snapshot_binding_verified=normalization[
            "training_snapshot_binding_verified"
        ],
        normalize_on_all_splits_mutant_fired=normalization[
            "normalize_on_all_splits_mutant_fired"
        ],
        failed_fem_case_count=failed_visible["failed_fem_cases"],
        represented_failed_fem_case_count=failed_visible[
            "represented_failed_fem_cases"
        ],
        failure_visibility_canary_count=failed_visible["failure_visibility_canaries"],
        represented_failure_visibility_canary_count=failed_visible[
            "represented_failure_visibility_canaries"
        ],
        collocation_replay_count=reproducible["replays"],
        byte_identical_replay_count=reproducible["byte_identical"],
        collocation_request_binding_verified=reproducible["request_binding_verified"],
        code_coverage_percent=coverage,
    )


def test_s10_gate_reduces_all_seven_required_checks_to_pass() -> None:
    gate = _document(_GATE_PATH)
    raw_checks = gate["checks"]
    assert isinstance(raw_checks, list)
    assert tuple(check["id"] for check in raw_checks) == _REQUIRED_CHECK_IDS

    evaluation = evaluate_s10_gate(_measurements(gate))

    assert evaluation.verdict is GateStatus.PASS
    assert gate["verdict"] == evaluation.verdict.value
    assert gate["verdict_reason"] == evaluation.verdict_reason
    for recorded, reduced in zip(raw_checks, evaluation.checks, strict=True):
        assert recorded["id"] == reduced.check_id
        assert recorded["status"] == reduced.status.value
        assert recorded["measured"] == reduced.measured
        assert recorded["threshold"] == reduced.threshold
        assert recorded["comparison"] == reduced.comparison
        assert recorded["reason"] == reduced.reason


def test_s10_gate_is_bound_to_the_materialized_production_report() -> None:
    gate = _document(_GATE_PATH)
    report = _document(_REPORT_PATH)
    dataset = gate["production_dataset"]
    raw_checks = gate["checks"]
    assert isinstance(dataset, dict)
    assert isinstance(raw_checks, list)
    checks = {check["id"]: check for check in raw_checks}

    assert rfc8785.dumps(report) == _REPORT_PATH.read_bytes()
    assert _content_hash(report) == _REPORT_CONTENT_HASH
    assert dataset["report_content_hash"] == _REPORT_CONTENT_HASH
    assert dataset["fem_source_revision"] == report["fem_source_revision"]
    assert (
        dataset["materializer_source_revision"]
        == report["materializer_source_revision"]
    )
    assert gate["campaign"]["campaign_hash"] == report["campaign_hash"]
    assert (
        gate["campaign"]["public_case_manifest_hash"]
        == report["public_case_manifest_hash"]
    )
    assert dataset["request_plan_hash"] == report["request_plan_hash"]
    assert dataset["execution_profile_hash"] == report["execution_profile_hash"]

    cases = report["cases"]
    comparisons = report["cross_mesh_comparisons"]
    normalization = report["normalization"]
    canary = report["failure_visibility_canary"]
    assert isinstance(cases, list)
    assert isinstance(comparisons, list)
    assert isinstance(normalization, dict)
    assert isinstance(canary, dict)
    assert len(cases) == 344
    assert sum(case["collocation_point_count"] for case in cases) == 88_064
    assert sum(case["projection_source_verified"] for case in cases) == 344
    assert sum(case["collocation_replay_verified"] for case in cases) == 344
    assert len(comparisons) == 343
    assert all(item["method"] == "physical_domain_integral" for item in comparisons)
    assert sum(item["source_verified"] for item in comparisons) == 343
    assert len(report["failed_cases"]) == 0
    assert normalization["statistic_count"] == 135
    assert normalization["non_train_source_count"] == 0
    assert normalization["non_active_source_count"] == 0
    assert normalization["snapshot_binding_verified"] is True
    assert len(normalization["normalized_samples"]) == 256
    assert all(
        sample["round_trip_verified"] for sample in normalization["normalized_samples"]
    )
    assert canary["is_production_failure"] is False
    assert canary["represented_failure_count"] == 1

    role_counts = {
        snapshot["role"]: snapshot["case_count"]
        for snapshot in report["role_snapshots"]
    }
    assert role_counts == {"DEVELOPMENT": 64, "PILOT": 24, "TRAIN": 256}
    assert dataset["population"] == {
        "planned_fem_cases": 344,
        "successful_fem_cases": len(cases),
        "failed_fem_cases": len(report["failed_cases"]),
        "role_counts": {"DEVELOPMENT": 64, "PILOT": 24, "TRAIN": 256},
        "collocation_points_per_case": 256,
        "collocation_points": 88_064,
        "traceable_samples": len(cases) + 88_064,
        "projected_fields": 344,
        "physical_integral_comparisons": len(comparisons),
        "normalized_train_samples": len(normalization["normalized_samples"]),
        "normalization_statistics": normalization["statistic_count"],
    }
    assert checks["G10.traceability"]["measured"] == {
        "samples": len(cases) + 88_064,
        "fem_result_hash": len(cases),
        "sampler_stream": 88_064,
        "ambiguous": 0,
        "cp_manifest_binding_verified": True,
        "request_plan_binding_verified": True,
        "fem_result_store_resolution_verified": True,
    }
    assert checks["G10.projection-error"]["measured"]["projected_fields"] == len(cases)
    assert checks["G10.cross-mesh"]["measured"]["physical_integral_comparisons"] == len(
        comparisons
    )
    assert (
        checks["G10.normalization"]["measured"]["statistics"]
        == normalization["statistic_count"]
    )
    assert checks["G10.failed-visible"]["measured"] == {
        "failed_fem_cases": len(report["failed_cases"]),
        "represented_failed_fem_cases": 0,
        "failure_visibility_canaries": 1,
        "represented_failure_visibility_canaries": canary["represented_failure_count"],
    }
    assert checks["G10.reproducible"]["measured"]["replays"] == len(cases)


def test_s10_gate_records_cryptographic_resolution_and_honest_provenance() -> None:
    gate = _document(_GATE_PATH)
    dataset = gate["production_dataset"]
    profile = gate["execution_profile"]
    assert isinstance(dataset, dict)
    assert isinstance(profile, dict)

    assert dataset["object_store_verification"] == {
        "all_references_resolved": True,
        "cryptographically_read_reference_count": 2324,
        "unique_reference_count": 2324,
        "verification_api": "ObjectStore.has_object",
        "verification_operation": (
            "called once for every unique store hash referenced by the report"
        ),
        "verification_semantics": (
            "has_object calls get, which reads bytes, recomputes the domain-separated "
            "content hash, and verifies committed size"
        ),
        "corruption_detection_test": (
            "tests/store/test_store.py::"
            "test_corrupt_object_is_detected_and_not_returned"
        ),
    }
    assert profile["profile_file_persistence"] == (
        "recovered after FEM completion and before S10 closeout"
    )
    assert profile["recovery_disclosed"] is True
    assert profile["ledger_binding_verified"] is True
    assert profile["report_binding_verified"] is True
    assert gate["scope"] == {
        "claim": (
            "The complete public S10 production dataset is materialized and all "
            "seven required G10 checks pass."
        ),
        "holdout_excluded_from_public_contracts": True,
        "private_reveal_used": False,
        "production_fem_batch_run": True,
        "reserves_excluded_from_active_request_plan": True,
        "s11_blocked": False,
    }


def test_s10_public_evidence_contains_no_private_reveal_material() -> None:
    payload = _REPORT_PATH.read_bytes() + _GATE_PATH.read_bytes()

    assert all(field not in payload for field in _FORBIDDEN_PRIVATE_FIELDS)
