"""Write the provenance-bound DogBone S12 gate artifact."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import rfc8785

from piginot.contracts import strict_json_loads
from piginot.objectives import (
    GateStatus,
    S12GateMeasurements,
    S12VerificationReport,
    evaluate_s12_gate,
    run_s12_verification,
)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_FOCUSED_TEST_PATTERN = re.compile(
    r"(?P<passed>[1-9][0-9]*) passed; total branch coverage "
    r"(?P<coverage>[0-9]+(?:\.[0-9]+)?)%"
)
_FULL_TEST_PATTERN = re.compile(
    r"(?P<passed>[1-9][0-9]*) passed"
    r"(?:, (?P<skipped>[1-9][0-9]*) skipped)?"
)
_S8_GATE_PATH = Path("artifacts/gates/S8.json")
_S11_GATE_PATH = Path("artifacts/gates/S11.json")
_CONTINGENCY_PATH = Path("artifacts/campaigns/dogbone-v1/S11-CONTINGENCY-EVIDENCE.json")
_S8_CONTENT_HASH = "6b72f5bb0cf55bbd0ea8c7e8b149b5c37b162069ed450d80712452246705cbf4"
_S11_CONTENT_HASH = "8f5396827f2dea671c50ad6c1a8a1b147ba4b9f51a44c74b845131d394c68a24"
_CONTINGENCY_CONTENT_HASH = (
    "aae56fcb3ab6b6b2c693c5b3dbdbce89a1716fd90086dd1786217b189957ec18"
)
_CONTINGENCY_PROTOCOL_HASH = (
    "f16a4f63c3a8cfc3ee321bc4e15162802cbc6f106ed807f5aa4bc59eceac81df"
)
_BASELINE_CANDIDATE_ID = "A-s11-baseline"
_COVERAGE_COMMAND = (
    "uv run --frozen --all-extras python -m pytest -q "
    "tests/objectives tests/test_s12_anti_scope.py "
    "--cov=src/piginot/objectives --cov-branch --cov-report=term-missing "
    "--cov-fail-under=85"
)
_FULL_TEST_COMMAND = "uv run --frozen --all-extras python -m pytest -q"


def s12_gate_payload(
    report: S12VerificationReport,
    *,
    code_coverage_percent: float,
    implementation_source_revision: str,
    branch: str,
    timestamp_utc: str,
    focused_test_result: str,
    full_test_result: str,
    repository_root: Path,
) -> dict[str, object]:
    """Build public G12 evidence after validating every upstream artifact."""

    if type(report) is not S12VerificationReport:
        raise TypeError("report must be exact S12VerificationReport")
    coverage = _coverage(code_coverage_percent)
    _require_revision(implementation_source_revision, "implementation source revision")
    _require_text(branch, "branch")
    _require_timestamp(timestamp_utc)
    focused_pass_count = _focused_test_pass_count(focused_test_result, coverage)
    full_pass_count, full_skipped_count = _full_test_counts(full_test_result)
    upstream, conditioning_names = _upstream_evidence(repository_root, report)
    measurements = _measurements(report, coverage)
    evaluation = evaluate_s12_gate(measurements)
    if evaluation.verdict is not GateStatus.PASS:
        failed = ", ".join(
            check.check_id
            for check in evaluation.checks
            if check.status is not GateStatus.PASS
        )
        raise RuntimeError(f"G12 gate is not PASS: {failed}")
    bindings = _campaign_bindings()
    bound_names = {
        name
        for binding in bindings
        for name in cast(list[str], binding["source_parameters"])
    }
    all_conditioning_covered = bound_names == set(conditioning_names)
    if not all_conditioning_covered:
        raise ValueError("campaign reachability bindings do not cover S11 conditioning")
    checks = [
        {
            "id": check.check_id,
            "status": check.status.value,
            "measured": _json_value(check.measured),
            "threshold": _json_value(check.threshold),
            "comparison": check.comparison,
            "reason": check.reason,
        }
        for check in evaluation.checks
    ]
    return {
        "schema_version": "1.0",
        "step": "S12",
        "gate_id": "G12",
        "timestamp_utc": timestamp_utc,
        "branch": branch,
        "implementation_source_revision": implementation_source_revision,
        "upstream_evidence": upstream,
        "objective_policy": {
            "registered_modes": ["DATA_ONLY", "PHYSICS_ONLY", "HYBRID"],
            "default_mode": "HYBRID",
            "equilibrium_forms": ["weak", "strong"],
            "default_reported_equilibrium_form": "weak",
            "nondimensionalization": (
                "declared length, displacement and stress scales; body-force and "
                "weak-force scales are derived"
            ),
            "served_stress_source": "P_served = P_predicted",
            "constitutive_components": "in-plane 2x2 only",
            "angular_term": "independent skew component of P_served F_3D^T",
            "traction_free_term": "P_served N on declared free boundaries",
        },
        "parameter_reachability": {
            "family_neutral_harness_parameters": list(report.required_parameter_names),
            "verified_loss_paths": dict(report.reachability_loss_paths),
            "campaign_bindings": bindings,
            "all_s11_conditioning_parameters_covered": all_conditioning_covered,
        },
        "measurements": _measurements_payload(measurements),
        "implementation_evidence": {
            "focused_coverage_command": _COVERAGE_COMMAND,
            "focused_test_result": focused_test_result,
            "focused_test_pass_count": focused_pass_count,
            "full_test_command": _FULL_TEST_COMMAND,
            "full_test_result": full_test_result,
            "full_test_pass_count": full_pass_count,
            "full_test_skipped_count": full_skipped_count,
            "scoped_branch_coverage_percent": coverage,
            "scoped_branch_coverage_threshold_percent": 85.0,
            "perturbation_fixture_scope": (
                "channel-isolated analytic residual fixtures; not one coupled "
                "physical state"
            ),
            "perturbation_recomputed_term_counts": dict(
                report.perturbation_recomputed_term_counts
            ),
            "weak_strong_fixture": {
                "request_id": report.weak_strong_fem_request_id,
                "case": "pure_dirichlet manufactured solution",
                "element_order": 2,
                "quadrature_point_count": report.weak_strong_quadrature_point_count,
                "stress_reconstruction": (
                    "continuous recovered P2 field represented cellwise by its "
                    "quadratic polynomial"
                ),
                "test_function": "x(1-x)y(1-y)",
                "comparison": (
                    "weak residual versus strong residual projected onto the same "
                    "test function"
                ),
                "claim_scope": (
                    "same-field integration-by-parts sign, scale and quadrature "
                    "consistency; not proof that the recovered stress satisfies "
                    "equilibrium"
                ),
                "comparison_bound_scope": (
                    "the tighter nondimensional S8 G8.patch threshold is reused as "
                    "a numerical consistency tolerance; it is not a claim that FEM "
                    "discretization error is 1e-12"
                ),
                "weak_residual_max": report.weak_residual_max,
                "projected_strong_residual_max": (report.projected_strong_residual_max),
                "reconstructed_stress_variation": (
                    report.reconstructed_stress_variation
                ),
            },
            "holdout_access_mutant": (
                "a verification-local negative control omits the role guard while "
                "reusing the same required terms and weights; the production "
                "registry API still refuses HOLDOUT"
            ),
        },
        "scope": {
            "claim": (
                "All nine G12 objective and gradient-verification checks pass for "
                "the selected S11 baseline architecture. This is not an accuracy "
                "or trained-model claim."
            ),
            "training_run": False,
            "production_checkpoint_created": False,
            "holdout_accessed": False,
            "development_accessed": False,
            "legacy_experiments_accessed": False,
            "selected_architecture_candidate": _BASELINE_CANDIDATE_ID,
            "s13_unblocked": evaluation.verdict is GateStatus.PASS,
            "next_step": (
                "S13 deterministic production training on the public S10 dataset "
                "with baseline architecture A"
            ),
        },
        "checks": checks,
        "verdict": evaluation.verdict.value,
        "verdict_reason": "all nine required G12 checks passed",
        "limitations": [
            "G12 verifies objective formulas, gradients and finite negative controls; it does not measure operator prediction accuracy.",
            "Perturbation isolation uses independent residual-channel fixtures because a shared physical-state perturbation can legitimately activate coupled terms.",
            "The weak/strong check uses one P2 manufactured FEM solve and one interior test function to verify same-field integration-by-parts consistency; it neither proves recovered-stress equilibrium nor estimates FEM discretization error.",
            "Reachability establishes nonzero paths at one deterministic probe; it does not guarantee useful conditioning sensitivity after training.",
            "No DEVELOPMENT or HOLDOUT case was opened, and no production training was run.",
        ],
    }


def write_s12_gate_artifact(payload: dict[str, object], target: Path) -> None:
    """Create canonical evidence once and refuse divergent replacement."""

    if type(payload) is not dict:
        raise TypeError("payload must be a JSON object")
    if not isinstance(target, Path):
        raise TypeError("target must be a pathlib.Path")
    encoded = rfc8785.dumps(cast(Any, payload))
    if target.exists():
        if target.is_symlink() or target.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different S12 evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)


def _measurements(
    report: S12VerificationReport, coverage: float
) -> S12GateMeasurements:
    return S12GateMeasurements(
        exact_max_residuals=report.exact_max_residuals,
        perturbation_fired_terms=report.perturbation_fired_terms,
        perturbation_recomputed_term_counts=(
            report.perturbation_recomputed_term_counts
        ),
        finite_difference_relative_errors=report.finite_difference_relative_errors,
        detached_terms=report.detached_terms,
        reachable_parameter_names=report.reachable_parameter_names,
        required_parameter_names=report.required_parameter_names,
        reachability_loss_paths=report.reachability_loss_paths,
        anisotropy_relative_change=report.anisotropy_relative_change,
        coordinate_anisotropy_mutant_fired=(report.coordinate_anisotropy_mutant_fired),
        fem_solve_success=report.fem_solve_success,
        weak_strong_difference=report.weak_strong_difference,
        weak_residual_max=report.weak_residual_max,
        projected_strong_residual_max=report.projected_strong_residual_max,
        reconstructed_stress_variation=report.reconstructed_stress_variation,
        weak_strong_fem_request_id=report.weak_strong_fem_request_id,
        weak_strong_quadrature_point_count=(report.weak_strong_quadrature_point_count),
        s8_discretization_error_bound=report.s8_discretization_error_bound,
        constitutive_residual_shape=report.constitutive_residual_shape,
        constitutive_rejects_out_of_plane=report.constitutive_rejects_out_of_plane,
        holdout_refused=report.holdout_refused,
        holdout_access_mutant_fired=report.holdout_access_mutant_fired,
        code_coverage_percent=coverage,
    )


def _measurements_payload(measurements: S12GateMeasurements) -> dict[str, object]:
    return {
        "exact_max_residuals": dict(measurements.exact_max_residuals),
        "perturbation_fired_terms": {
            name: list(terms)
            for name, terms in measurements.perturbation_fired_terms.items()
        },
        "perturbation_recomputed_term_counts": dict(
            measurements.perturbation_recomputed_term_counts
        ),
        "finite_difference_relative_errors": dict(
            measurements.finite_difference_relative_errors
        ),
        "detached_terms": list(measurements.detached_terms),
        "reachable_parameter_names": list(measurements.reachable_parameter_names),
        "required_parameter_names": list(measurements.required_parameter_names),
        "reachability_loss_paths": dict(measurements.reachability_loss_paths),
        "anisotropy_relative_change": measurements.anisotropy_relative_change,
        "coordinate_anisotropy_mutant_fired": (
            measurements.coordinate_anisotropy_mutant_fired
        ),
        "fem_solve_success": measurements.fem_solve_success,
        "weak_strong_difference": measurements.weak_strong_difference,
        "weak_residual_max": measurements.weak_residual_max,
        "projected_strong_residual_max": (measurements.projected_strong_residual_max),
        "reconstructed_stress_variation": (measurements.reconstructed_stress_variation),
        "weak_strong_fem_request_id": measurements.weak_strong_fem_request_id,
        "weak_strong_quadrature_point_count": (
            measurements.weak_strong_quadrature_point_count
        ),
        "s8_discretization_error_bound": measurements.s8_discretization_error_bound,
        "constitutive_residual_shape": list(measurements.constitutive_residual_shape),
        "constitutive_rejects_out_of_plane": (
            measurements.constitutive_rejects_out_of_plane
        ),
        "holdout_refused": measurements.holdout_refused,
        "holdout_access_mutant_fired": measurements.holdout_access_mutant_fired,
        "code_coverage_percent": measurements.code_coverage_percent,
    }


def _campaign_bindings() -> list[dict[str, object]]:
    rows = (
        ("geometry_axial_extent", ("L_total",), "direct_conditioning"),
        ("geometry_outer_width", ("W_grip",), "direct_conditioning"),
        ("geometry_inner_width", ("W_gauge",), "direct_conditioning"),
        ("geometry_transition_radius", ("R_fillet",), "direct_conditioning"),
        ("elastic_modulus", ("E",), "direct_conditioning"),
        ("poisson_ratio", ("nu",), "direct_conditioning"),
        ("load_amplitude", ("nominal_axial_strain",), "direct_conditioning"),
        (
            "prescribed_displacement",
            ("L_total", "nominal_axial_strain"),
            "derived_boundary_quantity",
        ),
    )
    return [
        {
            "harness_parameter": harness,
            "source_parameters": list(sources),
            "binding_kind": kind,
        }
        for harness, sources, kind in rows
    ]


def _upstream_evidence(
    repository_root: Path, report: S12VerificationReport
) -> tuple[dict[str, object], tuple[str, ...]]:
    if not isinstance(repository_root, Path) or not repository_root.is_dir():
        raise ValueError("repository_root must be an existing directory")
    s8 = _read_document(repository_root, _S8_GATE_PATH, _S8_CONTENT_HASH)
    s11 = _read_document(repository_root, _S11_GATE_PATH, _S11_CONTENT_HASH)
    contingency = _read_document(
        repository_root, _CONTINGENCY_PATH, _CONTINGENCY_CONTENT_HASH
    )
    s8_patch = _named_check(s8, "G8.patch")
    if (
        s8.get("step") != "S8"
        or s8.get("verdict") != "PASS"
        or s8_patch.get("status") != "PASS"
        or s8_patch.get("threshold") != report.s8_discretization_error_bound
    ):
        raise ValueError("S8 patch evidence does not supply the G12 bound")
    architecture = _mapping(s11, "architecture")
    conditioning = tuple(_string_list(architecture, "parameter_names"))
    if s11.get("step") != "S11" or s11.get("verdict") != "PASS":
        raise ValueError("S11 architecture gate must be PASS")
    selection = _mapping(_mapping(contingency, "train_screen_decision"), "selection")
    if (
        contingency.get("protocol_content_hash") != _CONTINGENCY_PROTOCOL_HASH
        or contingency.get("final_candidate_id") != _BASELINE_CANDIDATE_ID
        or contingency.get("development_confirmation") is not None
        or selection.get("selected_candidate_id") != _BASELINE_CANDIDATE_ID
        or selection.get("fallback_to_baseline") is not True
    ):
        raise ValueError("S11 contingency did not preserve the baseline boundary")
    model_files = contingency.get("model_files")
    if not isinstance(model_files, list):
        raise ValueError("S11 contingency model files are missing")
    baseline = next(
        (
            item
            for item in model_files
            if isinstance(item, Mapping)
            and item.get("candidate_id") == _BASELINE_CANDIDATE_ID
        ),
        None,
    )
    if baseline is None or baseline.get("operator_module_digest") != architecture.get(
        "module_digest"
    ):
        raise ValueError("S11 baseline module digest is not preserved")
    return (
        {
            "s8_gate": {
                "path": _S8_GATE_PATH.as_posix(),
                "content_hash": _S8_CONTENT_HASH,
                "patch_check_id": "G8.patch",
                "patch_measured": s8_patch["measured"],
                "discretization_error_bound": s8_patch["threshold"],
                "verdict": s8["verdict"],
            },
            "s11_gate": {
                "path": _S11_GATE_PATH.as_posix(),
                "content_hash": _S11_CONTENT_HASH,
                "architecture_id": architecture["architecture_id"],
                "module_digest": architecture["module_digest"],
                "verdict": s11["verdict"],
            },
            "s11_contingency": {
                "path": _CONTINGENCY_PATH.as_posix(),
                "content_hash": _CONTINGENCY_CONTENT_HASH,
                "protocol_content_hash": contingency["protocol_content_hash"],
                "selected_candidate_id": contingency["final_candidate_id"],
                "fallback_to_baseline": selection["fallback_to_baseline"],
                "development_opened": False,
            },
        },
        conditioning,
    )


def _read_document(root: Path, relative: Path, expected_hash: str) -> dict[str, object]:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"required upstream evidence is not a regular file: {relative}"
        )
    raw = path.read_bytes()
    document = strict_json_loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"upstream evidence must be a JSON object: {relative}")
    typed = cast(dict[str, object], document)
    framed = rfc8785.dumps(cast(Any, {"domain": "content", "payload": typed}))
    digest = hashlib.blake2b(framed, digest_size=32).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"upstream evidence content hash changed: {relative}")
    return typed


def _named_check(document: Mapping[str, object], check_id: str) -> dict[str, object]:
    checks = document.get("checks")
    if not isinstance(checks, list):
        raise ValueError("upstream gate checks are missing")
    match = next(
        (
            check
            for check in checks
            if isinstance(check, dict) and check.get("id") == check_id
        ),
        None,
    )
    if match is None:
        raise ValueError(f"upstream gate check is missing: {check_id}")
    return cast(dict[str, object], match)


def _mapping(values: Mapping[str, object], name: str) -> dict[str, object]:
    value = values.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"upstream field {name} must be an object")
    return cast(dict[str, object], value)


def _string_list(values: Mapping[str, object], name: str) -> list[str]:
    value = values.get(name)
    if not isinstance(value, list) or any(
        type(item) is not str or not item for item in value
    ):
        raise ValueError(f"upstream field {name} must contain strings")
    return cast(list[str], value)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _coverage(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 100.0
    ):
        raise ValueError("coverage must be finite and lie in [0, 100]")
    return float(value)


def _require_revision(value: object, label: str) -> str:
    if type(value) is not str or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 40-character lowercase git revision")
    return value


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _require_timestamp(value: object) -> str:
    timestamp = _require_text(value, "timestamp_utc")
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(
            "timestamp_utc must be canonical UTC text YYYY-MM-DDTHH:MM:SSZ"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != timestamp:
        raise ValueError(
            "timestamp_utc must be canonical UTC text YYYY-MM-DDTHH:MM:SSZ"
        )
    return timestamp


def _focused_test_pass_count(value: object, coverage: float) -> int:
    summary = _require_text(value, "focused_test_result")
    match = _FOCUSED_TEST_PATTERN.fullmatch(summary)
    if match is None:
        raise ValueError(
            "focused_test_result must record a successful pass count and coverage"
        )
    measured_coverage = _coverage(float(match.group("coverage")))
    if not math.isclose(measured_coverage, coverage, rel_tol=0.0, abs_tol=0.005):
        raise ValueError("focused coverage must match code_coverage_percent")
    return int(match.group("passed"))


def _full_test_counts(value: object) -> tuple[int, int]:
    summary = _require_text(value, "full_test_result")
    match = _FULL_TEST_PATTERN.fullmatch(summary)
    if match is None:
        raise ValueError(
            "full_test_result must record a successful pass count and optional skips"
        )
    skipped = match.group("skipped")
    return int(match.group("passed")), 0 if skipped is None else int(skipped)


def _clean_source_revision() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("S12 gate closeout requires a clean committed source tree")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return _require_revision(revision, "implementation source revision")


def _current_branch() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    branch = subprocess.run(
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return _require_text(branch, "branch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage-percent", required=True, type=float)
    parser.add_argument("--focused-test-result", required=True)
    parser.add_argument("--full-test-result", required=True)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    payload = s12_gate_payload(
        run_s12_verification(),
        code_coverage_percent=args.coverage_percent,
        implementation_source_revision=_clean_source_revision(),
        branch=_current_branch(),
        timestamp_utc=datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        focused_test_result=args.focused_test_result,
        full_test_result=args.full_test_result,
        repository_root=repository_root,
    )
    write_s12_gate_artifact(payload, args.output)
    print(f"wrote {args.output}")
    print(f"verdict: {payload['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("s12_gate_payload", "write_s12_gate_artifact")
