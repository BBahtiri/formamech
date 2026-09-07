"""Write the DogBone S11 architecture gate artifact from measured evidence."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import rfc8785

from campaigns.dogbone_v1.s11_evidence import (
    DogboneS11Evidence,
    evidence_payload,
    run_dogbone_s11_evidence,
)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_UPSTREAM_REPOSITORY = "https://github.com/BBahtiri/PI-GINOT-DogBone"
_UPSTREAM_MODEL_FILES = (
    "models/pi_ginot.py",
    "models/geometry_encoder.py",
    "models/physics_decoder.py",
)
_COVERAGE_COMMAND = (
    "$env:PYTHONPATH='src'; pytest -q tests/operators "
    "tests/campaigns/test_dogbone_s11_evidence.py tests/test_s11_anti_scope.py "
    "--cov=src/piginot/operators --cov-report=term-missing --cov-fail-under=80"
)


def s11_gate_payload(
    evidence: DogboneS11Evidence,
    *,
    implementation_source_revision: str,
    branch: str,
    timestamp_utc: str,
    focused_test_result: str,
) -> dict[str, object]:
    """Build the public S11 gate payload without private/training material."""

    if type(evidence) is not DogboneS11Evidence:
        raise TypeError("evidence must be exact DogboneS11Evidence")
    _require_revision(implementation_source_revision, "implementation_source_revision")
    _require_text(branch, "branch")
    _require_text(timestamp_utc, "timestamp_utc")
    _require_text(focused_test_result, "focused_test_result")

    measured = evidence_payload(evidence)
    gate = measured["gate"]
    measurements = measured["measurements"]
    diagnostics = measured["diagnostics"]
    assert isinstance(gate, dict)
    assert isinstance(measurements, dict)
    assert isinstance(diagnostics, dict)
    return {
        "schema_version": "1.0",
        "step": "S11",
        "gate_id": "G11",
        "timestamp_utc": timestamp_utc,
        "branch": branch,
        "implementation_source_revision": implementation_source_revision,
        "upstream_reference": {
            "repository": _UPSTREAM_REPOSITORY,
            "files": list(_UPSTREAM_MODEL_FILES),
            "adaptation": (
                "Contract-hardened S11 adaptation of the DogBone PI-GINOT "
                "branch/trunk operator: encode-once boundary geometry, latent "
                "tokens, query positional encoding, cross-attention decoder, and "
                "FiLM conditioning. S11 deliberately excludes upstream training, "
                "UI, agent, and checkpoint-loading surfaces."
            ),
        },
        "scope": {
            "claim": (
                "The fixed S11 operator architecture satisfies all nine structural "
                "G11 checks. This is not a trained production checkpoint and makes "
                "no accuracy claim."
            ),
            "training_run": False,
            "production_checkpoint_created": False,
            "post_hoc_field_scaling": False,
            "http_or_ui_surface": False,
            "served_stress_source": "P_served = P_predicted from the independent P head",
            "dirichlet_policy": (
                "component-wise soft enforcement is the default; hard lifting "
                "requires explicit full-range boundary-refinement evidence"
            ),
            "next_step": "run the ADR 0001 S11-S12 architecture iteration before S12",
            "s12_contingency_requirement": (
                "ADR 0001 still requires the scheduled S11-S12 architecture "
                "iteration and a first-class traction-free condition on P_served."
            ),
        },
        "architecture": {
            "architecture_id": evidence.architecture_id,
            "module_digest": evidence.module_digest,
            "reference_mesh_id": evidence.reference_mesh_id,
            "model_seed": evidence.model_seed,
            "permutation_seed": evidence.permutation_seed,
            "parameter_names": measured["parameter_names"],
            "boundary_features": ["xy", "semantic_tag_one_hot", "normal"],
            "output_heads": {
                "u": {"shape": "(batch, query_points, 2)"},
                "P": {"shape": "(batch, query_points, 2, 2)"},
            },
        },
        "measurements": measurements,
        "diagnostics": diagnostics,
        "implementation_evidence": {
            "focused_coverage_command": _COVERAGE_COMMAND,
            "focused_test_result": focused_test_result,
            "scoped_coverage_percent": measurements["code_coverage_percent"],
            "scoped_coverage_threshold_percent": 80.0,
            "safe_loader": (
                "Only .safetensors files are accepted; header key, shape, dtype, "
                "architecture_id, and module_digest are checked before tensor "
                "materialization; pickle formats are rejected before deserialization."
            ),
        },
        "checks": gate["checks"],
        "verdict": gate["verdict"],
        "verdict_reason": "all nine required G11 checks passed",
        "limitations": [
            "S11 proves architecture invariants only; no model training or accuracy is claimed.",
            "Soft Dirichlet enforcement is the only default path. Hard lifting remains unavailable without separate full-range verification evidence.",
            "The S1 peak-stress underprediction contingency is not resolved by S11 and must carry into S12 objective and traction-free checks.",
        ],
    }


def write_s11_gate_artifact(payload: dict[str, object], target: Path) -> None:
    """Create the canonical gate artifact once, refusing divergent replacement."""

    if type(payload) is not dict:
        raise TypeError("payload must be a JSON object")
    if not isinstance(target, Path):
        raise TypeError("target must be a pathlib.Path")
    encoded = rfc8785.dumps(cast(Any, payload))
    if target.exists():
        if target.is_symlink() or target.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different S11 evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)


def _require_revision(value: str, label: str) -> str:
    if type(value) is not str or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 40-character lowercase git revision")
    return value


def _require_text(value: str, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


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
        raise RuntimeError("S11 gate closeout requires a clean committed source tree")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return _require_revision(revision, "implementation_source_revision")


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


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage-percent", required=True, type=float)
    parser.add_argument("--focused-test-result", required=True)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    evidence = run_dogbone_s11_evidence(code_coverage_percent=args.coverage_percent)
    payload = s11_gate_payload(
        evidence,
        implementation_source_revision=_clean_source_revision(),
        branch=_current_branch(),
        timestamp_utc=datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        focused_test_result=args.focused_test_result,
    )
    write_s11_gate_artifact(payload, args.output)
    architecture = cast(dict[str, object], payload["architecture"])
    print(f"wrote {args.output}")
    print(f"architecture_id: {architecture['architecture_id']}")
    print(f"module_digest: {architecture['module_digest']}")
    print(f"verdict: {payload['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "s11_gate_payload",
    "write_s11_gate_artifact",
)
