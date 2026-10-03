"""Build the DogBone v1 bundle of the frozen S13c candidate (S15).

The bundle holds the compiled reference mesh, morph map and ratified envelope,
the ``CP-FREEZE`` weights, a model specification that reproduces the S14
serving arithmetic exactly, the G14 report with every upstream gate, the
outstanding work and the machine-readable limitations.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from campaigns.dogbone_v1.s11_contingency import STRESS_REFERENCE_MPA
from campaigns.dogbone_v1.s13c_protocol import (
    DOGBONE_S13C_CONDITIONING_PARAMETERS,
    load_s13c_normalization,
    operator_config,
)
from campaigns.dogbone_v1.s14_candidate import (
    INDICATOR_LENGTH_SCALE_MM,
    frozen_candidate_record,
)
from campaigns.dogbone_v1.s14_envelope import _family, dogbone_envelope
from campaigns.dogbone_v1.s14_reference import _EXCLUDED_TAGS, _ROOT_TAG
from piginot.bundle import (
    CONSTITUTIVE_PLANE_STRESS,
    BundleManifest,
    ConditioningInput,
    ConfidenceSpec,
    ModelSpec,
    QoiSpec,
    ScaledParameter,
    build_bundle,
)
from piginot.operators import ENERGY_ARCHITECTURE_ID

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _ROOT / "artifacts" / "campaigns" / "dogbone-v1"
_GATES = _ROOT / "artifacts" / "gates"
GATE_REPORT_PATH = _ARTIFACTS / "S14-GATE.json"
DEVELOPMENT_PATH = _ARTIFACTS / "S14-DEVELOPMENT.json"
BUNDLE_NAME = "dogbone-v1-s13c-cp-freeze"
UPSTREAM_GATES = tuple(f"S{n}.json" for n in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) + (
    "S11-ENERGY.json",
    "S12.json",
    "S13.json",
)
PEAK_QOI = "root_peak_first_piola_frobenius"
UNITS = {
    "coordinates": "mm",
    "u": "mm",
    "F_2D": "1",
    "F33": "1",
    "J_3D": "1",
    "P_served": "MPa",
    "sigma": "MPa",
    "confidence.indicator": "1",
    "confidence.stress_error_bound": "MPa",
    "constitutive_gap": "1",
    PEAK_QOI: "MPa",
}
BOUNDARY_CONDITIONS = {
    "left_symmetry": ("x",),
    "bottom": ("y",),
    "right_grip": ("x",),
}
NOT_RUN_WORK = (
    (
        "G15",
        "evaluated on this bundle after it is built; the gate artifact is "
        "artifacts/gates/S15.json",
    ),
    ("G16", "the S16 mutation suite has not been run"),
    ("G17", "the S17 release gate has not been run"),
    (
        "experimental-validation",
        "out of scope: the framework verifies against FEM solutions only",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_spec() -> ModelSpec:
    """The serving arithmetic of the S14 evaluation, as data."""

    normalization = load_s13c_normalization()
    index = {name: i for i, name in enumerate(normalization.parameter_names)}
    config = dataclasses.asdict(operator_config())
    config["parameter_names"] = list(config["parameter_names"])
    calibration = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))["confidence"]
    curve = calibration["calibration"]
    return ModelSpec(
        architecture_id=ENERGY_ARCHITECTURE_ID,
        operator_config=config,
        conditioning=tuple(
            ConditioningInput(
                name,
                name,
                normalization.mean[index[name]],
                normalization.standard_deviation[index[name]],
            )
            for name in DOGBONE_S13C_CONDITIONING_PARAMETERS
        ),
        frame_origin=(ScaledParameter((), 0.0), ScaledParameter((), 0.0)),
        frame_extent=(
            ScaledParameter(("L_total",), 0.5),
            ScaledParameter(("W_grip",), 0.5),
        ),
        lifting_kind="axis-aligned-tension",
        lifting_displacement=ScaledParameter(("nominal_axial_strain", "L_total"), 0.5),
        material_name="dogbone-v1-compressible-neo-hookean",
        material_model="compressible_neo_hookean",
        material_parameters={
            "elastic_modulus": "elastic_modulus",
            "poisson_ratio": "poisson_ratio",
        },
        boundary_conditions=BOUNDARY_CONDITIONS,
        served_stress=CONSTITUTIVE_PLANE_STRESS,
        dtype="float64",
        query_chunk=2_048,
        qois=(QoiSpec(PEAK_QOI, "P_served", _ROOT_TAG, _EXCLUDED_TAGS, "MPa"),),
        confidence=ConfidenceSpec(
            variant=calibration["indicator"],
            length_scale_mm=INDICATOR_LENGTH_SCALE_MM,
            reference_stress_mpa=STRESS_REFERENCE_MPA,
            quantile=curve["quantile"],
            upper_edges=tuple(curve["upper_edges"]),
            values=tuple(curve["values"]),
        ),
        units=UNITS,
    )


def outstanding_work(gate: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every non-passing, waived or not-run item, with its reason."""

    items: list[dict[str, str]] = []
    for name in UPSTREAM_GATES:
        document = json.loads((_GATES / name).read_text(encoding="utf-8"))
        for check in document.get("checks", []):
            if check.get("status") != "PASS":
                items.append(
                    {
                        "id": str(check.get("id")),
                        "status": str(check.get("status")),
                        "reason": str(
                            check.get("reason") or check.get("detail") or "not given"
                        ),
                    }
                )
    for check in gate["report"]["checks"]:
        reason = str(check.get("reason") or "")
        if check["status"] != "PASS":
            items.append(
                {"id": check["id"], "status": check["status"], "reason": reason}
            )
        elif "decision owner" in reason:
            items.append({"id": check["id"], "status": "WAIVED", "reason": reason})
    items.extend(
        {"id": identifier, "status": "NOT_RUN", "reason": reason}
        for identifier, reason in NOT_RUN_WORK
    )
    return items


def verification_report() -> dict[str, Any]:
    gate = json.loads(GATE_REPORT_PATH.read_text(encoding="utf-8"))
    upstream = []
    for name in UPSTREAM_GATES:
        path = _GATES / name
        document = json.loads(path.read_text(encoding="utf-8"))
        upstream.append(
            {
                "artifact": f"artifacts/gates/{name}",
                "gate_id": document.get("gate_id"),
                "verdict": document.get("verdict"),
                "sha256": _sha256(path),
            }
        )
    return {
        "gate_id": gate["gate_id"],
        "verdict": gate["verdict"],
        "claim_level": gate["claim_level"],
        "protocol_sha256": gate["protocol_sha256"],
        "report": gate["report"],
        "report_sha256": _sha256(GATE_REPORT_PATH),
        "upstream_gates": upstream,
        "outstanding_work": outstanding_work(gate),
    }


LIMITATION_ITEMS = (
    {
        "id": "numerical-verification-only",
        "statement": "Predictions are verified against FEM solutions of the same "
        "plane-stress model, not against physical measurements.",
    },
    {
        "id": "envelope-only",
        "statement": "Only requests inside the ratified envelope "
        "dogbone-v1-strain-controlled-10-20-v1 are served; everything else is "
        "refused with a reason code.",
    },
    {
        "id": "finite-sampling",
        "statement": "The HOLDOUT budget was checked on a finite sample of cases; "
        "it is evidence, not a uniform error bound.",
    },
    {
        "id": "material-and-load",
        "statement": "One compressible Neo-Hookean material (E = 760 MPa, "
        "nu = 0.23) in 2D finite-strain plane stress with 1 mm reference "
        "thickness, under displacement-controlled nominal axial strain from 10% "
        "to 20%, on the quarter model with two symmetry planes.",
    },
    {
        "id": "served-stress",
        "statement": "The served stress is constitutive: P_served is the "
        "plane-stress Neo-Hookean first Piola stress of the predicted "
        "deformation (ADR 0018). There is no independent stress head.",
    },
    {
        "id": "peak-is-recovered",
        "statement": "The peak QoI was verified against an L2-recovered P2 FEM "
        "stress at the fillet-root vertices; it is mesh- and recovery-dependent "
        "and not a certified physical maximum.",
    },
    {
        "id": "development-budget-waiver",
        "statement": "The DEVELOPMENT FEM-reference budget failed on two slices "
        "and was accepted by the decision owner (ADR 0021); HOLDOUT was judged "
        "with the unchanged rule and passed.",
    },
    {
        "id": "confidence-calibration",
        "statement": "The confidence bound is the DEVELOPMENT-calibrated 90% "
        "quantile of the pointwise in-plane stress error given the "
        "equilibrium-residual indicator; its HOLDOUT coverage was 86.8%, below "
        "the nominal 90%, and indicators beyond the calibrated range are "
        "flagged.",
    },
)


def identities() -> dict[str, str]:
    record = frozen_candidate_record()
    family = _family()
    return {
        "checkpoint_manifest_hash": str(record["checkpoint_manifest_hash"]),
        "compiled_family_recipe": family.recipe_id,
        "model_state_hash": str(record["model_state_hash"]),
        "run_hash": str(record["run_hash"]),
        "training_revision": str(record["training_revision"]),
        "trial_id": str(record["trial_id"]),
    }


def build(directory: Path, weights_path: Path, *, created_utc: str) -> BundleManifest:
    """Build the DogBone bundle from the ``CP-FREEZE`` weights."""

    weights = weights_path.read_bytes()
    record = frozen_candidate_record()
    if hashlib.sha256(weights).hexdigest() != record["model_safetensors_sha256"]:
        raise ValueError("weights differ from the CP-FREEZE model export")
    report = verification_report()
    family = _family()
    return build_bundle(
        directory,
        name=BUNDLE_NAME,
        created_utc=created_utc,
        claim_level=str(report["claim_level"]),
        reference_mesh=family.reference_mesh,
        morph_map=family.morph_map,
        envelope=dogbone_envelope(),
        model=model_spec(),
        weights=weights,
        verification=report,
        limitation_items=LIMITATION_ITEMS,
        identities=identities(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--created-utc", required=True)
    args = parser.parse_args(argv)
    manifest = build(args.output, args.weights, created_utc=args.created_utc)
    print(
        json.dumps(
            {"bundle": str(args.output), "manifest_sha256": manifest.manifest_sha256}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BUNDLE_NAME",
    "LIMITATION_ITEMS",
    "PEAK_QOI",
    "build",
    "identities",
    "model_spec",
    "outstanding_work",
    "verification_report",
)
