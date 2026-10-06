"""S15 for the notched campaign: build the bundle and measure G15 on it.

The bundle is built through the same ``piginot.bundle.build_bundle`` as the
DogBone bundle. Only the family data differ: the blended-notch reference mesh
and morph map, the sealed envelope, the four conditioning inputs, the frame
``(25 mm, W/2)``, the loaded-end displacement ``strain x 25 mm``, the
boundary tags and the notch-root QoI.

Phases::

    python -m campaigns.notched_v1.s15 build --output B --weights W --created-utc T
    python -m campaigns.notched_v1.s15 measure --bundle B --training-run-root R \
        --s10-run-root S10 --rehearsal-root REH --run-root OUT
    python -m campaigns.notched_v1.s15 fresh-install --bundle B --run-root OUT
    python -m campaigns.notched_v1.s15 gate --run-root OUT
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rfc8785
import torch

from campaigns.dogbone_v1 import s15_validate as dogbone
from campaigns.dogbone_v1.s14_candidate import INDICATOR_LENGTH_SCALE_MM, S14Case
from campaigns.notched_v1 import s14
from campaigns.notched_v1 import s14_envelope as envelope
from campaigns.notched_v1.family import (
    HALF_LENGTH_MM,
    LEFT_TAG,
    LOADED_END_TAG,
    LOWER_TAG,
    NOTCH_TAG,
    compiled_family,
)
from campaigns.notched_v1.gate_io import (
    check_document,
    gate_document,
    run_pytest_with_coverage,
    write_gate_document,
)
from campaigns.notched_v1.pilot import ARTIFACTS
from campaigns.notched_v1.s9 import GATES
from campaigns.notched_v1.s13 import (
    STRESS_REFERENCE_MPA,
    load_committed_protocol,
    operator_config,
)
from piginot.bundle import (
    CONSTITUTIVE_PLANE_STRESS,
    REQUIRED_LIMITATION_PARAGRAPH,
    BundleIntegrityError,
    BundleManifest,
    BundleReasonCode,
    ConditioningInput,
    ConfidenceSpec,
    ModelSpec,
    QoiSpec,
    ScaledParameter,
    Verdict,
    build_bundle,
    load_bundle,
    predict,
)
from piginot.bundle.inference import FIELD_NAMES, MODEL_PATH
from piginot.bundle.manifest import verify_bundle, write_bundle
from piginot.doe import EvidenceRole
from piginot.operators import ENERGY_ARCHITECTURE_ID

REPOSITORY = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "notched-v1-cp-freeze"
GATE_PATH = GATES / "S15.json"
VALIDATION_PATH = ARTIFACTS / "S15-VALIDATION.json"
MEASUREMENTS = "S15-MEASUREMENTS.json"
FRESH_INSTALL = "S15-FRESH-INSTALL.json"
PEAK_QOI = "root_peak_first_piola_frobenius"
BOUNDARY_CONDITIONS = {LEFT_TAG: ("x",), LOWER_TAG: ("y",), LOADED_END_TAG: ("x",)}
REFERENCE_REQUEST = {
    "width": 22.0,
    "notch_radius": 4.0,
    "notch_depth": 3.0,
    "nominal_axial_strain": 0.15,
}
REFERENCE_QUERIES = [[5.0, 2.0], [12.0, 4.0], [20.0, 8.0], [24.0, 10.0], [0.0, 0.0]]
UPSTREAM = (
    ("artifacts/gates/S0.json", "shared"),
    ("artifacts/gates/S1.json", "shared"),
    ("artifacts/gates/S2.json", "shared"),
    ("artifacts/gates/S3.json", "shared"),
    ("artifacts/gates/S4.json", "shared"),
    ("artifacts/gates/S5.json", "shared"),
    ((GATES / "S6.json").as_posix(), "notched-v1"),
    ("artifacts/gates/S7.json", "shared"),
    ("artifacts/gates/S8.json", "shared"),
    ((GATES / "S9.json").as_posix(), "notched-v1"),
    ((GATES / "S10.json").as_posix(), "notched-v1"),
    ((GATES / "S11-ENERGY.json").as_posix(), "notched-v1"),
    ("artifacts/gates/S12.json", "shared"),
    ((GATES / "S13.json").as_posix(), "notched-v1"),
)
NOT_RUN_WORK = (
    ("G15", "evaluated on this bundle after it is built"),
    ("experimental-validation", "out of scope: verified against FEM solutions only"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _units() -> dict[str, str]:
    return {
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


def model_spec(run_root: Path) -> ModelSpec:
    """The serving arithmetic of the notched S14 evaluation, as data."""

    protocol = load_committed_protocol()
    normalization = protocol.normalization
    config = dataclasses.asdict(operator_config())
    config["parameter_names"] = list(config["parameter_names"])
    development = json.loads((run_root / s14.DEVELOPMENT_RESULT).read_text("utf-8"))
    confidence = development["confidence"]
    curve = confidence["calibration"]
    return ModelSpec(
        architecture_id=ENERGY_ARCHITECTURE_ID,
        operator_config=config,
        conditioning=tuple(
            ConditioningInput(name, name, mean, deviation)
            for name, mean, deviation in zip(
                normalization.parameter_names,
                normalization.mean,
                normalization.standard_deviation,
                strict=True,
            )
        ),
        frame_origin=(ScaledParameter((), 0.0), ScaledParameter((), 0.0)),
        frame_extent=(
            ScaledParameter((), HALF_LENGTH_MM),
            ScaledParameter(("width",), 0.5),
        ),
        lifting_kind="axis-aligned-tension",
        lifting_displacement=ScaledParameter(("nominal_axial_strain",), HALF_LENGTH_MM),
        material_name="notched-v1-compressible-neo-hookean",
        material_model="compressible_neo_hookean",
        material_parameters={
            "elastic_modulus": "elastic_modulus",
            "poisson_ratio": "poisson_ratio",
        },
        boundary_conditions=BOUNDARY_CONDITIONS,
        served_stress=CONSTITUTIVE_PLANE_STRESS,
        dtype="float64",
        query_chunk=2_048,
        qois=(QoiSpec(PEAK_QOI, "P_served", NOTCH_TAG, (), "MPa"),),
        confidence=ConfidenceSpec(
            variant=confidence["indicator"],
            length_scale_mm=INDICATOR_LENGTH_SCALE_MM,
            reference_stress_mpa=STRESS_REFERENCE_MPA,
            quantile=curve["quantile"],
            upper_edges=tuple(curve["upper_edges"]),
            values=tuple(curve["values"]),
        ),
        units=_units(),
    )


def outstanding_work(gate: Mapping[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for artifact, _ in UPSTREAM:
        document = json.loads((REPOSITORY / artifact).read_text("utf-8"))
        for check in document.get("checks", []):
            if check.get("status") != "PASS":
                items.append(
                    {
                        "id": str(check.get("id")),
                        "status": str(check.get("status")),
                        "reason": str(check.get("reason") or "not given"),
                    }
                )
    for limitation in json.loads((REPOSITORY / GATES / "S9.json").read_text()).get(
        "limitations", []
    ):
        items.append(
            {"id": "G9.design-diagnostics", "status": "FAIL", "reason": limitation}
        )
    for check in gate["checks"] + gate.get("reported_checks", []):
        if check["status"] != "PASS":
            items.append(
                {
                    "id": check["id"],
                    "status": check["status"],
                    "reason": str(check.get("reason")),
                }
            )
    items.extend(
        {"id": identifier, "status": "NOT_RUN", "reason": reason}
        for identifier, reason in NOT_RUN_WORK
    )
    return items


def verification_report(run_root: Path) -> dict[str, Any]:
    gate = json.loads((REPOSITORY / s14.GATE_PATH).read_text("utf-8"))
    run_gate = json.loads((run_root / s14.GATE_RESULT).read_text("utf-8"))
    upstream = []
    for artifact, scope in UPSTREAM:
        path = REPOSITORY / artifact
        document = json.loads(path.read_text("utf-8"))
        upstream.append(
            {
                "artifact": artifact,
                "scope": scope,
                "gate_id": document.get("gate_id"),
                "verdict": document.get("verdict"),
                "sha256": _sha256(path),
            }
        )
    return {
        "gate_id": "G14",
        "campaign_id": "notched-v1",
        "verdict": gate["verdict"],
        "claim_level": gate["claim_level"],
        "protocol_sha256": gate["protocol_sha256"],
        "report": run_gate["report"],
        "report_sha256": _sha256(run_root / s14.GATE_RESULT),
        "upstream_gates": upstream,
        "outstanding_work": outstanding_work(gate),
    }


def _verification_limitations(run_root: Path) -> tuple[dict[str, str], ...]:
    """What the S14 evidence does and does not show, as measured."""

    holdout_path = run_root / s14.HOLDOUT_RESULT
    if holdout_path.is_file():
        holdout = json.loads(holdout_path.read_text("utf-8"))
        coverage = holdout["confidence"]["calibration_coverage"]
        return (
            {
                "id": "finite-sampling",
                "statement": "The HOLDOUT budget was checked on a finite sample of "
                "64 cases; it is evidence, not a uniform error bound.",
            },
            {
                "id": "confidence-calibration",
                "statement": "The confidence bound is the DEVELOPMENT-calibrated 90% "
                "quantile of the pointwise in-plane stress error given the "
                "equilibrium-residual indicator; its HOLDOUT coverage was "
                f"{coverage:.1%}.",
            },
        )
    development = json.loads((run_root / s14.DEVELOPMENT_RESULT).read_text("utf-8"))
    rows = {
        row["quantity"]: row["statistic"]
        for row in development["budget"]["gated"]
        if row["slice"] == "all"
    }
    failed = sum(row["status"] == "FAIL" for row in development["budget"]["gated"])
    return (
        {
            "id": "not-verified",
            "statement": "NOT_VERIFIED: the S14 DEVELOPMENT budget failed "
            f"({failed} of {len(development['budget']['gated'])} gated rows; "
            f"all-cases p90 stress field {rows['p_relative_l2']:.2%}, peak "
            f"{rows['absolute_peak_error']:.2%}, displacement "
            f"{rows['u_relative_l2']:.2%} against 5%, 5% and 2%), so HOLDOUT was "
            "never opened. The predictions carry no verified error claim.",
        },
        {
            "id": "finite-sampling",
            "statement": "The DEVELOPMENT budget was measured on a finite sample of "
            f"{development['case_count']} cases; it is evidence, not a uniform "
            "error bound.",
        },
        {
            "id": "confidence-calibration",
            "statement": "The confidence bound is the DEVELOPMENT-calibrated 90% "
            "quantile of the pointwise in-plane stress error given the "
            "equilibrium-residual indicator; its in-sample DEVELOPMENT coverage was "
            f"{development['confidence']['calibration']['in_sample_coverage']:.1%} "
            "and it has no HOLDOUT coverage measurement.",
        },
    )


def limitation_items(run_root: Path) -> tuple[dict[str, str], ...]:
    return (
        *_verification_limitations(run_root),
        {
            "id": "numerical-verification-only",
            "statement": "Predictions are verified against FEM solutions of the same "
            "plane-stress model, not against physical measurements.",
        },
        {
            "id": "envelope-only",
            "statement": "Only requests inside the sealed envelope "
            "notched-v1-strain-controlled-10-20-v1 are served; everything else is "
            "refused with a reason code.",
        },
        {
            "id": "material-and-load",
            "statement": "One compressible Neo-Hookean material (E = 760 MPa, nu = "
            "0.23) in 2D finite-strain plane stress with 1 mm reference thickness, "
            "under displacement-controlled nominal axial strain from 10% to 20%, on "
            "the quarter model of a 50 mm plate with two symmetry planes.",
        },
        {
            "id": "served-stress",
            "statement": "The served stress is constitutive: P_served is the "
            "plane-stress Neo-Hookean first Piola stress of the predicted deformation "
            "(ADR 0018). There is no independent stress head.",
        },
        {
            "id": "peak-is-recovered",
            "statement": "The peak QoI is scored against an L2-recovered P2 FEM "
            "stress at the notch-profile vertices; it is mesh- and recovery-dependent "
            "and not a certified physical maximum.",
        },
        {
            "id": "design-coverage",
            "statement": "The sealed design's empirical fill distance (0.124) exceeds "
            "the proposal's 0.10 target; every stratum, boundary, interaction and "
            "difficult-region quota was met.",
        },
    )


def identities() -> dict[str, str]:
    record = s14.frozen_candidate_record()
    return {
        "campaign_id": "notched-v1",
        "checkpoint_manifest_hash": str(record["checkpoint_manifest_hash"]),
        "compiled_family_recipe": compiled_family().recipe_id,
        "model_state_hash": str(record["model_state_hash"]),
        "run_hash": str(record["run_hash"]),
        "training_revision": str(record["training_revision"]),
        "trial_id": str(record["trial_id"]),
    }


def build(
    directory: Path, weights_path: Path, run_root: Path, *, created_utc: str
) -> BundleManifest:
    weights = weights_path.read_bytes()
    record = s14.frozen_candidate_record()
    if hashlib.sha256(weights).hexdigest() != record["model_safetensors_sha256"]:
        raise ValueError("weights differ from the CP-FREEZE model export")
    report = verification_report(run_root)
    family = compiled_family()
    return build_bundle(
        directory,
        name=BUNDLE_NAME,
        created_utc=created_utc,
        claim_level=str(report["claim_level"]),
        reference_mesh=family.reference_mesh,
        morph_map=family.morph_map,
        envelope=envelope.notched_envelope(),
        model=model_spec(run_root),
        weights=weights,
        verification=report,
        limitation_items=limitation_items(run_root),
        identities=identities(),
    )


# --- measurements -----------------------------------------------------------------


def _request(values: Mapping[str, float]) -> dict[str, object]:
    return envelope.theta(values)


def fields_check(bundle_path: Path, expected_claim: str) -> dict[str, Any]:
    bundle = load_bundle(bundle_path)
    prediction = predict(bundle, _request(REFERENCE_REQUEST), REFERENCE_QUERIES)
    fast = predict(bundle, _request(REFERENCE_REQUEST), confidence=False)
    document = prediction.document()
    vertices, queries = document["vertices"], document["queries"]
    finite = all(
        bool(torch.isfinite(getattr(points, name)).all())
        for points in (prediction.vertices, prediction.queries)
        for name in FIELD_NAMES
    )
    present = {
        "fields_at_vertices": all(name in vertices for name in FIELD_NAMES),
        "fields_at_queries": all(name in queries for name in FIELD_NAMES),
        "qois": bool(document["qois"]),
        "units": all(name in document["units"] for name in FIELD_NAMES),
        "identities": {
            "bundle_manifest_sha256",
            "weights_sha256",
            "envelope_id",
            "mesh_id",
        }
        <= set(document["identities"]),
        "in_domain_verdict": document["verdict"] == Verdict.IN_DOMAIN.value,
        "reason_codes": document["reason_codes"] == [],
        "claim_level": document["claim_level"] == expected_claim,
        "confidence_at_vertices": "confidence" in vertices,
        "confidence_at_queries": "confidence" in queries,
        "limitations": document["limitations"] == REQUIRED_LIMITATION_PARAGRAPH,
    }
    assert prediction.vertices is not None and fast.vertices is not None
    return {
        "present": present,
        "finite": finite,
        "vertex_count": len(vertices["u"]),
        "query_count": len(queries["u"]),
        "qois": document["qois"],
        "claim_level": document["claim_level"],
        "stress_authority": document["stress_authority"],
        "fast_fields_equal": bool(
            torch.equal(fast.vertices.P_served, prediction.vertices.P_served)
        ),
        "passed": all(present.values()) and finite,
    }


def substitution_check(bundle_path: Path, scratch: Path) -> dict[str, Any]:
    bundle = load_bundle(bundle_path)
    _, contents = verify_bundle(bundle_path)
    model = json.loads(contents[MODEL_PATH])
    model["served_stress"] = "predicted-stress-head"
    resigned = dict(contents)
    resigned[MODEL_PATH] = rfc8785.dumps(model)
    target = scratch / "resigned-authority"
    write_bundle(
        target,
        resigned,
        name=bundle.manifest.name,
        created_utc=bundle.manifest.created_utc,
        claim_level=bundle.manifest.claim_level,
        identities=bundle.manifest.identities,
    )
    try:
        load_bundle(target)
        refused: list[str] = []
    except BundleIntegrityError as exc:
        refused = [code.value for code in exc.reason_codes]
    prediction = predict(bundle, _request(REFERENCE_REQUEST), confidence=False)
    return {
        "declared_served_stress": bundle.model.served_stress,
        "architecture_authority": bundle.serving.stress_authority,
        "prediction_stress_authority": prediction.stress_authority,
        "resigned_other_authority_refused_with": refused,
        "p_material_reported_separately": "P_material"
        not in (prediction.document()["vertices"] or {}),
        "passed": bundle.model.served_stress
        == bundle.serving.stress_authority
        == prediction.stress_authority
        and refused == [BundleReasonCode.STRESS_AUTHORITY_MISMATCH.value],
    }


def refusal_check(bundle_path: Path) -> dict[str, Any]:
    bundle = load_bundle(bundle_path)
    fixtures = envelope.refusal_fixtures()
    tested = envelope.request_form_fixtures()
    misses: list[str] = []
    for fixture in tested:
        prediction = predict(
            bundle, fixture.request_theta(), fixture.queries, confidence=False
        )
        if not (
            prediction.verdict is Verdict.REFUSED
            and fixture.expected_code in prediction.reason_codes
            and prediction.claim_level.value == "NOT_VERIFIED"
            and prediction.limitations == REQUIRED_LIMITATION_PARAGRAPH
        ):
            misses.append(fixture.fixture_id)
    false_refusals: list[str] = []
    cases = envelope.in_domain_cases()
    for case in cases:
        request = _request(case.values)
        outcome = bundle.geometry(request)
        if outcome.mesh is None:
            false_refusals.append(case.case_id)
            continue
        queries = [list(p) for p in envelope.in_domain_queries(outcome.mesh)]
        if not predict(bundle, request, queries, confidence=False).in_domain:
            false_refusals.append(case.case_id)
    return {
        "fixtures_tested": len(tested),
        "fixtures_refused_with_expected_code": len(tested) - len(misses),
        "missed": misses,
        "mesh_mutation_fixtures_not_applicable": len(fixtures) - len(tested),
        "in_domain_cases": len(cases),
        "false_refusals": false_refusals,
        "passed": not misses and not false_refusals,
    }


def limitations_check(bundle_path: Path, scratch: Path) -> dict[str, Any]:
    from piginot import cli

    bundle = load_bundle(bundle_path)
    request = scratch / "request.json"
    inputs: dict[str, Any] = {
        name: {"value": value, "unit": "mm"}
        for name, value in REFERENCE_REQUEST.items()
        if name != "nominal_axial_strain"
    }
    inputs |= {
        "nominal_axial_strain": {"value": 0.3, "unit": "1"},
        "elastic_modulus": {"value": 760.0, "unit": "MPa"},
        "poisson_ratio": {"value": 0.23, "unit": "1"},
    }
    request.write_text(json.dumps({"inputs": inputs}))
    refused = scratch / "refused.json"
    cli.main(
        [
            "predict",
            str(bundle_path),
            "--request",
            str(request),
            "--output",
            str(refused),
        ]
    )
    inspected = scratch / "inspect.json"
    cli.main(["inspect", str(bundle_path), "--output", str(inspected)])
    accepted = predict(bundle, _request(REFERENCE_REQUEST), confidence=False).document()
    broken = scratch / "broken"
    shutil.copytree(bundle_path, broken)
    (broken / MODEL_PATH).write_bytes((broken / MODEL_PATH).read_bytes() + b" ")
    broken_output = scratch / "broken.json"
    cli.main(["inspect", str(broken), "--output", str(broken_output)])
    reports = {
        "bundle report/limitations.json": bundle.limitations["paragraph"],
        "predict (in domain)": accepted["limitations"],
        "predict (refused, CLI)": json.loads(refused.read_text())["limitations"],
        "inspect (CLI)": json.loads(inspected.read_text())["limitations"]["paragraph"],
        "bundle refused (CLI)": json.loads(broken_output.read_text())["limitations"],
    }
    present = {
        name: text == REQUIRED_LIMITATION_PARAGRAPH for name, text in reports.items()
    }
    return {"reports": present, "passed": all(present.values())}


def reproduction_check(
    bundle_path: Path,
    checkpoint_path: Path,
    s10_run_root: Path,
    *,
    cases: int,
    confidence_cases: int,
) -> dict[str, Any]:
    """Bundle predictions equal the S14 serving path bit for bit on TRAIN cases."""

    from campaigns.dogbone_v1.s13c_evaluate import _served_vertex_output

    bundle = load_bundle(bundle_path)
    model = s14.load_frozen_candidate(checkpoint_path)
    protocol = load_committed_protocol()
    variant = (
        bundle.model.confidence.variant if bundle.model.confidence else "patch-mean"
    )
    solved_cases = s14.public_solved(
        s10_run_root, protocol.screen_case_ids[:cases], EvidenceRole.TRAIN
    )
    records = []
    for index, solved in enumerate(solved_cases):
        case: S14Case = solved.case
        geometry = s14.served_geometry(
            case,
            solved.request,
            protocol.normalization,
            allowed_roles=frozenset({EvidenceRole.TRAIN}),
        )
        nodes = torch.tensor(solved.request.mesh.nodes, dtype=torch.float64)
        u, stress, _ = _served_vertex_output(model, geometry, nodes)  # type: ignore[arg-type]
        with_confidence = index < confidence_cases
        values = {k: float(v) for k, v in case.primitive_coordinates.items()}
        prediction = predict(bundle, _request(values), confidence=with_confidence)
        vertices = prediction.vertices
        assert vertices is not None and prediction.mesh is not None
        record: dict[str, Any] = {
            "case_id": case.case_id,
            "mesh_id_equal": prediction.mesh.mesh_id == solved.request.mesh.mesh_id,
            "u_bitwise_equal": bool(torch.equal(vertices.u, u.detach())),
            "P_bitwise_equal": bool(torch.equal(vertices.P_served, stress.detach())),
        }
        if with_confidence:
            assert vertices.confidence_indicator is not None
            torch.set_num_threads(1)
            scored = s14.score_case(
                model,
                case,
                solved.request,
                solved.result,
                protocol.normalization,
                allowed_roles=frozenset({EvidenceRole.TRAIN}),
            )
            record["indicator_bitwise_equal"] = bool(
                torch.equal(vertices.confidence_indicator, scored.indicators[variant])
            )
        records.append(record)
    return {
        "cases": records,
        "passed": all(all(v for k, v in r.items() if k != "case_id") for r in records),
    }


def run_measure(args: argparse.Namespace) -> int:
    run_root: Path = args.run_root
    scratch = run_root / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    bundle = load_bundle(args.bundle)
    results: dict[str, Any] = {
        "bundle": str(args.bundle),
        "bundle_name": bundle.manifest.name,
    }
    results["manifest"] = dogbone.manifest_check(args.bundle, scratch)
    results["substitution"] = substitution_check(args.bundle, scratch)
    results["limitations"] = limitations_check(args.bundle, scratch)
    results["fields"] = fields_check(args.bundle, bundle.manifest.claim_level)
    results["reproduction"] = reproduction_check(
        args.bundle,
        args.training_run_root / "checkpoint-step-2400",
        args.s10_run_root,
        cases=args.reproduce_cases,
        confidence_cases=args.confidence_cases,
    )
    results["refusal"] = refusal_check(args.bundle)
    dogbone._write(run_root / MEASUREMENTS, results)
    return 0


def _request_document(
    strain: float, queries: list[list[float]] | None
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        name: {"value": value, "unit": "mm"}
        for name, value in REFERENCE_REQUEST.items()
        if name != "nominal_axial_strain"
    }
    inputs |= {
        "nominal_axial_strain": {"value": strain, "unit": "1"},
        "elastic_modulus": {"value": 760.0, "unit": "MPa"},
        "poisson_ratio": {"value": 0.23, "unit": "1"},
    }
    return {"inputs": inputs} | ({"queries": queries} if queries else {})


def run_fresh_install(args: argparse.Namespace) -> int:
    """Wheel, clean venv with ``piginot[runtime]``, CLI predict and refuse."""

    import os
    import subprocess

    work: Path = args.run_root / "fresh-install"
    if work.exists():
        shutil.rmtree(work)
    (work / "clean").mkdir(parents=True)
    uv = shutil.which("uv") or "uv"

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command, capture_output=True, text=True, check=False, **kwargs
        )

    if run(
        [uv, "build", "--wheel", "--out-dir", str(work / "dist"), str(REPOSITORY)]
    ).returncode:
        raise RuntimeError("wheel build failed")
    (wheel,) = sorted((work / "dist").glob("piginot-*.whl"))
    venv = work / "venv"
    if run([uv, "venv", str(venv), "--python", args.python, "--no-project"]).returncode:
        raise RuntimeError("venv creation failed")
    python = venv / "bin" / "python"
    if run(
        [uv, "pip", "install", "--python", str(python), f"{wheel}[runtime]"]
    ).returncode:
        raise RuntimeError("runtime install failed")
    packages = sorted(
        line
        for line in run(
            [uv, "pip", "freeze", "--python", str(python)]
        ).stdout.splitlines()
        if line
    )
    names = {line.split("==")[0].split(" @ ")[0].lower() for line in packages}
    clean = work / "clean"
    shutil.copytree(args.bundle, clean / "bundle")
    (clean / "request.json").write_text(
        json.dumps(_request_document(0.15, REFERENCE_QUERIES))
    )
    refused_document = _request_document(0.15, None)
    refused_document["inputs"]["width"]["value"] = 40.0
    (clean / "refused.json").write_text(json.dumps(refused_document))
    environment = {"PATH": f"{venv / 'bin'}:/usr/bin:/bin", "HOME": str(clean)}
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if variable in os.environ:
            environment[variable] = os.environ[variable]
    piginot = str(venv / "bin" / "piginot")
    steps = {
        "bundle verify": [piginot, "bundle", "verify", "bundle"],
        "inspect": [piginot, "inspect", "bundle", "--output", "inspect.json"],
        "predict": [
            piginot,
            "predict",
            "bundle",
            "--request",
            "request.json",
            "--output",
            "prediction.json",
        ],
        "predict (refused)": [
            piginot,
            "predict",
            "bundle",
            "--request",
            "refused.json",
            "--output",
            "refused-prediction.json",
            "--no-confidence",
        ],
    }
    exit_codes = {
        name: run(command, cwd=clean, env=environment).returncode
        for name, command in steps.items()
    }
    probe = run(
        [str(python), "-c", "import piginot; print(piginot.__file__)"],
        cwd=clean,
        env=environment,
    )
    blocked = run(
        [str(python), "-c", dogbone._BLOCKED_PROBE], cwd=clean, env=environment
    )
    blocked_exit, blocked_leaks = (
        (int(v) for v in blocked.stdout.split())
        if blocked.returncode == 0
        else (-1, -1)
    )
    requirements = dogbone.runtime_requirements(wheel)
    required = {
        r.split(";")[0].split("<")[0].split(">")[0].split("=")[0].strip().lower()
        for r in requirements
    }
    prediction = json.loads((clean / "prediction.json").read_text())
    refused = json.loads((clean / "refused-prediction.json").read_text())
    repository = predict(
        load_bundle(args.bundle), _request(REFERENCE_REQUEST), REFERENCE_QUERIES
    ).document()
    differences = None
    if prediction["vertices"] is not None and repository["vertices"] is not None:
        differences = {
            name: float(
                np.max(
                    np.abs(
                        np.asarray(prediction["vertices"][name])
                        - np.asarray(repository["vertices"][name])
                    )
                )
            )
            for name in FIELD_NAMES
        }
    installed_from = probe.stdout.splitlines()[0] if probe.stdout else ""
    document: dict[str, Any] = {
        "python": args.python,
        "wheel": wheel.name,
        "wheel_sha256": _sha256(wheel),
        "packages": packages,
        "runtime_requirements": requirements,
        "fem_only_packages_required": sorted(required & set(dogbone.FEM_ONLY_PACKAGES)),
        "fem_only_packages_present_transitively": sorted(
            n for n in names if n in dogbone.FEM_ONLY_PACKAGES
        ),
        "blocked_import_prediction_exit_code": blocked_exit,
        "blocked_modules_imported": blocked_leaks,
        "imported_from_installed_wheel": str(venv) in installed_from,
        "exit_codes": exit_codes,
        "prediction_verdict": prediction["verdict"],
        "prediction_claim_level": prediction["claim_level"],
        "prediction_has_every_field": prediction["vertices"] is not None
        and all(name in prediction["vertices"] for name in FIELD_NAMES),
        "refused_reason_codes": refused["reason_codes"],
        "max_abs_difference_to_repository_environment": differences,
        "limitations_present": prediction["limitations"]
        == REQUIRED_LIMITATION_PARAGRAPH
        and refused["limitations"] == REQUIRED_LIMITATION_PARAGRAPH,
    }
    document["passed"] = (
        exit_codes
        == {"bundle verify": 0, "inspect": 0, "predict": 0, "predict (refused)": 2}
        and not document["fem_only_packages_required"]
        and blocked_exit == 0
        and blocked_leaks == 0
        and document["imported_from_installed_wheel"]
        and document["prediction_verdict"] == "IN_DOMAIN"
        and document["prediction_has_every_field"]
        and document["refused_reason_codes"] == ["PARAMETER_OUT_OF_BOUNDS"]
        and document["limitations_present"]
    )
    dogbone._write(args.run_root / FRESH_INSTALL, document)
    return 0


def run_gate(args: argparse.Namespace) -> int:
    measurements = json.loads((args.run_root / MEASUREMENTS).read_text())
    fresh = json.loads((args.run_root / FRESH_INSTALL).read_text())
    summary, coverage, passed = run_pytest_with_coverage(
        REPOSITORY,
        ("tests/bundle", "tests/test_s15_anti_scope.py"),
        ("piginot.bundle",),
    )
    checks = [
        check_document(
            c["id"], c["status"] == "PASS", c["measured"], c["threshold"], "all"
        )
        for c in dogbone.gate_checks(measurements, fresh, coverage if passed else 0.0)
    ]
    dogbone._write(
        REPOSITORY / VALIDATION_PATH,
        {"measurements": measurements, "fresh_install": fresh},
    )
    document = gate_document(
        "G15",
        "S15",
        REPOSITORY,
        checks,
        bundle=BUNDLE_NAME,
        bundle_manifest_sha256=measurements["manifest"]["manifest_sha256"],
        claim_level_served=measurements["fields"]["claim_level"],
        evidence={VALIDATION_PATH.as_posix(): _sha256(REPOSITORY / VALIDATION_PATH)},
        test_evidence={
            "command": "pytest tests/bundle tests/test_s15_anti_scope.py",
            "result": summary,
        },
    )
    write_gate_document(REPOSITORY / GATE_PATH, document)
    print(document["verdict"], document["verdict_reason"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.s15")
    commands = parser.add_subparsers(dest="phase", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--weights", required=True, type=Path)
    build_parser.add_argument("--s14-run-root", required=True, type=Path)
    build_parser.add_argument("--created-utc", required=True)
    measure = commands.add_parser("measure")
    measure.add_argument("--bundle", required=True, type=Path)
    measure.add_argument("--training-run-root", required=True, type=Path)
    measure.add_argument("--s10-run-root", required=True, type=Path)
    measure.add_argument("--run-root", required=True, type=Path)
    measure.add_argument("--reproduce-cases", type=int, default=8)
    measure.add_argument("--confidence-cases", type=int, default=1)
    fresh = commands.add_parser("fresh-install")
    fresh.add_argument("--bundle", required=True, type=Path)
    fresh.add_argument("--run-root", required=True, type=Path)
    fresh.add_argument("--python", default="3.11")
    gate = commands.add_parser("gate")
    gate.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.phase == "build":
        manifest = build(
            args.output, args.weights, args.s14_run_root, created_utc=args.created_utc
        )
        print(
            json.dumps(
                {
                    "bundle": str(args.output),
                    "manifest_sha256": manifest.manifest_sha256,
                }
            )
        )
        return 0
    if args.phase == "measure":
        return run_measure(args)
    if args.phase == "fresh-install":
        return run_fresh_install(args)
    return run_gate(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ("BUNDLE_NAME", "build", "model_spec", "verification_report")
