"""S14 for the notched campaign: rehearsal, protocol, DEVELOPMENT, HOLDOUT, G14.

The DogBone S14 methods of ADR 0019 and ADR 0020, unchanged, on this family:
the same tolerances, budget components and safety factors, slice statistic,
confidence indicator variants and calibration, latency protocol and G14
checks. What differs is only what is family-specific: the frozen candidate
(``CP-FREEZE`` of this campaign), the notch root vertices, the coarse level
mesh (12 + 12 columns, 8 rows), the 32-case TRAIN rehearsal, the refusal
fixtures and the campaign's gate artifacts.

Usage (one BLAS thread)::

    python -m campaigns.notched_v1.s14 rehearsal --s10-run-root RUN --run-root OUT
    python -m campaigns.notched_v1.s14 protocol --created-utc T
    python -m campaigns.notched_v1.s14 development --s10-run-root RUN --run-root OUT
    python -m campaigns.notched_v1.s14 holdout --s10-run-root RUN --run-root OUT \
        --reveal-root ROOT
    python -m campaigns.notched_v1.s14 gate --run-root OUT
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from campaigns.dogbone_v1.s10_closeout import _load_run_execution_profile
from campaigns.dogbone_v1.s13_fem_recovery_reference import (
    summarize_fem_recovery_reference,
)
from campaigns.dogbone_v1.s13c_evaluate import _case_diagnostics, _served_vertex_output
from campaigns.dogbone_v1.s14_candidate import (
    INDICATOR_LENGTH_SCALE_MM,
    S14Case,
    ScoredCase,
    free_traction_components,
    slice_ids,
)
from campaigns.dogbone_v1.s14_evaluation import (
    DevelopmentAccess,
    _write_exclusive,
    consume_and_reveal_holdout,
    parse_reveal,
)
from campaigns.dogbone_v1.s14_gate import (
    COVERAGE_MINIMUM,
    FALSE_REFUSAL_MAXIMUM,
    REPORTED_CHECKS,
    REQUIRED_CHECKS,
    SPEARMAN_MINIMUM,
    SPEEDUP_TARGET,
    G14Measurements,
    g14_report,
)
from campaigns.dogbone_v1.s14_reference import (
    RECOVERY_SAFETY_FACTOR,
    REFINEMENT_RATIO,
    RICHARDSON_ORDER,
    RICHARDSON_SAFETY_FACTOR,
    FemReferenceTerms,
    nested_vertex_map,
)
from campaigns.dogbone_v1.s14_rehearsal import (
    CALIBRATION_BINS,
    CALIBRATION_QUANTILE,
    SCORING_THREADS,
    TOLERANCES,
    CaseRecord,
    SolvedCase,
    _model_calls,
    _single_thread_environment,
    _solve,
    budget_entry,
    component_summary,
    confidence_pilot,
    surrogate_components,
)
from campaigns.dogbone_v1.s14_validate import (
    development_waiver,
    failing_budget_rows,
    holdout_blockers,
    record_document,
    waiver_covers,
)
from campaigns.notched_v1 import s14_envelope as envelope
from campaigns.notched_v1.family import (
    _MESH_SPEC,
    _REFERENCE,
    GEOMETRY,
    INPUTS,
    MORPH_MAP,
    VALIDITY,
    boundary_conditions,
    compiled_family,
    material,
    parameter_point,
    solver_config,
)
from campaigns.notched_v1.gate_io import (
    gate_document,
    run_pytest_with_coverage,
    write_gate_document,
)
from campaigns.notched_v1.pilot import ARTIFACTS
from campaigns.notched_v1.proposal import design_spec
from campaigns.notched_v1.s9 import CP_ENVELOPE_PATH, GATES, PRIVATE_REVEAL_PATH
from campaigns.notched_v1.s13 import (
    STRESS_REFERENCE_MPA,
    ConditioningNormalization,
    NotchedGeometryCase,
    geometry_case,
    load_committed_protocol,
    published_cases,
)
from campaigns.notched_v1.s13_close import (
    CP_FREEZE_PATH,
    fem_results,
    load_registered_checkpoint,
    root_vertices,
    vertex_targets,
)
from piginot.contracts import CanonicalMesh, ExecutionProfile
from piginot.doe import EvidenceRole, HoldoutCommitment, normalized_coordinates
from piginot.doe.design import DesignPoint, _stratum
from piginot.fem import FemRequest, FemResult
from piginot.morphing import build_blended_notch_reference_mesh
from piginot.morphing.validation import validate_deformation
from piginot.operators import EnergyFormOperator
from piginot.verification import (
    INDICATOR_VARIANTS,
    REGISTERED_FAMILIES,
    CheckStatus,
    ClaimLevel,
    QuantileCurve,
    boundary_traction_residual,
    equilibrium_residual,
    evaluate_budget,
    fit_quantile_curve,
    indicator_variants,
    inject_failures,
    measure_latency,
    read_registered,
    reduce_registered,
    render_report,
    spearman,
    speedup,
)
from piginot.verification.budget import MINIMUM_QUANTILE_CASES, QUANTILE
from piginot.verification.families import GateFamily
from piginot.verification.holdout import ConsumptionRecord, reveal_after_consumption

REPOSITORY = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "notched-v1-s14-protocol-v1"
PROTOCOL_PATH = ARTIFACTS / "S14-PROTOCOL.json"
SIGNOFF_PATH = ARTIFACTS / "S14-PROTOCOL-SIGNOFF.json"
REHEARSAL_PATH = ARTIFACTS / "S14-TRAIN-REHEARSAL.json"
AMENDMENT_GLOB = "S14-PROTOCOL-AMENDMENT-*.json"
GATE_PATH = GATES / "S14.json"
SIGNATORY = "Betim Bahtiri"
CAMPAIGN_ID = "notched-v1"
LATENCY_CASES = 8
LATENCY_REPEATS = 20
POPULATION_COUNT = 64
REHEARSAL_CASES = 32
DEVELOPMENT_ACCESS_NAME = "S14-DEVELOPMENT-ACCESS.json"
HOLDOUT_CONSUMPTION_NAME = "S14-HOLDOUT-CONSUMED.json"
DEVELOPMENT_RESULT = "S14-DEVELOPMENT.json"
HOLDOUT_RESULT = "S14-HOLDOUT.json"
GATE_RESULT = "S14-GATE.json"
GATED_AXES = INPUTS
REPORTED_AXES = ("net_section_factor", "notch_sharpness")
AXIS_STRATA = {axis: 4 for axis in (*GATED_AXES, *REPORTED_AXES)}
EVALUATION_FILES = (
    "campaigns/dogbone_v1/s13_fem_recovery_reference.py",
    "campaigns/dogbone_v1/s13c_evaluate.py",
    "campaigns/dogbone_v1/s14_candidate.py",
    "campaigns/dogbone_v1/s14_envelope.py",
    "campaigns/dogbone_v1/s14_evaluation.py",
    "campaigns/dogbone_v1/s14_gate.py",
    "campaigns/dogbone_v1/s14_reference.py",
    "campaigns/dogbone_v1/s14_rehearsal.py",
    "campaigns/dogbone_v1/s14_validate.py",
    "campaigns/notched_v1/family.py",
    "campaigns/notched_v1/s13.py",
    "campaigns/notched_v1/s13_close.py",
    "campaigns/notched_v1/s14.py",
    "campaigns/notched_v1/s14_envelope.py",
    "src/piginot/morphing/blended_notch.py",
    "src/piginot/verification/__init__.py",
    "src/piginot/verification/budget.py",
    "src/piginot/verification/confidence.py",
    "src/piginot/verification/envelope.py",
    "src/piginot/verification/families.py",
    "src/piginot/verification/holdout.py",
    "src/piginot/verification/latency.py",
    "src/piginot/verification/reducer.py",
    "src/piginot/verification/report.py",
    "src/piginot/verification/status.py",
)
# Campaign-scoped families replace the DogBone gate artifacts; the framework
# families (G2-G5, G7, G8, G12) are shared.
CAMPAIGN_FAMILIES = {
    "F05": GATES / "S6.json",
    "F08": GATES / "S9.json",
    "F09": GATES / "S10.json",
    "F10": GATES / "S11-ENERGY.json",
    "F12": GATES / "S13.json",
}
_TRAIN = frozenset({EvidenceRole.TRAIN})
_DEVELOPMENT = frozenset({EvidenceRole.DEVELOPMENT})
_HOLDOUT = frozenset({EvidenceRole.HOLDOUT})


def slice_catalogue(axes: Sequence[str]) -> tuple[str, ...]:
    return ("all",) + tuple(f"{a}:{i}" for a in axes for i in range(AXIS_STRATA[a]))


GATED_SLICES = slice_catalogue(GATED_AXES)
REPORTED_SLICES = slice_catalogue(REPORTED_AXES)


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_digests(root: Path = REPOSITORY) -> dict[str, str]:
    return {name: text_sha256(root / name) for name in EVALUATION_FILES}


# --- reference error terms ----------------------------------------------------


def level_mesh(case: S14Case, scale: float) -> CanonicalMesh:
    """The S10 mesh recipe with ``scale`` times the columns and rows."""

    counts = {
        name: getattr(_MESH_SPEC, name) * scale
        for name in ("notch_columns", "outer_columns", "transverse_rows")
    }
    if any(value != int(value) or value < 1 for value in counts.values()):
        raise ValueError("scale must give whole positive column and row counts")
    spec = dataclasses.replace(_MESH_SPEC, **{k: int(v) for k, v in counts.items()})
    reference = build_blended_notch_reference_mesh(
        _REFERENCE,
        spec,
        source_provenance={
            "generator": "blended-notch-structured-reference-v1",
            "purpose": "s14-reference-error",
            "scale": repr(float(scale)),
        },
    )
    theta = parameter_point(*(float(case.primitive_coordinates[n]) for n in GEOMETRY))
    mesh, report = validate_deformation(reference, MORPH_MAP.bind(theta), VALIDITY)
    if mesh is None:
        raise ValueError(f"{case.case_id} level mesh is invalid: {report.reason_codes}")
    return mesh


def level_request(case: S14Case, request: FemRequest, scale: float) -> FemRequest:
    if request.request_id != case.case_id:
        raise ValueError("request does not belong to the case")
    return dataclasses.replace(request, mesh=level_mesh(case, scale))


def _vertex_fields(
    request: FemRequest, result: FemResult
) -> tuple[np.ndarray, np.ndarray]:
    count = len(request.mesh.nodes)
    coordinates = np.asarray(result.displacement_dof_coordinates[:count])
    if not np.array_equal(coordinates, np.asarray(request.mesh.nodes)):
        raise ValueError("P2 vertex degrees of freedom do not sit on the vertices")
    displacement = np.asarray(result.displacement[:count], dtype=np.float64)
    stress = np.asarray(result.recovered_first_piola_stress[:count], dtype=np.float64)
    return displacement, stress[:, :2, :2]


def _relative(difference: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(difference) / np.linalg.norm(reference))


def level_differences(
    coarse: FemRequest,
    coarse_result: FemResult,
    fine: FemRequest,
    fine_result: FemResult,
) -> tuple[float, float, float, int]:
    if (
        coarse_result.mesh_id != coarse.mesh.mesh_id
        or fine_result.mesh_id != fine.mesh.mesh_id
    ):
        raise ValueError("results do not belong to their meshes")
    u_fine, p_fine = _vertex_fields(fine, fine_result)
    u_coarse, p_coarse = _vertex_fields(coarse, coarse_result)
    mapped = nested_vertex_map(coarse.mesh, fine.mesh)
    u_ref, p_ref = u_fine[mapped], p_fine[mapped]
    root = np.asarray(root_vertices(coarse.mesh))
    peak_ref = float(np.linalg.norm(p_ref[root], axis=(1, 2)).max())
    peak_coarse = float(np.linalg.norm(p_coarse[root], axis=(1, 2)).max())
    return (
        _relative(u_coarse - u_ref, u_ref),
        _relative(p_coarse - p_ref, p_ref),
        abs(peak_coarse - peak_ref) / peak_ref,
        len(mapped),
    )


def reference_terms(
    case_id: str,
    request: FemRequest,
    result: FemResult,
    coarse: FemRequest,
    coarse_result: FemResult,
) -> FemReferenceTerms:
    if result.request_id != case_id or coarse_result.request_id != case_id:
        raise ValueError("FEM results do not belong to the case")
    recovery = {
        population.population: population.p_relative_l2
        for population in summarize_fem_recovery_reference(
            request, result, root_vertices(request.mesh)
        ).populations
    }
    u_difference, p_difference, peak_difference, count = level_differences(
        coarse, coarse_result, request, result
    )
    return FemReferenceTerms(
        case_id=case_id,
        recovery_all_cells=float(recovery["all_cells"]),
        recovery_root_cells=float(recovery["root_incident_cells"]),
        coarse_u_difference=u_difference,
        coarse_p_difference=p_difference,
        coarse_peak_difference=peak_difference,
        coarse_vertex_count=count,
    )


# --- the frozen candidate -------------------------------------------------------


def frozen_candidate_record(
    path: Path = REPOSITORY / CP_FREEZE_PATH,
) -> Mapping[str, Any]:
    document = json.loads(path.read_text("utf-8"))
    if (
        document.get("checkpoint_id") != "CP-FREEZE"
        or document.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("not the notched CP-FREEZE record")
    record: Mapping[str, Any] = document["frozen_candidate"]
    return record


def load_frozen_candidate(checkpoint_path: Path) -> EnergyFormOperator:
    protocol = load_committed_protocol()
    model, binding, manifest = load_registered_checkpoint(protocol, checkpoint_path)
    record = frozen_candidate_record()
    observed = {
        "checkpoint_manifest_hash": manifest.manifest_hash,
        "weights_hash": manifest.model_export_hash,
        "model_state_hash": manifest.model_state_hash,
        "run_hash": manifest.run_hash,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    for name, value in observed.items():
        if record[name] != value:
            raise ValueError(f"checkpoint {name} differs from CP-FREEZE")
    served: EnergyFormOperator = copy.deepcopy(model).to(torch.float64).eval()
    return served


def served_geometry(
    case: S14Case,
    request: FemRequest,
    normalization: ConditioningNormalization,
    *,
    allowed_roles: frozenset[EvidenceRole],
) -> NotchedGeometryCase:
    if case.role not in allowed_roles:
        raise PermissionError(f"{case.role.value} evidence is not open in this step")
    return geometry_case(
        case.case_id,
        case.leakage_group_id,
        case.primitive_coordinates,
        request,
        normalization,
        dtype=torch.float64,
    )


def score_case(
    model: EnergyFormOperator,
    case: S14Case,
    request: FemRequest,
    result: FemResult,
    normalization: ConditioningNormalization,
    *,
    allowed_roles: frozenset[EvidenceRole],
    with_indicators: bool = True,
) -> ScoredCase:
    """The S13 screen metrics plus the pointwise error and the indicators."""

    geometry = served_geometry(
        case, request, normalization, allowed_roles=allowed_roles
    )
    strain = float(case.primitive_coordinates["nominal_axial_strain"])
    targets = vertex_targets(case.case_id, strain, request, result)
    u, stress, determinant = _served_vertex_output(model, geometry, targets.query_xy)  # type: ignore[arg-type]
    if not bool(torch.isfinite(stress).all().item()):
        raise ValueError(f"{case.case_id} has inadmissible vertex deformation")
    predicted_p = stress / STRESS_REFERENCE_MPA
    diagnostics = _case_diagnostics(
        targets, u / geometry.prescribed_displacement_mm, predicted_p, determinant
    )
    pointwise = torch.linalg.vector_norm(
        predicted_p - targets.target_P, dim=(-2, -1)
    ).detach()
    indicators: Mapping[str, torch.Tensor] = {}
    traction = None
    if with_indicators:
        facets, free = free_traction_components(request)
        traction = boundary_traction_residual(
            stress.detach(),
            geometry.domain.vertices,
            geometry.domain.cells,
            facets,
            free,
            elastic_modulus=float(
                geometry.material.properties["elastic_modulus"].value
            ),
        )

        def residual(points: torch.Tensor) -> torch.Tensor:
            return equilibrium_residual(
                model,
                points,
                parameters=geometry.parameters(),
                frame=geometry.frame(),
                lifting=geometry.lifting(),
                material=geometry.material,
                length_scale=INDICATOR_LENGTH_SCALE_MM,
            )

        indicators = indicator_variants(
            residual,
            geometry.domain.vertices,
            geometry.domain.cells,
            length_scale=INDICATOR_LENGTH_SCALE_MM,
            traction=traction,
        )
    return ScoredCase(diagnostics, pointwise, indicators, traction)


# --- rehearsal machinery ----------------------------------------------------------


def rehearse_case(
    model: EnergyFormOperator,
    solved: SolvedCase,
    normalization: ConditioningNormalization,
    run_root: Path,
    *,
    allowed_roles: frozenset[EvidenceRole] = _TRAIN,
) -> tuple[CaseRecord, torch.Tensor, dict[str, torch.Tensor]]:
    """Score, solve the coarse level, measure the reference terms (cached)."""

    case = solved.case
    directory = run_root / "cases"
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / f"{case.case_id}.json"
    arrays_path = directory / f"{case.case_id}.npz"
    record: CaseRecord | None = None
    if record_path.exists():
        document = json.loads(record_path.read_text("utf-8"))
        document["slices"] = tuple(document["slices"])
        document["reported_slices"] = tuple(document["reported_slices"])
        record = CaseRecord(**document)
        if arrays_path.exists():
            arrays = np.load(arrays_path)
            return (
                record,
                torch.from_numpy(arrays["error"]),
                {
                    n: torch.from_numpy(arrays[n])
                    for n in (*INDICATOR_VARIANTS, "traction")
                },
            )
    torch.set_num_threads(SCORING_THREADS)
    scored = score_case(
        model,
        case,
        solved.request,
        solved.result,
        normalization,
        allowed_roles=allowed_roles,
    )
    if record is None:
        coarse = level_request(case, solved.request, 0.5)
        coarse_result, seconds = _solve(coarse)
        terms = reference_terms(
            case.case_id, solved.request, solved.result, coarse, coarse_result
        )
        record = CaseRecord(
            case_id=case.case_id,
            slices=slice_ids(case, GATED_AXES),
            reported_slices=slice_ids(case, REPORTED_AXES)[1:],
            surrogate=surrogate_components(scored),
            terms=asdict(terms),
            components=terms.components(),
            coarse_solve_seconds=seconds,
            minimum_vertex_determinant=scored.diagnostics.minimum_vertex_determinant,
        )
        record_path.write_text(json.dumps(asdict(record), sort_keys=True), "utf-8")
    elif surrogate_components(scored) != record.surrogate:
        raise RuntimeError(f"{case.case_id} rescoring differs from its cached record")
    assert scored.traction is not None
    np.savez(
        arrays_path,
        error=scored.pointwise_stress_error.numpy(),
        traction=scored.traction.numpy(),
        **{v: scored.indicators[v].numpy() for v in INDICATOR_VARIANTS},
    )
    return (
        record,
        scored.pointwise_stress_error,
        {**scored.indicators, "traction": scored.traction},
    )


def _terms(record: CaseRecord) -> FemReferenceTerms:
    return FemReferenceTerms(**record.terms)  # type: ignore[arg-type]


def budget_summary(records: Sequence[CaseRecord]) -> dict[str, object]:
    gated = evaluate_budget(
        tuple(
            budget_entry(r.case_id, r.slices, r.surrogate, _terms(r)) for r in records
        ),
        TOLERANCES,
        GATED_SLICES,
    )
    reported = evaluate_budget(
        tuple(
            budget_entry(r.case_id, ("all", *r.reported_slices), r.surrogate, _terms(r))
            for r in records
        ),
        TOLERANCES,
        REPORTED_SLICES,
    )

    def rows(results: Sequence[Any]) -> list[dict[str, object]]:
        return [
            {
                "slice": item.slice_id,
                "quantity": item.quantity,
                "cases": item.case_count,
                "statistic": item.statistic,
                "tolerance": item.tolerance,
                "status": item.status.value,
            }
            for item in results
        ]

    statuses = [item.status for item in gated]
    return {
        "gated_status": CheckStatus.PASS.value
        if all(s is CheckStatus.PASS for s in statuses)
        else next(s.value for s in statuses if s is not CheckStatus.PASS),
        "gated": rows(gated),
        "reported": [row for row in rows(reported) if row["slice"] != "all"],
    }


def refusal_run() -> dict[str, object]:
    fixtures = envelope.refusal_fixtures()
    start = time.perf_counter()
    mismatches, unstable = [], []
    for fixture in fixtures:
        first = envelope.check_fixture(fixture)
        if first.accepted or fixture.expected_code not in first.reason_codes:
            mismatches.append(fixture.fixture_id)
        if envelope.check_fixture(fixture) != first:
            unstable.append(fixture.fixture_id)
    cases = envelope.in_domain_cases()
    refused = [c.case_id for c in cases if not envelope.check_in_domain(c).accepted]
    return {
        "fixture_count": len(fixtures),
        "fixture_counts": envelope.FIXTURE_COUNTS,
        "fixture_set_hash": envelope.fixture_set_hash(fixtures),
        "refused_with_expected_code": len(fixtures) - len(mismatches),
        "mismatched_fixtures": mismatches,
        "unstable_reason_codes": unstable,
        "in_domain_count": len(cases),
        "false_refusals": refused,
        "false_refusal_rate": len(refused) / len(cases),
        "seconds": time.perf_counter() - start,
    }


def latency_run(
    model: EnergyFormOperator,
    cases: Sequence[SolvedCase],
    normalization: ConditioningNormalization,
    *,
    allowed_roles: frozenset[EvidenceRole] = _TRAIN,
    repeats: int = 20,
) -> dict[str, object]:
    if not _single_thread_environment():
        raise RuntimeError("latency needs OMP/OPENBLAS/MKL_NUM_THREADS=1")
    model_u: list[float] = []
    model_up: list[float] = []
    p99: list[float] = []
    fem: list[float] = []
    reproduced: list[bool] = []
    for solved in cases:
        geometry = served_geometry(
            solved.case, solved.request, normalization, allowed_roles=allowed_roles
        )
        displacement_only, displacement_and_stress = _model_calls(model, geometry)  # type: ignore[arg-type]
        for call, sink in (
            (displacement_only, model_u),
            (displacement_and_stress, model_up),
        ):
            result = measure_latency(call, repeats=repeats, warmup=3, threads=1)
            sink.append(result.median_seconds)
            if sink is model_up:
                p99.append(result.p99_seconds)
        fem_result, seconds = _solve(solved.request)
        fem.append(seconds)
        reproduced.append(
            fem_result.total_strain_energy == solved.result.total_strain_energy
        )
    fem_median = float(np.median(fem))
    up_median = float(np.median(model_up))
    return {
        "case_count": len(cases),
        "query_set": "all mesh vertices of each case (833 points)",
        "threads": 1,
        "model_u_median_seconds": float(np.median(model_u)),
        "model_u_and_p_median_seconds": up_median,
        "model_u_and_p_p99_seconds": max(p99),
        "fem_median_seconds": fem_median,
        "fem_seconds": fem,
        "fem_reproduces_stored_energy": reproduced,
        "speedup_u_and_p": speedup(fem_median, up_median),
        "speedup_target": SPEEDUP_TARGET,
    }


def public_solved(
    s10_run_root: Path, case_ids: Sequence[str], role: EvidenceRole
) -> list[SolvedCase]:
    """Public cases of one role with their S10 requests and P2 results."""

    cases = published_cases(case_ids)
    if any(case.role is not role for case in cases):
        raise PermissionError("a case of another role was requested")
    labelled = fem_results(s10_run_root, case_ids)
    return [
        SolvedCase(S14Case.from_published(case), *labelled[case.case_id])
        for case in cases
    ]


def run_rehearsal(
    *,
    s10_run_root: Path,
    checkpoint_path: Path,
    run_root: Path,
    latency_cases: int = 4,
    with_refusal: bool = True,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Every S14 measurement on the 32 TRAIN-screen cases."""

    protocol = load_committed_protocol()
    run_root.mkdir(parents=True, exist_ok=True)
    model = load_frozen_candidate(checkpoint_path)
    records, errors, indicators = [], [], []
    solved_cases = public_solved(
        s10_run_root, protocol.screen_case_ids, EvidenceRole.TRAIN
    )
    for solved in solved_cases:
        record, error, variants = rehearse_case(
            model, solved, protocol.normalization, run_root
        )
        records.append(record)
        errors.append(error)
        indicators.append(variants)
        progress(f"{len(records):2d}/{REHEARSAL_CASES} {record.case_id}")
    report: dict[str, object] = {
        "step": "S14 TRAIN rehearsal",
        "campaign_id": CAMPAIGN_ID,
        "evidence_role": "TRAIN",
        "development_opened": False,
        "holdout_opened": False,
        "case_count": len(records),
        "tolerances": {t.quantity: t.tolerance for t in TOLERANCES},
        "indicator_length_scale_mm": INDICATOR_LENGTH_SCALE_MM,
        "budget": budget_summary(records),
        "components": component_summary(records),
        "confidence_pilot": confidence_pilot(errors, indicators),
        "cases": [asdict(record) for record in records],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scoring_threads": SCORING_THREADS,
        },
    }
    if with_refusal:
        progress("refusal fixtures")
        report["refusal"] = refusal_run()
    if latency_cases:
        progress("latency")
        report["latency"] = latency_run(
            model, solved_cases[:latency_cases], protocol.normalization
        )
    return report


# --- protocol -------------------------------------------------------------------


def holdout_commitment(path: Path = REPOSITORY / CP_ENVELOPE_PATH) -> HoldoutCommitment:
    record = json.loads(path.read_text("utf-8"))["holdout_commitment"]
    return HoldoutCommitment(**{k: v for k, v in record.items() if k != "$type"})


def build_protocol(
    *, rehearsal_bytes: bytes, revision: str, created_utc: str
) -> dict[str, object]:
    rehearsal = json.loads(rehearsal_bytes)
    if rehearsal.get("evidence_role") != "TRAIN" or rehearsal.get("development_opened"):
        raise ValueError("the rehearsal must be TRAIN-only")
    if rehearsal.get("case_count") != REHEARSAL_CASES:
        raise ValueError("the rehearsal must cover the 32 TRAIN-screen cases")
    pilot = rehearsal["confidence_pilot"]
    variant = pilot["selected_variant"]
    if variant not in INDICATOR_VARIANTS:
        raise ValueError("the pilot selected an unregistered variant")
    refusal = rehearsal["refusal"]
    candidate = frozen_candidate_record()
    commitment = holdout_commitment()
    return {
        "protocol_id": PROTOCOL_ID,
        "campaign_id": CAMPAIGN_ID,
        "schema_version": "1.0",
        "status": "FROZEN_PENDING_USER_SIGN_OFF",
        "created_utc": created_utc,
        "decision_records": [
            "docs/adr/0019-s14-verification-decisions.md",
            "docs/adr/0020-s14-protocol-methods.md",
            "docs/adr/0025-s17-examples-and-second-family.md",
        ],
        "frozen_candidate": {
            "cp_freeze_sha256": text_sha256(REPOSITORY / CP_FREEZE_PATH),
            "checkpoint_manifest_hash": candidate["checkpoint_manifest_hash"],
            "weights_hash": candidate["weights_hash"],
            "run_hash": candidate["run_hash"],
        },
        "evaluation_code": {
            "revision": revision,
            "file_sha256": file_digests(),
            "line_endings": "LF-normalised before hashing",
        },
        "populations": {
            "development": {
                "count": POPULATION_COUNT,
                "source": "the 64 active DEVELOPMENT cases of the public manifest "
                "with their S10 P2 results",
                "access_record": DEVELOPMENT_ACCESS_NAME,
                "opened_only_after": "user sign-off of this protocol",
            },
            "holdout": {
                "count": POPULATION_COUNT,
                "commitment": commitment.commitment,
                "campaign_hash": commitment.campaign_hash,
                "consumption_record": HOLDOUT_CONSUMPTION_NAME,
                "reveal": f"pi_ginot_private_reveals/{CAMPAIGN_ID}/reveal-v1.json",
                "order": "consumption record written and fsynced, then the reveal "
                "is read and must open the CP-ENVELOPE commitment",
                "solver": "unchanged S10 P2 request recipe and execution profile",
                "opened_only_after": "DEVELOPMENT results and calibration are written, "
                "the DEVELOPMENT budget passes on every gated slice and the pooled "
                "Spearman correlation meets the threshold; otherwise HOLDOUT stays unspent",
            },
        },
        "budget": {
            "quantities": {t.quantity: {"tolerance": t.tolerance} for t in TOLERANCES},
            "gated_population": "DEVELOPMENT",
            "claim_population": "HOLDOUT (HOLDOUT_VERIFIED or DEVELOPMENT_VERIFIED "
            "when every required check passes)",
            "gated_slices": list(GATED_SLICES),
            "reported_slices": list(REPORTED_SLICES[1:]),
            "slice_statistic": {
                "quantile": QUANTILE,
                "interpolation": "linear between order statistics",
                "minimum_cases_for_quantile": MINIMUM_QUANTILE_CASES,
                "otherwise": "maximum",
            },
            "per_case_total": "surrogate + discretisation + projection",
            "missing_component": "NOT_RUN, never zero",
            "root_vertices": "the vertices of the notch profile",
            "components": {
                "discretisation": {
                    "p_relative_l2": "max(2.61 eta_all, 1.97 d_P)",
                    "absolute_peak_error": "max(2.61 eta_root, 1.97 d_peak)",
                    "u_relative_l2": "1.97 d_u",
                },
                "projection": {
                    "p_relative_l2": "eta_all",
                    "absolute_peak_error": "eta_root",
                    "u_relative_l2": "0 (vertex values are P2 degrees of freedom)",
                },
                "recovery_safety_factor": RECOVERY_SAFETY_FACTOR,
                "richardson_safety_factor": RICHARDSON_SAFETY_FACTOR,
                "richardson_order": RICHARDSON_ORDER,
                "refinement_ratio": REFINEMENT_RATIO,
                "coarse_level": "12 + 12 columns, 8 rows (S10: 24 + 24, 16)",
            },
        },
        "confidence": {
            "indicator": variant,
            "variants": list(INDICATOR_VARIANTS),
            "length_scale_mm": INDICATOR_LENGTH_SCALE_MM,
            "statistic": "Spearman correlation, all DEVELOPMENT vertices pooled",
            "threshold": SPEARMAN_MINIMUM,
            "calibration": {
                "quantile": CALIBRATION_QUANTILE,
                "bins": CALIBRATION_BINS,
                "fitted_on": "DEVELOPMENT",
                "coverage_reported_on": "HOLDOUT",
            },
            "train_pilot": {
                "pooled_spearman": pilot["pooled_spearman"],
                "median_per_case_spearman": pilot["median_per_case_spearman"],
            },
        },
        "refusal": {
            "fixture_set_hash": refusal["fixture_set_hash"],
            "fixture_counts": dict(envelope.FIXTURE_COUNTS),
            "required_refusal": "100% with the expected reason code",
            "maximum_false_refusal_rate": FALSE_REFUSAL_MAXIMUM,
            "in_domain_count": refusal["in_domain_count"],
        },
        "latency": {
            "cases": f"the first {LATENCY_CASES} DEVELOPMENT cases",
            "repeats": LATENCY_REPEATS,
            "threads": 1,
            "speedup_target": SPEEDUP_TARGET,
            "decision": "measured and reported",
        },
        "gate": {
            "required": list(REQUIRED_CHECKS),
            "reported": list(REPORTED_CHECKS),
            "coverage_minimum_percent": COVERAGE_MINIMUM,
            "families": {k: v.as_posix() for k, v in CAMPAIGN_FAMILIES.items()},
            "claim_ceiling": "HOLDOUT_VERIFIED",
        },
        "train_rehearsal": {
            "artifact": REHEARSAL_PATH.as_posix(),
            "sha256": hashlib.sha256(rehearsal_bytes).hexdigest(),
            "budget_gated_status": rehearsal["budget"]["gated_status"],
            "pooled_spearman": pilot["pooled_spearman"],
            "selected_variant": variant,
        },
        "out_of_scope": [
            "retraining or any change to the frozen candidate",
            "threshold or method changes after DEVELOPMENT is opened",
            "any claim above HOLDOUT_VERIFIED",
        ],
    }


def protocol_bytes(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def load_signed_protocol() -> tuple[dict[str, Any], str]:
    """The protocol and its digest, only if signed off and still frozen."""

    protocol_path = REPOSITORY / PROTOCOL_PATH
    signoff_path = REPOSITORY / SIGNOFF_PATH
    if not signoff_path.is_file():
        raise PermissionError("S14-PROTOCOL.json has not been signed off")
    digest = text_sha256(protocol_path)
    signoff = json.loads(signoff_path.read_text("utf-8"))
    if (
        signoff.get("protocol_sha256") != digest
        or signoff.get("signed_off_by") != SIGNATORY
    ):
        raise PermissionError("the sign-off does not name this protocol and signatory")
    document: dict[str, Any] = json.loads(protocol_path.read_text("utf-8"))
    amendments = []
    for path in sorted((REPOSITORY / ARTIFACTS).glob(AMENDMENT_GLOB)):
        amendment = json.loads(path.read_text("utf-8"))
        if (
            amendment.get("base_protocol_sha256") != digest
            or amendment.get("signed_off_by") != SIGNATORY
            or not isinstance(amendment.get("evaluation_file_sha256"), dict)
        ):
            raise PermissionError(
                f"{path.name} is not a signed amendment of this protocol"
            )
        amendments.append(amendment)
    frozen = (
        amendments[-1]["evaluation_file_sha256"]
        if amendments
        else document["evaluation_code"]["file_sha256"]
    )
    current = file_digests()
    changed = sorted(
        n for n in set(frozen) | set(current) if frozen.get(n) != current.get(n)
    )
    if changed:
        raise PermissionError(
            "evaluation files changed after the freeze: " + ", ".join(changed)
        )
    document["amendments"] = amendments
    return document, digest


# --- DEVELOPMENT, HOLDOUT, gate ------------------------------------------------------


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, indent=1, sort_keys=True, default=list) + "\n", "utf-8"
    )


def development_case_ids() -> tuple[str, ...]:
    manifest = json.loads(
        (REPOSITORY / ARTIFACTS / "public-case-manifest.json").read_text()
    )
    return tuple(
        sorted(
            case["case_id"]
            for case in manifest["cases"]
            if case["role"] == "DEVELOPMENT" and case["status"] == "ACTIVE"
        )
    )


def run_development(
    *, s10_run_root: Path, checkpoint_path: Path, run_root: Path
) -> dict[str, object]:
    protocol, digest = load_signed_protocol()
    signoff = json.loads((REPOSITORY / SIGNOFF_PATH).read_text("utf-8"))
    variant = protocol["confidence"]["indicator"]
    run_root.mkdir(parents=True, exist_ok=True)
    ids = development_case_ids()
    record_path = run_root / DEVELOPMENT_ACCESS_NAME
    if record_path.exists():
        existing = json.loads(record_path.read_text("utf-8"))
        if existing["protocol_sha256"] != digest or tuple(existing["case_ids"]) != ids:
            raise PermissionError("DEVELOPMENT was opened under another protocol")
    else:
        access = DevelopmentAccess(
            protocol_sha256=digest,
            signed_off_by=signoff["signed_off_by"],
            signed_off_utc=signoff["signed_off_utc"],
            opened_utc=_utc(),
            case_ids=ids,
        )
        _write_exclusive(record_path, access.to_bytes())
    model = load_frozen_candidate(checkpoint_path)
    normalization = load_committed_protocol().normalization
    records, errors, indicators = [], [], []
    solved_cases = public_solved(s10_run_root, ids, EvidenceRole.DEVELOPMENT)
    for solved in solved_cases:
        record, error, variants = rehearse_case(
            model,
            solved,
            normalization,
            run_root / "development",
            allowed_roles=_DEVELOPMENT,
        )
        records.append(record)
        errors.append(error)
        indicators.append(variants)
        print(f"DEVELOPMENT {len(records):2d}/64 {record.case_id}", flush=True)
    pooled_indicator = torch.cat(tuple(case[variant] for case in indicators))
    pooled_error = torch.cat(tuple(errors))
    curve = fit_quantile_curve(
        pooled_indicator,
        pooled_error,
        quantile=CALIBRATION_QUANTILE,
        bins=CALIBRATION_BINS,
    )
    latency = latency_run(
        model,
        solved_cases[:LATENCY_CASES],
        normalization,
        allowed_roles=_DEVELOPMENT,
        repeats=LATENCY_REPEATS,
    )
    report: dict[str, object] = {
        "evidence_role": "DEVELOPMENT",
        "campaign_id": CAMPAIGN_ID,
        "protocol_sha256": digest,
        "case_count": len(records),
        "budget": budget_summary(records),
        "components": component_summary(records),
        "confidence": {
            "indicator": variant,
            "pooled_spearman": spearman(pooled_indicator, pooled_error),
            "vertex_count": int(pooled_error.numel()),
            "calibration": {
                "quantile": curve.quantile,
                "upper_edges": list(curve.upper_edges),
                "values": list(curve.values),
                "in_sample_coverage": curve.coverage(pooled_indicator, pooled_error),
            },
        },
        "latency": latency,
        "cases": [record_document(record) for record in records],
    }
    _write(run_root / DEVELOPMENT_RESULT, report)
    return report


def holdout_cases(coordinates: Sequence[Mapping[str, float]]) -> tuple[S14Case, ...]:
    """Role-neutral HOLDOUT cases with the S9 strata of their coordinates."""

    sealed = envelope.notched_envelope()
    primitive = {bound.name for bound in sealed.primitive_bounds}
    derived = {bound.name for bound in sealed.derived_bounds}
    axes = design_spec(seed=b"public-axes-only").axes
    cases = []
    for index, values in enumerate(coordinates):
        if set(values) != primitive | derived:
            raise ValueError("holdout coordinates do not name every envelope axis")
        case_id = f"holdout-{index:02d}"
        point = DesignPoint(
            case_id=case_id,
            leakage_group_id=case_id,
            coordinates={name: float(values[name]) for name in primitive},
            derived_coordinates={name: float(values[name]) for name in derived},
            strata={"unassigned": 0},
        )
        normal = normalized_coordinates(sealed, axes, point)
        cases.append(
            S14Case(
                case_id=case_id,
                role=EvidenceRole.HOLDOUT,
                leakage_group_id=case_id,
                primitive_coordinates=point.coordinates,
                derived_coordinates=point.derived_coordinates,
                strata={
                    axis.name: _stratum(normal[axis.name], axis.strata) for axis in axes
                },
            )
        )
    return tuple(cases)


def s10_request(case: S14Case, profile: ExecutionProfile) -> FemRequest:
    """The unchanged S10 P2 request recipe for one case of any role."""

    family = compiled_family()
    values = case.primitive_coordinates
    from piginot.morphing import apply_morph

    morph = apply_morph(
        family.reference_mesh,
        family.morph_map,
        family.membership,
        parameter_point(*(float(values[n]) for n in GEOMETRY)),
    )
    if not morph.accepted or morph.mesh is None:
        raise ValueError(f"{case.case_id} is outside the sealed family")
    return FemRequest(
        request_id=case.case_id,
        mesh=morph.mesh,
        material=material(
            float(values["elastic_modulus"]), float(values["poisson_ratio"])
        ),
        boundary_conditions=boundary_conditions(float(values["nominal_axial_strain"])),
        loads=(),
        qois=(),
        config=solver_config(),
        execution_profile=profile,
    )


def run_holdout(
    *,
    s10_run_root: Path,
    checkpoint_path: Path,
    run_root: Path,
    reveal_root: Path,
    workers: int = 2,
) -> dict[str, object]:
    """Consume HOLDOUT once, solve and score it with the frozen code."""

    protocol, digest = load_signed_protocol()
    development_path = run_root / DEVELOPMENT_RESULT
    development = json.loads(development_path.read_text("utf-8"))
    if development["protocol_sha256"] != digest:
        raise PermissionError("DEVELOPMENT results belong to another protocol")
    waiver = development_waiver(protocol, text_sha256(development_path))
    blockers = holdout_blockers(development, waiver)
    if blockers:
        raise PermissionError(
            "HOLDOUT stays closed because DEVELOPMENT failed: " + ", ".join(blockers)
        )
    calibration = development["confidence"]["calibration"]
    curve = QuantileCurve(
        calibration["quantile"],
        tuple(calibration["upper_edges"]),
        tuple(calibration["values"]),
    )
    variant = development["confidence"]["indicator"]
    record_path = run_root / HOLDOUT_CONSUMPTION_NAME
    reveal_path = reveal_root / PRIVATE_REVEAL_PATH
    commitment = holdout_commitment()
    candidate = frozen_candidate_record()
    if record_path.exists():
        coordinates = reveal_after_consumption(
            record_path,
            candidate_run_hash=str(candidate["run_hash"]),
            reader=lambda: parse_reveal(reveal_path.read_bytes(), commitment),
        )
    else:
        coordinates = consume_and_reveal_holdout(
            record_path=record_path,
            record=ConsumptionRecord(
                campaign_id=CAMPAIGN_ID,
                candidate_run_hash=str(candidate["run_hash"]),
                freeze_record_sha256=text_sha256(REPOSITORY / CP_FREEZE_PATH),
                consumed_utc=_utc(),
                purpose=f"S14 HOLDOUT evaluation under protocol {digest}",
            ),
            reveal_path=reveal_path,
            commitment=commitment,
        )
    model = load_frozen_candidate(checkpoint_path)
    normalization = load_committed_protocol().normalization
    profile = _load_run_execution_profile(s10_run_root)
    records, errors, indicators, seconds_list = [], [], [], []
    directory = run_root / "holdout"

    def cached(case: S14Case) -> bool:
        return (directory / "cases" / f"{case.case_id}.json").exists() and (
            directory / "cases" / f"{case.case_id}.npz"
        ).exists()

    pending = [
        (case, s10_request(case, profile)) for case in holdout_cases(coordinates)
    ]
    # The full S10 solves run ``workers`` at a time; scoring and the coarse
    # solve of each case follow in order, and every finished case is cached.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pending), workers):
            chunk = pending[start : start + workers]
            unsolved = [(case, request) for case, request in chunk if not cached(case)]
            results = dict(
                zip(
                    (case.case_id for case, _ in unsolved),
                    pool.map(_solve, [request for _, request in unsolved]),
                    strict=True,
                )
            )
            for case, request in chunk:
                if case.case_id in results:
                    result, seconds = results[case.case_id]
                    seconds_list.append(seconds)
                    solved = SolvedCase(case, request, result)
                else:
                    solved = SolvedCase(case, request, None)  # type: ignore[arg-type]
                record, error, variants = rehearse_case(
                    model, solved, normalization, directory, allowed_roles=_HOLDOUT
                )
                records.append(record)
                errors.append(error)
                indicators.append(variants)
                print(f"HOLDOUT {len(records):2d}/64 {record.case_id}", flush=True)
    pooled_indicator = torch.cat(tuple(case[variant] for case in indicators))
    pooled_error = torch.cat(tuple(errors))
    report: dict[str, object] = {
        "evidence_role": "HOLDOUT",
        "campaign_id": CAMPAIGN_ID,
        "protocol_sha256": digest,
        "case_count": len(records),
        "budget": budget_summary(records),
        "components": component_summary(records),
        "confidence": {
            "indicator": variant,
            "pooled_spearman": spearman(pooled_indicator, pooled_error),
            "calibration_coverage": curve.coverage(pooled_indicator, pooled_error),
        },
        "fem_solve_seconds": seconds_list,
        "cases": [record_document(record) for record in records],
    }
    _write(run_root / HOLDOUT_RESULT, report)
    return report


def campaign_families() -> tuple[GateFamily, ...]:
    return tuple(
        dataclasses.replace(
            family, artifact=CAMPAIGN_FAMILIES[family.family_id].as_posix()
        )
        if family.family_id in CAMPAIGN_FAMILIES
        else family
        for family in REGISTERED_FAMILIES
    )


def family_reduction(root: Path) -> tuple[CheckStatus, dict[str, object]]:
    evidence = read_registered(root, campaign_families())
    injection = inject_failures(evidence, evidence_level=ClaimLevel.HOLDOUT_VERIFIED)
    reduction = reduce_registered(evidence, evidence_level=ClaimLevel.HOLDOUT_VERIFIED)
    families = {
        f"{item.family.family_id} {item.family.name} ({item.family.artifact})": outcome.status.value
        for item, outcome in zip(evidence, reduction.families, strict=True)
    }
    passed = injection.passed and reduction.verified
    return (
        CheckStatus.PASS if passed else CheckStatus.FAIL,
        {
            "injected_cases": injection.cases,
            "injected_not_verified": injection.not_verified,
            "control_claim": injection.control_level.value,
            "families": families,
        },
    )


def run_gate(*, run_root: Path) -> dict[str, object]:
    protocol, digest = load_signed_protocol()

    def load(name: str) -> dict[str, Any] | None:
        path = run_root / name
        return json.loads(path.read_text("utf-8")) if path.exists() else None

    development = load(DEVELOPMENT_RESULT)
    holdout = load(HOLDOUT_RESULT)
    reducer_status, reducer_detail = family_reduction(REPOSITORY)
    waiver_reason = None
    if development is not None:
        waiver = development_waiver(
            protocol, text_sha256(run_root / DEVELOPMENT_RESULT)
        )
        if development["budget"]["gated_status"] == "FAIL" and waiver_covers(
            waiver, development
        ):
            assert waiver is not None
            waiver_reason = (
                f"accepted by the decision owner ({waiver['decision_record']})"
            )
    first, second = refusal_run(), refusal_run()
    stable = (
        first["mismatched_fixtures"] == second["mismatched_fixtures"]
        and first["false_refusals"] == second["false_refusals"]
        and not first["unstable_reason_codes"]
    )
    summary, coverage, passed = run_pytest_with_coverage(
        REPOSITORY, ("tests/verification",), ("piginot.verification",)
    )
    latency = spearman_value = None
    if development is not None:
        timing = development["latency"]
        latency = {
            "median_seconds": timing["model_u_and_p_median_seconds"],
            "p99_seconds": timing["model_u_and_p_p99_seconds"],
            "speedup": timing["speedup_u_and_p"],
        }
        spearman_value = development["confidence"]["pooled_spearman"]

    def status(document: dict[str, Any] | None) -> CheckStatus | None:
        return (
            None
            if document is None
            else CheckStatus(document["budget"]["gated_status"])
        )

    measurements = G14Measurements(
        development_budget=status(development),
        development_budget_waiver=waiver_reason,
        holdout_budget=status(holdout),
        reducer=reducer_status,
        reducer_detail=reducer_detail,
        report_sections=True,
        fixture_count=first["fixture_count"],  # type: ignore[arg-type]
        refused_with_expected_code=first["refused_with_expected_code"],  # type: ignore[arg-type]
        false_refusal_rate=first["false_refusal_rate"],  # type: ignore[arg-type]
        reason_codes_stable=stable,
        development_spearman=spearman_value,
        latency=latency,
        holdout_order=(run_root / HOLDOUT_CONSUMPTION_NAME).exists()
        if holdout
        else None,
        no_certified=True,
        coverage_percent=coverage if passed else None,
        holdout_calibration_coverage=None
        if holdout is None
        else holdout["confidence"]["calibration_coverage"],
    )
    report = g14_report(measurements)
    rendered = json.loads(render_report(report))
    verdict = (
        "PASS"
        if all(c.status is CheckStatus.PASS for c in report.checks if c.required)
        else "FAIL"
    )
    run_document = {
        "gate_id": "G14",
        "step": "S14",
        "campaign_id": CAMPAIGN_ID,
        "protocol_sha256": digest,
        "claim_level": report.claim_level.value,
        "verdict": verdict,
        "report": rendered,
    }
    _write(run_root / GATE_RESULT, run_document)
    checks = [
        {
            "id": check.check_id,
            "status": check.status.value,
            "required": check.required,
            "measured": check.measured,
            "threshold": check.threshold,
            "reason": check.reason,
        }
        for check in report.checks
    ]
    gate = gate_document(
        "G14",
        "S14",
        REPOSITORY,
        [c for c in checks if c["required"]],
        reported_checks=[c for c in checks if not c["required"]],
        claim_level=report.claim_level.value,
        protocol_sha256=digest,
        test_evidence={"command": "pytest tests/verification", "result": summary},
        evidence={
            name: hashlib.sha256((run_root / name).read_bytes()).hexdigest()
            for name in (DEVELOPMENT_RESULT, HOLDOUT_RESULT, GATE_RESULT)
            if (run_root / name).exists()
        },
    )
    write_gate_document(REPOSITORY / GATE_PATH, gate)
    return gate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.s14")
    parser.add_argument(
        "phase", choices=("rehearsal", "protocol", "development", "holdout", "gate")
    )
    parser.add_argument("--s10-run-root", type=Path)
    parser.add_argument("--training-run-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--reveal-root", type=Path)
    parser.add_argument("--created-utc")
    parser.add_argument("--revision")
    parser.add_argument("--latency-cases", type=int, default=4)
    options = parser.parse_args(argv)
    checkpoint = None
    if options.training_run_root is not None:
        checkpoint = options.training_run_root / "checkpoint-step-2400"
    if options.phase == "rehearsal":
        report = run_rehearsal(
            s10_run_root=options.s10_run_root,
            checkpoint_path=checkpoint,  # type: ignore[arg-type]
            run_root=options.run_root,
            latency_cases=options.latency_cases,
        )
        _write(REPOSITORY / REHEARSAL_PATH, report)
    elif options.phase == "protocol":
        document = build_protocol(
            rehearsal_bytes=(REPOSITORY / REHEARSAL_PATH).read_bytes(),
            revision=options.revision,
            created_utc=options.created_utc,
        )
        target = REPOSITORY / PROTOCOL_PATH
        if target.exists():
            raise FileExistsError("the protocol is already frozen")
        target.write_bytes(protocol_bytes(document))
        print(text_sha256(target))
    elif options.phase == "development":
        run_development(
            s10_run_root=options.s10_run_root,
            checkpoint_path=checkpoint,  # type: ignore[arg-type]
            run_root=options.run_root,
        )
    elif options.phase == "holdout":
        run_holdout(
            s10_run_root=options.s10_run_root,
            checkpoint_path=checkpoint,  # type: ignore[arg-type]
            run_root=options.run_root,
            reveal_root=options.reveal_root,
        )
    else:
        gate = run_gate(run_root=options.run_root)
        print(gate["verdict"], gate["claim_level"], gate["verdict_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = (
    "CAMPAIGN_FAMILIES",
    "EVALUATION_FILES",
    "GATED_SLICES",
    "PROTOCOL_PATH",
    "REPORTED_SLICES",
    "budget_summary",
    "build_protocol",
    "campaign_families",
    "failing_budget_rows",
    "holdout_cases",
    "level_mesh",
    "load_frozen_candidate",
    "load_signed_protocol",
    "reference_terms",
    "refusal_run",
    "rehearse_case",
    "run_development",
    "run_gate",
    "run_holdout",
    "run_rehearsal",
    "s10_request",
    "score_case",
)
