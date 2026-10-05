"""The S16 detectors: each gate check's procedure on a fixed, CI-sized fixture.

A detector runs the production code that its gate check verifies and
compares the result with an answer fixed independently of that code: an
analytic value, a hand-built expectation or a committed artifact. Every
detector passes on the unmutated code. Nothing here reads DEVELOPMENT or
HOLDOUT evidence, and nothing needs data outside the repository.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from campaigns.dogbone_v1 import synthetic_bundle
from piginot.bundle import (
    BundleIntegrityError,
    BundleReasonCode,
    Verdict,
    load_bundle,
    predict,
)
from piginot.bundle import serving as serving_module
from piginot.bundle.inference import MODEL_PATH
from piginot.bundle.manifest import verify_bundle, write_bundle
from piginot.bundle.serving import ServedPoints, build_energy_form
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    Dimension,
    LoadSpec,
    Unit,
)
from piginot.datasets.normalization import NormalizationRecord, fit_train_normalization
from piginot.datasets.projection import (
    QuadratureField,
    compare_cross_mesh_stress,
    project_quadrature_to_cells,
)
from piginot.datasets.snapshots import (
    DatasetCaseRecord,
    RoleDatasetSnapshot,
    SampleTrace,
)
from piginot.doe import EvidenceRole, PublishedCaseStatus
from piginot.doe.roles import (
    CaseRoleRegistry,
    NodeObservation,
    RandomNodeSplitError,
)
from piginot.fem import FemRequest, FemSolverConfig, ScikitFemBackend
from piginot.fem.body_forces import ConstantBodyForce
from piginot.fem.mms_runner import verification_execution_profile, verification_material
from piginot.fem.seeded_defects import run_seeded_defect_suite
from piginot.fem.verification_checks import affine_patch_error
from piginot.fem.verification_meshes import unit_square_verification_mesh
from piginot.mutation import Detector, DetectorOutcome
from piginot.objectives.verification import _anisotropy_check, _role_isolation_check
from piginot.operators import (
    AffineDirichletLifting,
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    QueryFrame,
)
from piginot.physics import consistent_tangent, first_piola, plane_stress_closure
from piginot.store import ObjectStore
from piginot.training.search import GridSearchSpace
from piginot.training.selection import (
    CheckpointLineage,
    SelectionMetrics,
    SelectionPolicy,
    TrialResult,
    select_checkpoint,
    select_training_trial,
)
from piginot.verification import (
    ALL_CASES_SLICE,
    BudgetTolerance,
    CaseBudgetEntry,
    CheckStatus,
    ClaimLevel,
    ConsumptionRecord,
    consume_holdout,
    evaluate_budget,
    inject_failures,
    read_registered,
    reveal_after_consumption,
)

ROOT = Path(__file__).resolve().parents[2]
_SCRATCH: list[Path] = []


def _scratch(prefix: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix=f"piginot-s16-{prefix}-"))
    _SCRATCH.append(path)
    return path


def cleanup() -> None:
    """Remove the scratch directories the detectors created."""

    while _SCRATCH:
        shutil.rmtree(_SCRATCH.pop(), ignore_errors=True)


def _relative(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


# ------------------------------------------------------------------ G4 physics


def g4_tangent() -> DetectorOutcome:
    """The consistent tangent equals central differences of ``P`` (G4.tangent)."""

    material = verification_material()
    generator = torch.Generator().manual_seed(16_004)
    worst = 0.0
    for _ in range(12):
        perturbation = 0.08 * torch.randn(
            (3, 3), generator=generator, dtype=torch.float64
        )
        deformation = torch.eye(3, dtype=torch.float64) + perturbation
        direction = torch.randn((3, 3), generator=generator, dtype=torch.float64)
        step = 1.0e-6
        numerical = (
            first_piola(deformation + step * direction, material)
            - first_piola(deformation - step * direction, material)
        ) / (2.0 * step)
        predicted = torch.einsum(
            "ijkl,kl->ij", consistent_tangent(deformation, material), direction
        )
        error = float(
            torch.linalg.norm(predicted - numerical) / torch.linalg.norm(numerical)
        )
        worst = max(worst, error)
    return DetectorOutcome(worst < 1.0e-6, {"max_relative_error": worst})


def g4_closure_residual() -> DetectorOutcome:
    """The closure gives ``F33 > 0`` and ``P33 = 0`` (G4.closure-residual)."""

    material = verification_material()
    generator = torch.Generator().manual_seed(16_044)
    deformation_2d = torch.eye(2, dtype=torch.float64) + 0.15 * torch.randn(
        (32, 2, 2), generator=generator, dtype=torch.float64
    )
    f33 = plane_stress_closure(deformation_2d, material)
    deformation = torch.zeros((32, 3, 3), dtype=torch.float64)
    deformation[:, :2, :2] = deformation_2d
    deformation[:, 2, 2] = f33
    p33 = first_piola(deformation, material)[:, 2, 2]
    elastic_modulus = float(material.properties["elastic_modulus"].value)
    residual = float(p33.abs().max()) / elastic_modulus
    positive = bool((f33 > 0.0).all())
    return DetectorOutcome(
        positive and residual < 1.0e-10,
        {"min_f33": float(f33.min()), "max_p33_over_e": residual},
    )


# ------------------------------------------------------------------ G7, G8 FEM


@dataclass(frozen=True, slots=True)
class _TopNormalTraction:
    """Traction ``0.2 N`` along the outward reference normal."""

    provider_id: str = "s16-top-normal-traction-v1"

    def evaluate(
        self, reference_points: np.ndarray, outward_reference_normals: np.ndarray
    ) -> np.ndarray:
        return np.asarray(0.2 * outward_reference_normals, dtype=np.float64)


def _square_request(
    request_id: str,
    boundary: tuple[BoundaryConditionSpec, ...],
    *,
    element_order: int,
    loads: tuple[LoadSpec, ...] = (),
    **ids: Any,
) -> FemRequest:
    return FemRequest(
        request_id=request_id,
        mesh=unit_square_verification_mesh(1),
        material=verification_material(),
        boundary_conditions=boundary,
        loads=loads,
        qois=(),
        config=FemSolverConfig(
            element_order=element_order,
            solver_tolerance=1.0e-11,
            max_newton_iterations=24,
            load_steps=2,
        ),
        execution_profile=verification_execution_profile(),
        **ids,
    )


def _solve(backend: ScikitFemBackend, request: FemRequest) -> Any:
    attempt = backend.solve_many((request,))[0]
    if not attempt.success or attempt.result is None:
        raise RuntimeError(f"{request.request_id} did not solve: {attempt.message}")
    return attempt.result


def _mm(value: float) -> CanonicalQuantity:
    return CanonicalQuantity(value, Unit.MM)


def _clamp() -> tuple[BoundaryConditionSpec, ...]:
    return (
        BoundaryConditionSpec(
            "left_clamp", "left", "dirichlet", ("x", "y"), (_mm(0.0), _mm(0.0))
        ),
    )


def g7_loads() -> DetectorOutcome:
    """Body forces and normal-aware tractions balance their reactions (G7.loads-fields-store)."""

    gravity = ConstantBodyForce((0.05, -0.02))
    body = _solve(
        ScikitFemBackend(body_force_providers=(gravity,)),
        _square_request(
            "s16-g7-body-force",
            _clamp(),
            element_order=2,
            loads=(LoadSpec("gravity", "body_force"),),
            body_force_provider_ids={"gravity": gravity.provider_id},
        ),
    )
    traction = _TopNormalTraction()
    edge = _solve(
        ScikitFemBackend(traction_field_providers=(traction,)),
        _square_request(
            "s16-g7-traction",
            _clamp(),
            element_order=2,
            loads=(LoadSpec("edge", "traction", "top", ("x", "y")),),
            traction_field_provider_ids={"edge": traction.provider_id},
        ),
    )
    body_reaction = body.reactions["left_clamp"]
    edge_reaction = edge.reactions["left_clamp"]
    body_error = float(
        np.hypot(body_reaction[0] + 0.05, body_reaction[1] - 0.02) / 0.05
    )
    edge_error = float(np.hypot(edge_reaction[0], edge_reaction[1] + 0.2) / 0.2)
    return DetectorOutcome(
        body_error < 1.0e-8 and edge_error < 1.0e-8,
        {
            "body_force_reaction_error": body_error,
            "traction_reaction_error": edge_error,
        },
    )


_PATCH_GRADIENT = np.asarray(((0.03, 0.0), (0.0, -0.015)), dtype=np.float64)


def g8_patch() -> DetectorOutcome:
    """Affine patch tests through the backend reproduce the field (G8.patch)."""

    boundary = (
        BoundaryConditionSpec("left_x", "left", "dirichlet", ("x",), (_mm(0.0),)),
        BoundaryConditionSpec("right_x", "right", "dirichlet", ("x",), (_mm(0.03),)),
        BoundaryConditionSpec("bottom_y", "bottom", "dirichlet", ("y",), (_mm(0.0),)),
        BoundaryConditionSpec("top_y", "top", "dirichlet", ("y",), (_mm(-0.015),)),
    )
    errors: dict[str, float] = {}
    for order in (1, 2):
        result = _solve(
            ScikitFemBackend(),
            _square_request(f"s16-g8-patch-p{order}", boundary, element_order=order),
        )
        errors[f"p{order}"] = affine_patch_error(
            np.asarray(result.displacement_dof_coordinates),
            np.asarray(result.displacement),
            _PATCH_GRADIENT,
            np.zeros(2, dtype=np.float64),
        )
    return DetectorOutcome(max(errors.values()) < 1.0e-10, errors)


def g8_seeded_defects() -> DetectorOutcome:
    """All six seeded defects are distinguished from their controls (G8.seeded-defects)."""

    report = run_seeded_defect_suite()
    missed = [d.defect_id for d in report.detections if not d.detected]
    return DetectorOutcome(
        report.detected_count == report.required_count == 6,
        {"detected": report.detected_count, "missed": missed},
    )


# ------------------------------------------------------------------ G9 roles


@dataclass(frozen=True, slots=True)
class _Case:
    case_id: str
    leakage_group_id: str


def g9_no_node_split() -> DetectorOutcome:
    """Nodes of one case cannot carry two roles (G9.no-node-split)."""

    registry = CaseRoleRegistry.empty().assign(
        _Case("case-1", "group-1"), EvidenceRole.TRAIN
    )
    split = (
        NodeObservation("case-1", "node-0", EvidenceRole.TRAIN),
        NodeObservation("case-1", "node-1", EvidenceRole.HOLDOUT),
    )
    whole = (
        NodeObservation("case-1", "node-0", EvidenceRole.TRAIN),
        NodeObservation("case-1", "node-1", EvidenceRole.TRAIN),
    )
    registry.validate_node_observations(whole)
    try:
        registry.validate_node_observations(split)
    except RandomNodeSplitError:
        refused = True
    except Exception:  # a different refusal does not name the defect
        refused = False
    else:
        refused = False
    return DetectorOutcome(refused, {"split_refused_as_random_node_split": refused})


# ------------------------------------------------------------------ G10 datasets


def _tensor(value: float) -> tuple[float, ...]:
    return (value, 0.0, 0.0, 0.0, value + 1.0, 0.0, 0.0, 0.0, 0.0)


def _mesh(mesh_id: str, nodes: Any, cells: Any, facets: Any) -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id=mesh_id,
        nodes=nodes,
        cells=cells,
        cell_type="triangle",
        boundary_facets=facets,
        semantic_tags={"boundary": tuple(range(len(facets)))},
        region_ids=tuple(1 for _ in cells),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature=f"{mesh_id}-v1",
        quality_metrics={},
        source_provenance={"fixture": "s16"},
    )


def g10_projection_error() -> DetectorOutcome:
    """Every projected field carries the envelope of its samples (G10.projection-error)."""

    mesh = _mesh(
        "s16-two-cells",
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ((0, 1, 2), (0, 2, 3)),
        ((0, 1), (1, 2), (2, 3), (3, 0)),
    )
    field = QuadratureField(
        field_name="first_piola_stress",
        mesh_id="s16-two-cells",
        fem_result_hash="a" * 64,
        unit=Unit.MPA,
        component_shape=(3, 3),
        points=(((0.6, 0.2), (0.8, 0.4)), ((0.2, 0.6), (0.4, 0.8))),
        weights=((0.2, 0.3), (0.25, 0.25)),
        values=((_tensor(10.0), _tensor(14.0)), (_tensor(20.0), _tensor(22.0))),
    )
    projected = project_quadrature_to_cells(mesh, field)
    bounds = projected.transfer_error.absolute_bounds
    expected = (2.4, 1.0)  # |10 - 12.4|, |22 - 21.0|: weighted means 12.4 and 21.0
    observed = (bounds[0][0], bounds[1][0])
    covered = all(
        abs(sample[0] - projected.values[cell][0]) <= bounds[cell][0] + 1.0e-12
        for cell, samples in enumerate(field.values)
        for sample in samples
    )
    exact = all(abs(o - e) < 1.0e-12 for o, e in zip(observed, expected, strict=True))
    return DetectorOutcome(
        covered and exact, {"bounds": list(observed), "covered": covered}
    )


def g10_cross_mesh() -> DetectorOutcome:
    """Cross-mesh stress is compared by physical integrals (G10.cross-mesh)."""

    coarse_mesh = _mesh(
        "s16-coarse",
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        ((0, 1, 2),),
        ((0, 1), (1, 2), (2, 0)),
    )
    fine_mesh = _mesh(
        "s16-fine",
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0)),
        ((0, 3, 2), (3, 1, 2)),
        ((0, 3), (3, 1), (1, 2), (2, 0)),
    )
    constant = _tensor(7.0)
    coarse = QuadratureField(
        "first_piola_stress",
        "s16-coarse",
        "c" * 64,
        Unit.MPA,
        (3, 3),
        (((1.0 / 3.0, 1.0 / 3.0),),),
        ((0.5,),),
        ((constant,),),
    )
    fine = QuadratureField(
        "first_piola_stress",
        "s16-fine",
        "d" * 64,
        Unit.MPA,
        (3, 3),
        (((1.0 / 6.0, 1.0 / 3.0),), ((2.0 / 3.0, 1.0 / 6.0),)),
        ((0.25,), (0.25,)),
        ((constant,), (constant,)),
    )
    comparison = compare_cross_mesh_stress(coarse_mesh, coarse, fine_mesh, fine)
    integral_ok = abs(comparison.left_integral[0] - 3.5) < 1.0e-12  # 7 MPa x 0.5 mm^2
    try:
        compare_cross_mesh_stress(
            coarse_mesh, coarse, fine_mesh, fine, method="node_pairing"
        )
        pairing_refused = False
    except ValueError:
        pairing_refused = True
    passed = (
        integral_ok and comparison.relative_l2_difference < 1.0e-12 and pairing_refused
    )
    return DetectorOutcome(
        passed,
        {
            "left_integral_xx": comparison.left_integral[0],
            "relative_difference": comparison.relative_l2_difference,
            "node_pairing_refused": pairing_refused,
        },
    )


def _record(index: int, value: float, role: EvidenceRole) -> NormalizationRecord:
    return NormalizationRecord(
        sample_id=f"stress-{index}",
        case_id=f"case-{index}",
        leakage_group_id=f"group-{index}",
        source_hash=f"{index + 1:064x}",
        role=role,
        status=PublishedCaseStatus.ACTIVE,
        unit=Unit.MPA,
        values=(value,),
    )


def _training_snapshot(records: tuple[NormalizationRecord, ...]) -> RoleDatasetSnapshot:
    return RoleDatasetSnapshot(
        campaign_hash="e" * 64,
        public_case_manifest_hash="f" * 64,
        request_plan_hash="1" * 64,
        role=EvidenceRole.TRAIN,
        cases=tuple(
            DatasetCaseRecord(
                case_id=record.case_id,
                leakage_group_id=record.leakage_group_id,
                role=EvidenceRole.TRAIN,
                primitive_coordinates={"width": 1.0},
                derived_coordinates={"ratio": 1.0},
                strata={"width": 0},
                request_recipe_hash="a" * 64,
                request_content_hash="b" * 64,
                mesh_id=f"mesh-{record.case_id}",
                mesh_content_hash="c" * 64,
                success=True,
                sample_trace=SampleTrace(fem_result_hash=record.source_hash),
                store_content_hash="d" * 64,
                failure_code=None,
                message="succeeded",
            )
            for record in records
        ),
    )


def g10_normalization() -> DetectorOutcome:
    """The S10 close-out fits statistics on TRAIN only (G10.normalization)."""

    from campaigns.dogbone_v1.s10_closeout import _normalization_evidence

    train = tuple(
        _record(i, v, EvidenceRole.TRAIN) for i, v in enumerate((100.0, 140.0, 180.0))
    )
    development = _record(3, 1_000_000.0, EvidenceRole.DEVELOPMENT)
    scale = CanonicalQuantity(100.0, Unit.MPA, Dimension.STRESS)
    snapshot = _training_snapshot(train)
    store = ObjectStore.create(_scratch("store") / "objects")
    evidence = _normalization_evidence(
        (*train, development),
        SimpleNamespace(reference_scale=scale),
        snapshot,
        store,  # type: ignore[arg-type]
    )
    statistics = fit_train_normalization(
        train, reference_scale=scale, training_snapshot=snapshot
    )
    expected_mean = (100.0 + 140.0 + 180.0) / 3.0 / 100.0
    passed = (
        evidence.training_case_ids == ("case-0", "case-1", "case-2")
        and evidence.non_train_source_count == 0
        and abs(statistics.mean[0] - expected_mean) < 1.0e-12
        and evidence.statistic_count == 1
    )
    return DetectorOutcome(
        passed,
        {
            "training_case_ids": list(evidence.training_case_ids),
            "mean": statistics.mean[0],
        },
    )


# ------------------------------------------------------------------ G11 operator


def _lifting_errors() -> tuple[float, float]:
    torch.manual_seed(16_011)
    model = EnergyFormOperator(synthetic_bundle.tiny_config()).to(torch.float64)
    half_length, half_width, displacement = 27.0, 10.0, 4.05
    extent = torch.tensor([[half_length, half_width]], dtype=torch.float64)
    lifting = AxisAlignedTensionLifting(
        extent, torch.tensor([displacement], dtype=torch.float64)
    )
    frame = QueryFrame(torch.zeros((1, 2), dtype=torch.float64), extent)
    parameters = torch.zeros((1, 5), dtype=torch.float64)
    along = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    left = torch.stack(
        (torch.zeros(9, dtype=torch.float64), half_width * along), dim=-1
    )
    right = torch.stack(
        (torch.full((9,), half_length, dtype=torch.float64), half_width * along), dim=-1
    )
    bottom = torch.stack(
        (half_length * along, torch.zeros(9, dtype=torch.float64)), dim=-1
    )

    def error(applied: Callable[[torch.Tensor], AffineDirichletLifting] | Any) -> float:
        with torch.no_grad():
            u_left = model.displacement(
                query_xy=left[None], parameters=parameters, frame=frame, lifting=applied
            )
            u_right = model.displacement(
                query_xy=right[None],
                parameters=parameters,
                frame=frame,
                lifting=applied,
            )
            u_bottom = model.displacement(
                query_xy=bottom[None],
                parameters=parameters,
                frame=frame,
                lifting=applied,
            )
        deviations = torch.cat(
            (
                u_left[0, :, 0].abs(),
                (u_right[0, :, 0] - displacement).abs(),
                u_bottom[0, :, 1].abs(),
            )
        )
        return float(deviations.max()) / displacement

    def identity(query_xy: torch.Tensor) -> AffineDirichletLifting:
        return AffineDirichletLifting(
            torch.zeros_like(query_xy), torch.ones_like(query_xy)
        )

    return error(lifting), error(identity)


def g11_lifting() -> DetectorOutcome:
    """Essential conditions hold exactly through the lifting (G11.lifting)."""

    control, _ = _lifting_errors()
    return DetectorOutcome(control < 1.0e-12, {"control_relative_error": control})


def g11_lifting_mutant() -> DetectorOutcome:
    """Control exact and the identity-lifting mutant visible (G11.lifting-mutant)."""

    control, mutant = _lifting_errors()
    return DetectorOutcome(
        control < 1.0e-12 and mutant > 1.0e-3,
        {"control_relative_error": control, "mutant_relative_error": mutant},
    )


# ------------------------------------------------------------------ G12 objectives


def g12_anisotropy() -> DetectorOutcome:
    """The strong residual is invariant to per-axis units (G12.anisotropy)."""

    change, mutant_fired = _anisotropy_check()
    return DetectorOutcome(
        change < 1.0e-12 and mutant_fired,
        {"relative_change": change, "seeded_mutant_fired": mutant_fired},
    )


def g12_role_isolation() -> DetectorOutcome:
    """Objectives refuse HOLDOUT evidence (G12.role-isolation)."""

    refused, consistent = _role_isolation_check()
    return DetectorOutcome(
        refused and consistent,
        {"holdout_refused": refused, "mutant_matches": consistent},
    )


# ------------------------------------------------------------------ G13 selection

_HASHES = tuple(f"{value:x}" * 64 for value in range(1, 7))


def _space() -> GridSearchSpace:
    return GridSearchSpace(
        learning_rates=(5e-5, 1e-4),
        physics_weight_scales=(0.1, 1.0),
        optimizer_steps=400,
        case_batch_size=4,
        maximum_trials=4,
    )


def _metrics(**changes: float) -> SelectionMetrics:
    values: dict[str, Any] = {
        "median_absolute_peak_error": 0.075,
        "p90_absolute_peak_error": 0.13,
        "median_signed_peak_error": -0.065,
        "p_relative_l2": 0.18,
        "u_relative_l2": 0.037,
        "hybrid_loss": 1.0,
        "invalid_collocation_points": 0,
    }
    values.update(changes)
    return SelectionMetrics(**values)


def _result(
    index: int, metrics: SelectionMetrics, role: EvidenceRole = EvidenceRole.TRAIN
) -> TrialResult:
    return TrialResult(
        lineage=CheckpointLineage(
            trial_id=_space().trial_ids[index],
            run_hash=_HASHES[0],
            dataset_snapshot_hash=_HASHES[1],
            execution_profile_hash=_HASHES[2],
            weights_hash=_HASHES[3],
            model_state_hash=_HASHES[4],
            checkpoint_hash=_HASHES[5],
        ),
        role=role,
        metrics=metrics,
    )


def _ranked_results() -> tuple[TrialResult, ...]:
    """Four trials whose frozen ranking is 3, 2, 1, 0 by construction."""

    return (
        _result(0, _metrics(p90_absolute_peak_error=0.13, p_relative_l2=0.17)),
        _result(1, _metrics(p90_absolute_peak_error=0.12, p_relative_l2=0.19)),
        _result(
            2,
            _metrics(
                p90_absolute_peak_error=0.12,
                median_signed_peak_error=-0.05,
                u_relative_l2=0.036,
                hybrid_loss=0.8,
            ),
        ),
        _result(
            3,
            _metrics(
                p90_absolute_peak_error=0.12,
                median_signed_peak_error=-0.05,
                u_relative_l2=0.035,
                hybrid_loss=0.7,
            ),
        ),
    )


def g13_holdout() -> DetectorOutcome:
    """Selection refuses HOLDOUT results and a HOLDOUT role (G13.holdout)."""

    results = _ranked_results()
    holdout_results = (replace(results[0], role=EvidenceRole.HOLDOUT), *results[1:])
    refusals = 0
    try:
        select_checkpoint(holdout_results, policy=SelectionPolicy(_space()))
    except PermissionError:
        refusals += 1
    try:
        select_training_trial(_space(), results, role=EvidenceRole.HOLDOUT)
    except PermissionError:
        refusals += 1
    return DetectorOutcome(refusals == 2, {"refusals": refusals, "required": 2})


def g13_selection() -> DetectorOutcome:
    """The first-ranked eligible trial is served (G13.selection)."""

    results = _ranked_results()
    decision = select_checkpoint(results, policy=SelectionPolicy(_space()))
    expected_order = tuple(results[i].trial_id for i in (3, 2, 1, 0))
    passed = (
        decision.selected_result.trial_id == expected_order[0]
        and decision.ranked_eligible_trial_ids == expected_order
    )
    return DetectorOutcome(
        passed,
        {
            "selected_is_first_ranked": decision.selected_result.trial_id
            == expected_order[0],
            "ranking_matches": decision.ranked_eligible_trial_ids == expected_order,
        },
    )


# ------------------------------------------------------------------ G14 verification


def _entry(
    case_id: str, slices: tuple[str, ...], surrogate: float | None
) -> CaseBudgetEntry:
    return CaseBudgetEntry(
        case_id,
        (ALL_CASES_SLICE, *slices),
        {"p": {"surrogate": surrogate, "discretisation": 0.0, "projection": 0.0}},
    )


def g14_budget() -> DetectorOutcome:
    """A missing component makes its slices NOT_RUN, never PASS (G14.budget)."""

    entries = (
        _entry("a", ("s1",), 0.01),
        _entry("b", ("s1",), 0.2),
        _entry("c", (), None),
    )
    results = {
        r.slice_id: r
        for r in evaluate_budget(
            entries, (BudgetTolerance("p", 0.05),), (ALL_CASES_SLICE, "s1", "s2")
        )
    }
    statuses = {key: value.status.value for key, value in results.items()}
    passed = (
        results[ALL_CASES_SLICE].status is CheckStatus.NOT_RUN
        and results["s1"].status is CheckStatus.FAIL
        and results["s1"].statistic == 0.2
        and results["s2"].status is CheckStatus.NOT_RUN
    )
    return DetectorOutcome(passed, {"statuses": statuses})


def g14_reducer() -> DetectorOutcome:
    """Every injected failure in the twelve families gives NOT_VERIFIED (G14.reducer)."""

    result = inject_failures(
        read_registered(ROOT), evidence_level=ClaimLevel.HOLDOUT_VERIFIED
    )
    return DetectorOutcome(
        result.passed,
        {
            "cases": result.cases,
            "not_verified": result.not_verified,
            "control_level": result.control_level.value,
        },
    )


def g14_holdout_order() -> DetectorOutcome:
    """The holdout is read only after its consumption record exists (G14.holdout-order)."""

    record_path = _scratch("holdout") / "consumption.json"
    reads: list[bool] = []

    def reader() -> str:
        reads.append(record_path.is_file())
        return "revealed"

    candidate = "1" * 64
    try:
        reveal_after_consumption(
            record_path, candidate_run_hash=candidate, reader=reader
        )
        refused_before = False
    except PermissionError:
        refused_before = True
    read_before_consumption = bool(reads)
    consume_holdout(
        record_path,
        ConsumptionRecord(
            "s16", candidate, "2" * 64, "2026-10-05T00:00:00Z", "S16 detector"
        ),
    )
    reads.clear()
    value = reveal_after_consumption(
        record_path, candidate_run_hash=candidate, reader=reader
    )
    reads_after = list(reads)
    reads.clear()
    try:
        reveal_after_consumption(
            record_path, candidate_run_hash="3" * 64, reader=reader
        )
        foreign_refused = False
    except PermissionError:
        foreign_refused = True
    passed = (
        refused_before
        and not read_before_consumption
        and value == "revealed"
        and reads_after == [True]
        and foreign_refused
        and not reads
    )
    return DetectorOutcome(
        passed,
        {
            "refused_without_record": refused_before,
            "read_without_record": read_before_consumption,
            "foreign_candidate_refused": foreign_refused,
        },
    )


# ------------------------------------------------------------------ G15 bundle


@lru_cache(maxsize=1)
def _synthetic_bundle_path() -> Path:
    path = _scratch("bundle") / "synthetic"
    synthetic_bundle.build(path)
    return path


HEAD = "s16-predicted-stress-head"
HEAD_ARCHITECTURE = "s16-two-head-v1"
_OFFSET = 5.0


class _TwoHead:
    """A stand-in architecture whose served stress comes from its own head."""

    architecture_id = HEAD_ARCHITECTURE
    stress_authority = HEAD

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.parameter_count = inner.parameter_count

    def serve(self, query_xy: torch.Tensor, **kwargs: Any) -> ServedPoints:
        served = self._inner.serve(query_xy, **kwargs)
        head = served.stress + _OFFSET * torch.eye(2, dtype=served.stress.dtype)
        return ServedPoints(served.u, served.gradient, head)

    def residual(self, query_xy: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self._inner.residual(query_xy, **kwargs)


@contextmanager
def _two_head_registered() -> Iterator[None]:
    def builder(config: Mapping[str, Any], tensors: Any, metadata: Any) -> _TwoHead:
        return _TwoHead(build_energy_form(config, tensors, metadata))

    serving_module.ARCHITECTURES[HEAD_ARCHITECTURE] = builder
    try:
        yield
    finally:
        serving_module.ARCHITECTURES.pop(HEAD_ARCHITECTURE, None)


def g15_substitution() -> DetectorOutcome:
    """``P_served`` is the declared stress, never the material law (G15.substitution)."""

    directory = _scratch("two-head") / "bundle"
    with _two_head_registered():
        synthetic_bundle.build(
            directory,
            model=synthetic_bundle.tiny_model_spec(
                architecture_id=HEAD_ARCHITECTURE, served_stress=HEAD
            ),
        )
        prediction = predict(
            load_bundle(directory), synthetic_bundle.request(), confidence=False
        )
    points = prediction.vertices
    if points is None:
        return DetectorOutcome(False, {"verdict": prediction.verdict.value})
    head_served = bool(
        torch.allclose(
            points.P_served,
            points.P_material + _OFFSET * torch.eye(2, dtype=torch.float64),
            atol=1.0e-9,
        )
    )
    resigned = _scratch("resigned") / "bundle"
    _, contents = verify_bundle(_synthetic_bundle_path())
    contents = dict(contents)
    contents[MODEL_PATH] = synthetic_bundle.tiny_model_spec(
        served_stress=HEAD
    ).to_bytes()
    write_bundle(
        resigned,
        contents,
        name="s16",
        created_utc=synthetic_bundle.CREATED,
        claim_level=synthetic_bundle.CLAIM,
        identities={"origin": "s16"},
    )
    try:
        load_bundle(resigned)
        mismatch_refused = False
    except BundleIntegrityError as exc:
        mismatch_refused = exc.reason_codes == (
            BundleReasonCode.STRESS_AUTHORITY_MISMATCH,
        )
    passed = head_served and prediction.stress_authority == HEAD and mismatch_refused
    return DetectorOutcome(
        passed,
        {
            "declared_head_served": head_served,
            "authority_mismatch_refused": mismatch_refused,
        },
    )


def _tamper_cases(directory: Path) -> list[tuple[str, Callable[[Path], None]]]:
    cases: list[tuple[str, Callable[[Path], None]]] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        for offset in sorted({0, size // 2, size - 1}):

            def flip(
                root: Path, relative: str = relative, offset: int = offset
            ) -> None:
                target = root / relative
                payload = bytearray(target.read_bytes())
                payload[offset] ^= 0x01
                target.write_bytes(bytes(payload))

            cases.append((f"flip {relative}@{offset}", flip))

        def delete(root: Path, relative: str = relative) -> None:
            (root / relative).unlink()

        cases.append((f"delete {relative}", delete))

    def extra(root: Path) -> None:
        (root / "extra.txt").write_bytes(b"x")

    cases.append(("extra file", extra))
    if _symlinks_supported(directory.parent):

        def symlink(root: Path) -> None:
            os.symlink(root / "a.json", root / "alias.json")

        cases.append(("symlink", symlink))
    return cases


def _symlinks_supported(scratch: Path) -> bool:
    probe = scratch / "symlink-probe"
    try:
        os.symlink(scratch, probe)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


def g15_manifest() -> DetectorOutcome:
    """Changing any byte of a bundle is detected (G15.manifest)."""

    scratch = _scratch("tamper")
    template = scratch / "template"
    write_bundle(
        template,
        {"a.json": b'{"a":1}', "nested/b.bin": bytes(range(64)), "c.txt": b"hello"},
        name="s16",
        created_utc="2026-10-05T00:00:00Z",
        claim_level="NOT_VERIFIED",
        identities={},
    )
    verify_bundle(template)
    missed: list[str] = []
    cases = _tamper_cases(template)
    for index, (label, tamper) in enumerate(cases):
        copy = scratch / f"case-{index}"
        shutil.copytree(template, copy, symlinks=True)
        tamper(copy)
        try:
            verify_bundle(copy)
        except BundleIntegrityError:
            continue
        missed.append(label)
    return DetectorOutcome(not missed, {"cases": len(cases), "missed": missed})


def g15_refuse() -> DetectorOutcome:
    """In-envelope requests are served; out-of-envelope ones refused (G15.refuse)."""

    bundle = load_bundle(_synthetic_bundle_path())
    inside = predict(bundle, synthetic_bundle.request(), confidence=False)
    outside = predict(
        bundle, synthetic_bundle.request(nominal_axial_strain=0.25), confidence=False
    )
    passed = (
        inside.verdict is Verdict.IN_DOMAIN
        and outside.verdict is Verdict.REFUSED
        and outside.reason_codes == ("PARAMETER_OUT_OF_BOUNDS",)
    )
    return DetectorOutcome(
        passed,
        {
            "in_envelope": inside.verdict.value,
            "in_envelope_codes": list(inside.reason_codes),
            "out_of_envelope_codes": list(outside.reason_codes),
        },
    )


def g15_fields() -> DetectorOutcome:
    """Served fields are finite and mutually consistent (G15.fields)."""

    bundle = load_bundle(_synthetic_bundle_path())
    prediction = predict(bundle, synthetic_bundle.request(), confidence=False)
    points = prediction.vertices
    if points is None:
        return DetectorOutcome(False, {"reason_codes": list(prediction.reason_codes)})
    finite = all(
        bool(torch.isfinite(getattr(points, name)).all())
        for name in ("u", "F_2D", "F33", "J_3D", "P_served", "sigma")
    )
    jacobian = bool(
        torch.allclose(
            points.J_3D, torch.linalg.det(points.F_2D) * points.F33, rtol=1e-12
        )
    )
    loaded_end = float(points.coordinates[:, 0].max())
    at_end = points.coordinates[:, 0] == loaded_end
    displacement = 0.5 * 54.0 * 0.15
    end_error = float((points.u[at_end, 0] - displacement).abs().max()) / displacement
    passed = finite and jacobian and end_error < 1.0e-12 and prediction.qois != ()
    return DetectorOutcome(
        passed,
        {
            "finite": finite,
            "jacobian_consistent": jacobian,
            "loaded_end_error": end_error,
        },
    )


# ------------------------------------------------------------------ registry

DETECTORS: tuple[Detector, ...] = (
    Detector(
        "G4.tangent", "consistent tangent against central differences", g4_tangent
    ),
    Detector(
        "G4.closure-residual", "closure root positive, P33 zero", g4_closure_residual
    ),
    Detector(
        "G7.loads-fields-store", "body-force and normal-traction reactions", g7_loads
    ),
    Detector("G8.patch", "P1 and P2 affine patch through the backend", g8_patch),
    Detector(
        "G8.seeded-defects",
        "six seeded defects with production controls",
        g8_seeded_defects,
    ),
    Detector("G9.no-node-split", "a case's nodes carry one role", g9_no_node_split),
    Detector(
        "G10.projection-error",
        "projected fields carry their envelope",
        g10_projection_error,
    ),
    Detector(
        "G10.cross-mesh", "physical-integral cross-mesh comparison", g10_cross_mesh
    ),
    Detector(
        "G10.normalization",
        "TRAIN-only statistics in the S10 close-out",
        g10_normalization,
    ),
    Detector(
        "G11.lifting", "essential conditions exact through the lifting", g11_lifting
    ),
    Detector(
        "G11.lifting-mutant",
        "lifting control exact, identity mutant visible",
        g11_lifting_mutant,
    ),
    Detector("G12.anisotropy", "residual invariant to per-axis units", g12_anisotropy),
    Detector("G12.role-isolation", "objectives refuse HOLDOUT", g12_role_isolation),
    Detector("G13.holdout", "selection refuses HOLDOUT", g13_holdout),
    Detector(
        "G13.selection", "the first-ranked eligible trial is served", g13_selection
    ),
    Detector("G14.budget", "missing components are NOT_RUN", g14_budget),
    Detector("G14.reducer", "48 injected failures give NOT_VERIFIED", g14_reducer),
    Detector(
        "G14.holdout-order", "consumption record before the reveal", g14_holdout_order
    ),
    Detector("G15.fields", "served fields finite and consistent", g15_fields),
    Detector(
        "G15.substitution", "the declared stress authority is served", g15_substitution
    ),
    Detector("G15.refuse", "out-of-envelope requests refused", g15_refuse),
    Detector("G15.manifest", "every tampered byte detected", g15_manifest),
)

CACHES: tuple[Callable[..., Any], ...] = (
    run_seeded_defect_suite,
    _synthetic_bundle_path,
)

__all__ = ("CACHES", "DETECTORS", "cleanup")
