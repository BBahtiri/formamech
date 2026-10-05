"""Example 1: a fixed-mesh strain sweep, end to end on CPU (G17.example1).

One notched-plate geometry (W = 23 mm, r = 5 mm, d = 3 mm) on a coarse
structured mesh, loaded by a nominal axial strain between 10% and 20%:

1. compile a single-point geometry family and a strain envelope;
2. solve the P2 FEM reference at six training and three held-out strains;
3. train a small energy-form operator on the training strains with the
   framework trainer (no FEM label enters the loss);
4. verify the operator against FEM on the held-out strains;
5. calibrate the confidence indicator on the training strains, build a bundle,
   serve a held-out strain through ``piginot.bundle.predict`` and check that a
   strain outside the envelope is refused.

The example claims nothing: its bundle is ``NOT_VERIFIED`` because the full S14
protocol (FEM reference error terms, DEVELOPMENT and a one-time HOLDOUT) is not
run here. Usage::

    python -m examples.fixed_mesh_sweep.sweep --output OUT [--steps 600]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from piginot.bundle import (
    CONSTITUTIVE_PLANE_STRESS,
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
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    Dimension,
    ExecutionProfile,
    GeometryFamilySpec,
    MaterialSpec,
    ParameterSpec,
    Unit,
    hash_spec,
)
from piginot.doe import EvidenceRole, ExecutableEnvelope
from piginot.fem import FemRequest, FemResult, FemSolverConfig
from piginot.fem.backend import ScikitFemBackend
from piginot.morphing import (
    BlendedNotchMap,
    BlendedNotchMeshSpec,
    BlendedNotchParameters,
    FamilyMembershipPredicate,
    MorphValidityCriteria,
    build_blended_notch_reference_mesh,
)
from piginot.morphing.membership import PrimitiveBound
from piginot.objectives import (
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.operators import (
    ENERGY_ARCHITECTURE_ID,
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    EnergyOperatorConfig,
    QueryFrame,
    energy_module_digest,
)
from piginot.training import (
    CosineTailSchedule,
    DeterministicSamplerPlan,
    EnergyTrainerConfig,
    GridSearchSpace,
    TrainingPopulation,
    TrainingRunBinding,
    model_state_hash,
    train_energy_to_checkpoint,
)
from piginot.verification import fit_quantile_curve

HALF_LENGTH_MM = 25.0
WIDTH, RADIUS, DEPTH = 23.0, 5.0, 3.0
E_MPA, NU = 760.0, 0.23
TRAIN_STRAINS = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20)
HELD_OUT_STRAINS = (0.11, 0.15, 0.19)
LOAD_STEPS = 16
SEED = 17_001
TIME_BUDGET_SECONDS = 600.0
ENVELOPE_ID = "example-fixed-mesh-strain-sweep-v1"
TAGS = ("bottom", "right_end", "notch", "top", "left_symmetry")
OPERATOR = EnergyOperatorConfig(
    parameter_names=("nominal_axial_strain",),
    hidden_dim=32,
    latent_token_count=16,
    parameter_hidden_dim=32,
    num_attention_heads=4,
    decoder_layers=2,
    positional_frequencies=4,
)
ENERGY = PotentialEnergyConfig(
    points_per_cell=3,
    barrier_weight=1.0e3,
    barrier_threshold=0.05,
    admissibility_floor=1.0e-6,
)


def _length(value: float) -> CanonicalQuantity:
    return CanonicalQuantity(value, Unit.MM, Dimension.LENGTH)


def _bound(
    name: str, lower: float, upper: float, unit: Unit, dimension: Dimension
) -> PrimitiveBound:
    return PrimitiveBound(
        name,
        CanonicalQuantity(lower, unit, dimension),
        CanonicalQuantity(upper, unit, dimension),
    )


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The compiled single-point family of the example."""

    reference: BlendedNotchParameters
    mesh: Any
    morph_map: BlendedNotchMap
    membership: FamilyMembershipPredicate
    envelope: ExecutableEnvelope


def compile_geometry() -> Geometry:
    reference = BlendedNotchParameters(HALF_LENGTH_MM, WIDTH, RADIUS, DEPTH)
    spec = BlendedNotchMeshSpec(12, 12, 8, *TAGS)
    mesh = build_blended_notch_reference_mesh(
        reference, spec, source_provenance={"generator": "example-fixed-mesh-sweep-v1"}
    )
    morph_map = BlendedNotchMap(reference, "width", "notch_radius", "notch_depth")
    family = GeometryFamilySpec(
        name="example-fixed-notch-v1",
        parameters=tuple(
            ParameterSpec(name, _length(value), _length(value))
            for name, value in (
                ("width", WIDTH),
                ("notch_radius", RADIUS),
                ("notch_depth", DEPTH),
            )
        ),
        derived_quantities=(),
        topology_signature=mesh.topology_signature,
        semantic_tags=tuple(mesh.semantic_tags),
    )
    membership = FamilyMembershipPredicate.compile(
        family, mesh, MorphValidityCriteria(0.01, 0.05, 5.0, 20.0)
    )
    envelope = ExecutableEnvelope(
        envelope_id=ENVELOPE_ID,
        compiled_family_hash=hash_spec(
            {"family": family, "mesh_id": mesh.mesh_id}, "content"
        ),
        geometry_membership=membership,
        geometry_membership_hash=hash_spec(membership, "content"),
        material_bounds=(
            _bound("elastic_modulus", E_MPA, E_MPA, Unit.MPA, Dimension.STRESS),
            _bound("poisson_ratio", NU, NU, Unit.ONE, Dimension.DIMENSIONLESS),
        ),
        load_bounds=(
            _bound(
                "nominal_axial_strain", 0.10, 0.20, Unit.ONE, Dimension.DIMENSIONLESS
            ),
        ),
    )
    return Geometry(reference, mesh, morph_map, membership, envelope)


def request_inputs(strain: float) -> dict[str, object]:
    return {
        "width": _length(WIDTH),
        "notch_radius": _length(RADIUS),
        "notch_depth": _length(DEPTH),
        "elastic_modulus": CanonicalQuantity(E_MPA, Unit.MPA),
        "poisson_ratio": CanonicalQuantity(NU, Unit.ONE),
        "nominal_axial_strain": CanonicalQuantity(strain, Unit.ONE),
    }


def fem_request(mesh: Any, strain: float) -> FemRequest:
    return FemRequest(
        request_id=f"example-strain-{strain:.2f}",
        mesh=mesh,
        material=MaterialSpec(
            "example-material",
            "finite_strain_plane_stress_neo_hookean",
            {
                "elastic_modulus": CanonicalQuantity(E_MPA, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(NU, Unit.ONE),
            },
        ),
        boundary_conditions=(
            BoundaryConditionSpec(
                "left_x_symmetry", "left_symmetry", "symmetry", ("x",)
            ),
            BoundaryConditionSpec("bottom_y_symmetry", "bottom", "symmetry", ("y",)),
            BoundaryConditionSpec(
                "right_end_displacement",
                "right_end",
                "dirichlet",
                ("x",),
                (CanonicalQuantity(strain * HALF_LENGTH_MM, Unit.MM),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=2,
            solver_tolerance=1.0e-10,
            max_newton_iterations=20,
            plane_stress_max_iterations=32,
            load_steps=LOAD_STEPS,
            max_dofs=100_000,
        ),
        execution_profile=ExecutionProfile(
            python_version=platform.python_version(),
            library_versions={p: version(p) for p in ("numpy", "scikit-fem", "scipy")},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=0,
        ),
    )


def solve(requests: Sequence[FemRequest]) -> dict[float, FemResult]:
    attempts = ScikitFemBackend().solve_many(tuple(requests))
    results = {}
    for request, attempt in zip(requests, attempts, strict=True):
        if not attempt.success or attempt.result is None:
            raise RuntimeError(f"{request.request_id} FEM failed: {attempt.message}")
        strain = request.boundary_conditions[2].values[0].value / HALF_LENGTH_MM
        results[round(strain, 6)] = attempt.result
    return results


class _Case:
    """What the energy objective and the evaluation need for one strain."""

    def __init__(
        self,
        mesh: Any,
        strain: float,
        mean: float,
        deviation: float,
        dtype: torch.dtype,
    ):
        self.strain = strain
        self.domain = TriangleDomain(
            torch.tensor(mesh.nodes, dtype=dtype),
            torch.tensor(mesh.cells, dtype=torch.int64),
        )
        extent = torch.tensor([[HALF_LENGTH_MM, 0.5 * WIDTH]], dtype=dtype)
        self.frame = QueryFrame(torch.zeros((1, 2), dtype=dtype), extent)
        self.lifting = AxisAlignedTensionLifting(
            extent, torch.tensor([strain * HALF_LENGTH_MM], dtype=dtype)
        )
        self.parameters = torch.tensor([[(strain - mean) / deviation]], dtype=dtype)
        self.material = MaterialSpec(
            "example-material",
            "compressible_neo_hookean",
            {
                "elastic_modulus": CanonicalQuantity(E_MPA, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(NU, Unit.ONE),
            },
        )


class _Objective:
    def __init__(self, cases: Mapping[str, _Case]):
        self.cases = cases

    def __call__(
        self, model: torch.nn.Module, sample_id: str, step: int
    ) -> PotentialEnergyTerms:
        case = self.cases[sample_id]
        points, weights = stratified_triangle_quadrature(
            case.domain, ENERGY.points_per_cell
        )
        _, gradient = model.displacement_and_gradient(  # type: ignore[operator]
            query_xy=points.unsqueeze(0),
            parameters=case.parameters,
            frame=case.frame,
            lifting=case.lifting,
            create_graph=True,
        )
        return plane_stress_potential_energy(
            gradient[0],
            weights,
            case.material,
            config=ENERGY,
            reference_area=case.domain.area,
        )


def _revision() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if len(head) == 40:
            return head
    except (OSError, subprocess.CalledProcessError):
        pass
    return "0" * 40


def initial_model() -> EnergyFormOperator:
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(SEED)
        return EnergyFormOperator(OPERATOR).to(torch.float32)
    finally:
        torch.random.set_rng_state(state)


def train(geometry: Geometry, output: Path, steps: int) -> tuple[Path, float, float]:
    """Train with the framework trainer; returns the checkpoint, mean, deviation."""

    mean = math.fsum(TRAIN_STRAINS) / len(TRAIN_STRAINS)
    deviation = math.sqrt(
        math.fsum((s - mean) ** 2 for s in TRAIN_STRAINS) / len(TRAIN_STRAINS)
    )
    cases = {
        f"strain-{s:.2f}": _Case(geometry.mesh, s, mean, deviation, torch.float32)
        for s in TRAIN_STRAINS
    }
    ids = tuple(sorted(cases))
    snapshot = hash_spec(
        {"mesh_id": geometry.mesh.mesh_id, "strains": TRAIN_STRAINS}, "content"
    )
    population = TrainingPopulation(
        dataset_snapshot_hash=snapshot,
        public_manifest_hash=hash_spec(geometry.envelope, "content"),
        role=EvidenceRole.TRAIN,
        sample_ids=ids,
    )
    batches = []
    for epoch in range(math.ceil(2 * steps / len(ids))):
        generator = torch.Generator().manual_seed(SEED + epoch)
        order = torch.randperm(len(ids), generator=generator).tolist()
        batches.extend(tuple(order[i : i + 2]) for i in range(0, len(ids), 2))
    plan = DeterministicSamplerPlan(
        sampler_id=f"example-epoch-permutation:{SEED}",
        population_size=len(ids),
        batches=tuple(batches[:steps]),
    )
    space = GridSearchSpace((1.0e-3,), (1.0,), steps, 2, 1)
    (trial,) = space.trials
    config = EnergyTrainerConfig(
        search_space=space,
        trial=trial,
        schedule=CosineTailSchedule(1.0e-3, 1.0e-5, (2 * steps) // 3, steps),
    )
    normalization_hash = hash_spec(
        {"mean": mean, "standard_deviation": deviation}, "content"
    )
    objective_hash = hash_spec(
        {"energy": ENERGY, "lifting": "axis-aligned-tension"}, "recipe"
    )
    model = initial_model()
    binding = TrainingRunBinding(
        dataset_snapshot_hash=snapshot,
        public_manifest_hash=population.public_manifest_hash,
        normalization_hash=normalization_hash,
        architecture_id=model.architecture_id,
        architecture_module_digest=energy_module_digest(model),
        objective_registry_hash=objective_hash,
        trainer_config_hash=config.config_hash,
        sampler_plan_hash=plan.plan_hash,
        initial_weights_hash=model_state_hash(model),
        implementation_revision=_revision(),
        seed=SEED,
        execution_profile=ExecutionProfile(
            python_version=platform.python_version(),
            library_versions={"torch": torch.__version__},
            blas_threads=1,
            device="cpu",
            dtype="float32",
            rng_algorithm="torch-cpu-mt19937",
            seed=SEED,
        ),
    )
    checkpoint = output / "checkpoint"
    train_energy_to_checkpoint(
        checkpoint,
        model=model,
        population=population,
        plan=plan,
        binding=binding,
        config=config,
        objective=_Objective(cases),
        objective_hash=objective_hash,
        target_step=steps,
    )
    return checkpoint, mean, deviation


def _vertex_fields(result: FemResult, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.as_tensor(result.displacement[:count], dtype=torch.float64)
    stress = torch.as_tensor(
        result.recovered_first_piola_stress[:count], dtype=torch.float64
    )
    return u, stress[:, :2, :2]


def _notch_vertices(mesh: Any) -> list[int]:
    return sorted(
        {v for i in mesh.semantic_tags["notch"] for v in mesh.boundary_facets[i]}
    )


def errors(
    mesh: Any, u: torch.Tensor, stress: torch.Tensor, result: FemResult
) -> dict[str, float]:
    target_u, target_p = _vertex_fields(result, len(mesh.nodes))
    root = _notch_vertices(mesh)
    true_peak = torch.linalg.vector_norm(target_p[root], dim=(-2, -1)).max()
    peak = torch.linalg.vector_norm(stress[root], dim=(-2, -1)).max()
    return {
        "u_relative_l2": float(
            torch.linalg.vector_norm(u - target_u) / torch.linalg.vector_norm(target_u)
        ),
        "p_relative_l2": float(
            torch.linalg.vector_norm(stress - target_p)
            / torch.linalg.vector_norm(target_p)
        ),
        "absolute_peak_error": float(abs(peak - true_peak) / true_peak),
    }


def model_spec(mean: float, deviation: float, confidence: ConfidenceSpec) -> ModelSpec:
    config = dataclasses.asdict(OPERATOR)
    config["parameter_names"] = list(config["parameter_names"])
    return ModelSpec(
        architecture_id=ENERGY_ARCHITECTURE_ID,
        operator_config=config,
        conditioning=(
            ConditioningInput(
                "nominal_axial_strain", "nominal_axial_strain", mean, deviation
            ),
        ),
        frame_origin=(ScaledParameter((), 0.0), ScaledParameter((), 0.0)),
        frame_extent=(
            ScaledParameter((), HALF_LENGTH_MM),
            ScaledParameter(("width",), 0.5),
        ),
        lifting_kind="axis-aligned-tension",
        lifting_displacement=ScaledParameter(("nominal_axial_strain",), HALF_LENGTH_MM),
        material_name="example-compressible-neo-hookean",
        material_model="compressible_neo_hookean",
        material_parameters={
            "elastic_modulus": "elastic_modulus",
            "poisson_ratio": "poisson_ratio",
        },
        boundary_conditions={
            "left_symmetry": ("x",),
            "bottom": ("y",),
            "right_end": ("x",),
        },
        served_stress=CONSTITUTIVE_PLANE_STRESS,
        dtype="float64",
        query_chunk=2_048,
        qois=(
            QoiSpec("notch_peak_first_piola_frobenius", "P_served", "notch", (), "MPa"),
        ),
        confidence=confidence,
        units={
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
            "notch_peak_first_piola_frobenius": "MPa",
        },
    )


def make_bundle(
    directory: Path,
    geometry: Geometry,
    weights: bytes,
    mean: float,
    deviation: float,
    confidence: ConfidenceSpec,
    report: Mapping[str, Any],
) -> None:
    build_bundle(
        directory,
        name="example-fixed-mesh-strain-sweep",
        created_utc="2026-10-05T00:00:00Z",
        claim_level="NOT_VERIFIED",
        reference_mesh=geometry.mesh,
        morph_map=geometry.morph_map,
        envelope=geometry.envelope,
        model=model_spec(mean, deviation, confidence),
        weights=weights,
        verification={
            "claim_level": "NOT_VERIFIED",
            **report,
            "outstanding_work": [
                {
                    "id": "S14-protocol",
                    "status": "NOT_RUN",
                    "reason": "example only: no FEM reference error budget, "
                    "DEVELOPMENT or HOLDOUT evaluation was run",
                }
            ],
        },
        limitation_items=(
            {
                "id": "example-only",
                "statement": "Example 1 verifies against FEM on three held-out strains "
                "of one coarse mesh; no error budget, DEVELOPMENT or HOLDOUT protocol "
                "was run, so the bundle claims NOT_VERIFIED.",
            },
        ),
        identities={"example": "fixed-mesh-strain-sweep-v1"},
    )


def run(output: Path, steps: int) -> dict[str, Any]:
    start = time.perf_counter()
    timings: dict[str, float] = {}
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    geometry = compile_geometry()
    mark = time.perf_counter()
    strains = (*TRAIN_STRAINS, *HELD_OUT_STRAINS)
    fem = solve([fem_request(geometry.mesh, s) for s in strains])
    timings["fem_seconds"] = time.perf_counter() - mark
    mark = time.perf_counter()
    checkpoint, mean, deviation = train(geometry, output, steps)
    timings["training_seconds"] = time.perf_counter() - mark
    weights = (checkpoint / "model.safetensors").read_bytes()
    mark = time.perf_counter()
    placeholder = ConfidenceSpec("patch-mean", 1.0, E_MPA, 0.9, (1.0,), (0.0,))
    provisional = output / "provisional-bundle"
    make_bundle(
        provisional,
        geometry,
        weights,
        mean,
        deviation,
        placeholder,
        {"stage": "provisional"},
    )
    bundle = load_bundle(provisional)
    indicators, pointwise = [], []
    for strain in TRAIN_STRAINS:
        prediction = predict(bundle, request_inputs(strain))
        assert (
            prediction.vertices is not None
            and prediction.vertices.confidence_indicator is not None
        )
        _, target_p = _vertex_fields(fem[round(strain, 6)], len(geometry.mesh.nodes))
        indicators.append(prediction.vertices.confidence_indicator)
        pointwise.append(
            torch.linalg.vector_norm(
                (prediction.vertices.P_served[:, :2, :2] - target_p) / E_MPA,
                dim=(-2, -1),
            )
        )
    curve = fit_quantile_curve(
        torch.cat(indicators), torch.cat(pointwise), quantile=0.9, bins=20
    )
    confidence = ConfidenceSpec(
        "patch-mean",
        1.0,
        E_MPA,
        curve.quantile,
        tuple(curve.upper_edges),
        tuple(curve.values),
    )
    held_out: dict[str, dict[str, float]] = {}
    for strain in HELD_OUT_STRAINS:
        prediction = predict(bundle, request_inputs(strain), confidence=False)
        assert prediction.vertices is not None
        held_out[f"{strain:.2f}"] = errors(
            geometry.mesh,
            prediction.vertices.u,
            prediction.vertices.P_served[:, :2, :2],
            fem[round(strain, 6)],
        )
    report = {
        "example": "fixed-mesh strain sweep",
        "train_strains": list(TRAIN_STRAINS),
        "held_out_strains": list(HELD_OUT_STRAINS),
        "held_out_errors": held_out,
    }
    final = output / "bundle"
    make_bundle(final, geometry, weights, mean, deviation, confidence, report)
    bundle = load_bundle(final)
    served = predict(bundle, request_inputs(0.15), [[5.0, 2.0], [20.0, 8.0]])
    refused = predict(bundle, request_inputs(0.25), confidence=False)
    timings["verification_and_bundle_seconds"] = time.perf_counter() - mark
    total = time.perf_counter() - start
    summary = {
        "steps": steps,
        "mesh": {
            "vertices": len(geometry.mesh.nodes),
            "cells": len(geometry.mesh.cells),
        },
        "held_out_errors": held_out,
        "calibration_in_sample_coverage": curve.coverage(
            torch.cat(indicators), torch.cat(pointwise)
        ),
        "served_verdict": served.verdict.value,
        "served_claim_level": served.claim_level.value,
        "served_qois": served.document()["qois"],
        "refused_verdict": refused.verdict.value,
        "refused_reason_codes": list(refused.reason_codes),
        "bundle_manifest_sha256": bundle.manifest.manifest_sha256,
        "timings_seconds": {**timings, "total": total},
        "time_budget_seconds": TIME_BUDGET_SECONDS,
    }
    summary["passed"] = (
        served.verdict is Verdict.IN_DOMAIN
        and refused.verdict is Verdict.REFUSED
        and "PARAMETER_OUT_OF_BOUNDS" in refused.reason_codes
        and total < TIME_BUDGET_SECONDS
        and all(
            math.isfinite(v) for errors_ in held_out.values() for v in errors_.values()
        )
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.fixed_mesh_sweep.sweep")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    arguments = parser.parse_args(argv)
    summary = run(arguments.output, arguments.steps)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
