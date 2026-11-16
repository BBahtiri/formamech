"""Public S6 evidence for the notched campaign: family sweep and FEM pilot.

Both runs use public, secret-free inputs only. The family sweep morphs 500
shapes of a public Latin hypercube; the FEM pilot solves the reference shape
and the eight corners of the geometry box at both nominal-strain endpoints with
the production recipe (P2, 64 load steps).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import pickle
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import ClassVar

from campaigns.dogbone_v1.fem_pilot import (
    FemPilotRecord,
    FemPilotSummary,
    PilotGeometryCase,
    record_pilot_attempt,
    summarize_pilot_records,
)
from campaigns.dogbone_v1.strain_pilot import _minimum_deformation_jacobian
from campaigns.notched_v1.family import (
    BOUNDS,
    GEOMETRY,
    boundary_conditions,
    compiled_family,
    material,
    parameter_point,
    prescribed_displacement_mm,
    solver_config,
)
from piginot.contracts import (
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.fem import FemRequest, FemSolverConfig
from piginot.fem.backend import ScikitFemBackend
from piginot.morphing import apply_morph

ARTIFACTS = Path("artifacts/campaigns/notched-v1")
VALIDATION_PATH = ARTIFACTS / "family-validation.json"
FEM_PILOT_PATH = ARTIFACTS / "fem-pilot.json"
PUBLIC_VALIDATION_SEED = "notched-family-validation-20261005-v1"
SAMPLE_GENERATOR_ID = "blake2b-latin-hypercube-v1"
FEM_PILOT_ID = "notched-fem-pilot-20261005-v1"
SELECTION_POLICY_ID = "reference-and-geometry-box-corners-at-strain-endpoints-v1"
STRAIN_LEVELS = (0.10, 0.20)
LOAD_STEPS = 64
_LOADED_END_CONDITION = "right_end_displacement"
_PERSON = b"piginot-ntv1"
_DECIMALS = 12


def _reported(value: float) -> float:
    rounded = round(value, _DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


@dataclass(frozen=True, slots=True)
class FamilyValidationSummary:
    """Canonical result of one public family-level morph validation sweep."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    recipe_id: str
    public_seed: str
    sample_generator_id: str
    candidate_count: int
    attempted: int
    membership_accepted: int
    accepted: int
    membership_rejections: Mapping[str, int]
    morph_rejections: Mapping[str, int]
    minimum_jacobian_determinant: float
    minimum_feature_size_mm: float
    minimum_angle_degrees: float
    maximum_aspect_ratio: float
    quadrature_points_per_shape: int
    reference_mesh_id: str
    validity_policy_hash: str
    compiled_family_hash: str
    worst_case_parameters: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        for name in (
            "minimum_jacobian_determinant",
            "minimum_feature_size_mm",
            "minimum_angle_degrees",
            "maximum_aspect_ratio",
        ):
            object.__setattr__(self, name, _reported(getattr(self, name)))
        for name in ("membership_rejections", "morph_rejections"):
            values = dict(sorted(getattr(self, name).items()))
            object.__setattr__(self, name, MappingProxyType(values))
        worst = {
            key: tuple(_reported(value) for value in values)
            for key, values in sorted(self.worst_case_parameters.items())
        }
        object.__setattr__(self, "worst_case_parameters", MappingProxyType(worst))


def _digest(seed_key: bytes, domain: str, index: int) -> bytes:
    payload = domain.encode("utf-8") + b"\0" + index.to_bytes(8, "big")
    return hashlib.blake2b(
        payload, key=seed_key, digest_size=16, person=_PERSON
    ).digest()


def public_samples(
    count: int,
    *,
    seed: str = PUBLIC_VALIDATION_SEED,
    names: Sequence[str] = GEOMETRY,
) -> tuple[tuple[float, ...], ...]:
    """A keyed Latin hypercube over the named family bounds."""

    seed_key = hashlib.blake2b(seed.encode("utf-8"), digest_size=32).digest()
    permutations = {
        name: sorted(
            range(count),
            key=lambda index, name=name: (
                _digest(seed_key, f"permutation/{name}", index),
                index,
            ),
        )
        for name in names
    }
    rows = []
    for row in range(count):
        values = []
        for name in names:
            lower, upper = BOUNDS[name]
            jitter = int.from_bytes(_digest(seed_key, f"jitter/{name}", row)[:8], "big")
            fraction = (permutations[name][row] + (jitter + 0.5) / 2**64) / count
            values.append(lower + (upper - lower) * fraction)
        rows.append(tuple(values))
    return tuple(rows)


def run_family_validation_sweep(
    *, required_accepted: int = 500, candidate_count: int = 900
) -> FamilyValidationSummary:
    """Morph a public LHS until ``required_accepted`` shapes pass."""

    family = compiled_family()
    membership_rejections: Counter[str] = Counter()
    morph_rejections: Counter[str] = Counter()
    minima = {"jacobian": float("inf"), "feature": float("inf"), "angle": float("inf")}
    maximum_aspect = 0.0
    worst: dict[str, tuple[float, ...]] = {}
    accepted = membership_accepted = attempted = quadrature_points = 0
    for values in public_samples(candidate_count):
        attempted += 1
        theta = parameter_point(*values)
        decision = family.membership.evaluate(theta)
        if not decision.accepted:
            membership_rejections.update(code.value for code in decision.reason_codes)
            continue
        membership_accepted += 1
        result = apply_morph(
            family.reference_mesh, family.morph_map, family.membership, theta
        )
        if not result.accepted:
            morph_rejections.update(code.value for code in result.reason_codes)
            continue
        accepted += 1
        report = result.report
        quadrature_points = report.quadrature_points_checked
        measures = {
            "jacobian": report.minimum_jacobian_determinant,
            "feature": report.minimum_feature_size_mm,
            "angle": report.minimum_angle_degrees,
        }
        for name, value in measures.items():
            if value is None:
                raise RuntimeError("accepted morph omitted a validity measure")
            if value < minima[name]:
                minima[name] = value
                worst[name] = values
        if report.maximum_aspect_ratio is None:
            raise RuntimeError("accepted morph omitted its aspect ratio")
        if report.maximum_aspect_ratio > maximum_aspect:
            maximum_aspect = report.maximum_aspect_ratio
            worst["aspect"] = values
        if accepted == required_accepted:
            break
    if accepted != required_accepted:
        raise RuntimeError("public validation stream did not reach its target")
    return FamilyValidationSummary(
        recipe_id=family.recipe_id,
        public_seed=PUBLIC_VALIDATION_SEED,
        sample_generator_id=SAMPLE_GENERATOR_ID,
        candidate_count=candidate_count,
        attempted=attempted,
        membership_accepted=membership_accepted,
        accepted=accepted,
        membership_rejections=membership_rejections,
        morph_rejections=morph_rejections,
        minimum_jacobian_determinant=minima["jacobian"],
        minimum_feature_size_mm=minima["feature"],
        minimum_angle_degrees=minima["angle"],
        maximum_aspect_ratio=maximum_aspect,
        quadrature_points_per_shape=quadrature_points,
        reference_mesh_id=family.reference_mesh.mesh_id,
        validity_policy_hash=family.membership.validity_policy_hash,
        compiled_family_hash=hash_spec(family, "content"),
        worst_case_parameters=worst,
    )


@dataclass(frozen=True, slots=True)
class NotchedPilotRecord:
    """One pilot solve: the S7 outcome plus the strain-response diagnostics."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    nominal_axial_strain: float
    prescribed_displacement: CanonicalQuantity
    outcome: FemPilotRecord
    loaded_end_reaction: float | None
    minimum_deformation_jacobian: float | None


@dataclass(frozen=True, slots=True)
class NotchedFemPilotReport:
    """Public FEM solvability evidence over the geometry box and strain range."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    pilot_id: str
    selection_policy_id: str
    compiled_family_hash: str
    reference_mesh_id: str
    validity_policy_hash: str
    material: MaterialSpec
    solver_config: FemSolverConfig
    execution_profile: ExecutionProfile
    runtime_environment: Mapping[str, str]
    backend_name: str
    records: tuple[NotchedPilotRecord, ...]
    summary: FemPilotSummary
    minimum_deformation_jacobian: float


def pilot_cases() -> tuple[PilotGeometryCase, ...]:
    """Reference shape and the eight box corners at both strain endpoints.

    ``parameters`` is ``(width, notch_radius, notch_depth, strain)``.
    """

    cases = []
    for strain in STRAIN_LEVELS:
        cases.append(
            PilotGeometryCase(
                f"reference-e{strain:.2f}", (23.0, 5.0, 3.0, strain), "reference"
            )
        )
        for corner in itertools.product(*(BOUNDS[name] for name in GEOMETRY)):
            width, radius, depth = corner
            sentinel = (
                "difficult"
                if (width, radius, depth) == (16.0, 2.0, 5.0)
                else "boundary"
            )
            label = f"w{width:g}-r{radius:g}-d{depth:g}-e{strain:.2f}"
            cases.append(PilotGeometryCase(label, (*corner, strain), sentinel))
    return tuple(cases)


def execution_profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version=platform.python_version(),
        library_versions={
            package: version("formamech" if package == "piginot" else package)
            for package in ("numpy", "piginot", "scikit-fem", "scipy", "torch")
        },
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def pilot_request(case: PilotGeometryCase, profile: ExecutionProfile) -> FemRequest:
    family = compiled_family()
    width, radius, depth, strain = case.parameters
    morph = apply_morph(
        family.reference_mesh,
        family.morph_map,
        family.membership,
        parameter_point(width, radius, depth),
    )
    if not morph.accepted or morph.mesh is None:
        raise RuntimeError(f"pilot geometry {case.case_id!r} failed morphing")
    return FemRequest(
        request_id=f"{FEM_PILOT_ID}-{case.case_id}",
        mesh=morph.mesh,
        material=material(),
        boundary_conditions=boundary_conditions(strain),
        loads=(),
        qois=(),
        config=solver_config(load_steps=LOAD_STEPS),
        execution_profile=profile,
    )


def _solve(case: PilotGeometryCase) -> NotchedPilotRecord:
    request = pilot_request(case, execution_profile())
    started = perf_counter()
    (attempt,) = ScikitFemBackend().solve_many((request,))
    elapsed = perf_counter() - started
    outcome = record_pilot_attempt(case, request, attempt, elapsed)
    strain = case.parameters[3]
    displacement = CanonicalQuantity(prescribed_displacement_mm(strain), Unit.MM)
    if not outcome.success or attempt.result is None:
        return NotchedPilotRecord(strain, displacement, outcome, None, None)
    reaction = attempt.result.reactions[_LOADED_END_CONDITION][0]
    jacobian = _minimum_deformation_jacobian(attempt.result)
    return NotchedPilotRecord(strain, displacement, outcome, reaction, jacobian)


def _solve_cached(task: tuple[PilotGeometryCase, str | None]) -> NotchedPilotRecord:
    """Solve one case, or reuse its record from a scratch cache (restarts)."""

    case, cache = task
    if cache is None:
        return _solve(case)
    path = Path(cache) / f"{case.case_id}.pickle"
    if path.is_file():
        record: NotchedPilotRecord = pickle.loads(path.read_bytes())
        return record
    record = _solve(case)
    staging = path.with_suffix(".tmp")
    staging.write_bytes(pickle.dumps(record))
    staging.replace(path)
    return record


def run_fem_pilot(
    *, workers: int = 2, cache: Path | None = None
) -> NotchedFemPilotReport:
    """Solve every pilot case; ``cache`` (outside the repository) survives restarts."""

    cases = pilot_cases()
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    tasks = [(case, None if cache is None else str(cache)) for case in cases]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = tuple(pool.map(_solve_cached, tasks))
    family = compiled_family()
    jacobians = [
        record.minimum_deformation_jacobian
        for record in records
        if record.minimum_deformation_jacobian is not None
    ]
    return NotchedFemPilotReport(
        pilot_id=FEM_PILOT_ID,
        selection_policy_id=SELECTION_POLICY_ID,
        compiled_family_hash=hash_spec(family, "content"),
        reference_mesh_id=family.reference_mesh.mesh_id,
        validity_policy_hash=family.membership.validity_policy_hash,
        material=material(),
        solver_config=solver_config(load_steps=LOAD_STEPS),
        execution_profile=execution_profile(),
        runtime_environment={
            "logical_cpu_count": str(os.cpu_count() or "unknown"),
            "machine": platform.machine() or "unknown",
            "worker_processes": str(workers),
        },
        backend_name=ScikitFemBackend().backend_name,
        records=records,
        summary=summarize_pilot_records(tuple(record.outcome for record in records)),
        minimum_deformation_jacobian=min(jacobians, default=float("nan")),
    )


def write_exact(target: Path, payload: bytes) -> None:
    """Create a public artifact once; refuse to replace different evidence."""

    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace existing evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.pilot")
    parser.add_argument("phase", choices=("validation", "fem"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache", type=Path)
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if arguments.phase == "validation":
        report: object = run_family_validation_sweep()
        target = root / VALIDATION_PATH
    else:
        report = run_fem_pilot(workers=arguments.workers, cache=arguments.cache)
        target = root / FEM_PILOT_PATH
    write_exact(target, canonical_json_bytes(report) + b"\n")
    print(f"wrote {target}")
    print(f"content hash: {hash_spec(report, 'content')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FEM_PILOT_ID",
    "FamilyValidationSummary",
    "NotchedFemPilotReport",
    "NotchedPilotRecord",
    "PUBLIC_VALIDATION_SEED",
    "pilot_cases",
    "public_samples",
    "run_family_validation_sweep",
    "run_fem_pilot",
    "write_exact",
)
