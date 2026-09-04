"""Deterministic S10 FEM requests for the ratified DogBone campaign."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from campaigns.dogbone_v1.family import compiled_family, parameter_point
from campaigns.dogbone_v1.strain_pilot import (
    LOAD_POLICY_ID,
    prescribed_displacement_mm,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
    hash_spec,
)
from piginot.datasets.request_plan import (
    PublicFemRequestPlan,
    build_public_fem_request_plan,
)
from piginot.doe import CPEnvelope, PublishedCase, PublishedCaseManifest
from piginot.doe.publication import PublishedCaseStatus
from piginot.fem import FemRequest, FemSolverConfig
from piginot.morphing import apply_morph

DOGBONE_S10_LOAD_POLICY_ID = LOAD_POLICY_ID

_CAMPAIGN_HASH = "8f58fe30803e2bd147f863ac000e1e89a19092c8b347df6b4d74f38487d96d46"
_MANIFEST_HASH = "ce95c03f5b22308a7aa9bf627966a6b21300a4e7b0272d3affa7dd503af564a9"
_COMPILED_FAMILY_HASH = (
    "a8143a75e62c614d40f52049cfc88c1cda74d38c9bd488cc3bed4c5678d973a4"
)
_REFERENCE_MESH_ID = "3bfc1043f5aa9e989469adacb0ee1d0b483825ffabcd5dfac4bf9caca580d82c"
_VALIDITY_POLICY_HASH = (
    "b3c2df59eed939f1aa14a69d05a6fd2dc57be571ffba33c6355982713cc6931d"
)
_CAMPAIGN_ROOT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "campaigns" / "dogbone-v1"
)
_GEOMETRY_AXES = ("L_total", "W_grip", "W_gauge", "R_fillet")
_REQUIRED_PRIMITIVES = frozenset(
    (*_GEOMETRY_AXES, "elastic_modulus", "poisson_ratio", "nominal_axial_strain")
)


def _load_public_sources() -> tuple[CPEnvelope, PublishedCaseManifest]:
    checkpoint = CPEnvelope.from_bytes(
        (_CAMPAIGN_ROOT / "CP-ENVELOPE.json").read_bytes()
    )
    manifest = PublishedCaseManifest.from_bytes(
        (_CAMPAIGN_ROOT / "public-case-manifest.json").read_bytes()
    )
    if checkpoint.campaign_hash != _CAMPAIGN_HASH:
        raise RuntimeError("DogBone CP-ENVELOPE campaign hash changed")
    if checkpoint.public_case_manifest_hash != _MANIFEST_HASH:
        raise RuntimeError("DogBone CP-ENVELOPE manifest binding changed")
    if manifest.content_hash != _MANIFEST_HASH:
        raise RuntimeError("DogBone public manifest content hash changed")
    if checkpoint.compiled_family_hash != _COMPILED_FAMILY_HASH:
        raise RuntimeError("DogBone compiled-family checkpoint binding changed")
    return checkpoint, manifest


def _number(case: PublishedCase, name: str) -> float:
    value = case.primitive_coordinates[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"DogBone case {case.case_id!r} has non-numeric {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"DogBone case {case.case_id!r} has non-finite {name}")
    return number


def _material(case: PublishedCase) -> MaterialSpec:
    return MaterialSpec(
        "dogbone-pinned-reference-material-v1",
        "finite_strain_plane_stress_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(
                _number(case, "elastic_modulus"), Unit.MPA
            ),
            "poisson_ratio": CanonicalQuantity(
                _number(case, "poisson_ratio"), Unit.ONE
            ),
        },
    )


def _boundary_conditions(case: PublishedCase) -> tuple[BoundaryConditionSpec, ...]:
    displacement = prescribed_displacement_mm(
        _number(case, "L_total"),
        _number(case, "nominal_axial_strain"),
    )
    return (
        BoundaryConditionSpec(
            "left_x_symmetry",
            "left_symmetry",
            "symmetry",
            ("x",),
        ),
        BoundaryConditionSpec(
            "bottom_y_symmetry",
            "bottom",
            "symmetry",
            ("y",),
        ),
        BoundaryConditionSpec(
            "right_grip_displacement",
            "right_grip",
            "dirichlet",
            ("x",),
            (CanonicalQuantity(displacement, Unit.MM),),
        ),
    )


def _solver_config() -> FemSolverConfig:
    return FemSolverConfig(
        element_order=2,
        solver_tolerance=1.0e-10,
        max_newton_iterations=20,
        plane_stress_max_iterations=32,
        load_steps=64,
        max_dofs=100_000,
    )


def build_dogbone_s10_public_requests(
    *, execution_profile: ExecutionProfile
) -> tuple[FemRequest, ...]:
    """Materialize one exact S7 request per active public S9 case."""

    if type(execution_profile) is not ExecutionProfile:
        raise TypeError("execution_profile must be an exact ExecutionProfile")
    return _build_dogbone_s10_public_requests_cached(execution_profile)


@lru_cache(maxsize=8)
def _build_dogbone_s10_public_requests_cached(
    execution_profile: ExecutionProfile,
) -> tuple[FemRequest, ...]:
    checkpoint, manifest = _load_public_sources()
    family = compiled_family()
    if hash_spec(family, "content") != _COMPILED_FAMILY_HASH:
        raise RuntimeError("runtime DogBone compiled family changed")
    if family.reference_mesh.mesh_id != _REFERENCE_MESH_ID:
        raise RuntimeError("runtime DogBone reference mesh changed")
    if family.membership.validity_policy_hash != _VALIDITY_POLICY_HASH:
        raise RuntimeError("runtime DogBone validity policy changed")

    active = tuple(
        sorted(
            (
                case
                for case in manifest.cases
                if case.status is PublishedCaseStatus.ACTIVE
            ),
            key=lambda case: case.case_id,
        )
    )
    requests: list[FemRequest] = []
    for case in active:
        if frozenset(case.primitive_coordinates) != _REQUIRED_PRIMITIVES:
            raise RuntimeError(f"DogBone case {case.case_id!r} primitive axes changed")
        theta = parameter_point(*(_number(case, name) for name in _GEOMETRY_AXES))
        morph = apply_morph(
            family.reference_mesh,
            family.morph_map,
            family.membership,
            theta,
        )
        if not morph.accepted or morph.mesh is None:
            raise RuntimeError(
                f"active DogBone case {case.case_id!r} failed ratified morphing"
            )
        requests.append(
            FemRequest(
                request_id=case.case_id,
                mesh=morph.mesh,
                material=_material(case),
                boundary_conditions=_boundary_conditions(case),
                loads=(),
                qois=(),
                config=_solver_config(),
                execution_profile=execution_profile,
            )
        )
    return tuple(requests)


def build_dogbone_s10_request_plan(
    *, execution_profile: ExecutionProfile
) -> PublicFemRequestPlan:
    """Bind the exact active DogBone request population to CP-ENVELOPE."""

    checkpoint, manifest = _load_public_sources()
    requests = build_dogbone_s10_public_requests(execution_profile=execution_profile)
    return build_public_fem_request_plan(
        checkpoint,
        manifest,
        {request.request_id: request for request in requests},
    )


__all__ = (
    "DOGBONE_S10_LOAD_POLICY_ID",
    "build_dogbone_s10_public_requests",
    "build_dogbone_s10_request_plan",
)
