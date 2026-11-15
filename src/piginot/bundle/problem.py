"""Portable problem and evidence identities for material-aware candidates.

This carries definitions and verification identities, not a scientific claim.
The ordinary ProblemSpec parser is the only formula deserialization path.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from piginot.contracts import (
    CanonicalMesh,
    ProblemSpec,
    canonical_json_bytes,
)
from piginot.operators.architecture import LiftingVerification
from piginot.physics.errors import (
    InadmissibleDeformationError,
    NonFiniteTensorError,
    PlaneStressClosureError,
)

if TYPE_CHECKING:
    from piginot.contracts import MaterialSpec
    from piginot.doe.envelope import ExecutableEnvelope
    from piginot.lifting import DirichletLifting
    from piginot.operators.energy_operator import QueryFrame
    from piginot.problem.verification import LiftingEvidence

SCHEMA = "piginot.bundle-problem-v1"
MODEL_SCHEMA = "piginot.bundle-model-v2"
PROBLEM_LIFTING = "problem-spec-v1"
PROBLEM_STRESS = "constitutive-plane-stress-problem-spec-v1"
MATERIAL_DOMAIN_ERRORS = (
    InadmissibleDeformationError,
    NonFiniteTensorError,
    PlaneStressClosureError,
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_VERIFICATION_KEYS = {
    "verification_id",
    "parameter_range_hash",
    "verification_hash",
    "boundary_refinement_levels",
    "full_parameter_range_verified",
    "max_boundary_residual",
}


class PredictionReasonCode(StrEnum):
    """Refusals raised by the inference closure itself."""

    INADMISSIBLE_DEFORMATION = "PREDICTION_INADMISSIBLE_DEFORMATION"
    PROBLEM_OUT_OF_DOMAIN = "PREDICTION_PROBLEM_OUT_OF_DOMAIN"
    MATERIAL_DOMAIN_FAILURE = "PREDICTION_MATERIAL_DOMAIN_FAILURE"


def _lifting_evidence(value: object) -> LiftingEvidence:
    """Decode S24 evidence using primitive data, never executable objects."""
    from piginot.problem.verification import LiftingEvidence

    fields = {
        "problem_hash",
        "kind",
        "parameter_vectors",
        "refused_geometries",
        "refinements",
        "node_residual",
        "conformity",
        "orientation",
        "boundary_residual",
        "mutant_miss",
        "thresholds",
        "passed",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("lifting evidence has invalid fields")
    if (
        not isinstance(value["problem_hash"], str)
        or _HASH.fullmatch(value["problem_hash"]) is None
        or value["kind"] not in ("quarter", "adf")
        or value["passed"] is not True
        or value["thresholds"]
        != {"exact": 1e-12, "conformity": 1e-9, "mutant_minimum_miss": 1e-3}
    ):
        raise ValueError("lifting evidence has invalid identity or thresholds")
    for name in ("parameter_vectors", "refused_geometries"):
        if type(value[name]) is not int or value[name] < (
            1 if name == "parameter_vectors" else 0
        ):
            raise ValueError("lifting evidence counts are invalid")
    if value["refused_geometries"] != 0:
        raise ValueError("lifting evidence must cover the full parameter range")
    if not isinstance(value["refinements"], list) or not isinstance(
        value["boundary_residual"], dict
    ):
        raise ValueError("lifting evidence refinements and residuals are invalid")

    def residual(number: object) -> bool:
        return (
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(number)
            and number >= 0
        )

    if not all(
        residual(value[name]) for name in ("node_residual", "mutant_miss")
    ) or not all(residual(number) for number in value["boundary_residual"].values()):
        raise ValueError(
            "lifting evidence residuals must be finite nonnegative numbers"
        )
    if value["kind"] == "adf":
        if (
            not residual(value["conformity"])
            or not residual(value["orientation"])
            or set(value["boundary_residual"])
            != {str(level) for level in value["refinements"]}
        ):
            raise ValueError("ADF lifting evidence lacks measured refinements")
    elif (
        value["conformity"] is not None
        or value["orientation"] is not None
        or value["boundary_residual"]
    ):
        raise ValueError("quarter lifting evidence has unexpected ADF measurements")
    evidence = LiftingEvidence(
        value["problem_hash"],
        value["kind"],
        value["parameter_vectors"],
        value["refused_geometries"],
        tuple(value["refinements"]),
        value["node_residual"],
        value["conformity"],
        value["orientation"],
        MappingProxyType(
            {int(level): number for level, number in value["boundary_residual"].items()}
        ),
        value["mutant_miss"],
    )
    if not evidence.passed:
        raise ValueError("lifting evidence did not pass")
    return evidence


@dataclass(frozen=True, slots=True)
class PortableProblem:
    spec: ProblemSpec
    material_admission_sha256: str
    lifting_verification: LiftingVerification
    lifting_evidence: LiftingEvidence

    def __post_init__(self) -> None:
        if type(self.spec) is not ProblemSpec:
            raise ValueError("portable problem needs an exact ProblemSpec")
        if (
            not isinstance(self.material_admission_sha256, str)
            or _HASH.fullmatch(self.material_admission_sha256) is None
        ):
            raise ValueError("material admission needs a lowercase SHA256 digest")
        proof = self.lifting_verification
        if type(proof) is not LiftingVerification:
            raise ValueError("portable problem needs a typed lifting verification")
        from piginot.problem.verification import LiftingEvidence

        if type(self.lifting_evidence) is not LiftingEvidence:
            raise ValueError("portable problem needs typed lifting evidence")
        evidence = _lifting_evidence(self.lifting_evidence.document())
        if (
            evidence.problem_hash != self.problem_hash
            or evidence.kind != self.spec.lifting.kind
            or evidence.verification(self.spec) != proof
        ):
            raise ValueError(
                "lifting evidence differs from the problem or verification"
            )
        object.__setattr__(self, "lifting_evidence", evidence)
        if any(
            condition.kind == "neumann" for condition in self.spec.boundary_conditions
        ):
            raise ValueError("portable bundles do not support Neumann traction")

    @property
    def problem_hash(self) -> str:
        return self.spec.problem_hash

    @property
    def identities(self) -> dict[str, str]:
        return {
            "problem_hash": self.problem_hash,
            "material_admission_sha256": self.material_admission_sha256,
            "lifting_verification_hash": self.lifting_verification.verification_hash,
        }

    def document(self) -> dict[str, Any]:
        proof = self.lifting_verification
        return {
            "schema": SCHEMA,
            "spec": self.spec.document(),
            "problem_hash": self.problem_hash,
            "material_admission_sha256": self.material_admission_sha256,
            "lifting_verification": {
                "verification_id": proof.verification_id,
                "parameter_range_hash": proof.parameter_range_hash,
                "verification_hash": proof.verification_hash,
                "boundary_refinement_levels": list(proof.boundary_refinement_levels),
                "full_parameter_range_verified": proof.full_parameter_range_verified,
                "max_boundary_residual": proof.max_boundary_residual,
            },
            "lifting_evidence": self.lifting_evidence.document(),
        }

    @classmethod
    def from_document(cls, value: object) -> PortableProblem:
        keys = {
            "schema",
            "spec",
            "problem_hash",
            "material_admission_sha256",
            "lifting_verification",
            "lifting_evidence",
        }
        if (
            not isinstance(value, dict)
            or set(value) != keys
            or value["schema"] != SCHEMA
        ):
            raise ValueError("portable problem has an unsupported schema or fields")
        proof = value["lifting_verification"]
        if not isinstance(proof, dict) or set(proof) != _VERIFICATION_KEYS:
            raise ValueError("lifting verification has invalid fields")
        levels = proof["boundary_refinement_levels"]
        if not isinstance(levels, list):
            raise ValueError("lifting refinements must be an array")
        problem = cls(
            ProblemSpec.from_document(value["spec"]),
            value["material_admission_sha256"],
            LiftingVerification(
                **{**proof, "boundary_refinement_levels": tuple(levels)}
            ),
            _lifting_evidence(value["lifting_evidence"]),
        )
        if value["problem_hash"] != problem.problem_hash:
            raise ValueError("portable problem hash differs from its definition")
        return problem

    def boundary_components(self) -> dict[str, tuple[str, ...]]:
        return {
            region: tuple(
                component
                for component in ("x", "y")
                if any(
                    condition.region == region
                    and condition.is_dirichlet
                    and component in condition.components
                    for condition in self.spec.boundary_conditions
                )
            )
            for region in sorted(
                {
                    condition.region
                    for condition in self.spec.boundary_conditions
                    if condition.is_dirichlet
                }
            )
        }

    def check_bundle(
        self, mesh: CanonicalMesh, morph_map: object, envelope: ExecutableEnvelope
    ) -> None:
        if self.spec.geometry.reference_mesh_id != mesh.mesh_id:
            raise ValueError("portable reference mesh differs from ProblemSpec")
        digest = hashlib.sha256(canonical_json_bytes(morph_map)).hexdigest()
        if self.spec.geometry.morph_map_sha256 != digest:
            raise ValueError("portable morph map differs from ProblemSpec")
        bounds = {bound.name: bound for bound in envelope.primitive_bounds}
        if set(bounds) != {parameter.name for parameter in self.spec.parameters}:
            raise ValueError("portable envelope inputs differ from ProblemSpec")
        for parameter in self.spec.parameters:
            bound = bounds[parameter.name]
            if (
                bound.lower_bound.unit != parameter.unit
                or bound.lower_bound.value < parameter.lo
                or bound.upper_bound.value > parameter.hi
            ):
                raise ValueError("portable envelope exceeds the ProblemSpec domain")
        self.spec.check_mesh(mesh)

    def request_context(
        self, values: Mapping[str, float]
    ) -> tuple[QueryFrame, DirichletLifting, MaterialSpec, float]:
        from piginot.physics import reference_modulus
        from piginot.problem.lifting import build_lifting, query_frame
        from piginot.problem.material import material_spec

        material = material_spec(self.spec, values)
        return (
            query_frame(self.spec, [values]),
            build_lifting(self.spec, [values]),
            material,
            reference_modulus(material),
        )

    def accepts(self, inputs: Mapping[str, object]) -> bool:
        """The embedded constraint domain, in addition to envelope bounds."""
        from piginot.problem.expressions import value

        values = self.spec.values(inputs)
        try:
            return all(
                constraint.lo <= value(constraint.expression, values) <= constraint.hi
                for constraint in self.spec.constraints
            )
        except (ArithmeticError, ValueError):
            return False


__all__ = (
    "MATERIAL_DOMAIN_ERRORS",
    "MODEL_SCHEMA",
    "PROBLEM_LIFTING",
    "PROBLEM_STRESS",
    "PortableProblem",
    "PredictionReasonCode",
)
