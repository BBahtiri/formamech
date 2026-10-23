"""Immutable, unit-safe PI-GINOT specification vocabulary."""

from piginot.contracts.canonical import canonical_json_bytes, strict_json_loads
from piginot.contracts.expressions import DerivedExpression
from piginot.contracts.hashing import HASH_DOMAINS, HashDomain, hash_spec
from piginot.contracts.problem_spec import ProblemSpec, ProblemSpecError
from piginot.contracts.profile import ExecutionProfile
from piginot.contracts.specs import (
    BoundaryConditionSpec,
    CampaignPlan,
    CanonicalMesh,
    DerivedQuantitySpec,
    GeometryFamilySpec,
    LoadSpec,
    MaterialSpec,
    ParameterSpec,
    ProblemFamilySpec,
    QoISpec,
    ToleranceSpec,
)
from piginot.contracts.units import (
    CanonicalQuantity,
    Dimension,
    Quantity,
    Unit,
    to_canonical,
)

__all__ = (
    "HASH_DOMAINS",
    "BoundaryConditionSpec",
    "CampaignPlan",
    "CanonicalMesh",
    "CanonicalQuantity",
    "DerivedExpression",
    "DerivedQuantitySpec",
    "Dimension",
    "ExecutionProfile",
    "GeometryFamilySpec",
    "HashDomain",
    "LoadSpec",
    "MaterialSpec",
    "ParameterSpec",
    "ProblemFamilySpec",
    "ProblemSpec",
    "ProblemSpecError",
    "QoISpec",
    "Quantity",
    "ToleranceSpec",
    "Unit",
    "canonical_json_bytes",
    "hash_spec",
    "strict_json_loads",
    "to_canonical",
)
