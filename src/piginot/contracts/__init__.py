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
from piginot.contracts.state_machine import (
    ALLOWED_CLAIM_LEVELS,
    STATE_MACHINE_SCHEMA,
    STATE_SNAPSHOT_SCHEMA,
    CheckStatus,
    ClaimLevel,
    InvalidTransitionError,
    StateMachineSpec,
    StateSnapshot,
    TransitionRecord,
    TransitionSpec,
    get_state_machine,
    register_state_machine,
    registered_state_machines,
)
from piginot.contracts.units import (
    CanonicalQuantity,
    Dimension,
    Quantity,
    Unit,
    to_canonical,
)

__all__ = (
    "ALLOWED_CLAIM_LEVELS",
    "HASH_DOMAINS",
    "STATE_MACHINE_SCHEMA",
    "STATE_SNAPSHOT_SCHEMA",
    "BoundaryConditionSpec",
    "CampaignPlan",
    "CanonicalMesh",
    "CanonicalQuantity",
    "CheckStatus",
    "ClaimLevel",
    "DerivedExpression",
    "DerivedQuantitySpec",
    "Dimension",
    "ExecutionProfile",
    "GeometryFamilySpec",
    "HashDomain",
    "InvalidTransitionError",
    "LoadSpec",
    "MaterialSpec",
    "ParameterSpec",
    "ProblemFamilySpec",
    "ProblemSpec",
    "ProblemSpecError",
    "QoISpec",
    "Quantity",
    "StateMachineSpec",
    "StateSnapshot",
    "ToleranceSpec",
    "TransitionRecord",
    "TransitionSpec",
    "Unit",
    "canonical_json_bytes",
    "get_state_machine",
    "hash_spec",
    "register_state_machine",
    "registered_state_machines",
    "strict_json_loads",
    "to_canonical",
)
