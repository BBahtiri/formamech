"""S11 neural-operator architecture surface."""

from piginot.operators.architecture import (
    ARCHITECTURE_ID,
    LOCAL_STRESS_ARCHITECTURE_ID,
    BoundaryConditionMode,
    DirichletData,
    GeometryEncoding,
    LiftingVerification,
    OperatorConfig,
    OperatorOutput,
    PiGinotOperator,
    module_digest,
)
from piginot.operators.gate import (
    GateStatus,
    S11GateCheck,
    S11GateEvaluation,
    S11GateMeasurements,
    evaluate_s11_gate,
)
from piginot.operators.safetensors_io import (
    TensorSpec,
    load_operator_state_dict,
    load_operator_state_dict_with_metadata,
    state_dict_schema,
)

__all__ = (
    "ARCHITECTURE_ID",
    "LOCAL_STRESS_ARCHITECTURE_ID",
    "BoundaryConditionMode",
    "DirichletData",
    "GeometryEncoding",
    "GateStatus",
    "LiftingVerification",
    "OperatorConfig",
    "OperatorOutput",
    "PiGinotOperator",
    "S11GateCheck",
    "S11GateEvaluation",
    "S11GateMeasurements",
    "TensorSpec",
    "load_operator_state_dict",
    "load_operator_state_dict_with_metadata",
    "evaluate_s11_gate",
    "module_digest",
    "state_dict_schema",
)
