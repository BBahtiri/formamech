"""S7 primary finite-element solver surface."""

from piginot.fem.backend import ScikitFemBackend
from piginot.fem.body_forces import BodyForceProvider, ConstantBodyForce
from piginot.fem.contact import (
    ContactVerificationReport,
    RigidIndenterSpec,
    assemble_contact_terms,
)
from piginot.fem.field_providers import (
    DirichletFieldProvider,
    TractionFieldProvider,
)
from piginot.fem.models import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolveFailure,
    FemSolverConfig,
)
from piginot.fem.protocol import (
    BackendCapabilities,
    DiscreteLinearization,
    FemBackend,
    IndependentSolverAdapter,
    SolverBackend,
)
from piginot.fem.registry import (
    FEM_BACKEND_GROUP,
    available_backends,
    register_backend,
    resolve_backend,
    unregister_backend,
)
from piginot.fem.torch_backend import (
    TORCH_PHYSICS_MODES,
    TorchFemBackend,
    TorchFemLinearization,
)

__all__ = (
    "FEM_BACKEND_GROUP",
    "TORCH_PHYSICS_MODES",
    "BackendCapabilities",
    "BodyForceProvider",
    "ConstantBodyForce",
    "ContactVerificationReport",
    "DirichletFieldProvider",
    "DiscreteLinearization",
    "FemAttempt",
    "FemBackend",
    "FemConvergenceEntry",
    "FemFailureCode",
    "FemRequest",
    "FemResult",
    "FemSolveFailure",
    "FemSolverConfig",
    "IndependentSolverAdapter",
    "RigidIndenterSpec",
    "ScikitFemBackend",
    "SolverBackend",
    "TorchFemBackend",
    "TorchFemLinearization",
    "TractionFieldProvider",
    "assemble_contact_terms",
    "available_backends",
    "register_backend",
    "resolve_backend",
    "unregister_backend",
)
