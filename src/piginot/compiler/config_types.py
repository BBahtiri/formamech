"""Immutable campaign configuration values with no numerical imports."""

from __future__ import annotations

from dataclasses import dataclass

from piginot.compiler._validation import (
    CompilerConfigError,
    digest,
    integer,
    names,
    positive,
    text,
)
from piginot.contracts import ExecutionProfile

NUMERICAL_LIBRARIES = frozenset(
    {
        "numpy",
        "torch",
        "scipy",
        "scikit-fem",
        "sympy",
        "safetensors",
        "threadpoolctl",
    }
)


@dataclass(frozen=True, slots=True)
class GeometryResourceRef:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        text(self.path, "adapter.geometry_resource.path")
        digest(self.sha256, "adapter.geometry_resource.sha256")


@dataclass(frozen=True, slots=True)
class AdapterRef:
    id: str
    implementation_sha256: str
    geometry_resource: GeometryResourceRef | None = None

    def __post_init__(self) -> None:
        text(self.id, "adapter.id", identifier=True)
        digest(self.implementation_sha256, "adapter.implementation_sha256")
        if (
            self.geometry_resource is not None
            and type(self.geometry_resource) is not GeometryResourceRef
        ):
            raise CompilerConfigError(
                "type", "geometry_resource must be a GeometryResourceRef"
            )


@dataclass(frozen=True, slots=True)
class PublicManifestSource:
    kind: str
    path: str
    sha256: str
    fit_case_ids: tuple[str, ...]
    screen_case_ids: tuple[str, ...]
    development_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind != "public-manifest":
            raise CompilerConfigError(
                "unsupported", "source.kind must be public-manifest"
            )
        text(self.path, "source.path")
        digest(self.sha256, "source.sha256")
        groups = (
            names(self.fit_case_ids, "source.fit_case_ids"),
            names(self.screen_case_ids, "source.screen_case_ids"),
            names(
                self.development_case_ids, "source.development_case_ids", nonempty=False
            ),
        )
        combined = tuple(item for group in groups for item in group)
        if len(combined) != len(set(combined)):
            raise CompilerConfigError(
                "source", "fit, screen and development IDs must be disjoint"
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture_id: str
    parameter_names: tuple[str, ...]
    hidden_dim: int = 64
    latent_token_count: int = 32
    parameter_hidden_dim: int = 96
    num_attention_heads: int = 4
    decoder_layers: int = 6
    positional_frequencies: int = 6
    output_init_std: float = 1e-5

    def __post_init__(self) -> None:
        text(self.architecture_id, "model.architecture_id", identifier=True)
        names(self.parameter_names, "model.parameter_names")
        for label in (
            "hidden_dim",
            "latent_token_count",
            "parameter_hidden_dim",
            "num_attention_heads",
            "decoder_layers",
            "positional_frequencies",
        ):
            integer(getattr(self, label), f"model.{label}")
        if self.hidden_dim % self.num_attention_heads:
            raise CompilerConfigError(
                "value", "model.hidden_dim must be divisible by num_attention_heads"
            )
        positive(self.output_init_std, "model.output_init_std")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    optimizer_steps: int
    case_batch_size: int
    checkpoint_interval: int
    learning_rate: float
    final_learning_rate: float
    constant_steps: int
    seed: int

    def __post_init__(self) -> None:
        for label in ("optimizer_steps", "case_batch_size", "checkpoint_interval"):
            integer(getattr(self, label), f"training.{label}")
        integer(self.seed, "training.seed", minimum=0)
        integer(self.constant_steps, "training.constant_steps", minimum=0)
        if self.constant_steps > self.optimizer_steps:
            raise CompilerConfigError(
                "value", "constant_steps cannot exceed optimizer_steps"
            )
        positive(self.learning_rate, "training.learning_rate")
        positive(self.final_learning_rate, "training.final_learning_rate")
        if self.final_learning_rate > self.learning_rate:
            raise CompilerConfigError(
                "value", "final_learning_rate cannot exceed learning_rate"
            )


@dataclass(frozen=True, slots=True)
class EnergyObjectiveConfig:
    points_per_cell: int
    barrier_weight: float
    barrier_threshold: float
    admissibility_floor: float

    def __post_init__(self) -> None:
        integer(self.points_per_cell, "objective.points_per_cell")
        for label in ("barrier_weight", "barrier_threshold", "admissibility_floor"):
            positive(getattr(self, label), f"objective.{label}")
        if self.admissibility_floor >= self.barrier_threshold:
            raise CompilerConfigError(
                "value", "admissibility_floor must be below barrier_threshold"
            )


@dataclass(frozen=True, slots=True)
class ReferencePolicy:
    initialization: str
    refinement_levels: tuple[int, ...]
    workers: int

    def __post_init__(self) -> None:
        if self.initialization not in ("cold", "neo-hookean-full-load-hgo"):
            raise CompilerConfigError(
                "unsupported", "unsupported reference initialization"
            )
        levels = self.refinement_levels
        if type(levels) is not tuple or not levels:
            raise CompilerConfigError(
                "type", "refinement_levels must be a nonempty tuple"
            )
        for level in levels:
            integer(level, "reference.refinement_levels")
        if levels[0] != 1 or levels != tuple(sorted(set(levels))):
            raise CompilerConfigError(
                "value", "refinement levels must start at 1 and increase strictly"
            )
        if not set(levels) <= {1, 2, 4}:
            raise CompilerConfigError(
                "unsupported", "supported reference refinement levels are 1, 2 and 4"
            )
        if type(self.workers) is not int or self.workers not in (1, 2):
            raise CompilerConfigError("value", "reference.workers must be 1 or 2")


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    checkpoint_rule: str
    development: bool

    def __post_init__(self) -> None:
        if self.checkpoint_rule != "final":
            raise CompilerConfigError(
                "unsupported", "only the final checkpoint rule is supported"
            )
        if type(self.development) is not bool:
            raise CompilerConfigError("type", "selection.development must be a boolean")


def _profile_metadata(profile: ExecutionProfile) -> None:
    if type(profile) is not ExecutionProfile:
        raise CompilerConfigError(
            "type", "execution profiles must be exact ExecutionProfile values"
        )
    integer(profile.seed, "execution.seed", minimum=0)
    for label in ("python_version", "device", "dtype", "rng_algorithm"):
        text(getattr(profile, label), f"execution.{label}")
    if not profile.library_versions:
        raise CompilerConfigError("value", "execution requires library versions")
    for name, version in profile.library_versions.items():
        text(name, "library name")
        text(version, "library version")
    if not NUMERICAL_LIBRARIES <= profile.library_versions.keys():
        raise CompilerConfigError(
            "missing-key",
            "execution requires every numerical library version: "
            + ", ".join(sorted(NUMERICAL_LIBRARIES)),
        )
    if profile.device != "cpu" or profile.blas_threads != 1:
        raise CompilerConfigError(
            "unsupported", "execution requires CPU and one BLAS thread"
        )
    if profile.platform_fingerprint is None:
        raise CompilerConfigError(
            "missing-key", "execution requires a platform fingerprint"
        )
    for name, value in profile.platform_fingerprint.items():
        text(value, f"platform_fingerprint.{name}")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    training: ExecutionProfile
    reference: ExecutionProfile
    inference_dtype: str

    def __post_init__(self) -> None:
        for profile in (self.training, self.reference):
            _profile_metadata(profile)
        if self.training.dtype not in ("float32", "float64"):
            raise CompilerConfigError(
                "unsupported", "training dtype must be float32 or float64"
            )
        if self.reference.dtype != "float64" or self.inference_dtype != "float64":
            raise CompilerConfigError(
                "unsupported", "reference and inference require float64"
            )
        for label in (
            "python_version",
            "library_versions",
            "platform_fingerprint",
            "container_digest",
        ):
            if getattr(self.training, label) != getattr(self.reference, label):
                raise CompilerConfigError(
                    "execution", f"training/reference {label} identities differ"
                )
