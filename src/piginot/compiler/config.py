"""Strict data-only contract for a campaign's reproducible configuration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from piginot.compiler._validation import (
    CompilerConfigError,
    array,
    object_fields,
    positive,
    primitive,
    text,
)
from piginot.compiler.config_types import (
    AdapterRef,
    EnergyObjectiveConfig,
    ExecutionConfig,
    GeometryResourceRef,
    ModelConfig,
    PublicManifestSource,
    ReferencePolicy,
    SelectionPolicy,
    TrainingConfig,
)
from piginot.contracts import ExecutionProfile, canonical_json_bytes, strict_json_loads

SCHEMA = "piginot.campaign-config-v1"
_ROOT_FIELDS = (
    "schema",
    "name",
    "adapter",
    "source",
    "model",
    "normalization",
    "training",
    "objective",
    "reference",
    "execution",
    "selection",
)


def _profile(value: object) -> ExecutionProfile:
    fields = object_fields(
        value,
        "execution profile",
        (
            "python_version",
            "library_versions",
            "blas_threads",
            "device",
            "dtype",
            "rng_algorithm",
            "seed",
            "platform_fingerprint",
        ),
        ("container_digest",),
    )
    versions = fields["library_versions"]
    if not isinstance(versions, dict) or not versions:
        raise CompilerConfigError("type", "library_versions must be a nonempty object")
    for name, version in versions.items():
        text(name, "library name")
        text(version, "library version")
    fingerprint = object_fields(
        fields["platform_fingerprint"],
        "platform_fingerprint",
        ("os", "machine", "cpu_capability"),
    )
    for name, value in fingerprint.items():
        text(value, f"platform_fingerprint.{name}")
    for name in ("python_version", "device", "dtype", "rng_algorithm"):
        text(fields[name], f"execution.{name}")
    try:
        return ExecutionProfile(
            **{
                **fields,
                "library_versions": dict(versions),
                "platform_fingerprint": fingerprint,
            }
        )
    except (TypeError, ValueError) as error:
        raise CompilerConfigError("execution", str(error)) from error


def _source(value: object) -> PublicManifestSource:
    fields = object_fields(
        value,
        "source",
        (
            "kind",
            "path",
            "sha256",
            "fit_case_ids",
            "screen_case_ids",
            "development_case_ids",
        ),
    )
    return PublicManifestSource(
        **{
            **fields,
            **{
                name: array(fields[name], f"source.{name}")
                for name in ("fit_case_ids", "screen_case_ids", "development_case_ids")
            },
        }
    )


def _model(value: object) -> ModelConfig:
    fields = object_fields(
        value,
        "model",
        ("architecture_id", "parameter_names"),
        (
            "hidden_dim",
            "latent_token_count",
            "parameter_hidden_dim",
            "num_attention_heads",
            "decoder_layers",
            "positional_frequencies",
            "output_init_std",
        ),
    )
    if "output_init_std" in fields:
        fields = {
            **fields,
            "output_init_std": positive(
                fields["output_init_std"], "model.output_init_std"
            ),
        }
    return ModelConfig(
        **{
            **fields,
            "parameter_names": array(
                fields["parameter_names"], "model.parameter_names"
            ),
        }
    )


def _training(value: object) -> TrainingConfig:
    fields = object_fields(
        value,
        "training",
        (
            "optimizer_steps",
            "case_batch_size",
            "checkpoint_interval",
            "learning_rate",
            "final_learning_rate",
            "constant_steps",
            "seed",
        ),
    )
    return TrainingConfig(
        **{
            **fields,
            **{
                name: positive(fields[name], f"training.{name}")
                for name in ("learning_rate", "final_learning_rate")
            },
        }
    )


def _objective(value: object) -> EnergyObjectiveConfig:
    fields = object_fields(
        value,
        "objective",
        (
            "points_per_cell",
            "barrier_weight",
            "barrier_threshold",
            "admissibility_floor",
        ),
        (
            "cross_mesh_weight",
            "local_physics_weight",
            "quadrature_degree",
            "use_deterministic_quadrature",
        ),
    )
    return EnergyObjectiveConfig(
        **{
            **fields,
            **{
                name: positive(fields[name], f"objective.{name}")
                for name in (
                    "barrier_weight",
                    "barrier_threshold",
                    "admissibility_floor",
                )
            },
        }
    )


def _reference(value: object) -> ReferencePolicy:
    fields = object_fields(
        value, "reference", ("initialization", "refinement_levels", "workers")
    )
    return ReferencePolicy(
        **{
            **fields,
            "refinement_levels": array(
                fields["refinement_levels"], "reference.refinement_levels"
            ),
        }
    )


def _execution(value: object) -> ExecutionConfig:
    fields = object_fields(
        value, "execution", ("training", "reference", "inference_dtype")
    )
    return ExecutionConfig(
        _profile(fields["training"]),
        _profile(fields["reference"]),
        fields["inference_dtype"],
    )


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    schema: str
    name: str
    adapter: AdapterRef
    source: PublicManifestSource
    model: ModelConfig
    normalization: str
    training: TrainingConfig
    objective: EnergyObjectiveConfig
    reference: ReferencePolicy
    execution: ExecutionConfig
    selection: SelectionPolicy

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise CompilerConfigError("schema", f"schema must be {SCHEMA}")
        text(self.name, "name", identifier=True)
        if self.normalization != "train-fit":
            raise CompilerConfigError(
                "unsupported", "normalization must use train-fit only"
            )
        expected = (
            ("adapter", AdapterRef),
            ("source", PublicManifestSource),
            ("model", ModelConfig),
            ("training", TrainingConfig),
            ("objective", EnergyObjectiveConfig),
            ("reference", ReferencePolicy),
            ("execution", ExecutionConfig),
            ("selection", SelectionPolicy),
        )
        for name, kind in expected:
            if type(getattr(self, name)) is not kind:
                raise CompilerConfigError(
                    "type", f"{name} must be an exact {kind.__name__}"
                )
        if self.execution.training.seed != self.training.seed:
            raise CompilerConfigError(
                "execution", "training profile seed must match training seed"
            )
        if bool(self.source.development_case_ids) != self.selection.development:
            raise CompilerConfigError(
                "selection", "development IDs require an explicit matching request"
            )

    @classmethod
    def from_document(cls, document: object) -> CampaignConfig:
        fields = object_fields(document, "campaign config", _ROOT_FIELDS)
        adapter = object_fields(
            fields["adapter"],
            "adapter",
            ("id", "implementation_sha256"),
            ("geometry_resource",),
        )
        if "geometry_resource" in adapter:
            adapter = {
                **adapter,
                "geometry_resource": GeometryResourceRef(
                    **object_fields(
                        adapter["geometry_resource"],
                        "geometry resource",
                        ("path", "sha256"),
                    )
                ),
            }
        selection = object_fields(
            fields["selection"], "selection", ("checkpoint_rule", "development")
        )
        return cls(
            schema=fields["schema"],
            name=fields["name"],
            adapter=AdapterRef(**adapter),
            source=_source(fields["source"]),
            model=_model(fields["model"]),
            normalization=fields["normalization"],
            training=_training(fields["training"]),
            objective=_objective(fields["objective"]),
            reference=_reference(fields["reference"]),
            execution=_execution(fields["execution"]),
            selection=SelectionPolicy(**selection),
        )

    @classmethod
    def load(cls, path: str | Path) -> CampaignConfig:
        """Read only this configuration; referenced sources remain unopened."""

        try:
            document = strict_json_loads(Path(path).read_bytes())
        except ValueError as error:
            raise CompilerConfigError(
                "json", f"invalid campaign JSON: {error}"
            ) from error
        return cls.from_document(document)

    def document(self) -> dict[str, Any]:
        document = cast(dict[str, Any], primitive(self))
        return {
            **document,
            "adapter": {
                name: value
                for name, value in document["adapter"].items()
                if name != "geometry_resource" or value is not None
            },
            "objective": {
                name: value
                for name, value in document["objective"].items()
                if value is not None
            },
        }

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


__all__ = (
    "AdapterRef",
    "CampaignConfig",
    "CompilerConfigError",
    "EnergyObjectiveConfig",
    "GeometryResourceRef",
    "ExecutionConfig",
    "ModelConfig",
    "PublicManifestSource",
    "ReferencePolicy",
    "SelectionPolicy",
    "TrainingConfig",
)
