"""Immutable, metadata-only plans for the installed campaign compiler."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import formamech
import piginot
import piginot_campaigns
from piginot.compiler.adapters import (
    AdapterDescriptor,
    GeometryIdentity,
    resolve_adapter,
    resolve_configured_adapter,
)
from piginot.compiler.config import CampaignConfig, CompilerConfigError
from piginot.compiler.provenance import implementation_package_sha256
from piginot.compiler.public_source import select_public_cases
from piginot.contracts import ProblemSpec, canonical_json_bytes, strict_json_loads
from piginot.doe import PublishedCaseManifest

STAGES = (
    "plan",
    "eligibility",
    "material",
    "geometry-lifting",
    "train-preparation",
    "train",
    "train-screen",
    "freeze",
    "public-development",
    "export-report",
)


class PlanError(ValueError):
    """The requested immutable plan cannot safely be created."""

    def __init__(self, rule: str, message: str) -> None:
        self.rule = rule
        super().__init__(f"[{rule}] {message}")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation() -> tuple[str, str]:
    package = Path(piginot.__file__).resolve().parent
    return piginot.__version__, implementation_package_sha256(package)


def _capabilities(
    spec: ProblemSpec, config: CampaignConfig, descriptor: AdapterDescriptor
) -> None:
    if (
        descriptor.adapter_id != config.adapter.id
        or descriptor.implementation_sha256 != config.adapter.implementation_sha256
    ):
        raise PlanError("adapter", "adapter identity or implementation digest differs")
    geometry = GeometryIdentity(
        spec.geometry.reference_mesh_id, spec.geometry.morph_map_sha256
    )
    if geometry not in descriptor.geometries:
        raise PlanError("adapter", "unsupported reference mesh and morph identity")
    material = spec.material.model
    if spec.material.energy is not None:
        material = (
            "formula-fibre" if spec.material.energy.fibres else "formula-isotropic"
        )
    capabilities = (
        (material, descriptor.materials),
        (spec.lifting.kind, descriptor.liftings),
        (config.model.architecture_id, descriptor.architectures),
        (config.reference.initialization, descriptor.reference_initializations),
    )
    if any(value not in supported for value, supported in capabilities):
        raise PlanError(
            "capability",
            "requested material, lifting, model or reference capability is unavailable",
        )
    if config.reference.initialization != "cold" and spec.material.model != "hgo":
        raise PlanError(
            "reference", "the declared full-load initializer is supported for HGO only"
        )
    if any(condition.kind == "neumann" for condition in spec.boundary_conditions):
        raise PlanError(
            "capability",
            "Neumann conditions are unsupported; use traction_free for zero natural traction",
        )
    names = set(config.model.parameter_names)
    required = {
        parameter.name for parameter in spec.parameters if parameter.lo != parameter.hi
    }
    if not required <= names <= {parameter.name for parameter in spec.parameters}:
        raise PlanError(
            "conditioning",
            "model conditioning must include every varying ProblemSpec parameter",
        )
    source = config.source
    if (
        len(source.fit_case_ids) + len(source.screen_case_ids)
        != spec.split.counts["train"]
    ):
        raise PlanError("split", "TRAIN case counts differ from the ProblemSpec budget")
    if (
        config.selection.development
        and len(source.development_case_ids) != spec.split.counts["development"]
    ):
        raise PlanError(
            "split", "DEVELOPMENT case count differs from the ProblemSpec budget"
        )


def check_output_path(output: Path, protected: tuple[Path, ...]) -> Path:
    if output.is_symlink():
        raise PlanError("output", "output cannot be a symlink")
    resolved = output.resolve()
    if any(path == resolved or path.is_relative_to(resolved) for path in protected):
        raise PlanError("output", "output overlaps a source or evidence input")
    package_roots = (
        Path(formamech.__file__).resolve().parent,
        Path(piginot.__file__).resolve().parent,
        Path(piginot_campaigns.__file__).resolve().parent,
    )
    if any(resolved.is_relative_to(package) for package in package_roots):
        raise PlanError("output", "output cannot be inside an installed package")
    if resolved.exists() and not resolved.is_dir():
        raise PlanError("output", "output must be a directory")
    return resolved


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    problem: ProblemSpec
    config: CampaignConfig
    adapter: AdapterDescriptor
    manifest: PublishedCaseManifest
    source_path: Path
    output_path: Path
    input_paths: tuple[Path, ...]
    input_sha256: tuple[tuple[str, str], ...]
    package_version: str
    package_sha256: str

    def _payload(self) -> dict[str, Any]:
        source = self.config.source
        return {
            "schema": "piginot.campaign-plan-v1",
            "status": "PLANNED",
            "scientific_status": "NOT_EVALUATED",
            "claim_level": "NOT_VERIFIED",
            "problem_hash": self.problem.problem_hash,
            "problem": self.problem.document(),
            "config_sha256": self.config.config_sha256,
            "config": self.config.document(),
            "adapter": self.adapter.document(),
            "implementation": {
                "version": self.package_version,
                "package_sha256": self.package_sha256,
            },
            "input_paths": [str(path) for path in self.input_paths],
            "input_sha256": dict(self.input_sha256),
            "output_path": str(self.output_path),
            "source": {
                "path": str(self.source_path),
                "manifest_content_hash": self.manifest.content_hash,
                "fit_case_ids": list(source.fit_case_ids),
                "screen_case_ids": list(source.screen_case_ids),
                "development_case_ids": list(source.development_case_ids),
            },
            "execution_profile_status": "DECLARED_NOT_NUMERICALLY_PROBED",
            "stages": {
                name: "PASS"
                if name in ("plan", "eligibility")
                else "NOT_REQUESTED"
                if name == "public-development"
                and not self.config.selection.development
                else "PENDING"
                for name in STAGES
            },
        }

    @property
    def plan_sha256(self) -> str:
        return _sha(canonical_json_bytes(self._payload()))

    def document(self) -> dict[str, Any]:
        return {**self._payload(), "plan_sha256": self.plan_sha256}


def build_plan(
    spec_path: Path,
    config_path: Path,
    output_path: Path,
    *,
    descriptor: AdapterDescriptor | None = None,
) -> ExecutionPlan:
    """Read declared public inputs; never create output or invoke numerics."""

    spec_path, config_path = spec_path.resolve(), config_path.resolve()
    spec_bytes, config_bytes = spec_path.read_bytes(), config_path.read_bytes()
    try:
        spec = ProblemSpec.from_bytes(spec_bytes)
        config = CampaignConfig.from_document(strict_json_loads(config_bytes))
        geometry_inputs: tuple[tuple[str, Path], ...] = ()
        geometry_hashes: tuple[tuple[str, str], ...] = ()
        if config.adapter.geometry_resource is not None:
            bound = cast(Any, resolve_configured_adapter(config.adapter, config_path))
            if descriptor is not None and descriptor != bound.descriptor:
                raise PlanError(
                    "adapter",
                    "bound geometry adapter differs from the supplied descriptor",
                )
            descriptor = bound.descriptor
            geometry_inputs = cast(tuple[tuple[str, Path], ...], bound.geometry_inputs)
            geometry_hashes = cast(
                tuple[tuple[str, str], ...], bound.geometry_input_sha256
            )
            labels = tuple(label for label, _ in geometry_inputs)
            if (
                labels != tuple(label for label, _ in geometry_hashes)
                or len(labels) != len(set(labels))
                or set(labels).intersection(("problem", "config", "manifest"))
            ):
                raise PlanError(
                    "adapter", "geometry input bindings have an incoherent roster"
                )
            for (_, path), (label, expected) in zip(
                geometry_inputs, geometry_hashes, strict=True
            ):
                if _sha(path.read_bytes()) != expected:
                    raise PlanError(
                        "input", f"{label} changed during geometry validation"
                    )
        elif descriptor is None:
            descriptor = resolve_adapter(config.adapter.id).descriptor
        _capabilities(spec, config, descriptor)
        source_path = Path(config.source.path)
        if not source_path.is_absolute():
            source_path = config_path.parent / source_path
        source_path = source_path.resolve()
        protected = (
            spec_path,
            config_path,
            source_path,
            *(path for _, path in geometry_inputs),
        )
        output = check_output_path(output_path, protected)
        payload = source_path.read_bytes()
        if _sha(payload) != config.source.sha256:
            raise PlanError("source", "public manifest digest differs from its binding")
        manifest = PublishedCaseManifest.from_bytes(payload)
        select_public_cases(spec, config, manifest)
    except PlanError:
        raise
    except ValueError as error:
        rule = error.rule if isinstance(error, CompilerConfigError) else "input"
        raise PlanError(rule, str(error)) from error
    version, package_sha256 = _implementation()
    return ExecutionPlan(
        problem=spec,
        config=config,
        adapter=descriptor,
        manifest=manifest,
        source_path=source_path,
        output_path=output,
        input_paths=protected,
        input_sha256=(
            ("problem", _sha(spec_bytes)),
            ("config", _sha(config_bytes)),
            ("manifest", _sha(payload)),
            *geometry_hashes,
        ),
        package_version=version,
        package_sha256=package_sha256,
    )
