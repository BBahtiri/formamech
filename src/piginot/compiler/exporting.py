"""Atomic, load-verified candidate export from the bound final checkpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save

from piginot import __version__
from piginot.bundle import build_bundle, load_bundle
from piginot.bundle.inference import (
    ENVELOPE_PATH,
    LIMITATIONS_PATH,
    MODEL_PATH,
    MORPH_MAP_PATH,
    REFERENCE_MESH_PATH,
    VERIFICATION_PATH,
    WEIGHTS_PATH,
)
from piginot.bundle.limitations import REQUIRED_LIMITATION_PARAGRAPH
from piginot.bundle.manifest import (
    BUNDLE_FORMAT,
    BundleManifest,
    FileEntry,
    manifest_hash,
)
from piginot.bundle.serving import problem_energy_module_digest
from piginot.bundle.specs import ConditioningInput, ModelSpec, PortableProblem
from piginot.compiler.adapters import resolve_adapter, resolve_configured_adapter
from piginot.compiler.execution_records import digest, load_bound, save_bound
from piginot.compiler.planning import ExecutionPlan
from piginot.compiler.records import _verify_inputs
from piginot.contracts import (
    CanonicalMesh,
    CanonicalQuantity,
    Dimension,
    ProblemSpec,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.doe.envelope import ExecutableEnvelope
from piginot.morphing import MorphValidityCriteria
from piginot.morphing.membership import (
    DerivedBound,
    FamilyMembershipPredicate,
    PrimitiveBound,
)
from piginot.operators import EnergyFormOperator
from piginot.operators.architecture import LiftingVerification
from piginot.problem.verification import LiftingEvidence
from piginot.training._checkpoint_codec import _canonical_safetensors
from piginot.training.checkpoint import load_training_checkpoint
from piginot.training.identity import TrainingRunBinding


def problem_envelope(
    spec: ProblemSpec, mesh: CanonicalMesh, criteria: MorphValidityCriteria
) -> ExecutableEnvelope:
    """Bind the declared ranges; embedded ProblemSpec checks every constraint."""
    bounds = {
        parameter.name: PrimitiveBound(
            parameter.name,
            CanonicalQuantity(parameter.lo, parameter.unit),
            CanonicalQuantity(parameter.hi, parameter.unit),
        )
        for parameter in spec.parameters
    }
    geometry = tuple(bounds[p.name] for p in spec.parameters if p.role == "geometry")
    names = {bound.name for bound in geometry}
    constants = {
        constant.name: constant.value
        for constant in spec.constants
        if constant.unit.dimension is Dimension.DIMENSIONLESS
    }

    def substitute(tree: Sequence[Any]) -> tuple[Any, ...]:
        if tree[0] == "name" and tree[1] in constants:
            return ("number", constants[tree[1]])
        return tuple(
            substitute(node) if isinstance(node, (list, tuple)) else node
            for node in tree
        )

    def symbols(tree: tuple[Any, ...]) -> set[str]:
        if tree[0] == "name":
            return {tree[1]}
        return {
            name for node in tree if isinstance(node, tuple) for name in symbols(node)
        }

    constraints = tuple(
        DerivedBound(
            constraint.name,
            tree,
            CanonicalQuantity(constraint.lo, constraint.unit),
            CanonicalQuantity(constraint.hi, constraint.unit),
        )
        for constraint in spec.constraints
        if symbols(tree := substitute(constraint.expression.canonical())) <= names
    )
    membership = FamilyMembershipPredicate(
        spec.name,
        geometry,
        constraints,
        mesh.mesh_id,
        mesh.topology_signature,
        {name: len(facets) for name, facets in mesh.semantic_tags.items()},
        criteria,
        hash_spec(criteria, "content"),
    )
    return ExecutableEnvelope(
        f"{spec.name}-compiler-v1",
        spec.problem_hash,
        membership,
        hash_spec(membership, "content"),
        tuple(bounds[p.name] for p in spec.parameters if p.role == "material"),
        tuple(bounds[p.name] for p in spec.parameters if p.role == "load"),
    )


def _conditioner(
    plan: ExecutionPlan, normalization: Mapping[str, Any], binding: TrainingRunBinding
) -> tuple[ConditioningInput, ...]:
    names = plan.config.model.parameter_names
    if (
        digest(normalization) != binding.normalization_hash
        or tuple(normalization["parameter_names"]) != names
        or tuple(normalization["fit_case_ids"])
        != tuple(sorted(plan.config.source.fit_case_ids))
    ):
        raise ValueError("normalization differs from bound TRAIN-fit evidence")
    return tuple(
        ConditioningInput(name, name, mean, deviation)
        for name, mean, deviation in zip(
            names,
            normalization["mean"],
            normalization["standard_deviation"],
            strict=True,
        )
    )


def _checkpoint_model(
    plan: ExecutionPlan,
    path: Path,
    binding: TrainingRunBinding,
    model: EnergyFormOperator,
) -> tuple[EnergyFormOperator, str]:
    shape = asdict(plan.config.model)
    shape.pop("architecture_id")
    if (
        type(model) is not EnergyFormOperator
        or asdict(model.config) != shape
        or binding.problem_spec_hash != plan.problem.problem_hash
        or binding.architecture_id != model.architecture_id
        or binding.execution_profile != plan.config.execution.training
        or binding.public_manifest_hash != plan.manifest.content_hash
        or binding.seed != plan.config.training.seed
    ):
        raise ValueError("model or run binding differs from the execution plan")
    if (
        path
        != plan.output_path
        / "checkpoints"
        / f"step-{plan.config.training.optimizer_steps:09d}"
    ):
        raise ValueError("export requires the fixed final checkpoint")
    dtype = {"float32": torch.float32, "float64": torch.float64}[
        plan.config.execution.training.dtype
    ]
    with torch.random.fork_rng(devices=[]):
        restored = EnergyFormOperator(model.config).to(dtype)
        if (
            problem_energy_module_digest(restored, plan.problem.problem_hash)
            != binding.architecture_module_digest
        ):
            raise ValueError("checkpoint architecture differs from the bound operator")
        optimizer = torch.optim.Adam(restored.parameters(), foreach=False, fused=False)
        loaded = load_training_checkpoint(
            path, model=restored, optimizer=optimizer, binding=binding
        )
    if loaded.manifest.step != plan.config.training.optimizer_steps:
        raise ValueError("checkpoint has not reached the fixed final step")
    return restored, loaded.manifest.manifest_hash


def export_candidate(
    plan: ExecutionPlan,
    *,
    checkpoint_path: Path,
    run_binding: TrainingRunBinding,
    normalization: Mapping[str, Any],
    material_admission: Mapping[str, Any],
    lifting_verification: LiftingVerification,
    lifting_evidence: LiftingEvidence,
    model: EnergyFormOperator,
) -> dict[str, Any]:
    """Export no scientific claim, reusing only an exactly bound completed bundle."""
    _verify_inputs(plan)
    conditioning = _conditioner(plan, normalization, run_binding)
    evidence = material_admission.get("evidence")
    if (
        material_admission.get("status") != "PASS"
        or material_admission.get("all_eight_checks_complete") is not True
        or not isinstance(evidence, Mapping)
        or evidence.get("passed") is not True
        or evidence.get("problem_hash") != plan.problem.problem_hash
        or evidence.get("model") != plan.problem.material.model
    ):
        raise ValueError("export requires all eight matching material checks")
    problem = PortableProblem(
        plan.problem, digest(evidence), lifting_verification, lifting_evidence
    )
    restored, checkpoint_hash = _checkpoint_model(
        plan, checkpoint_path, run_binding, model
    )
    adapter = (
        resolve_adapter(plan.config.adapter.id)
        if plan.config.adapter.geometry_resource is None
        else resolve_configured_adapter(plan.config.adapter, plan.input_paths[1])
    )
    if adapter.descriptor != plan.adapter:
        raise ValueError("installed export adapter differs from the plan")
    reference, morph = adapter.export_geometry(plan.problem)  # type: ignore[attr-defined]
    envelope = problem_envelope(plan.problem, reference, adapter.validity_criteria())  # type: ignore[attr-defined]
    spec = ModelSpec.from_problem(
        problem=problem,
        architecture_id=restored.architecture_id,
        operator_config=asdict(restored.config),
        conditioning=conditioning,
        material_name=f"{plan.problem.name}-material",
        dtype=plan.config.execution.inference_dtype,
        query_chunk=512,
        units={"u": "mm", "P_served": "MPa", "sigma": "MPa"},
    )
    weights = _canonical_safetensors(
        save(
            {name: value.contiguous() for name, value in restored.state_dict().items()},
            metadata={
                "architecture_id": restored.architecture_id,
                "architecture_module_digest": run_binding.architecture_module_digest,
                "problem_hash": plan.problem.problem_hash,
            },
        )
    )
    identities = {
        "plan_sha256": plan.plan_sha256,
        "config_sha256": plan.config.config_sha256,
        "package_sha256": plan.package_sha256,
        "adapter_sha256": plan.adapter.implementation_sha256,
        "training_run_hash": run_binding.run_hash,
        "checkpoint_sha256": checkpoint_hash,
        "normalization_sha256": digest(normalization),
        **problem.identities,
    }
    binding = digest(identities)
    destination = plan.output_path / "candidate-bundle"
    state_path = plan.output_path / "export-state.json"
    state = load_bound(state_path, binding)
    if state is None:
        if destination.exists() or destination.is_symlink():
            raise ValueError("existing candidate lacks its bound export state")
        state = {"created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
        save_bound(state_path, binding, state)
    verification: dict[str, Any] = {
        "gate_id": "G28.candidate",
        "verdict": "NOT_VERIFIED",
        "claim_level": "NOT_VERIFIED",
        "material_admission": dict(evidence),
        "lifting_evidence": lifting_evidence.document(),
        "outstanding_work": [
            {
                "id": "scientific-acceptance",
                "status": "NOT_RUN",
                "reason": "candidate has no matching scientific release verification",
            }
        ],
    }
    if plan.config.adapter.geometry_resource is not None:
        from piginot.cad.adapter import FixedCadAdapter

        if isinstance(adapter, FixedCadAdapter) and isinstance(
            adapter.resource.provenance.get("mesh_policy"), Mapping
        ):
            resource = adapter.resource
            preparation = resource.provenance
            if not isinstance(preparation.get("mesh_evidence"), Mapping):
                raise ValueError("CAD mesh policy has no achieved mesh evidence")
            verification["geometry_preparation"] = strict_json_loads(
                canonical_json_bytes(
                    {
                        "geometry_resource_sha256": resource.sha256,
                        "source_sha256": resource.source_sha256,
                        "mesh_policy": preparation["mesh_policy"],
                        "mesh_evidence": preparation["mesh_evidence"],
                        **(
                            {"adaptation": preparation["adaptation"]}
                            if "adaptation" in preparation
                            else {}
                        ),
                    }
                )
            )
    limitation_items = (
        {
            "id": "compiler-candidate",
            "statement": "New weights have no inherited calibration or certified bound.",
        },
    )
    expected_files = {
        REFERENCE_MESH_PATH: canonical_json_bytes(reference),
        MORPH_MAP_PATH: canonical_json_bytes(morph),
        ENVELOPE_PATH: envelope.to_bytes(),
        MODEL_PATH: spec.to_bytes(),
        WEIGHTS_PATH: weights,
        VERIFICATION_PATH: canonical_json_bytes(verification),
        LIMITATIONS_PATH: canonical_json_bytes(
            {
                "paragraph": REQUIRED_LIMITATION_PARAGRAPH,
                "items": list(limitation_items),
            }
        ),
    }
    expected_manifest = BundleManifest(
        BUNDLE_FORMAT,
        plan.config.name,
        state["created_utc"],
        "NOT_VERIFIED",
        {**identities, "piginot_version": __version__},
        {
            name: FileEntry(hashlib.sha256(payload).hexdigest(), len(payload))
            for name, payload in expected_files.items()
        },
    )
    if destination.is_symlink():
        raise ValueError("candidate bundle cannot be a symlink")
    if not destination.exists():
        build_bundle(
            destination,
            name=plan.config.name,
            created_utc=state["created_utc"],
            claim_level="NOT_VERIFIED",
            reference_mesh=reference,
            morph_map=morph,
            envelope=envelope,
            model=spec,
            weights=weights,
            verification=verification,
            limitation_items=limitation_items,
            identities=identities,
        )
    bundle = load_bundle(destination)
    if bundle.manifest.manifest_sha256 != manifest_hash(expected_manifest):
        raise ValueError("existing candidate differs from the bound export")
    return {
        "status": "PASS",
        "bundle_path": str(destination),
        "sha256": bundle.manifest.manifest_sha256,
        "claim_level": "NOT_VERIFIED",
    }


__all__ = ("export_candidate", "problem_envelope")
