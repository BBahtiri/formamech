"""Prospective authorization and initializer checks for the zero-FEM CAD pilot."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, cast

from safetensors.torch import load_file

import piginot
import piginot_campaigns
from piginot.compiler.adapters import resolve_configured_adapter
from piginot.compiler.boundary_objective import BoundaryPilotObjective
from piginot.compiler.energy_training import EnergyTraining
from piginot.compiler.planning import build_plan
from piginot.compiler.provenance import require_clean_implementation_identity
from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.training.checkpoint import model_state_hash

SCHEMA = "piginot.s29-boundary-pilot-v1"


def sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"pilot input must be a regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document(path: Path) -> dict[str, Any]:
    sha(path)
    value = strict_json_loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("pilot metadata must be an object")
    return cast(dict[str, Any], value)


def validate_pilot_terms(proposal: dict[str, Any]) -> None:
    """Reject expansion beyond the prospective two-arm owner-approved scope."""
    try:
        budgets = proposal["budgets"]
        caps = {
            "new_fem_solves": 0,
            "maximum_optimizer_iterations_total": 100,
            "maximum_training_objective_gradient_evaluations_total": 200,
            "maximum_validation_energy_evaluations_total": 20,
            "maximum_wall_seconds_total": 43200,
        }
        if any(
            type(budgets[name]) is not int or budgets[name] != limit
            for name, limit in caps.items()
        ):
            raise ValueError("pilot budgets differ from the approved allowance")
        if (
            budgets["automatic_resume"] is not False
            or budgets["automatic_retries"] is not False
        ):
            raise ValueError("the pilot cannot resume or retry automatically")
        ids = proposal["pilot_cases"]["ids"]
        if (
            proposal["pilot_cases"]["role"] != "TRAIN-FIT"
            or len(ids) != 4
            or len(set(ids)) != 4
        ):
            raise ValueError("the pilot requires exactly four distinct FIT cases")
        access = proposal["data_access"]
        if access["fit_only_for_training_and_selection"] is not True or any(
            access[name] is not False
            for name in (
                "screen_for_selection",
                "development",
                "calibration",
                "holdout",
            )
        ):
            raise ValueError("only FIT data can train and select this pilot")
        if [(arm["id"], arm["traction_weight"]) for arm in proposal["arms"]] != [
            ("A", 0),
            ("B", 1),
        ] or any(
            type(arm["traction_weight"]) not in (int, float) for arm in proposal["arms"]
        ):
            raise ValueError(
                "exactly the declared energy and boundary arms are allowed"
            )
        optimizer = proposal["optimizer"]
        expected = {
            "kind": "L-BFGS",
            "dtype": "float64",
            "device": "cpu",
            "blas_threads": 1,
            "history_size": 20,
            "learning_rate": 1,
            "line_search": "strong_wolfe",
            "inner_iterations_per_call": 1,
            "maximum_accepted_iterations_per_arm": 50,
            "maximum_training_objective_gradient_evaluations_per_arm": 100,
            "maximum_validation_energy_evaluations_per_arm": 10,
            "gradient_point_chunk": 2048,
            "tolerance_gradient": 1e-8,
            "tolerance_change": 1e-12,
        }
        if any(optimizer[name] != value for name, value in expected.items()):
            raise ValueError("optimizer settings differ from the approved pilot")
        if any(
            proposal["preservation"][name] is not False
            for name in (
                "modify_original_run",
                "resume_original_run",
                "modify_old_results",
                "reuse_old_optimizer_state",
            )
        ):
            raise ValueError("the original run and Adam state must be preserved")
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete or malformed pilot proposal") from error


def _member(root: Path, entry: dict[str, Any]) -> Path:
    relative = entry.get("path")
    if not isinstance(relative, str):
        raise ValueError("a pilot member path must be a string")
    path = root / relative
    if (
        not path.resolve().is_relative_to(root.resolve())
        or sha(path) != entry["sha256"]
    ):
        raise ValueError("a frozen pilot member changed or escaped its directory")
    return path


def verify_initializer(initializer: dict[str, Any], proposal: dict[str, Any]) -> None:
    if (
        initializer.get("schema") != "piginot.checked-initializer-v1"
        or initializer.get("status") != "CHECKED"
        or initializer.get("new_fem_solves") != 0
        or initializer.get("new_optimizer_steps") != 0
    ):
        raise ValueError("a checked, unmodified initializer is required")
    if (
        initializer["training"] != proposal["baseline"]["training_record"]
        or initializer["parent_protocol_sha256"]
        != proposal["baseline"]["protocol_sha256"]
    ):
        raise ValueError("initializer lineage differs from the accepted proposal")
    for name, expected in initializer["consumed_files_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"initializer source changed: {name}")
    for label in ("weights", "build_identity"):
        if sha(Path(initializer[f"{label}_path"])) != initializer[f"{label}_sha256"]:
            raise ValueError("exported initializer integrity check failed")
    if (
        sha(Path(initializer["parent_protocol_path"]))
        != initializer["parent_protocol_sha256"]
    ):
        raise ValueError("parent protocol changed")
    identity = document(Path(initializer["build_identity_path"]))
    parent = initializer["parent_implementation"]
    manifest_sha = hashlib.sha256(canonical_json_bytes(identity["files"])).hexdigest()
    if (
        identity["source_revision"] != parent["source_revision"]
        or identity["manifest_sha256"] != parent["manifest_sha256"]
        or manifest_sha != identity["manifest_sha256"]
        or identity["dirty"] is not False
    ):
        raise ValueError("initializer build identity differs")
    roots = {
        "piginot": Path(piginot.__file__).resolve().parent,
        "piginot_campaigns": Path(piginot_campaigns.__file__).resolve().parent,
    }
    for name, expected in identity["files"].items():
        prefix, relative = name.split("/", 1)
        if prefix not in roots:
            raise ValueError("unknown initializer implementation root")
        path = roots[prefix] / relative
        if not path.resolve().is_relative_to(roots[prefix]) or sha(path) != expected:
            raise ValueError(
                f"legacy initializer/physics implementation changed: {name}"
            )


def load_inputs(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = document(path)
    if (
        protocol.get("schema") != SCHEMA
        or protocol.get("status") != "FROZEN"
        or protocol.get("numerical_execution_authorized") is not True
    ):
        raise ValueError("execution requires a frozen owner-authorized pilot")
    root = path.parent
    proposal_path = _member(root, protocol["proposal"])
    proposal = document(proposal_path)
    validate_pilot_terms(proposal)
    approval = document(_member(root, protocol["authorization"]))
    if (
        approval.get("approved") is not True
        or approval.get("accepted_proposal_sha256") != sha(proposal_path)
        or approval.get("budgets") != proposal["budgets"]
    ):
        raise ValueError(
            "the owner decision does not authorize these exact pilot terms"
        )
    identity = require_clean_implementation_identity()
    observed = {
        "source_commit": identity.source_revision,
        "package_sha256": identity.package_sha256,
        "manifest_sha256": identity.manifest_sha256,
    }
    if observed != protocol["implementation"] or identity.origin != "packaged":
        raise ValueError("the installed pilot implementation differs from its freeze")
    for relative, expected in protocol["input_files_sha256"].items():
        _member(root, {"path": relative, "sha256": expected})
    for name, expected in protocol.get("external_files_sha256", {}).items():
        if sha(Path(name)) != expected:
            raise ValueError("an external frozen pilot input changed")
    initializer = document(_member(root, protocol["initializer"]))
    verify_initializer(initializer, proposal)
    return protocol, proposal, initializer


def prepare_objective(
    protocol: dict[str, Any],
    proposal: dict[str, Any],
    initializer: dict[str, Any],
    output: Path,
) -> tuple[BoundaryPilotObjective, dict[str, Any]]:
    parent = Path(initializer["parent_protocol_path"]).parent
    spec, config = parent / "inputs/problem.json", parent / "inputs/config.json"
    plan = build_plan(spec, config, output / "initializer-context")
    adapter = resolve_configured_adapter(plan.config.adapter, config)
    context = EnergyTraining(plan, adapter, protocol["implementation"]["source_commit"])
    prepared = context.prepare()
    for name in ("normalization", "mesh_sha256", "dataset_snapshot_sha256"):
        if prepared[name] != initializer[name]:
            raise ValueError(
                "pilot input preparation differs from the checked initializer"
            )
    context.model.load_state_dict(load_file(initializer["weights_path"]), strict=True)
    if model_state_hash(context.model) != initializer["training"]["model_state_hash"]:
        raise ValueError("imported initializer weights differ")
    ids = tuple(proposal["pilot_cases"]["ids"])
    scales = {}
    for name in ids:
        values = context._values(name)
        if values != proposal["pilot_cases"]["values"][name]:
            raise ValueError("pilot FIT coordinates changed")
        scales[name] = math.hypot(values["ux"] / 60.0, values["uy"] / 40.0)
    objective = BoundaryPilotObjective(
        context,
        ids,
        strain_scales=scales,
        points_per_cell=12,
        training_seed=2916,
        validation_seed=2917,
        chunk=proposal["optimizer"]["gradient_point_chunk"],
    )
    identity = {
        "plan_sha256": plan.plan_sha256,
        "parent_training": initializer["training"],
        "initializer_weights_sha256": initializer["weights_sha256"],
        "objective": objective.document(),
    }
    return objective, identity
