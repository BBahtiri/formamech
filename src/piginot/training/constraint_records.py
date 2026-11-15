"""Immutable completed-block records bound to the safe inner checkpoints."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from dataclasses import asdict, astuple, replace
from pathlib import Path
from typing import Any, cast

import torch

from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.objectives.energy_constraint import (
    ENERGY_RATIO_LIMIT,
    ConstraintTerms,
    PenaltyState,
    constraint_merit,
    update_penalty,
)
from piginot.training.constraint_budget import (
    ConstraintBudget,
    ConstraintLimits,
    ConstraintUsage,
)
from piginot.training.lbfgs_checkpoint import load_stage_checkpoint
from piginot.training.lbfgs_stage import (
    EvaluationLimit,
    LBFGSConfig,
    create_lbfgs_optimizer,
)


def digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("constraint records require regular files")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_record(path: Path) -> dict[str, Any]:
    file_sha(path)
    value = strict_json_loads(path.read_bytes())
    if (
        not isinstance(value, dict)
        or set(value) != {"record", "sha256"}
        or not isinstance(value["record"], dict)
        or digest(value["record"]) != value["sha256"]
    ):
        raise ValueError("invalid constraint checkpoint envelope")
    return cast(dict[str, Any], value["record"])


def write_record(path: Path, record: dict[str, Any]) -> Path:
    payload = canonical_json_bytes({"record": record, "sha256": digest(record)}) + b"\n"
    if path.is_symlink():
        raise ValueError("constraint records cannot be symlinks")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("constraint checkpoint already has different content")
        return path
    descriptor, name = tempfile.mkstemp(prefix=".constraint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def inner_config(study: dict[str, Any], usage: ConstraintUsage) -> LBFGSConfig:
    config = LBFGSConfig(**study["config"]["inner"])
    limits = ConstraintLimits(**study["limits"])
    iterations = min(config.max_iterations, limits.max_iterations - usage.iterations)
    evaluations = min(
        config.max_evaluations,
        limits.max_gradients - usage.gradient_evaluations,
        (limits.max_forwards - usage.forward_evaluations) // 2,
    )
    if iterations <= 0 or evaluations <= 0:
        raise EvaluationLimit("the global constrained-study allowance is exhausted")
    return replace(config, max_iterations=iterations, max_evaluations=evaluations)


def block_binding(study: dict[str, Any], number: int, state: PenaltyState) -> str:
    return digest({"study": digest(study), "block": number, "state": asdict(state)})


def restore_boundary(
    root: Path,
    checkpoint: Path,
    study: dict[str, Any],
    model: torch.nn.Module,
    budget: ConstraintBudget,
    target: int,
) -> tuple[int, PenaltyState, ConstraintTerms]:
    """Validate the complete prefix before restoring the caller's model or RNG."""
    match = re.fullmatch(r"outer-(\d{6})\.json", checkpoint.name)
    if (
        match is None
        or checkpoint.parent.resolve() != root.resolve()
        or checkpoint.is_symlink()
        or (root / "incomplete.json").exists()
    ):
        raise ValueError("resume requires a completed outer-block checkpoint")
    count = int(match[1])
    if not 0 < count <= min(study["config"]["outer_blocks"], target):
        raise ValueError("invalid completed block count")
    for path in root.iterdir():
        later = re.fullmatch(r"(?:block-|outer-)(\d{6})(?:\.json)?", path.name)
        if later is not None and int(later[1]) > count:
            raise ValueError("resume cannot replay an earlier completed block")
    state, usage = PenaltyState(), ConstraintUsage(**study["initial_usage"])
    previous_model = study["initial_model_hash"]
    last_inner, last_config, last_binding = root, LBFGSConfig(), ""
    terms = ConstraintTerms(0.0, 0.0)
    try:
        for number in range(1, count + 1):
            record = read_record(root / f"outer-{number:06d}.json")
            config = inner_config(study, usage)
            binding = block_binding(study, number, state)
            if (
                record["schema"] != "piginot.constraint-block-v1"
                or record["study_sha256"] != digest(study)
                or record["block"] != number
                or record["state_before"] != asdict(state)
                or record["usage_before"] != asdict(usage)
                or record["starting_model_hash"] != previous_model
            ):
                raise ValueError("outer checkpoint lineage differs")
            stage = record["stage"]
            relative = Path(record["inner_checkpoint"])
            path = root / relative
            block = root / f"block-{number:06d}"
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or block.is_symlink()
                or not path.resolve().is_relative_to(block.resolve())
                or stage["status"] not in ("COMPLETE", "CONVERGED", "STALLED")
                or file_sha(path / "checkpoint.json") != record["inner_metadata_sha256"]
            ):
                raise ValueError("invalid inner checkpoint reference")
            inner = strict_json_loads((path / "checkpoint.json").read_bytes())
            if not isinstance(inner, dict):
                raise ValueError("inner checkpoint metadata must be an object")
            if (
                inner["binding"] != binding
                or inner["config"] != asdict(config)
                or inner["admissible"] is not True
                or file_sha(path / "state.safetensors") != inner["payload_sha256"]
                or any(
                    inner[key] != stage[key]
                    for key in (
                        "iterations",
                        "evaluations",
                        "closure_calls",
                        "model_hash",
                        "loss",
                    )
                )
            ):
                raise ValueError("inner checkpoint and outer record differ")
            terms = ConstraintTerms(**record["terms"])
            merit = constraint_merit(
                terms,
                study["config"]["reference_energy"],
                state,
                energy_ratio_limit=study["config"].get(
                    "energy_ratio_limit", ENERGY_RATIO_LIMIT
                ),
            )
            if not terms.admissible or not math.isclose(
                merit.value, stage["loss"], rel_tol=1e-11, abs_tol=1e-12
            ):
                raise ValueError("accepted-point merit differs")
            next_usage = replace(
                usage,
                iterations=usage.iterations + stage["iterations"],
                gradient_evaluations=usage.gradient_evaluations + stage["evaluations"],
                forward_evaluations=usage.forward_evaluations
                + 2 * stage["evaluations"],
            )
            next_state = update_penalty(state, merit.constraint)
            if record["usage_after"] != asdict(next_usage) or record[
                "state_after"
            ] != asdict(next_state):
                raise ValueError("outer multiplier or cumulative counters differ")
            usage, state, previous_model = next_usage, next_state, stage["model_hash"]
            last_inner, last_config, last_binding = path, config, binding
        ConstraintBudget(budget.limits, deadline_utc=budget.deadline_utc, usage=usage)
        if any(
            saved < spent
            for saved, spent in zip(astuple(usage), astuple(budget.usage), strict=True)
        ):
            raise ValueError("resume cannot reduce spent counters")
    except (KeyError, TypeError, OSError, EvaluationLimit) as error:
        raise ValueError("malformed constraint checkpoint prefix") from error
    optimizer = create_lbfgs_optimizer(model, last_config)
    load_stage_checkpoint(
        last_inner,
        model=model,
        optimizer=optimizer,
        binding=last_binding,
        config=asdict(last_config),
    )
    budget.restore_usage(usage)
    return count, state, terms
