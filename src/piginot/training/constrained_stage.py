"""Bounded inequality outer blocks with fresh deterministic L-BFGS history."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from piginot.objectives.energy_constraint import (
    ENERGY_RATIO_LIMIT,
    ConstraintTerms,
    PenaltyState,
    constraint_merit,
    update_penalty,
)
from piginot.training.checkpoint import model_state_hash
from piginot.training.constrained_objective import ConstrainedObjective, TermEvaluator
from piginot.training.constraint_budget import ConstraintBudget, ConstraintUsage
from piginot.training.constraint_records import (
    block_binding,
    digest,
    file_sha,
    inner_config,
    read_record,
    restore_boundary,
    write_record,
)
from piginot.training.lbfgs_stage import (
    EvaluationLimit,
    LBFGSConfig,
    StageDeadline,
    train_lbfgs_stage,
)


@dataclass(frozen=True, slots=True)
class ConstrainedConfig:
    reference_energy: float
    outer_blocks: int = 5
    inner: LBFGSConfig = LBFGSConfig(max_iterations=10)
    energy_ratio_limit: float = field(default=ENERGY_RATIO_LIMIT, kw_only=True)

    def __post_init__(self) -> None:
        constraint_merit(
            ConstraintTerms(0.0, 0.0),
            self.reference_energy,
            PenaltyState(),
            energy_ratio_limit=self.energy_ratio_limit,
        )
        if type(self.outer_blocks) is not int or self.outer_blocks <= 0:
            raise ValueError("outer block count must be a positive integer")


@dataclass(frozen=True, slots=True)
class ConstrainedResult:
    status: str
    completed_blocks: int
    usage: ConstraintUsage
    state: PenaltyState
    terms: ConstraintTerms | None
    checkpoint: Path | None
    model_hash: str
    train_feasible: bool
    message: str = ""


def _study(
    root: Path,
    model: torch.nn.Module,
    config: ConstrainedConfig,
    budget: ConstraintBudget,
    binding: str,
    resume: Path | None,
) -> dict[str, Any]:
    params = tuple(model.parameters())
    if root.is_symlink() or re.fullmatch(r"[0-9a-f]{64}", binding) is None:
        raise ValueError("a real output path and exact binding are required")
    if not params or any(
        p.dtype != torch.float64
        or p.device.type != "cpu"
        or not p.requires_grad
        or not bool(torch.isfinite(p).all())
        for p in params
    ):
        raise ValueError("the constrained stage requires finite float64 CPU parameters")
    identity = {
        "schema": "piginot.constrained-study-v1",
        "binding": binding,
        "config": {
            key: value
            for key, value in asdict(config).items()
            if key != "energy_ratio_limit" or value != ENERGY_RATIO_LIMIT
        },
        "limits": asdict(budget.limits),
        "deadline_utc": budget.deadline_utc,
        "inner_stop_policy": "advance_on_evaluated_admissible_stall",
    }
    if resume is not None:
        saved = read_record(root / "study.json")
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "the constrained-study binding, config or deadline differs"
            )
        return saved
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "constrained training requires fresh output or explicit boundary resume"
        )
    if budget.usage.gradient_evaluations or budget.usage.iterations:
        raise ValueError("a fresh study cannot discard previous training debits")
    root.mkdir(parents=True, exist_ok=True)
    study = {
        **identity,
        "initial_model_hash": model_state_hash(model),
        "initial_usage": asdict(budget.usage),
    }
    write_record(root / "study.json", study)
    return study


def run_constrained_stage(
    root: Path,
    *,
    model: torch.nn.Module,
    evaluator: TermEvaluator,
    budget: ConstraintBudget,
    config: ConstrainedConfig,
    binding: str,
    resume_from: Path | None = None,
    target_blocks: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ConstrainedResult:
    """Resume only at a completed block; interrupted partial blocks stay stopped."""
    target = config.outer_blocks if target_blocks is None else target_blocks
    if type(target) is not int or not 0 < target <= config.outer_blocks:
        raise ValueError("target blocks must be within the declared study")
    study = _study(root, model, config, budget, binding, resume_from)
    count, state, terms, checkpoint = 0, PenaltyState(), None, None
    if resume_from is not None:
        count, state, terms = restore_boundary(
            root, resume_from, study, model, budget, target
        )
        checkpoint = resume_from

    def result(status: str, message: str = "") -> ConstrainedResult:
        return ConstrainedResult(
            status,
            count,
            budget.usage,
            state,
            terms,
            checkpoint,
            model_state_hash(model),
            terms is not None
            and terms.admissible
            and terms.bulk_energy
            <= config.energy_ratio_limit * config.reference_energy,
            message,
        )

    def incomplete(
        status: str, message: str, inner: dict[str, Any] | None = None
    ) -> ConstrainedResult:
        write_record(
            root / "incomplete.json",
            {
                "schema": "piginot.constrained-incomplete-v1",
                "study_sha256": digest(study),
                "status": status,
                "usage": asdict(budget.usage),
                "completed_blocks": count,
                "message": message,
                "last_inner": inner,
            },
        )
        return result(status, message)

    try:
        for number in range(count + 1, target + 1):
            budget.check()
            inner = inner_config(study, budget.usage)
            before, starting_hash = budget.usage, model_state_hash(model)
            objective = ConstrainedObjective(
                evaluator,
                config.reference_energy,
                state,
                budget,
                energy_ratio_limit=config.energy_ratio_limit,
            )

            def publish(
                event: dict[str, Any], number: int = number, state: PenaltyState = state
            ) -> None:
                if event["event"] == "ACCEPTED_ITERATION":
                    budget.accept_iteration()
                if progress is not None:
                    progress(
                        {
                            "block": number,
                            "state": asdict(state),
                            "usage": asdict(budget.usage),
                            "inner": event,
                        }
                    )

            outcome = train_lbfgs_stage(
                root / f"block-{number:06d}",
                model=model,
                objective=objective,
                config=inner,
                binding=block_binding(study, number, state),
                deadline=budget.deadline_monotonic,
                deadline_utc=budget.deadline_utc,
                progress=publish,
            )
            # An inexact subproblem can stall before the multipliers are settled.
            # Advance only from its evaluated admissible point, preserving STALLED
            # in the inner receipt; the independent final constraint still decides.
            if outcome.status not in ("COMPLETE", "CONVERGED", "STALLED"):
                return incomplete(
                    outcome.status,
                    outcome.message,
                    {**asdict(outcome), "checkpoint": str(outcome.checkpoint)},
                )
            budget.check()
            accepted = objective.terms_for(model)
            violation = constraint_merit(
                accepted,
                config.reference_energy,
                state,
                energy_ratio_limit=config.energy_ratio_limit,
            ).constraint
            updated = update_penalty(state, violation)
            record = {
                "schema": "piginot.constraint-block-v1",
                "study_sha256": digest(study),
                "block": number,
                "starting_model_hash": starting_hash,
                "state_before": asdict(state),
                "state_after": asdict(updated),
                "usage_before": asdict(before),
                "usage_after": asdict(budget.usage),
                "terms": asdict(accepted),
                "stage": {
                    key: value
                    for key, value in asdict(outcome).items()
                    if key != "checkpoint"
                },
                "inner_checkpoint": outcome.checkpoint.relative_to(root).as_posix(),
                "inner_metadata_sha256": file_sha(
                    outcome.checkpoint / "checkpoint.json"
                ),
            }
            checkpoint = write_record(root / f"outer-{number:06d}.json", record)
            count, state, terms = number, updated, accepted
    except EvaluationLimit as error:
        return incomplete("EVALUATION_LIMIT", str(error))
    except StageDeadline as error:
        return incomplete("TIME_LIMIT", str(error))
    except BaseException as error:
        incomplete(
            "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR",
            str(error),
        )
        raise
    return result("COMPLETE" if count == config.outer_blocks else "CHECKPOINTED")
