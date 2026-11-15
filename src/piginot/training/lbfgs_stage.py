"""Deterministic, budgeted L-BFGS polishing with accepted-point rollback."""

from __future__ import annotations

import copy
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from piginot.training.checkpoint import model_state_hash
from piginot.training.lbfgs_checkpoint import (
    EvaluatedPoint,
    StageCursor,
    load_stage_checkpoint,
    save_stage_checkpoint,
)


@dataclass(frozen=True, slots=True)
class LBFGSConfig:
    max_iterations: int = 50
    max_evaluations: int = 100
    history_size: int = 20
    checkpoint_interval: int = 10
    learning_rate: float = 1.0
    tolerance_grad: float = 1e-8
    tolerance_change: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "max_iterations",
            "max_evaluations",
            "history_size",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "tolerance_grad", "tolerance_change"):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class LBFGSResult:
    status: str
    iterations: int
    evaluations: int
    closure_calls: int
    checkpoint: Path
    model_hash: str
    loss: float | None
    message: str = ""


@dataclass(frozen=True, slots=True)
class GradientEvaluation:
    value: float
    admissible: bool = True


class EvaluationLimit(RuntimeError):
    """The declared number of actual objective/gradient evaluations was spent."""


class StageDeadline(RuntimeError):
    """The enclosing wall-clock allowance ended."""


class LineSearchFailure(RuntimeError):
    """An optimizer trial did not establish the declared Wolfe conditions."""


def create_lbfgs_optimizer(
    model: torch.nn.Module, config: LBFGSConfig
) -> torch.optim.LBFGS:
    """Use the same parameter-group contract when training or restoring history."""
    return torch.optim.LBFGS(
        model.parameters(),
        lr=config.learning_rate,
        max_iter=1,
        max_eval=config.max_evaluations,
        history_size=config.history_size,
        tolerance_grad=config.tolerance_grad,
        tolerance_change=config.tolerance_change,
        line_search_fn="strong_wolfe",
    )


def _gradient(model: torch.nn.Module) -> torch.Tensor:
    parameters = tuple(model.parameters())
    if all(parameter.grad is None for parameter in parameters):
        raise ValueError("the objective must backpropagate its gradient")
    values = [
        torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten()
        for p in parameters
    ]
    gradient = torch.cat(values)
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("the objective gradient must be finite")
    return gradient.clone()


def _restore_gradient(model: torch.nn.Module, gradient: torch.Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.grad = gradient[offset : offset + count].reshape_as(parameter).clone()
        offset += count


class _Stage:
    def __init__(
        self,
        root: Path,
        model: torch.nn.Module,
        objective: Callable[[torch.nn.Module], float | GradientEvaluation],
        binding: str,
        config: LBFGSConfig,
        deadline: float | None,
        progress: Callable[[dict[str, Any]], None] | None,
        deadline_utc: float | None = None,
    ):
        self.root, self.model, self.objective = root, model, objective
        self.binding, self.config, self.deadline = binding, config, deadline
        self.progress = progress
        self.deadline_utc = deadline_utc
        self.optimizer = create_lbfgs_optimizer(model, config)
        self.cursor = StageCursor()
        self.point: EvaluatedPoint | None = None
        self.points: dict[str, EvaluatedPoint] = {}

    def publish(self, event: str) -> None:
        if self.progress is not None:
            self.progress(
                {
                    "event": event,
                    **asdict(self.cursor),
                    "accepted_model_hash": None
                    if self.point is None
                    else self.point.model_hash,
                    "accepted_loss": None if self.point is None else self.point.value,
                }
            )

    def check_deadline(self) -> None:
        if (self.deadline is not None and time.monotonic() >= self.deadline) or (
            self.deadline_utc is not None and time.time() >= self.deadline_utc
        ):
            raise StageDeadline("wall-clock limit reached")

    def closure(self) -> torch.Tensor:
        self.check_deadline()
        key = model_state_hash(self.model)
        self.cursor = replace(self.cursor, closure_calls=self.cursor.closure_calls + 1)
        point = self.points.get(key)
        if point is not None:
            _restore_gradient(self.model, point.gradient)
            self.publish("CACHED_GRADIENT")
            return torch.tensor(point.value, dtype=torch.float64)
        if self.cursor.evaluations >= self.config.max_evaluations:
            raise EvaluationLimit("objective/gradient evaluation limit reached")
        self.cursor = replace(self.cursor, evaluations=self.cursor.evaluations + 1)
        self.publish("EVALUATION_RESERVED")
        self.model.zero_grad(set_to_none=True)
        generator_before = torch.get_rng_state().clone()
        observed = self.objective(self.model)
        if not torch.equal(generator_before, torch.get_rng_state()):
            raise ValueError("L-BFGS closures must use fixed deterministic samples")
        value = (
            observed.value
            if isinstance(observed, GradientEvaluation)
            else float(observed)
        )
        admissible = (
            observed.admissible if isinstance(observed, GradientEvaluation) else True
        )
        if not math.isfinite(value):
            raise ValueError("the objective must be finite")
        gradient = _gradient(self.model)
        self.check_deadline()
        self.points[key] = EvaluatedPoint(key, value, gradient, admissible)
        self.publish("EVALUATION_FINISHED")
        return torch.tensor(value, dtype=torch.float64)

    def _acceptance(self, initial_hash: str) -> EvaluatedPoint:
        key = model_state_hash(self.model)
        if key not in self.points:
            raise LineSearchFailure("accepted weights were never evaluated")
        initial, accepted = self.points[initial_hash], self.points[key]
        if not accepted.admissible:
            raise LineSearchFailure("accepted weights violate the admissibility guard")
        if key == initial_hash:
            return accepted
        state = self.optimizer.state[next(iter(self.model.parameters()))]
        direction, step = state.get("d"), state.get("t")
        if not isinstance(direction, torch.Tensor) or step is None:
            raise LineSearchFailure("line-search direction or step is unavailable")
        slope = float(initial.gradient @ direction)
        bound = initial.value + 1e-4 * float(step) * slope
        slack = 1e-12 * max(1.0, abs(initial.value))
        curvature = abs(float(accepted.gradient @ direction))
        if (
            slope >= 0
            or accepted.value > bound + slack
            or curvature > 0.9 * abs(slope) + slack
        ):
            raise LineSearchFailure(
                "accepted trial does not satisfy strong Wolfe conditions"
            )
        return accepted

    def step(self) -> bool:
        initial_hash = model_state_hash(self.model)
        previous_model = copy.deepcopy(self.model.state_dict())
        previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
        previous_generator = torch.get_rng_state().clone()
        self.points = {} if self.point is None else {self.point.model_hash: self.point}
        try:
            self.optimizer.step(self.closure)  # type: ignore[no-untyped-call]
            accepted = self._acceptance(initial_hash)
        except BaseException:
            self.model.load_state_dict(previous_model)
            self.optimizer.load_state_dict(previous_optimizer)
            torch.set_rng_state(previous_generator)
            self.point = self.points.get(initial_hash, self.point)
            self.points = (
                {} if self.point is None else {self.point.model_hash: self.point}
            )
            raise
        self.point = accepted
        self.points = {accepted.model_hash: accepted}
        changed = accepted.model_hash != initial_hash
        if changed:
            self.cursor = replace(self.cursor, iterations=self.cursor.iterations + 1)
        return changed

    def save(self) -> Path:
        return save_stage_checkpoint(
            self.root,
            model=self.model,
            optimizer=self.optimizer,
            binding=self.binding,
            config=asdict(self.config),
            cursor=self.cursor,
            point=self.point,
        )


def _validate_inputs(
    root: Path,
    model: torch.nn.Module,
    binding: str,
    config: LBFGSConfig,
    resume: Path | None,
    target: int,
) -> None:
    if (
        not isinstance(root, Path)
        or root.is_symlink()
        or re.fullmatch(r"[0-9a-f]{64}", binding) is None
    ):
        raise ValueError("a real stage path and exact binding digest are required")
    parameters = tuple(model.parameters())
    if not parameters or any(
        p.dtype != torch.float64 or p.device.type != "cpu" or not p.requires_grad
        for p in parameters
    ):
        raise ValueError("L-BFGS polishing requires trainable float64 CPU parameters")
    if type(target) is not int or not 0 < target <= config.max_iterations:
        raise ValueError("target iterations must be within the declared budget")
    if root.exists() and any(root.iterdir()) and resume is None:
        raise FileExistsError("existing stage output requires explicit resume")
    if resume is not None and (
        not resume.resolve().is_relative_to(root.resolve()) or not resume.is_dir()
    ):
        raise ValueError("resume requires a checkpoint from this stage directory")


def _run(stage: _Stage, target: int) -> tuple[str, str]:
    try:
        while stage.cursor.iterations < target:
            if not stage.step():
                stage.check_deadline()
                assert stage.point is not None
                converged = (
                    float(stage.point.gradient.abs().max())
                    <= stage.config.tolerance_grad
                )
                return (
                    "CONVERGED" if converged else "STALLED"
                ), "optimizer did not change weights"
            stage.publish("ACCEPTED_ITERATION")
            stage.check_deadline()
            if stage.cursor.iterations % stage.config.checkpoint_interval == 0:
                stage.save()
        stage.check_deadline()
        return (
            "COMPLETE" if target == stage.config.max_iterations else "CHECKPOINTED"
        ), ""
    except EvaluationLimit as error:
        return "EVALUATION_LIMIT", str(error)
    except StageDeadline as error:
        return "TIME_LIMIT", str(error)
    except LineSearchFailure as error:
        return "LINE_SEARCH_FAILURE", str(error)
    except Exception as error:
        return "ERROR", f"{type(error).__name__}: {error}"


def train_lbfgs_stage(
    root: Path,
    *,
    model: torch.nn.Module,
    objective: Callable[[torch.nn.Module], float | GradientEvaluation],
    binding: str,
    config: LBFGSConfig,
    resume_from: Path | None = None,
    target_iterations: int | None = None,
    deadline: float | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    deadline_utc: float | None = None,
) -> LBFGSResult:
    """Run a fixed objective; only real gradient computations spend that budget.

    Accepted-point gradient reuse is exact for this deterministic phase. Closure
    invocations and computed evaluations are recorded separately. An interrupted
    or rejected trial restores the last accepted weights and optimizer history;
    evaluations already reserved remain debited in the terminal checkpoint.
    """
    target = config.max_iterations if target_iterations is None else target_iterations
    _validate_inputs(root, model, binding, config, resume_from, target)
    if deadline_utc is not None and (
        type(deadline_utc) not in (int, float)
        or not math.isfinite(deadline_utc)
        or deadline_utc <= 0
    ):
        raise ValueError("UTC deadline must be finite and positive")
    root.mkdir(parents=True, exist_ok=True)
    stage = _Stage(
        root, model, objective, binding, config, deadline, progress, deadline_utc
    )
    if resume_from is not None:
        stage.cursor, stage.point = load_stage_checkpoint(
            resume_from,
            model=model,
            optimizer=stage.optimizer,
            binding=binding,
            config=asdict(config),
        )
        if target < stage.cursor.iterations:
            raise ValueError("resume cannot move the stage cursor backwards")
    previous_mode = model.training
    model.eval()
    try:
        try:
            status, message = _run(stage, target)
        except BaseException:
            stage.save()
            stage.publish("INTERRUPTED")
            raise
        checkpoint = stage.save()
        stage.publish(status)
        return LBFGSResult(
            status,
            stage.cursor.iterations,
            stage.cursor.evaluations,
            stage.cursor.closure_calls,
            checkpoint,
            model_state_hash(model),
            None if stage.point is None else stage.point.value,
            message,
        )
    finally:
        model.train(previous_mode)
