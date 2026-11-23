"""Trajectory-safe state hashing, checkpoints, and fail-closed guarded stepper (F0/F1).

Implements the step contract of Feature F:
* verify ``BoundTrajectoryState`` identity before stepping (refuse changed
  geometry, material schema, load-path prefix, or controller version),
* validate ``det F``, nonnegative discrete dissipation, and step residual,
* halve ``dt`` within budget or escalate to an incremental FEM fallback if
  declared, and otherwise fail closed with ``UNVERIFIED`` / ``SOLVE_FAILED``.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import torch
from safetensors.torch import load as load_safetensors_bytes
from safetensors.torch import save_file

from piginot.contracts import canonical_json_bytes
from piginot.contracts.state_machine import (
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)
from piginot.contracts.stateful import (
    BoundTrajectoryState,
    StatefulVerificationPolicy,
)

STEP_CONTROLLER_VERSION: Final = "stateful-guarded-stepper-v1"
TRAJECTORY_CHECKPOINT_SCHEMA: Final = "piginot.stateful-checkpoint-v1"


def hash_state_tensors(state: Mapping[str, torch.Tensor]) -> str:
    """Deterministic BLAKE2b-256 hash over canonical sorted state tensors."""
    hasher = hashlib.blake2b(digest_size=32)
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        arr = np.asarray(tensor.numpy(), dtype="<f8")
        # Canonicalize negative zero to positive zero for exact cross-platform replay.
        arr = np.where(arr == 0.0, 0.0, arr)
        meta = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        hasher.update(meta)
        hasher.update(arr.tobytes())
    return f"blake2b256:{hasher.hexdigest()}"


class StepStatus(enum.StrEnum):
    ACCEPTED_OPERATOR = "ACCEPTED_OPERATOR"
    ACCEPTED_FEM_FALLBACK = "ACCEPTED_FEM_FALLBACK"
    UNVERIFIED = "UNVERIFIED"
    REFUSED_STATE_MISMATCH = "REFUSED_STATE_MISMATCH"
    REFUSED_ADMISSIBILITY = "REFUSED_ADMISSIBILITY"
    SOLVE_FAILED = "SOLVE_FAILED"


class StepPhase(enum.StrEnum):
    """Explicit stages of one guarded stateful rollout step."""

    RECEIVED = "RECEIVED"
    BINDING_VERIFIED = "BINDING_VERIFIED"
    OPERATOR_TRIED = "OPERATOR_TRIED"
    FEM_FALLBACK_TRIED = "FEM_FALLBACK_TRIED"
    ACCEPTED_OPERATOR = "ACCEPTED_OPERATOR"
    ACCEPTED_FEM_FALLBACK = "ACCEPTED_FEM_FALLBACK"
    UNVERIFIED = "UNVERIFIED"
    REFUSED_STATE_MISMATCH = "REFUSED_STATE_MISMATCH"
    REFUSED_ADMISSIBILITY = "REFUSED_ADMISSIBILITY"
    SOLVE_FAILED = "SOLVE_FAILED"


STATEFUL_STEP_MACHINE: Final[StateMachineSpec[StepPhase]] = register_state_machine(
    StateMachineSpec(
        machine_id="formamech.stateful.guarded_step",
        description=(
            "Guarded stateful trajectory step: binding/state-hash verification -> "
            "neural operator with substep halving -> optional FEM fallback -> terminal outcome."
        ),
        states=tuple(StepPhase),
        initial_states=(StepPhase.RECEIVED,),
        terminal_states=(
            StepPhase.ACCEPTED_OPERATOR,
            StepPhase.ACCEPTED_FEM_FALLBACK,
            StepPhase.UNVERIFIED,
            StepPhase.REFUSED_STATE_MISMATCH,
            StepPhase.REFUSED_ADMISSIBILITY,
            StepPhase.SOLVE_FAILED,
        ),
        transitions=(
            TransitionSpec(
                StepPhase.RECEIVED,
                StepPhase.REFUSED_STATE_MISMATCH,
                "binding_or_hash_mismatch",
                "Trajectory binding or state tensor hash failed verification.",
            ),
            TransitionSpec(
                StepPhase.RECEIVED,
                StepPhase.BINDING_VERIFIED,
                "verify_binding",
                "Trajectory binding and state tensor hash verified.",
            ),
            TransitionSpec(
                StepPhase.BINDING_VERIFIED,
                StepPhase.OPERATOR_TRIED,
                "try_operator",
                "Evaluate neural operator with bounded substep halving.",
            ),
            TransitionSpec(
                StepPhase.OPERATOR_TRIED,
                StepPhase.ACCEPTED_OPERATOR,
                "accept_operator",
                "Operator passed det(F), dissipation, and residual guards.",
            ),
            TransitionSpec(
                StepPhase.OPERATOR_TRIED,
                StepPhase.FEM_FALLBACK_TRIED,
                "escalate_fem_fallback",
                "Operator violated guards; escalate to authoritative FEM stepper.",
            ),
            TransitionSpec(
                StepPhase.OPERATOR_TRIED,
                StepPhase.REFUSED_ADMISSIBILITY,
                "refuse_fail_closed",
                "Operator violated guards with no FEM fallback and fail_closed=True.",
            ),
            TransitionSpec(
                StepPhase.OPERATOR_TRIED,
                StepPhase.UNVERIFIED,
                "emit_unverified",
                "Operator violated guards with fail_closed=False and no fallback.",
            ),
            TransitionSpec(
                StepPhase.FEM_FALLBACK_TRIED,
                StepPhase.ACCEPTED_FEM_FALLBACK,
                "accept_fem",
                "Authoritative FEM fallback satisfied all physical guards.",
            ),
            TransitionSpec(
                StepPhase.FEM_FALLBACK_TRIED,
                StepPhase.SOLVE_FAILED,
                "fem_failed",
                "FEM fallback failed all substep halvings.",
            ),
        ),
    )
)


@dataclass(frozen=True, slots=True)
class CandidateStepOutput:
    """Output returned by a candidate transition operator or reference step callback."""

    displacement: torch.Tensor
    state_new: Mapping[str, torch.Tensor]
    min_det_f: float
    min_dissipation: float
    total_dissipation: float
    residual_norm: float | None
    admissible: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class StepResult:
    """Auditable outcome of advancing one stateful step."""

    status: StepStatus
    time_index: int
    time_value: float
    dt_used: float
    substeps_taken: int
    displacement: torch.Tensor | None
    state_new: Mapping[str, torch.Tensor] | None
    bound_state: BoundTrajectoryState | None
    min_det_f: float | None
    total_dissipation: float | None
    residual_norm: float | None
    reason: str

    def __post_init__(self) -> None:
        if self.accepted and (
            self.displacement is None
            or self.state_new is None
            or self.bound_state is None
        ):
            raise ValueError(
                f"accepted StepResult ({self.status.value}) requires non-None "
                "displacement, state_new, and bound_state"
            )

    @property
    def accepted(self) -> bool:
        return self.status in (
            StepStatus.ACCEPTED_OPERATOR,
            StepStatus.ACCEPTED_FEM_FALLBACK,
        )


StepPredictor = Callable[
    [torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, float], float],
    CandidateStepOutput,
]


class StatefulStepper(Protocol):
    """Protocol for advancing a bound stateful trajectory by ``dt``."""

    def advance(
        self,
        *,
        bound_state: BoundTrajectoryState,
        displacement_old: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        load_next: Mapping[str, float],
        dt: float,
    ) -> StepResult: ...


class GuardedStatefulStepper:
    """Fail-closed candidate stepper with adaptive dt halving and FEM fallback."""

    def __init__(
        self,
        *,
        problem_spec_hash: str,
        mesh_hash: str,
        material_schema_hash: str,
        load_path_hash: str,
        predictor: StepPredictor,
        verification: StatefulVerificationPolicy,
        residual_tolerance: float = 1e-6,
        max_substep_halvings: int = 3,
        fem_fallback: StepPredictor | None = None,
        controller_version: str = STEP_CONTROLLER_VERSION,
    ) -> None:
        self.problem_spec_hash = problem_spec_hash
        self.mesh_hash = mesh_hash
        self.material_schema_hash = material_schema_hash
        self.load_path_hash = load_path_hash
        self.predictor = predictor
        self.verification = verification
        self.residual_tolerance = float(residual_tolerance)
        self.max_substep_halvings = int(max_substep_halvings)
        self.fem_fallback = fem_fallback
        self.controller_version = controller_version

    def _check_candidate(self, out: CandidateStepOutput) -> tuple[bool, str]:
        if not out.admissible:
            return False, out.reason or "candidate reported inadmissible state"
        if (
            not math.isfinite(out.min_det_f)
            or out.min_det_f < self.verification.min_det_f
        ):
            return (
                False,
                f"min_det_f={out.min_det_f} below floor {self.verification.min_det_f}",
            )
        if (
            not math.isfinite(out.min_dissipation)
            or out.min_dissipation < -self.verification.dissipation_tolerance
        ):
            return (
                False,
                f"negative discrete dissipation {out.min_dissipation} exceeds tolerance",
            )
        if out.residual_norm is not None:
            if (
                not math.isfinite(out.residual_norm)
                or out.residual_norm > self.residual_tolerance
            ):
                return (
                    False,
                    f"step residual {out.residual_norm} exceeds {self.residual_tolerance}",
                )
        return True, "all step guards passed"

    def _run_substeps(
        self,
        callback: StepPredictor,
        displacement_old: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        load_next: Mapping[str, float],
        dt: float,
        n_substeps: int,
    ) -> tuple[bool, CandidateStepOutput | None, float, str]:
        u_curr = displacement_old
        s_curr: Mapping[str, torch.Tensor] = state_old
        sub_dt = dt / n_substeps
        total_diss = 0.0
        last_out: CandidateStepOutput | None = None
        for _ in range(n_substeps):
            out = callback(u_curr, s_curr, load_next, sub_dt)
            ok, msg = self._check_candidate(out)
            if not ok:
                return False, out, total_diss, msg
            u_curr = out.displacement
            s_curr = out.state_new
            total_diss += out.total_dissipation
            last_out = out
        return True, last_out, total_diss, "guards passed"

    def advance(
        self,
        *,
        bound_state: BoundTrajectoryState,
        displacement_old: torch.Tensor,
        state_old: Mapping[str, torch.Tensor],
        load_next: Mapping[str, float],
        dt: float,
    ) -> StepResult:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")

        ok_bind, bind_msg = bound_state.verify_compatible(
            problem_spec_hash=self.problem_spec_hash,
            mesh_hash=self.mesh_hash,
            material_schema_hash=self.material_schema_hash,
            load_path_hash=self.load_path_hash,
            step_controller_version=self.controller_version,
        )
        if not ok_bind:
            return StepResult(
                status=StepStatus.REFUSED_STATE_MISMATCH,
                time_index=bound_state.time_index,
                time_value=bound_state.time_value,
                dt_used=0.0,
                substeps_taken=0,
                displacement=None,
                state_new=None,
                bound_state=None,
                min_det_f=None,
                total_dissipation=None,
                residual_norm=None,
                reason=bind_msg,
            )

        actual_hash = hash_state_tensors(state_old)
        if actual_hash != bound_state.state_hash:
            return StepResult(
                status=StepStatus.REFUSED_STATE_MISMATCH,
                time_index=bound_state.time_index,
                time_value=bound_state.time_value,
                dt_used=0.0,
                substeps_taken=0,
                displacement=None,
                state_new=None,
                bound_state=None,
                min_det_f=None,
                total_dissipation=None,
                residual_norm=None,
                reason="supplied state_old tensors do not match bound_state.state_hash",
            )

        last_reason = ""
        for halving in range(self.max_substep_halvings + 1):
            n_sub = 2**halving
            ok, out, total_diss, reason = self._run_substeps(
                self.predictor, displacement_old, state_old, load_next, dt, n_sub
            )
            if ok and out is not None:
                next_bound = BoundTrajectoryState(
                    problem_spec_hash=self.problem_spec_hash,
                    mesh_hash=self.mesh_hash,
                    material_schema_hash=self.material_schema_hash,
                    load_path_hash=self.load_path_hash,
                    time_index=bound_state.time_index + 1,
                    time_value=bound_state.time_value + dt,
                    step_controller_version=self.controller_version,
                    state_hash=hash_state_tensors(out.state_new),
                )
                return StepResult(
                    status=StepStatus.ACCEPTED_OPERATOR,
                    time_index=next_bound.time_index,
                    time_value=next_bound.time_value,
                    dt_used=dt,
                    substeps_taken=n_sub,
                    displacement=out.displacement,
                    state_new=out.state_new,
                    bound_state=next_bound,
                    min_det_f=out.min_det_f,
                    total_dissipation=total_diss,
                    residual_norm=out.residual_norm,
                    reason=f"operator step accepted in {n_sub} substep(s)",
                )
            last_reason = reason

        # Escalate to incremental FEM fallback if allowed and registered.
        if self.verification.allow_fem_fallback and self.fem_fallback is not None:
            for halving in range(self.max_substep_halvings + 1):
                n_sub = 2**halving
                ok, out, total_diss, reason = self._run_substeps(
                    self.fem_fallback, displacement_old, state_old, load_next, dt, n_sub
                )
                if ok and out is not None:
                    next_bound = BoundTrajectoryState(
                        problem_spec_hash=self.problem_spec_hash,
                        mesh_hash=self.mesh_hash,
                        material_schema_hash=self.material_schema_hash,
                        load_path_hash=self.load_path_hash,
                        time_index=bound_state.time_index + 1,
                        time_value=bound_state.time_value + dt,
                        step_controller_version=self.controller_version,
                        state_hash=hash_state_tensors(out.state_new),
                    )
                    return StepResult(
                        status=StepStatus.ACCEPTED_FEM_FALLBACK,
                        time_index=next_bound.time_index,
                        time_value=next_bound.time_value,
                        dt_used=dt,
                        substeps_taken=n_sub,
                        displacement=out.displacement,
                        state_new=out.state_new,
                        bound_state=next_bound,
                        min_det_f=out.min_det_f,
                        total_dissipation=total_diss,
                        residual_norm=out.residual_norm,
                        reason=f"escalated to incremental FEM fallback ({n_sub} substep(s))",
                    )
                last_reason = reason
            return StepResult(
                status=StepStatus.SOLVE_FAILED,
                time_index=bound_state.time_index,
                time_value=bound_state.time_value,
                dt_used=0.0,
                substeps_taken=0,
                displacement=None,
                state_new=None,
                bound_state=None,
                min_det_f=None,
                total_dissipation=None,
                residual_norm=None,
                reason=f"operator and incremental FEM fallback failed: {last_reason}",
            )

        return StepResult(
            status=(
                StepStatus.REFUSED_ADMISSIBILITY
                if self.verification.fail_closed_on_violation
                else StepStatus.UNVERIFIED
            ),
            time_index=bound_state.time_index,
            time_value=bound_state.time_value,
            dt_used=0.0,
            substeps_taken=0,
            displacement=None,
            state_new=None,
            bound_state=None,
            min_det_f=None,
            total_dissipation=None,
            residual_norm=None,
            reason=f"step refused after substep budget exhausted: {last_reason}",
        )


def save_trajectory_checkpoint(
    directory: Path,
    *,
    bound_state: BoundTrajectoryState,
    displacement: torch.Tensor,
    state_tensors: Mapping[str, torch.Tensor],
) -> Path:
    """Persist a hash-verified trajectory checkpoint using canonical JSON + safetensors."""
    actual = hash_state_tensors(state_tensors)
    if actual != bound_state.state_hash:
        raise ValueError(
            "cannot checkpoint state_tensors that differ from bound_state.state_hash"
        )
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        "__displacement__": displacement.detach().cpu().contiguous(),
        **{
            f"state.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in sorted(state_tensors.items())
        },
    }
    tensor_path = directory / "state.safetensors"
    save_file(tensors, str(tensor_path))
    tensor_sha256 = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest = {
        "schema": TRAJECTORY_CHECKPOINT_SCHEMA,
        "bound_state": bound_state.document(),
        "binding_hash": bound_state.binding_hash,
        "tensor_file": "state.safetensors",
        "tensor_sha256": tensor_sha256,
    }
    manifest_path = directory / "checkpoint.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest_path


def load_trajectory_checkpoint(
    directory: Path,
) -> tuple[BoundTrajectoryState, torch.Tensor, dict[str, torch.Tensor]]:
    """Load and cryptographically verify a trajectory checkpoint; fail closed on tampering."""
    manifest_path = directory / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TRAJECTORY_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported trajectory checkpoint schema")
    tensor_path = directory / str(manifest["tensor_file"])
    actual_sha = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    if actual_sha != manifest["tensor_sha256"]:
        raise ValueError("trajectory checkpoint tensor file failed SHA-256 check")

    raw_bound = manifest["bound_state"]
    bound = BoundTrajectoryState(
        problem_spec_hash=raw_bound["problem_spec_hash"],
        mesh_hash=raw_bound["mesh_hash"],
        material_schema_hash=raw_bound["material_schema_hash"],
        load_path_hash=raw_bound["load_path_hash"],
        time_index=int(raw_bound["time_index"]),
        time_value=float(raw_bound["time_value"]),
        step_controller_version=raw_bound["step_controller_version"],
        state_hash=raw_bound["state_hash"],
    )
    if bound.binding_hash != manifest["binding_hash"]:
        raise ValueError("trajectory checkpoint binding_hash mismatch")

    loaded = load_safetensors_bytes(tensor_path.read_bytes())
    if "__displacement__" not in loaded:
        raise ValueError("trajectory checkpoint missing displacement tensor")
    displacement = loaded["__displacement__"]
    state_tensors = {
        k[len("state.") :]: v for k, v in loaded.items() if k.startswith("state.")
    }
    if hash_state_tensors(state_tensors) != bound.state_hash:
        raise ValueError(
            "trajectory checkpoint state tensors failed state_hash verification"
        )
    return bound, displacement, state_tensors


__all__ = (
    "STEP_CONTROLLER_VERSION",
    "TRAJECTORY_CHECKPOINT_SCHEMA",
    "CandidateStepOutput",
    "GuardedStatefulStepper",
    "StatefulStepper",
    "StepPredictor",
    "StepResult",
    "StepStatus",
    "hash_state_tensors",
    "load_trajectory_checkpoint",
    "save_trajectory_checkpoint",
)
