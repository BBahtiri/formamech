"""GPU-first compute runtime, chunked quadrature, AMP, and DDP abstractions (V3-1).

Enforces FormaMech's dual execution contract:

* ``cpu-deterministic-bitwise``: strict CPU execution with deterministic seeds.
* ``gpu-accelerated-tolerance``: opt-in CUDA execution with explicit GPU
  hardware fingerprinting, optional mixed-precision forward passes (always
  accumulating variational energy & verification metrics in ``float64``),
  memory-bounded volumetric quadrature chunking, and DDP gradient reduction.

Every abstraction works natively on Windows/Linux/macOS CPU environments and
provides a deterministic ``SimulatedDistributedStrategy`` for unit-testing DDP
synchronization without requiring multi-GPU hardware.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

import torch
from torch import nn

from piginot.contracts.profile import REPLAY_TIERS
from piginot.contracts.state_machine import (
    StateMachineSpec,
    StateSnapshot,
    TransitionSpec,
    register_state_machine,
)

COMPUTE_RUNTIME_SCHEMA: Final = "formamech.compute-runtime-v1"
SUPPORTED_AMP_DTYPES: Final[frozenset[str]] = frozenset({"none", "bfloat16", "float16"})


class ComputeRuntimeState(StrEnum):
    """Explicit lifecycle state machine for GPU/distributed training runtimes."""

    UNCONFIGURED = "UNCONFIGURED"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


COMPUTE_RUNTIME_MACHINE: Final[StateMachineSpec[ComputeRuntimeState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.training.compute_runtime",
            description=(
                "Lifecycle of a CPU/CUDA/DDP compute runtime enforcing replay-tier "
                "validation, hardware fingerprinting, and clean teardown."
            ),
            states=tuple(ComputeRuntimeState),
            initial_states=(ComputeRuntimeState.UNCONFIGURED,),
            terminal_states=(ComputeRuntimeState.COMPLETED, ComputeRuntimeState.FAILED),
            transitions=(
                TransitionSpec(
                    ComputeRuntimeState.UNCONFIGURED,
                    ComputeRuntimeState.VALIDATED,
                    "validate",
                    "Device, replay tier, AMP dtype, and chunk parameters validated.",
                ),
                TransitionSpec(
                    ComputeRuntimeState.UNCONFIGURED,
                    ComputeRuntimeState.FAILED,
                    "reject",
                    "Requested device or replay tier is unavailable or inconsistent.",
                ),
                TransitionSpec(
                    ComputeRuntimeState.VALIDATED,
                    ComputeRuntimeState.ACTIVE,
                    "activate",
                    "Compute runtime entered active training/inference context.",
                ),
                TransitionSpec(
                    ComputeRuntimeState.ACTIVE,
                    ComputeRuntimeState.COMPLETED,
                    "complete",
                    "Training/inference finished cleanly and synchronized.",
                ),
                TransitionSpec(
                    ComputeRuntimeState.ACTIVE,
                    ComputeRuntimeState.FAILED,
                    "abort",
                    "Runtime encountered non-finite loss or hardware failure.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class ComputeRuntimeConfig:
    """Validated compute configuration for 2D/3D neural operator training."""

    device: str = "cpu"
    replay_tier: str = "cpu-deterministic-bitwise"
    amp_dtype: str = "none"
    quadrature_chunk_size: int | None = None
    gradient_accumulation_steps: int = 1
    use_torch_compile: bool = False
    ddp_world_size: int = 1
    ddp_rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.replay_tier not in REPLAY_TIERS:
            raise ValueError(
                f"replay_tier must be one of {sorted(REPLAY_TIERS)}, "
                f"got {self.replay_tier!r}"
            )
        if self.device.startswith("cuda") and (
            self.replay_tier != "gpu-accelerated-tolerance"
        ):
            raise ValueError(
                "CUDA execution requires replay_tier='gpu-accelerated-tolerance'"
            )
        if self.amp_dtype not in SUPPORTED_AMP_DTYPES:
            raise ValueError(
                f"amp_dtype must be one of {sorted(SUPPORTED_AMP_DTYPES)}, "
                f"got {self.amp_dtype!r}"
            )
        if self.amp_dtype != "none" and self.replay_tier != "gpu-accelerated-tolerance":
            raise ValueError(
                "mixed precision (amp_dtype != 'none') requires "
                "replay_tier='gpu-accelerated-tolerance'"
            )
        if self.quadrature_chunk_size is not None and (
            type(self.quadrature_chunk_size) is not int
            or self.quadrature_chunk_size < 1
        ):
            raise ValueError("quadrature_chunk_size must be a positive integer or None")
        if (
            type(self.gradient_accumulation_steps) is not int
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if type(self.use_torch_compile) is not bool:
            raise TypeError("use_torch_compile must be bool")
        if type(self.ddp_world_size) is not int or self.ddp_world_size < 1:
            raise ValueError("ddp_world_size must be >= 1")
        if (
            type(self.ddp_rank) is not int
            or self.ddp_rank < 0
            or self.ddp_rank >= self.ddp_world_size
        ):
            raise ValueError("ddp_rank must satisfy 0 <= ddp_rank < ddp_world_size")

    @property
    def amp_enabled(self) -> bool:
        return self.amp_dtype != "none"

    def resolve_device(self, *, require_available: bool = True) -> torch.device:
        """Resolve ``torch.device`` fail-closed when CUDA is requested but absent."""
        dev = torch.device(self.device)
        if dev.type == "cuda" and require_available:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"requested device {self.device!r} but torch.cuda.is_available() is False"
                )
            idx = 0 if dev.index is None else int(dev.index)
            if idx < 0 or idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"requested CUDA index {idx} out of range "
                    f"(device_count={torch.cuda.device_count()})"
                )
        return dev

    def document(self) -> dict[str, Any]:
        return {
            "schema": COMPUTE_RUNTIME_SCHEMA,
            "device": self.device,
            "replay_tier": self.replay_tier,
            "amp_dtype": self.amp_dtype,
            "quadrature_chunk_size": self.quadrature_chunk_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_torch_compile": self.use_torch_compile,
            "ddp_world_size": self.ddp_world_size,
            "ddp_rank": self.ddp_rank,
        }


def capture_gpu_fingerprint(device: str | torch.device = "cuda:0") -> dict[str, str]:
    """Capture canonical GPU hardware fingerprint keys required by ``ExecutionProfile``."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"cannot capture GPU fingerprint for {str(dev)!r}: CUDA unavailable"
        )
    idx = 0 if dev.index is None else int(dev.index)
    return {
        "device_name": str(torch.cuda.get_device_name(idx)),
        "cuda_version": str(torch.version.cuda or "unknown"),
        "cublas_workspace": str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset")),
    }


def chunked_quadrature_slices(
    total_points: int, chunk_size: int | None
) -> tuple[slice, ...]:
    """Partition ``[0, total_points)`` into exact non-overlapping quadrature chunks."""
    if type(total_points) is not int or total_points < 1:
        raise ValueError("total_points must be a positive integer")
    if chunk_size is None or chunk_size >= total_points:
        return (slice(0, total_points),)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer or None")
    return tuple(
        slice(start, min(start + chunk_size, total_points))
        for start in range(0, total_points, chunk_size)
    )


@contextlib.contextmanager
def autocast_context(config: ComputeRuntimeConfig) -> Iterator[None]:
    """Enter safe mixed-precision forward context (no-op when disabled)."""
    if not config.amp_enabled:
        yield
        return
    dev = torch.device(config.device)
    amp_t = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    if dev.type == "cpu" and amp_t is torch.float16:
        # PyTorch CPU autocast supports bfloat16 only; stay fail-closed on fp16 CPU.
        raise ValueError(
            "amp_dtype='float16' is unsupported on CPU; use 'bfloat16' or 'none'"
        )
    with torch.autocast(device_type=dev.type, dtype=amp_t):
        yield


@runtime_checkable
class DistributedStrategy(Protocol):
    """Abstraction over single-process, PyTorch DDP, and simulated multi-rank training."""

    @property
    def world_size(self) -> int: ...

    @property
    def rank(self) -> int: ...

    @property
    def is_main_process(self) -> bool: ...

    def wrap_model(self, model: nn.Module) -> nn.Module: ...

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def barrier(self) -> None: ...


class SingleProcessStrategy:
    """Zero-overhead deterministic single-process strategy."""

    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    @property
    def is_main_process(self) -> bool:
        return True

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def barrier(self) -> None:
        return None


class TorchDDPStrategy:
    """PyTorch ``DistributedDataParallel`` wrapper for multi-GPU scaling."""

    def __init__(self, *, rank: int, world_size: int, device: torch.device) -> None:
        if world_size < 1 or not (0 <= rank < world_size):
            raise ValueError("invalid rank/world_size for TorchDDPStrategy")
        self._rank = rank
        self._world_size = world_size
        self._device = device

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    def wrap_model(self, model: nn.Module) -> nn.Module:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "torch.distributed process group must be initialized before DDP wrap"
            )
        device_ids = [self._device.index or 0] if self._device.type == "cuda" else None
        return nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return tensor
        out = tensor.clone()
        torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM)
        return out / float(self._world_size)

    def barrier(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


class SimulatedDistributedStrategy:
    """CPU-testable deterministic multi-rank gradient reducer for DDP CI verification."""

    def __init__(self, *, rank: int = 0, world_size: int = 2) -> None:
        if world_size < 1 or not (0 <= rank < world_size):
            raise ValueError("invalid rank/world_size for SimulatedDistributedStrategy")
        self._rank = rank
        self._world_size = world_size
        self.all_reduce_calls = 0

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        self.all_reduce_calls += 1
        return tensor.clone()

    def reduce_rank_tensors(self, rank_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Exact mean reduction across simulated per-rank tensors."""
        if len(rank_tensors) != self._world_size:
            raise ValueError(
                f"expected {self._world_size} rank tensors, got {len(rank_tensors)}"
            )
        self.all_reduce_calls += 1
        stacked = torch.stack(
            [t.detach().to(torch.float64) for t in rank_tensors], dim=0
        )
        return torch.mean(stacked, dim=0).to(dtype=rank_tensors[0].dtype)

    def barrier(self) -> None:
        return None


class ComputeRuntimeSession:
    """Stateful session binding ``ComputeRuntimeConfig`` to ``COMPUTE_RUNTIME_MACHINE``."""

    def __init__(
        self,
        config: ComputeRuntimeConfig,
        *,
        strategy: DistributedStrategy | None = None,
        require_available: bool = True,
    ) -> None:
        self.config = config
        self._state = ComputeRuntimeState.UNCONFIGURED
        try:
            self.device = config.resolve_device(require_available=require_available)
        except Exception:
            self._state = COMPUTE_RUNTIME_MACHINE.require_transition(
                self._state, ComputeRuntimeState.FAILED, trigger="reject"
            ).to_state
            raise
        if strategy is not None:
            self.strategy: DistributedStrategy = strategy
        elif config.ddp_world_size > 1:
            self.strategy = TorchDDPStrategy(
                rank=config.ddp_rank,
                world_size=config.ddp_world_size,
                device=self.device,
            )
        else:
            self.strategy = SingleProcessStrategy()
        self._state = COMPUTE_RUNTIME_MACHINE.require_transition(
            self._state, ComputeRuntimeState.VALIDATED, trigger="validate"
        ).to_state

    @property
    def state(self) -> ComputeRuntimeState:
        return self._state

    def snapshot(self) -> StateSnapshot[ComputeRuntimeState]:
        return COMPUTE_RUNTIME_MACHINE.snapshot(
            self._state,
            metadata={
                "device": str(self.device),
                "replay_tier": self.config.replay_tier,
                "world_size": self.strategy.world_size,
            },
        )

    def activate(self) -> None:
        self._state = COMPUTE_RUNTIME_MACHINE.require_transition(
            self._state, ComputeRuntimeState.ACTIVE, trigger="activate"
        ).to_state

    def complete(self) -> None:
        self._state = COMPUTE_RUNTIME_MACHINE.require_transition(
            self._state, ComputeRuntimeState.COMPLETED, trigger="complete"
        ).to_state

    def abort(self) -> None:
        self._state = COMPUTE_RUNTIME_MACHINE.require_transition(
            self._state, ComputeRuntimeState.FAILED, trigger="abort"
        ).to_state

    def maybe_compile(self, model: nn.Module) -> nn.Module:
        if not self.config.use_torch_compile:
            return model
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return model
        return compile_fn(model)  # type: ignore[no-any-return]


def validate_loss_finite(value: float, *, label: str = "loss") -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite {label}: {value}")
    return float(value)


__all__ = (
    "COMPUTE_RUNTIME_MACHINE",
    "COMPUTE_RUNTIME_SCHEMA",
    "SUPPORTED_AMP_DTYPES",
    "ComputeRuntimeConfig",
    "ComputeRuntimeSession",
    "ComputeRuntimeState",
    "DistributedStrategy",
    "SimulatedDistributedStrategy",
    "SingleProcessStrategy",
    "TorchDDPStrategy",
    "autocast_context",
    "capture_gpu_fingerprint",
    "chunked_quadrature_slices",
    "validate_loss_finite",
)
