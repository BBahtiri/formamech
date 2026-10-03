"""Registered serving architectures and their declared served-stress authority.

A bundle declares which stress it serves (``served_stress``). Each registered
architecture states the one authority it provides; loading refuses a bundle
whose declaration differs, so no other stress is ever substituted for
``P_served`` (``G15.substitution``). The material law ``P_material(F)`` is
evaluated separately by the inference closure as a consistency oracle and is
reported under its own name.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from piginot.contracts import MaterialSpec
from piginot.operators.energy_operator import (
    ENERGY_ARCHITECTURE_ID,
    DirichletLifting,
    EnergyFormOperator,
    EnergyOperatorConfig,
    QueryFrame,
    constitutive_in_plane_stress,
    energy_module_digest,
)
from piginot.verification.confidence import equilibrium_residual

CONSTITUTIVE_PLANE_STRESS = "constitutive-plane-stress-neo-hookean"
DETERMINANT_FLOOR = 1.0e-12
_MAX_HEADER_BYTES = 16 * 1024 * 1024


class ServingError(ValueError):
    """The weights or architecture cannot serve the declared model."""


class StressAuthorityMismatch(ServingError):
    """The declared served stress is not the one the architecture provides."""


class InadmissibleDeformation(ValueError):
    """The predicted in-plane deformation is not invertible at some point."""


@dataclass(frozen=True, slots=True)
class ServedPoints:
    """Displacement, its gradient and the served stress at query points."""

    u: torch.Tensor
    gradient: torch.Tensor
    stress: torch.Tensor


class ServingModel(Protocol):
    """What the inference closure needs from an architecture."""

    @property
    def architecture_id(self) -> str: ...

    @property
    def stress_authority(self) -> str: ...

    @property
    def parameter_count(self) -> int: ...

    def serve(
        self,
        query_xy: torch.Tensor,
        *,
        parameters: torch.Tensor,
        frame: QueryFrame,
        lifting: DirichletLifting,
        material: MaterialSpec,
        chunk: int,
    ) -> ServedPoints: ...

    def residual(
        self,
        query_xy: torch.Tensor,
        *,
        parameters: torch.Tensor,
        frame: QueryFrame,
        lifting: DirichletLifting,
        material: MaterialSpec,
        length_scale: float,
    ) -> torch.Tensor: ...


def read_safetensors(payload: bytes) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Tensors and string metadata of a ``safetensors`` byte string."""

    if len(payload) < 8:
        raise ServingError("weights are too short to be safetensors")
    (length,) = struct.unpack("<Q", payload[:8])
    if length > _MAX_HEADER_BYTES or 8 + length > len(payload):
        raise ServingError("weights header is malformed")
    try:
        header = json.loads(payload[8 : 8 + length])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServingError("weights header is not JSON") from exc
    metadata = header.get("__metadata__", {}) if isinstance(header, dict) else None
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ServingError("weights metadata must map text to text")
    from safetensors.torch import load as load_tensors

    try:
        tensors = load_tensors(payload)
    except Exception as exc:  # safetensors raises its own error types
        raise ServingError(f"weights do not load: {exc}") from exc
    return dict(tensors), dict(metadata)


class EnergyFormServing:
    """The S13c energy-form operator; its served stress is constitutive."""

    architecture_id = ENERGY_ARCHITECTURE_ID
    stress_authority = CONSTITUTIVE_PLANE_STRESS

    def __init__(self, model: EnergyFormOperator) -> None:
        self._model = model
        self._count = sum(parameter.numel() for parameter in model.parameters())

    @property
    def parameter_count(self) -> int:
        return self._count

    def serve(
        self,
        query_xy: torch.Tensor,
        *,
        parameters: torch.Tensor,
        frame: QueryFrame,
        lifting: DirichletLifting,
        material: MaterialSpec,
        chunk: int,
    ) -> ServedPoints:
        u_parts: list[torch.Tensor] = []
        gradient_parts: list[torch.Tensor] = []
        stress_parts: list[torch.Tensor] = []
        for start in range(0, query_xy.shape[0], chunk):
            points = query_xy[start : start + chunk].unsqueeze(0)
            u, gradient = self._model.displacement_and_gradient(
                query_xy=points,
                parameters=parameters,
                frame=frame,
                lifting=lifting,
                create_graph=False,
            )
            deformation = torch.eye(2, dtype=gradient.dtype) + gradient[0]
            determinant = (
                deformation[:, 0, 0] * deformation[:, 1, 1]
                - deformation[:, 0, 1] * deformation[:, 1, 0]
            )
            if bool((determinant <= DETERMINANT_FLOOR).any().item()):
                raise InadmissibleDeformation("det F <= 1e-12 at a served point")
            u_parts.append(u[0].detach())
            gradient_parts.append(gradient[0].detach())
            stress_parts.append(
                constitutive_in_plane_stress(gradient[0], material).detach()
            )
        return ServedPoints(
            torch.cat(u_parts), torch.cat(gradient_parts), torch.cat(stress_parts)
        )

    def residual(
        self,
        query_xy: torch.Tensor,
        *,
        parameters: torch.Tensor,
        frame: QueryFrame,
        lifting: DirichletLifting,
        material: MaterialSpec,
        length_scale: float,
    ) -> torch.Tensor:
        return equilibrium_residual(
            self._model,
            query_xy,
            parameters=parameters,
            frame=frame,
            lifting=lifting,
            material=material,
            length_scale=length_scale,
        )


def build_energy_form(
    operator_config: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str],
) -> EnergyFormServing:
    """Rebuild the operator, load its weights and check its module digest."""

    try:
        config = EnergyOperatorConfig(
            **{
                **operator_config,
                "parameter_names": tuple(operator_config["parameter_names"]),
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ServingError(f"operator configuration is invalid: {exc}") from exc
    with torch.random.fork_rng(devices=[]):
        model = EnergyFormOperator(config).to(torch.float32)
    try:
        model.load_state_dict(dict(tensors), strict=True)
    except RuntimeError as exc:
        raise ServingError(f"weights do not match the operator: {exc}") from exc
    if metadata.get("architecture_id") != ENERGY_ARCHITECTURE_ID:
        raise ServingError("weights were exported by another architecture")
    if metadata.get("architecture_module_digest") != energy_module_digest(model):
        raise ServingError("weights module digest differs from the declared operator")
    if not all(math.isfinite(float(t.abs().max())) for t in tensors.values()):
        raise ServingError("weights must be finite")
    served = model.to(torch.float64).eval()
    served.requires_grad_(False)
    return EnergyFormServing(served)


ARCHITECTURES: dict[
    str,
    Callable[[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, str]], Any],
] = {ENERGY_ARCHITECTURE_ID: build_energy_form}


def build_serving(
    architecture_id: str,
    operator_config: Mapping[str, Any],
    weights: bytes,
    *,
    served_stress: str,
) -> ServingModel:
    """The registered architecture, refusing any served-stress mismatch."""

    builder = ARCHITECTURES.get(architecture_id)
    if builder is None:
        raise ServingError(f"unregistered architecture {architecture_id!r}")
    tensors, metadata = read_safetensors(weights)
    serving: ServingModel = builder(operator_config, tensors, metadata)
    if serving.architecture_id != architecture_id:
        raise ServingError("the builder returned another architecture")
    if serving.stress_authority != served_stress:
        raise StressAuthorityMismatch(
            f"declared served stress {served_stress!r} is not provided by "
            f"{architecture_id!r} ({serving.stress_authority!r})"
        )
    return serving


__all__ = (
    "ARCHITECTURES",
    "CONSTITUTIVE_PLANE_STRESS",
    "DETERMINANT_FLOOR",
    "EnergyFormServing",
    "InadmissibleDeformation",
    "ServedPoints",
    "ServingError",
    "ServingModel",
    "StressAuthorityMismatch",
    "build_energy_form",
    "build_serving",
    "read_safetensors",
)
