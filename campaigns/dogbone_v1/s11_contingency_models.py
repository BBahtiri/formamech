"""Deterministic model construction for the preregistered S11 contingency."""

from __future__ import annotations

import hashlib
from typing import TypeAlias

import torch

from campaigns.dogbone_v1.s11_contingency import (
    BASELINE_CANDIDATE_ID,
    RAW_PARAMETER_NAMES,
    ArchitectureCandidate,
    ArchitectureScreenProtocol,
)
from piginot.operators import OperatorConfig, PiGinotOperator, module_digest

_StateTemplateKey: TypeAlias = tuple[str, tuple[int, ...], torch.dtype]


def build_candidate_models(
    protocol: ArchitectureScreenProtocol,
    *,
    boundary_tag_names: tuple[str, ...],
) -> tuple[PiGinotOperator, ...]:
    """Build the fixed factorial with bit-identical shared initial state."""

    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an exact ArchitectureScreenProtocol")
    _validate_boundary_tags(boundary_tag_names)

    templates: dict[_StateTemplateKey, torch.Tensor] = {}
    baseline_state: dict[str, torch.Tensor] | None = None
    models: list[PiGinotOperator] = []
    for index, candidate in enumerate(protocol.candidates):
        seed = (
            protocol.model_seed
            if candidate.candidate_id == BASELINE_CANDIDATE_ID
            else protocol.candidate_specific_initialization_seed
        )
        model = _build_model(candidate, boundary_tag_names, seed)
        synchronized: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            template_key = (name, tuple(tensor.shape), tensor.dtype)
            template = templates.get(template_key)
            if template is None:
                template = tensor.detach().clone()
                if index > 0 and name == "parameter_encoder.0.weight":
                    if baseline_state is None:
                        raise RuntimeError("baseline state template is unavailable")
                    template = _preserve_raw_parameter_columns(
                        candidate,
                        template,
                        baseline_state[name],
                    )
                templates[template_key] = template.detach().clone()
            synchronized[name] = template.detach().clone()
        model.load_state_dict(synchronized, strict=True)
        if module_digest(model) != candidate.operator_module_digest:
            raise ValueError(
                f"candidate {candidate.candidate_id} operator schema changed"
            )
        if index == 0:
            baseline_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
        models.append(model)
    return tuple(models)


def model_state_hash(model: PiGinotOperator) -> str:
    """Hash every parameter and buffer value with names, dtypes, and shapes."""

    if type(model) is not PiGinotOperator:
        raise TypeError("model must be an exact PiGinotOperator")
    digest = hashlib.blake2b(digest_size=32)
    for name, tensor in sorted(model.state_dict().items()):
        for value in (
            name.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            str(tuple(tensor.shape)).encode("ascii"),
            tensor.detach().cpu().contiguous().numpy().tobytes(),
        ):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _build_model(
    candidate: ArchitectureCandidate,
    boundary_tag_names: tuple[str, ...],
    seed: int,
) -> PiGinotOperator:
    config = OperatorConfig(
        parameter_names=candidate.parameter_names,
        boundary_tag_names=boundary_tag_names,
        hidden_dim=64,
        latent_token_count=32,
        num_attention_heads=4,
        num_fourier_frequencies=6,
        encoder_self_attention_layers=3,
        decoder_cross_attention_layers=6,
        local_boundary_kernel_scales=candidate.local_boundary_kernel_scales,
        stress_enrichment_layers=candidate.stress_enrichment_layers,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PiGinotOperator(config).to(dtype=torch.float64)


def _preserve_raw_parameter_columns(
    candidate: ArchitectureCandidate,
    candidate_weight: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> torch.Tensor:
    raw_width = len(RAW_PARAMETER_NAMES)
    if candidate.parameter_names[:raw_width] != RAW_PARAMETER_NAMES:
        raise ValueError("candidate parameters must preserve the raw parameter prefix")
    if (
        candidate_weight.ndim != 2
        or baseline_weight.ndim != 2
        or candidate_weight.shape[0] != baseline_weight.shape[0]
        or baseline_weight.shape[1] != raw_width
        or candidate_weight.shape[1] < raw_width
    ):
        raise ValueError("parameter encoder state cannot preserve the raw prefix")
    result = candidate_weight.detach().clone()
    result[:, :raw_width] = baseline_weight
    return result


def _validate_boundary_tags(boundary_tag_names: tuple[str, ...]) -> None:
    if type(boundary_tag_names) is not tuple or not boundary_tag_names:
        raise ValueError("boundary_tag_names must be a non-empty tuple")
    if any(
        type(name) is not str or not name or name.strip() != name
        for name in boundary_tag_names
    ):
        raise ValueError("boundary_tag_names must contain canonical names")
    if len(set(boundary_tag_names)) != len(boundary_tag_names):
        raise ValueError("boundary_tag_names must be unique")


__all__ = ("build_candidate_models", "model_state_hash")
