"""Energy-form PI-GINOT operator with exact Dirichlet lifting.

This module ports the parameter-conditioned operator of the energy-form
PI-GINOT study (``pi-ginot-enhanced`` commit ``12c9c1c``) into the contract
style of :mod:`piginot.operators`:

* the shape/load parameter vector is mapped to a fixed set of latent tokens;
* query points are encoded with NeRF features, shifted by a pooled-token bias
  and refined by residual cross-attention blocks, each followed by FiLM;
* a linear head returns a dimensionless correction ``phi`` that enters an
  exact affine Dirichlet lifting ``u = offset + scale * phi``;
* the served first Piola stress is the constitutive stress of the predicted
  displacement gradient, evaluated with the S4 plane-stress kernel.

The decoder is point-local in the query axis: queries attend only to latent
tokens and every normalisation acts per query, so displacement gradients can
be taken with summed component derivatives.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, Final, cast

import torch
from torch import nn

from piginot.contracts import MaterialSpec
from piginot.contracts.canonical import canonical_json_bytes
from piginot.lifting._checks import (
    require_same_context as _require_same_context,
)
from piginot.lifting._checks import (
    validate_field as _validate_field,
)
from piginot.lifting._checks import (
    validate_matrix as _validate_matrix,
)
from piginot.lifting.base import AffineDirichletLifting, DirichletLifting
from piginot.lifting.quarter import AxisAlignedTensionLifting
from piginot.operators.architecture import OperatorOutput
from piginot.physics import first_piola, plane_stress_closure

ENERGY_ARCHITECTURE_ID: Final = "pi-ginot-energy-parametric-film-v1"


@dataclass(frozen=True, slots=True)
class EnergyOperatorConfig:
    """Immutable semantic and shape contract of the energy-form operator."""

    __canonical_namespace__: ClassVar[str] = "piginot.operators"

    parameter_names: tuple[str, ...]
    hidden_dim: int = 64
    latent_token_count: int = 32
    parameter_hidden_dim: int = 96
    num_attention_heads: int = 4
    decoder_layers: int = 6
    positional_frequencies: int = 6
    output_init_std: float = 1.0e-5

    def __post_init__(self) -> None:
        names = self.parameter_names
        if type(names) is not tuple or not names:
            raise ValueError("parameter_names must be a non-empty tuple")
        if any(
            type(name) is not str or not name or name.strip() != name for name in names
        ):
            raise ValueError("parameter_names must contain canonical names")
        if len(set(names)) != len(names):
            raise ValueError("parameter_names must not contain duplicates")
        for label, value in (
            ("hidden_dim", self.hidden_dim),
            ("latent_token_count", self.latent_token_count),
            ("parameter_hidden_dim", self.parameter_hidden_dim),
            ("num_attention_heads", self.num_attention_heads),
            ("decoder_layers", self.decoder_layers),
            ("positional_frequencies", self.positional_frequencies),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        std = self.output_init_std
        if (
            isinstance(std, bool)
            or not isinstance(std, (int, float))
            or not math.isfinite(float(std))
            or float(std) <= 0.0
        ):
            raise ValueError("output_init_std must be finite and positive")

    @property
    def parameter_dim(self) -> int:
        return len(self.parameter_names)

    @property
    def positional_dim(self) -> int:
        return 2 + 4 * self.positional_frequencies


@dataclass(frozen=True, slots=True)
class QueryFrame:
    """Per-sample affine map from physical coordinates onto ``[-1, 1]^2``."""

    origin: torch.Tensor
    extent: torch.Tensor

    def __post_init__(self) -> None:
        _validate_matrix(self.origin, "origin", trailing=2)
        _validate_matrix(self.extent, "extent", trailing=2)
        if self.origin.shape != self.extent.shape:
            raise ValueError("origin and extent must have the same shape")
        _require_same_context(self.origin, self.extent, "origin", "extent")
        if bool((self.extent <= 0.0).any().item()):
            raise ValueError("extent must be strictly positive")

    def normalize(self, query_xy: torch.Tensor) -> torch.Tensor:
        """Map physical coordinates into the canonical decoder frame."""

        if query_xy.shape[0] != self.origin.shape[0]:
            raise ValueError("query_xy and frame must share a batch dimension")
        return (
            2.0 * (query_xy - self.origin.unsqueeze(1)) / self.extent.unsqueeze(1) - 1.0
        )


class _NerfEncoding(nn.Module):
    """NeRF features ``(x, sin(2^k x), sin(2^k x + pi / 2))`` for two axes."""

    def __init__(self, frequencies: int) -> None:
        super().__init__()
        self.frequencies = frequencies

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        scales = torch.pow(
            coordinates.new_tensor(2.0),
            torch.arange(
                self.frequencies,
                dtype=coordinates.dtype,
                device=coordinates.device,
            ),
        )
        scaled = (coordinates.unsqueeze(-2) * scales.unsqueeze(-1)).flatten(-2)
        encoded = torch.sin(torch.cat((scaled, scaled + 0.5 * math.pi), dim=-1))
        return torch.cat((coordinates, encoded), dim=-1)


class _CrossAttention(nn.Module):
    """Explicit multi-head attention, twice differentiable on every backend."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * hidden_dim, hidden_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        batch, query_count, hidden = query.shape
        source_count = source.shape[1]
        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
        q = nn.functional.linear(query, w_q, b_q).reshape(
            batch, query_count, self.num_heads, self.head_dim
        )
        k = nn.functional.linear(source, w_k, b_k).reshape(
            batch, source_count, self.num_heads, self.head_dim
        )
        v = nn.functional.linear(source, w_v, b_v).reshape(
            batch, source_count, self.num_heads, self.head_dim
        )
        logits = torch.einsum("bqhd,bshd->bhqs", q, k) / math.sqrt(self.head_dim)
        attended = torch.einsum("bhqs,bshd->bqhd", torch.softmax(logits, dim=-1), v)
        return cast(
            torch.Tensor,
            self.out_proj(attended.reshape(batch, query_count, hidden)),
        )


class _ResidualCrossAttentionBlock(nn.Module):
    """Pre-norm residual cross-attention followed by a 4x GELU MLP."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.attention = _CrossAttention(hidden_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        query = query + self.attention(self.query_norm(query), self.source_norm(source))
        return cast(torch.Tensor, query + self.mlp(self.mlp_norm(query)))


class _FiLM(nn.Module):
    """Feature-wise modulation initialised to the identity map."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.gamma(condition).unsqueeze(1) * features
            + self.beta(condition).unsqueeze(1),
        )


class EnergyFormOperator(nn.Module):
    """Parameter-conditioned displacement operator trained on potential energy."""

    def __init__(self, config: EnergyOperatorConfig) -> None:
        super().__init__()
        if type(config) is not EnergyOperatorConfig:
            raise TypeError("config must be an EnergyOperatorConfig")
        self.config = config
        width = config.hidden_dim
        self.parameter_encoder = nn.Sequential(
            nn.Linear(config.parameter_dim, config.parameter_hidden_dim),
            nn.GELU(),
            nn.Linear(config.parameter_hidden_dim, config.parameter_hidden_dim),
            nn.GELU(),
            nn.Linear(
                config.parameter_hidden_dim,
                config.latent_token_count * width,
            ),
        )
        self.query_encoding = _NerfEncoding(config.positional_frequencies)
        self.query_embedding = nn.Linear(config.positional_dim, width)
        self.query_projection = nn.Sequential(nn.GELU(), nn.Linear(width, width))
        self.latent_bias = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        last_bias_layer = cast(nn.Linear, self.latent_bias[-1])
        nn.init.zeros_(last_bias_layer.weight)
        nn.init.zeros_(last_bias_layer.bias)
        self.blocks = nn.ModuleList(
            _ResidualCrossAttentionBlock(width, config.num_attention_heads)
            for _ in range(config.decoder_layers)
        )
        self.films = nn.ModuleList(_FiLM(width) for _ in range(config.decoder_layers))
        self.output = nn.Linear(width, 2)
        nn.init.normal_(self.output.weight, mean=0.0, std=config.output_init_std)
        nn.init.zeros_(self.output.bias)

    @property
    def architecture_id(self) -> str:
        return ENERGY_ARCHITECTURE_ID

    def encode_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        """Map normalised parameters to ``(batch, tokens, hidden)`` latents."""

        _validate_matrix(parameters, "parameters", trailing=self.config.parameter_dim)
        tokens = cast(torch.Tensor, self.parameter_encoder(parameters))
        return tokens.reshape(
            parameters.shape[0],
            self.config.latent_token_count,
            self.config.hidden_dim,
        )

    def correction(
        self,
        tokens: torch.Tensor,
        normalized_query: torch.Tensor,
    ) -> torch.Tensor:
        """Return the dimensionless correction ``phi`` at normalised queries."""

        features = self.query_projection(
            self.query_embedding(self.query_encoding(normalized_query))
        )
        pooled = tokens.mean(dim=1)
        features = features + self.latent_bias(pooled).unsqueeze(1)
        for block, film in zip(self.blocks, self.films, strict=True):
            features = film(block(features, tokens), pooled)
        return cast(torch.Tensor, self.output(features))

    def displacement(
        self,
        *,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        frame: QueryFrame,
        lifting: DirichletLifting,
    ) -> torch.Tensor:
        """Predict displacement in millimetres with exact essential conditions."""

        _validate_field(query_xy, "query_xy", trailing=2)
        if type(frame) is not QueryFrame:
            raise TypeError("frame must be a QueryFrame")
        if parameters.shape[0] != query_xy.shape[0]:
            raise ValueError("parameters and query_xy must share a batch dimension")
        _require_same_context(query_xy, parameters, "query_xy", "parameters")
        _require_same_context(query_xy, frame.origin, "query_xy", "frame")
        tokens = self.encode_parameters(parameters)
        phi = self.correction(tokens, frame.normalize(query_xy))
        affine = lifting(query_xy)
        if type(affine) is not AffineDirichletLifting:
            raise TypeError("lifting must return an AffineDirichletLifting")
        if affine.offset.shape != phi.shape:
            raise ValueError("lifting must align with the query points")
        return affine.offset + affine.scale * phi

    def displacement_and_gradient(
        self,
        *,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        frame: QueryFrame,
        lifting: DirichletLifting,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``u`` and ``du_i / dX_J`` with shape ``(batch, points, 2, 2)``."""

        query = query_xy
        if not query.requires_grad:
            query = query_xy.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            u = self.displacement(
                query_xy=query,
                parameters=parameters,
                frame=frame,
                lifting=lifting,
            )
            rows: list[torch.Tensor] = []
            for component in range(2):
                (gradient,) = torch.autograd.grad(
                    u[..., component].sum(),
                    query,
                    create_graph=create_graph,
                    retain_graph=True,
                )
                rows.append(gradient)
        displacement_gradient = torch.stack(rows, dim=-2)
        if not create_graph:
            return u.detach(), displacement_gradient.detach()
        return u, displacement_gradient

    def forward(
        self,
        *,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        frame: QueryFrame,
        lifting: DirichletLifting,
        materials: Sequence[MaterialSpec],
    ) -> OperatorOutput:
        """Serve ``u`` and the constitutive first Piola stress of ``I + grad u``."""

        if len(materials) != query_xy.shape[0]:
            raise ValueError("materials must provide one MaterialSpec per sample")
        u, gradient = self.displacement_and_gradient(
            query_xy=query_xy,
            parameters=parameters,
            frame=frame,
            lifting=lifting,
            create_graph=False,
        )
        stresses = tuple(
            constitutive_in_plane_stress(gradient[index], material)
            for index, material in enumerate(materials)
        )
        return OperatorOutput(u, torch.stack(stresses, dim=0))


def constitutive_in_plane_stress(
    displacement_gradient: torch.Tensor,
    material: MaterialSpec,
) -> torch.Tensor:
    """Return the in-plane block of plane-stress Neo-Hookean ``P(I + grad u)``."""

    if (
        not isinstance(displacement_gradient, torch.Tensor)
        or displacement_gradient.shape[-2:] != (2, 2)
        or displacement_gradient.ndim < 3
    ):
        raise ValueError("displacement_gradient must have shape (..., 2, 2)")
    identity = torch.eye(
        2,
        dtype=displacement_gradient.dtype,
        device=displacement_gradient.device,
    )
    deformation_2d = identity + displacement_gradient
    f33 = plane_stress_closure(deformation_2d, material)
    deformation = embed_plane_stress_deformation(deformation_2d, f33)
    return first_piola(deformation, material)[..., :2, :2]


def embed_plane_stress_deformation(
    deformation_2d: torch.Tensor,
    f33: torch.Tensor,
) -> torch.Tensor:
    """Embed an in-plane deformation and its closure stretch into ``3 x 3``."""

    if deformation_2d.shape[-2:] != (2, 2) or f33.shape != deformation_2d.shape[:-2]:
        raise ValueError("deformation_2d and f33 shapes do not align")
    zeros = torch.zeros_like(f33)
    first = torch.stack(
        (deformation_2d[..., 0, 0], deformation_2d[..., 0, 1], zeros), dim=-1
    )
    second = torch.stack(
        (deformation_2d[..., 1, 0], deformation_2d[..., 1, 1], zeros), dim=-1
    )
    third = torch.stack((zeros, zeros, f33), dim=-1)
    return torch.stack((first, second, third), dim=-2)


def energy_module_digest(module: EnergyFormOperator) -> str:
    """Hash architecture semantics and tensor schema, never learned values."""

    if type(module) is not EnergyFormOperator:
        raise TypeError("module must be an exact EnergyFormOperator")
    config = asdict(module.config)
    config["parameter_names"] = list(module.config.parameter_names)
    payload: dict[str, object] = {
        "architecture_id": module.architecture_id,
        "config": config,
        "served_stress": "constitutive-plane-stress-neo-hookean",
        "state_schema": [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in sorted(module.state_dict().items())
        ],
    }
    digest = hashlib.blake2b(canonical_json_bytes(payload), digest_size=32)
    return f"blake2b256:{digest.hexdigest()}"


__all__ = (
    "ENERGY_ARCHITECTURE_ID",
    "AffineDirichletLifting",
    "AxisAlignedTensionLifting",
    "DirichletLifting",
    "EnergyFormOperator",
    "EnergyOperatorConfig",
    "QueryFrame",
    "constitutive_in_plane_stress",
    "embed_plane_stress_deformation",
    "energy_module_digest",
)
