"""Mixed (u, P) S1 adapter around the exact pinned PI-GINOT implementation.

The upstream sources are vendored byte-for-byte (apart from platform newline
normalisation) under experiments/spike/upstream_pi_ginot. This module adds
only the direct first-Piola head required by S1. Displacement continues through
the unmodified upstream PI_GINOT.forward path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from experiments.spike.upstream_pi_ginot.config import (
    DECODER_CONFIG as UPSTREAM_DECODER_CONFIG,
)
from experiments.spike.upstream_pi_ginot.config import (
    ENCODER_CONFIG as UPSTREAM_ENCODER_CONFIG,
)
from experiments.spike.upstream_pi_ginot.models.pi_ginot import (
    PI_GINOT as UpstreamPIGINOT,
)

UPSTREAM_REPOSITORY: Final = "BBahtiri/PI-GINOT-DogBone"
UPSTREAM_COMMIT: Final = "9f6794141beca9d3e66433465c3b07039a7e41d2"
MATERIAL_YOUNGS_MODULUS: Final = 760.0
STRESS_HEAD_VARIANTS: Final = frozenset(
    {
        "linear",
        "mlp",
        "conditioned_mlp",
        "strain_conditioned_mlp",
        "coordinate_conditioned_mlp",
        "fully_conditioned_mlp",
    }
)
UPSTREAM_SOURCE_SHA256: Final = {
    "LICENSE": "64fd2f3673386742e9374f2fceb125fbc1a4b80cb687e7a8206e2ebfd1338eeb",
    "config.py": "bd610289505b574ef4f9addf5ebd437178519686c4c2c60ea501b387ce2294e1",
    "models/__init__.py": "bbcf1c958d0c8c10ff8e49bcbb70d531d86dfcec6170d1037ca032bf02ea0019",
    "models/geometry_encoder.py": "fe7466e1eb10f2adce81b78da8a4ccec6b4aae72d3adb8976d3bd900716c824b",
    "models/physics_decoder.py": "305c784733c6aea6a28d65f3ddf31ba46e39454b10cc09cb705bece16df0e557",
    "models/pi_ginot.py": "b31033fa294c061ef16df421ab4e007b6e04f8edeeded933a8dfa8873cc660c5",
    "models/modules/__init__.py": "c684199acfc7d30c6e49e75e4cf30c9e0cde8b24c9f36d4b1136418b9e918584",
    "models/modules/point_encoding.py": "7a2823e8641869279d89b33f7e213893524a7bdebb931195f883925b6600db28",
    "models/modules/point_position_embedding.py": "e253bb32ceeab2c37358a75f773785955ffb13b8e1596d77d4ebf7f9ad1d7df0",
    "models/modules/pointnet2_utils.py": "ff191b5f3046792d73675bb75fe4a604e31c07756f6bf44c1e9abd36d5f98a22",
    "models/modules/transformer.py": "a329673ffeb93229b8005ef46862b101ba5ad512163c6026956437996428368f",
}


@dataclass(frozen=True)
class ModelConfig:
    """Exact upstream architecture defaults plus the S1 stress-head choice."""

    boundary_points: int = 320
    latent_tokens: int = 32
    embed_dim: int = 64
    num_heads: int = 4
    encoder_cross_layers: int = 1
    encoder_self_layers: int = 3
    decoder_cross_layers: int = 6
    positional_encoding_max_degree: int = 6
    neighbor_count: int = 18
    neighbor_radius: float = 0.15
    stress_head_variant: str = "conditioned_mlp"

    def __post_init__(self) -> None:
        if self.boundary_points < 1:
            raise ValueError("boundary_points must be positive")
        if not 1 <= self.latent_tokens <= self.boundary_points:
            raise ValueError("latent_tokens must be within boundary point count")
        if not 1 <= self.neighbor_count <= self.boundary_points:
            raise ValueError("neighbor_count must be within boundary point count")
        if self.embed_dim < 1 or self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if self.neighbor_radius <= 0.0:
            raise ValueError("neighbor_radius must be positive")
        if self.stress_head_variant not in STRESS_HEAD_VARIANTS:
            raise ValueError(
                f"stress_head_variant must be one of {sorted(STRESS_HEAD_VARIANTS)}"
            )

    def encoder_config(self) -> dict[str, object]:
        """Return a fresh upstream GeometryEncoder configuration."""
        return {
            **UPSTREAM_ENCODER_CONFIG,
            "out_c": self.embed_dim,
            "width": self.embed_dim,
            "n_point": self.latent_tokens,
            "n_sample": self.neighbor_count,
            "radius": self.neighbor_radius,
            "d_hidden": [self.embed_dim, self.embed_dim],
            "num_heads": self.num_heads,
            "cross_attn_layers": self.encoder_cross_layers,
            "self_attn_layers": self.encoder_self_layers,
        }

    def decoder_config(self) -> dict[str, object]:
        """Return a fresh upstream PhysicsDecoder configuration."""
        return {
            **UPSTREAM_DECODER_CONFIG,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "cross_attn_layers": self.decoder_cross_layers,
            "pe_max_deg": self.positional_encoding_max_degree,
        }


@dataclass(frozen=True)
class MixedOutput:
    """Operator outputs with direct predicted stress as the authoritative field."""

    u: torch.Tensor
    P_predicted: torch.Tensor

    @property
    def P_served(self) -> torch.Tensor:
        return self.P_predicted


class DirectStressHead(nn.Module):
    """Direct stress head with optional physical-scale conditioning."""

    def __init__(self, embed_dim: int, variant: str) -> None:
        super().__init__()
        self.variant = variant
        if variant == "linear":
            self.hidden = nn.Identity()
        elif variant == "mlp":
            self.hidden = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        elif variant == "conditioned_mlp":
            self.hidden = nn.Sequential(
                nn.Linear(embed_dim + 1, embed_dim),
                nn.GELU(),
            )
        elif variant == "strain_conditioned_mlp":
            self.hidden = nn.Sequential(
                nn.Linear(embed_dim + 2, embed_dim),
                nn.GELU(),
            )
        elif variant == "coordinate_conditioned_mlp":
            self.hidden = nn.Sequential(
                nn.Linear(embed_dim + 3, embed_dim),
                nn.GELU(),
            )
        elif variant == "fully_conditioned_mlp":
            self.hidden = nn.Sequential(
                nn.Linear(embed_dim + 4, embed_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(f"unsupported stress head variant: {variant}")
        self.output = nn.Linear(embed_dim, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _batch_scalar(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [batch] or [batch, 1]")
        if torch.any(value.detach() <= 0.0):
            raise ValueError(f"{name} must be positive")
        return value

    def forward(
        self,
        features: torch.Tensor,
        query_points: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor,
    ) -> torch.Tensor:
        conditioned_variants = {
            "conditioned_mlp",
            "strain_conditioned_mlp",
            "coordinate_conditioned_mlp",
            "fully_conditioned_mlp",
        }
        if self.variant not in conditioned_variants:
            return self.output(self.hidden(features))

        batch_size, query_count, _ = features.shape
        if query_points.shape[:2] != features.shape[:2] or query_points.shape[-1] != 2:
            raise ValueError("query_points must have shape [batch, query, 2]")
        load = self._batch_scalar(u_delta, batch_size, "u_delta")
        length = self._batch_scalar(x_max, batch_size, "x_max")
        height = self._batch_scalar(y_max, batch_size, "y_max")
        aspect_ratio = (height / length)[:, None, None].expand(
            batch_size, query_count, 1
        )
        conditioning = [features, aspect_ratio]
        if self.variant in {"strain_conditioned_mlp", "fully_conditioned_mlp"}:
            engineering_strain = (load / length)[:, None, None].expand(
                batch_size, query_count, 1
            )
            conditioning.append(engineering_strain)
        if self.variant in {"coordinate_conditioned_mlp", "fully_conditioned_mlp"}:
            x_fraction = query_points[..., :1] / length[:, None, None]
            y_fraction = query_points[..., 1:] / height[:, None, None]
            conditioning.extend((x_fraction, y_fraction))
        conditioned = torch.cat(tuple(conditioning), dim=-1)
        dimensionless_stress = self.output(self.hidden(conditioned))
        nominal_stress = MATERIAL_YOUNGS_MODULUS * load / length
        return dimensionless_stress * nominal_stress[:, None, None]


class MixedPhysicsOperator(UpstreamPIGINOT):
    """Exact upstream PI-GINOT plus the S1 direct first-Piola output head."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        spike_config = config or ModelConfig()
        super().__init__(
            spike_config.encoder_config(),
            spike_config.decoder_config(),
        )
        self.spike_config = spike_config
        self.p_head = DirectStressHead(
            spike_config.embed_dim,
            spike_config.stress_head_variant,
        )

    @property
    def u_head(self) -> nn.Linear:
        """Compatibility alias for the unmodified upstream displacement head."""
        return self.decoder.output_proj

    def upstream_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only parameters that belong to the upstream implementation."""
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("p_head.")
        }

    def load_upstream_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> object:
        """Load upstream weights and require only the new P head to be absent."""
        outcome = self.load_state_dict(dict(state_dict), strict=False)
        expected_missing = {
            name for name in self.state_dict() if name.startswith("p_head.")
        }
        if set(outcome.missing_keys) != expected_missing or outcome.unexpected_keys:
            raise ValueError(
                "upstream checkpoint is incompatible with the pinned PI-GINOT backbone: "
                f"missing={outcome.missing_keys}, unexpected={outcome.unexpected_keys}"
            )
        return outcome

    def forward(
        self,
        query_points: torch.Tensor,
        boundary_points: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor,
    ) -> MixedOutput:
        if boundary_points.ndim != 3 or boundary_points.shape[-1] != 2:
            raise ValueError("boundary_points must have shape [batch, boundary, 2]")
        if boundary_points.shape[1] != self.spike_config.boundary_points:
            raise ValueError(
                f"expected {self.spike_config.boundary_points} boundary points, "
                f"received {boundary_points.shape[1]}"
            )

        decoder_features: list[torch.Tensor] = []

        def capture_features(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            decoder_features.append(inputs[0])

        handle = self.decoder.output_proj.register_forward_pre_hook(capture_features)
        try:
            displacement = super().forward(
                query_points,
                boundary_points,
                u_delta,
                x_max,
                y_max,
            )
        finally:
            handle.remove()
        if len(decoder_features) != 1:
            raise RuntimeError("upstream decoder did not expose exactly one head input")

        predicted_stress = self.p_head(
            decoder_features[0],
            query_points,
            u_delta,
            x_max,
            y_max,
        ).reshape(query_points.shape[0], query_points.shape[1], 2, 2)
        return MixedOutput(u=displacement, P_predicted=predicted_stress)


def _field_gradient(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    components: list[torch.Tensor] = []
    for component_index in range(field.shape[-1]):
        gradient = torch.autograd.grad(
            field[..., component_index].sum(),
            coordinates,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(coordinates)
        components.append(gradient)
    return torch.stack(components, dim=-2)


def material_first_piola_from_displacement(
    displacement: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    youngs_modulus: float = MATERIAL_YOUNGS_MODULUS,
    poisson_ratio: float = 0.23,
) -> torch.Tensor:
    """Differentiable in-plane constitutive oracle; never a served stress path."""
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = (
        youngs_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    gradient = _field_gradient(displacement, coordinates)
    identity = torch.eye(
        2, dtype=coordinates.dtype, device=coordinates.device
    ).expand_as(gradient)
    deformation = identity + gradient
    determinant = torch.linalg.det(deformation)
    if torch.any(determinant.detach() <= 0.0):
        raise ValueError("predicted displacement has a non-positive Jacobian")
    log_determinant = torch.log(determinant)
    log_f33 = -lam * log_determinant / (2.0 * mu + lam)
    for _ in range(8):
        exponential = torch.exp(2.0 * log_f33)
        residual = mu * torch.expm1(2.0 * log_f33) + lam * (log_determinant + log_f33)
        derivative = 2.0 * mu * exponential + lam
        log_f33 = log_f33 - residual / derivative
    f33_squared = torch.exp(2.0 * log_f33)
    inverse_transpose = torch.linalg.inv(deformation).transpose(-1, -2)
    return mu * (deformation - f33_squared[..., None, None] * inverse_transpose)


def equilibrium_residual(
    served_stress: torch.Tensor,
    coordinates: torch.Tensor,
    body_force: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``Div_X(P_served) + B`` without constitutive substitution."""
    if served_stress.shape != (*coordinates.shape[:-1], 2, 2):
        raise ValueError("served_stress must have shape [batch, query, 2, 2]")
    divergence_components: list[torch.Tensor] = []
    for spatial_component in range(2):
        component_divergence = torch.zeros_like(coordinates[..., 0])
        for reference_direction in range(2):
            stress_component = served_stress[
                ..., spatial_component, reference_direction
            ]
            if not stress_component.requires_grad:
                derivative = torch.zeros_like(coordinates)
            else:
                derivative_result = torch.autograd.grad(
                    stress_component.sum(),
                    coordinates,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                derivative = (
                    derivative_result
                    if derivative_result is not None
                    else torch.zeros_like(coordinates)
                )
            component_divergence = (
                component_divergence + derivative[..., reference_direction]
            )
        divergence_components.append(component_divergence)
    divergence = torch.stack(divergence_components, dim=-1)
    if body_force is None:
        return divergence
    if body_force.shape != divergence.shape:
        raise ValueError("body_force must match the divergence shape")
    return divergence + body_force
