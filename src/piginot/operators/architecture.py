"""Hardened PI-GINOT boundary encoder and query decoder for S11."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Final, cast

import torch
from torch import nn

from piginot.contracts.canonical import canonical_json_bytes

ARCHITECTURE_ID: Final = "pi-ginot-s11-boundary-cross-attention-v1"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


class BoundaryConditionMode(StrEnum):
    """Supported component-wise Dirichlet enforcement policies."""

    SOFT = "soft"
    HARD_LIFTING = "hard-lifting"


@dataclass(frozen=True, slots=True)
class LiftingVerification:
    """Evidence required before approximate-distance lifting can be enabled."""

    verification_id: str
    parameter_range_hash: str
    verification_hash: str
    boundary_refinement_levels: tuple[int, ...]
    full_parameter_range_verified: bool
    max_boundary_residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.verification_id, str) or not self.verification_id:
            raise ValueError("verification_id must be non-empty text")
        for name, value in (
            ("parameter_range_hash", self.parameter_range_hash),
            ("verification_hash", self.verification_hash),
        ):
            if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase 256-bit hex digest")
        levels = self.boundary_refinement_levels
        if type(levels) is not tuple or len(levels) < 2:
            raise ValueError("verification must span at least two boundary refinements")
        if any(
            isinstance(level, bool) or not isinstance(level, int) or level <= 0
            for level in levels
        ):
            raise ValueError("boundary refinement levels must be positive integers")
        if tuple(sorted(set(levels))) != levels:
            raise ValueError("boundary refinement levels must be unique and increasing")
        if self.full_parameter_range_verified is not True:
            raise ValueError("the full parameter range must be verified")
        if (
            isinstance(self.max_boundary_residual, bool)
            or not isinstance(self.max_boundary_residual, (int, float))
            or not math.isfinite(float(self.max_boundary_residual))
            or float(self.max_boundary_residual) < 0.0
        ):
            raise ValueError("max_boundary_residual must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class OperatorConfig:
    """Immutable semantic and shape contract for one PI-GINOT architecture."""

    parameter_names: tuple[str, ...]
    boundary_tag_names: tuple[str, ...]
    hidden_dim: int = 64
    latent_token_count: int = 32
    num_attention_heads: int = 4
    num_fourier_frequencies: int = 6
    encoder_self_attention_layers: int = 3
    decoder_cross_attention_layers: int = 6
    boundary_condition_mode: BoundaryConditionMode = BoundaryConditionMode.SOFT
    soft_dirichlet_residual_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        _validate_names(self.parameter_names, "parameter_names")
        _validate_names(self.boundary_tag_names, "boundary_tag_names")
        for name, value in (
            ("hidden_dim", self.hidden_dim),
            ("latent_token_count", self.latent_token_count),
            ("num_attention_heads", self.num_attention_heads),
            ("num_fourier_frequencies", self.num_fourier_frequencies),
            ("encoder_self_attention_layers", self.encoder_self_attention_layers),
            ("decoder_cross_attention_layers", self.decoder_cross_attention_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if type(self.boundary_condition_mode) is not BoundaryConditionMode:
            raise TypeError("boundary_condition_mode must be a BoundaryConditionMode")
        tolerance = self.soft_dirichlet_residual_tolerance
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) <= 0.0
        ):
            raise ValueError("soft_dirichlet_residual_tolerance must be positive")

    @property
    def parameter_dim(self) -> int:
        return len(self.parameter_names)

    @property
    def boundary_tag_dim(self) -> int:
        return len(self.boundary_tag_names)


@dataclass(frozen=True, slots=True)
class DirichletData:
    """Component mask, prescribed displacement and distance-lifting field."""

    component_mask: torch.Tensor
    prescribed_u: torch.Tensor
    approximate_distance: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.component_mask, torch.Tensor):
            raise TypeError("component_mask must be a torch.Tensor")
        if self.component_mask.dtype is not torch.bool:
            raise ValueError("component_mask must be boolean")
        _validate_field(self.prescribed_u, "prescribed_u", trailing=2)
        _validate_field(self.approximate_distance, "approximate_distance", trailing=2)
        if self.component_mask.shape != self.prescribed_u.shape:
            raise ValueError("component_mask and prescribed_u must have the same shape")
        _require_same_tensor_context(
            self.prescribed_u,
            self.approximate_distance,
            "prescribed_u",
            "approximate_distance",
        )
        if self.approximate_distance.shape != self.prescribed_u.shape:
            raise ValueError("approximate_distance and prescribed_u must align")
        if bool((self.approximate_distance < 0.0).any().item()):
            raise ValueError("approximate_distance must be non-negative")


@dataclass(frozen=True, slots=True)
class GeometryEncoding:
    """Encode-once latent tokens plus the boundary-derived coordinate frame."""

    tokens: torch.Tensor
    coordinate_min: torch.Tensor
    coordinate_span: torch.Tensor

    def __post_init__(self) -> None:
        _validate_field(self.tokens, "tokens")
        _validate_matrix(self.coordinate_min, "coordinate_min", trailing=2)
        _validate_matrix(self.coordinate_span, "coordinate_span", trailing=2)
        if self.tokens.shape[0] != self.coordinate_min.shape[0]:
            raise ValueError("geometry encoding tensors must share a batch dimension")
        if self.coordinate_min.shape != self.coordinate_span.shape:
            raise ValueError("coordinate frame tensors must have the same shape")
        _require_same_tensor_context(
            self.tokens, self.coordinate_min, "tokens", "coordinate_min"
        )
        _require_same_tensor_context(
            self.tokens, self.coordinate_span, "tokens", "coordinate_span"
        )
        if bool((self.coordinate_span <= 0.0).any().item()):
            raise ValueError("coordinate_span must be positive on each axis")


@dataclass(frozen=True, slots=True)
class OperatorOutput:
    """Query-wise displacement and authoritative predicted first-Piola stress."""

    u: torch.Tensor
    P: torch.Tensor

    def __post_init__(self) -> None:
        _validate_field(self.u, "u", trailing=2)
        if self.P.ndim != 4 or self.P.shape[-2:] != (2, 2):
            raise ValueError("P must have shape (batch, query_points, 2, 2)")
        if self.P.shape[:2] != self.u.shape[:2]:
            raise ValueError("u and P must share batch and query dimensions")
        _require_same_tensor_context(self.u, self.P, "u", "P")

    @property
    def served_stress(self) -> torch.Tensor:
        """Expose the predicted head without constitutive recomputation."""

        return self.P

    def flat(self) -> torch.Tensor:
        """Flatten both independent heads for architecture-gate comparisons."""

        batch = self.u.shape[0]
        return torch.cat((self.u.reshape(batch, -1), self.P.reshape(batch, -1)), dim=-1)


class FourierFeatures(nn.Module):
    """Deterministic NeRF-style positional features for two coordinates."""

    def __init__(self, num_frequencies: int) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies

    @property
    def output_dim(self) -> int:
        return 2 + 4 * self.num_frequencies

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        frequencies = torch.pow(
            coordinates.new_tensor(2.0),
            torch.arange(
                self.num_frequencies,
                dtype=coordinates.dtype,
                device=coordinates.device,
            ),
        )
        angles = math.pi * coordinates.unsqueeze(-1) * frequencies
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return torch.cat((coordinates, encoded.flatten(start_dim=-2)), dim=-1)


class _WeightedSetCrossAttention(nn.Module):
    """Cross-attention whose measure weights make point resampling stable."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_count, hidden = query.shape
        source_count = source.shape[1]
        q = self.query(query).reshape(batch, query_count, self.num_heads, self.head_dim)
        k = self.key(source).reshape(batch, source_count, self.num_heads, self.head_dim)
        v = self.value(source).reshape(
            batch, source_count, self.num_heads, self.head_dim
        )
        logits = torch.einsum("bqhd,bshd->bhqs", q, k) / math.sqrt(self.head_dim)
        usable = mask & (weights > 0.0)
        log_measure = torch.where(
            usable,
            torch.log(torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)),
            torch.full_like(weights, -torch.inf),
        )
        attention = torch.softmax(logits + log_measure[:, None, None, :], dim=-1)
        attended = torch.einsum("bhqs,bshd->bqhd", attention, v)
        return cast(
            torch.Tensor, self.output(attended.reshape(batch, query_count, hidden))
        )


class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + attended)
        return cast(torch.Tensor, self.output_norm(tokens + self.feed_forward(tokens)))


class _DecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        geometry_tokens: torch.Tensor,
        parameter_embedding: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query, geometry_tokens, geometry_tokens, need_weights=False
        )
        query = self.attention_norm(query + attended)
        gamma, beta = self.film(parameter_embedding).chunk(2, dim=-1)
        query = query * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return cast(torch.Tensor, self.output_norm(query + self.feed_forward(query)))


class PiGinotOperator(nn.Module):
    """Faithful, contract-hardened adaptation of the PI-GINOT operator."""

    architecture_id: Final[str] = ARCHITECTURE_ID

    def __init__(
        self,
        config: OperatorConfig,
        *,
        lifting_verification: LiftingVerification | None = None,
    ) -> None:
        super().__init__()
        if type(config) is not OperatorConfig:
            raise TypeError("config must be an OperatorConfig")
        if (
            lifting_verification is not None
            and type(lifting_verification) is not LiftingVerification
        ):
            raise TypeError("lifting_verification must be a LiftingVerification")
        if config.boundary_condition_mode is BoundaryConditionMode.HARD_LIFTING:
            if lifting_verification is None:
                raise ValueError(
                    "hard mode requires verified distance-lifting evidence"
                )
            if (
                lifting_verification.max_boundary_residual
                > config.soft_dirichlet_residual_tolerance
            ):
                raise ValueError("verified distance-lifting residual exceeds tolerance")
        elif lifting_verification is not None:
            raise ValueError("lifting verification is only valid for hard-lifting mode")
        self.config = config
        self.lifting_verification = lifting_verification

        self.boundary_position = FourierFeatures(config.num_fourier_frequencies)
        self.query_position = FourierFeatures(config.num_fourier_frequencies)
        boundary_input_dim = (
            self.boundary_position.output_dim + config.boundary_tag_dim + 2
        )
        self.boundary_projection = nn.Sequential(
            nn.Linear(boundary_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.latent_queries = nn.Parameter(
            torch.empty(config.latent_token_count, config.hidden_dim)
        )
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        self.boundary_cross_attention = _WeightedSetCrossAttention(
            config.hidden_dim, config.num_attention_heads
        )
        self.boundary_norm = nn.LayerNorm(config.hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            _SelfAttentionBlock(config.hidden_dim, config.num_attention_heads)
            for _ in range(config.encoder_self_attention_layers)
        )

        self.query_projection = nn.Sequential(
            nn.Linear(self.query_position.output_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.parameter_encoder = nn.Sequential(
            nn.Linear(config.parameter_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.decoder_blocks = nn.ModuleList(
            _DecoderBlock(config.hidden_dim, config.num_attention_heads)
            for _ in range(config.decoder_cross_attention_layers)
        )
        self.u_head = nn.Linear(config.hidden_dim, 2)
        self.p_head = nn.Linear(config.hidden_dim, 4)

    def encode_boundary(
        self,
        *,
        boundary_xy: torch.Tensor,
        boundary_tags: torch.Tensor,
        boundary_normals: torch.Tensor,
        boundary_weights: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
    ) -> GeometryEncoding:
        """Encode a tagged boundary cloud once for repeated spatial queries."""

        _validate_field(boundary_xy, "boundary_xy", trailing=2)
        _validate_field(
            boundary_tags,
            "boundary_tags",
            trailing=self.config.boundary_tag_dim,
        )
        _validate_field(boundary_normals, "boundary_normals", trailing=2)
        if boundary_xy.shape[:2] != boundary_tags.shape[:2]:
            raise ValueError("boundary_tags must align with boundary_xy")
        if boundary_xy.shape[:2] != boundary_normals.shape[:2]:
            raise ValueError("boundary_normals must align with boundary_xy")
        _require_same_tensor_context(
            boundary_xy, boundary_tags, "boundary_xy", "boundary_tags"
        )
        _require_same_tensor_context(
            boundary_xy, boundary_normals, "boundary_xy", "boundary_normals"
        )
        mask = _boundary_mask(boundary_mask, boundary_xy)
        weights = _boundary_weights(boundary_weights, boundary_xy, mask)
        coordinate_min, coordinate_span = _coordinate_frame(boundary_xy, mask)
        normalized_xy = _normalize_coordinates(
            boundary_xy, coordinate_min, coordinate_span
        )
        features = torch.cat(
            (self.boundary_position(normalized_xy), boundary_tags, boundary_normals),
            dim=-1,
        )
        source = self.boundary_projection(features)
        latents = self.latent_queries.unsqueeze(0).expand(boundary_xy.shape[0], -1, -1)
        attended = self.boundary_cross_attention(latents, source, weights, mask)
        latents = self.boundary_norm(latents + attended)
        for block in self.encoder_blocks:
            latents = block(latents)
        return GeometryEncoding(latents, coordinate_min, coordinate_span)

    def decode_queries(
        self,
        *,
        encoding: GeometryEncoding,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        dirichlet: DirichletData | None = None,
    ) -> OperatorOutput:
        """Decode arbitrary query points using a precomputed geometry latent."""

        if type(encoding) is not GeometryEncoding:
            raise TypeError("encoding must be a GeometryEncoding")
        _validate_field(query_xy, "query_xy", trailing=2)
        _validate_matrix(parameters, "parameters", trailing=self.config.parameter_dim)
        if query_xy.shape[0] != encoding.tokens.shape[0]:
            raise ValueError("query_xy and encoding must share a batch dimension")
        if parameters.shape[0] != query_xy.shape[0]:
            raise ValueError("parameters and query_xy must share a batch dimension")
        _require_same_tensor_context(
            encoding.tokens, query_xy, "encoding.tokens", "query_xy"
        )
        _require_same_tensor_context(query_xy, parameters, "query_xy", "parameters")
        normalized_query = _normalize_coordinates(
            query_xy, encoding.coordinate_min, encoding.coordinate_span
        )
        query = self.query_projection(self.query_position(normalized_query))
        parameter_embedding = self.parameter_encoder(parameters)
        for block in self.decoder_blocks:
            query = block(query, encoding.tokens, parameter_embedding)
        raw_u = self.u_head(query)
        if dirichlet is not None:
            _validate_dirichlet_for_output(dirichlet, raw_u)
        if self.config.boundary_condition_mode is BoundaryConditionMode.HARD_LIFTING:
            if dirichlet is None:
                raise ValueError("hard-lifting inference requires DirichletData")
            raw_u = torch.where(
                dirichlet.component_mask,
                dirichlet.prescribed_u + dirichlet.approximate_distance * raw_u,
                raw_u,
            )
        output_p = self.p_head(query).reshape(*query.shape[:2], 2, 2)
        return OperatorOutput(raw_u, output_p)

    def forward(
        self,
        *,
        boundary_xy: torch.Tensor,
        boundary_tags: torch.Tensor,
        boundary_normals: torch.Tensor,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        boundary_weights: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        dirichlet: DirichletData | None = None,
    ) -> OperatorOutput:
        """Encode the boundary and decode both authoritative output heads."""

        encoding = self.encode_boundary(
            boundary_xy=boundary_xy,
            boundary_tags=boundary_tags,
            boundary_normals=boundary_normals,
            boundary_weights=boundary_weights,
            boundary_mask=boundary_mask,
        )
        return self.decode_queries(
            encoding=encoding,
            query_xy=query_xy,
            parameters=parameters,
            dirichlet=dirichlet,
        )

    def dirichlet_residual(
        self, output: OperatorOutput, data: DirichletData
    ) -> torch.Tensor:
        """Return the maximum component-wise essential-boundary residual."""

        if type(output) is not OperatorOutput:
            raise TypeError("output must be an OperatorOutput")
        if type(data) is not DirichletData:
            raise TypeError("data must be DirichletData")
        _validate_dirichlet_for_output(data, output.u)
        if not bool(data.component_mask.any().item()):
            raise ValueError("DirichletData must select at least one component")
        return torch.abs(output.u - data.prescribed_u)[data.component_mask].amax()


def module_digest(module: PiGinotOperator) -> str:
    """Hash architecture semantics and tensor schema, never learned values."""

    if type(module) is not PiGinotOperator:
        raise TypeError("module must be an exact PiGinotOperator")
    payload: dict[str, object] = {
        "architecture_id": ARCHITECTURE_ID,
        "config": _config_payload(module.config),
        "lifting_verification": (
            None
            if module.lifting_verification is None
            else asdict(module.lifting_verification)
        ),
        "state_schema": [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in sorted(module.state_dict().items())
        ],
    }
    digest = hashlib.blake2b(canonical_json_bytes(payload), digest_size=32)
    return f"blake2b256:{digest.hexdigest()}"


def _config_payload(config: OperatorConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["parameter_names"] = list(config.parameter_names)
    payload["boundary_tag_names"] = list(config.boundary_tag_names)
    payload["boundary_condition_mode"] = config.boundary_condition_mode.value
    return payload


def _validate_names(values: tuple[str, ...], label: str) -> None:
    if type(values) is not tuple or not values:
        raise ValueError(f"{label} must be a non-empty tuple")
    if any(
        type(value) is not str or not value or value.strip() != value
        for value in values
    ):
        raise ValueError(f"{label} must contain non-empty canonical names")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")


def _validate_field(
    value: torch.Tensor,
    name: str,
    *,
    trailing: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout is not torch.strided:
        raise ValueError(f"{name} must be a dense strided tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if value.ndim != 3 or (trailing is not None and value.shape[-1] != trailing):
        suffix = "features" if trailing is None else str(trailing)
        raise ValueError(f"{name} must have shape (batch, count, {suffix})")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_matrix(value: torch.Tensor, name: str, *, trailing: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout is not torch.strided:
        raise ValueError(f"{name} must be a dense strided tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if value.ndim != 2 or value.shape[-1] != trailing or value.numel() == 0:
        raise ValueError(f"{name} must have shape (batch, {trailing})")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _require_same_tensor_context(
    left: torch.Tensor,
    right: torch.Tensor,
    left_name: str,
    right_name: str,
) -> None:
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must share dtype and device")


def _boundary_mask(
    value: torch.Tensor | None, boundary_xy: torch.Tensor
) -> torch.Tensor:
    if value is None:
        return torch.ones(
            boundary_xy.shape[:2], dtype=torch.bool, device=boundary_xy.device
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError("boundary_mask must be a torch.Tensor")
    if value.dtype is not torch.bool:
        raise ValueError("boundary_mask must be boolean")
    if value.shape != boundary_xy.shape[:2]:
        raise ValueError("boundary_mask must have shape (batch, boundary_points)")
    if value.device != boundary_xy.device:
        raise ValueError("boundary_mask and boundary_xy must share a device")
    if not bool(value.any(dim=1).all().item()):
        raise ValueError("each sample must expose at least one boundary point")
    return value


def _boundary_weights(
    value: torch.Tensor | None,
    boundary_xy: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        weights = torch.ones(
            boundary_xy.shape[:2],
            dtype=boundary_xy.dtype,
            device=boundary_xy.device,
        )
    else:
        if not isinstance(value, torch.Tensor):
            raise TypeError("boundary_weights must be a torch.Tensor")
        if value.shape != boundary_xy.shape[:2]:
            raise ValueError("boundary_weights must align with boundary_xy")
        if value.layout is not torch.strided:
            raise ValueError("boundary_weights must be dense and strided")
        _require_same_tensor_context(
            boundary_xy, value, "boundary_xy", "boundary_weights"
        )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("boundary_weights must be finite")
        weights = value
    if bool((weights < 0.0).any().item()):
        raise ValueError("boundary_weights must be non-negative")
    if not bool(((weights * mask).sum(dim=1) > 0.0).all().item()):
        raise ValueError("each sample needs positive visible boundary weight")
    return weights


def _coordinate_frame(
    coordinates: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_mask = mask.unsqueeze(-1)
    minimum = torch.where(
        expanded_mask,
        coordinates,
        torch.full_like(coordinates, torch.inf),
    ).amin(dim=1)
    maximum = torch.where(
        expanded_mask,
        coordinates,
        torch.full_like(coordinates, -torch.inf),
    ).amax(dim=1)
    span = maximum - minimum
    if bool((span <= torch.finfo(coordinates.dtype).eps).any().item()):
        raise ValueError("visible boundary must span both coordinate axes")
    return minimum, span


def _normalize_coordinates(
    coordinates: torch.Tensor,
    minimum: torch.Tensor,
    span: torch.Tensor,
) -> torch.Tensor:
    return 2.0 * (coordinates - minimum.unsqueeze(1)) / span.unsqueeze(1) - 1.0


def _validate_dirichlet_for_output(data: DirichletData, output_u: torch.Tensor) -> None:
    if type(data) is not DirichletData:
        raise TypeError("dirichlet must be DirichletData")
    if data.prescribed_u.shape != output_u.shape:
        raise ValueError("DirichletData must align with displacement output")
    _require_same_tensor_context(output_u, data.prescribed_u, "u", "prescribed_u")


__all__ = (
    "ARCHITECTURE_ID",
    "BoundaryConditionMode",
    "DirichletData",
    "GeometryEncoding",
    "LiftingVerification",
    "OperatorConfig",
    "OperatorOutput",
    "PiGinotOperator",
    "module_digest",
)
