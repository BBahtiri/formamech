"""Boundary-measure geometry encoding for the point-local energy decoder.

Geometry is encoded once from a weighted boundary point set. The resulting
tokens join the existing parameter tokens; the energy decoder, exact lifting,
and constitutive material kernels retain their existing behavior.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, Final

import torch
from torch import nn

from piginot.contracts import MaterialSpec
from piginot.contracts.canonical import canonical_json_bytes
from piginot.lifting._checks import (
    require_same_context,
    validate_field,
    validate_matrix,
)
from piginot.lifting.base import AffineDirichletLifting, DirichletLifting
from piginot.operators.architecture import OperatorOutput, _WeightedSetCrossAttention
from piginot.operators.energy_operator import (
    ENERGY_ARCHITECTURE_ID,
    EnergyFormOperator,
    EnergyOperatorConfig,
    QueryFrame,
    constitutive_in_plane_stress,
)

BOUNDARY_ENERGY_ARCHITECTURE_ID: Final = "pi-ginot-boundary-energy-film-v1"


@dataclass(frozen=True, slots=True)
class BoundaryEnergyConfig:
    """Boundary channels and the unchanged energy decoder configuration."""

    __canonical_namespace__: ClassVar[str] = "piginot.operators"

    energy: EnergyOperatorConfig
    boundary_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.energy) is not EnergyOperatorConfig:
            raise TypeError("energy must be an EnergyOperatorConfig")
        names = self.boundary_feature_names
        if (
            type(names) is not tuple
            or not names
            or any(
                type(name) is not str or not name or name.strip() != name
                for name in names
            )
            or len(set(names)) != len(names)
        ):
            raise ValueError("boundary_feature_names must be unique canonical names")

    @property
    def boundary_feature_dim(self) -> int:
        return len(self.boundary_feature_names)


@dataclass(frozen=True, slots=True)
class BoundaryEnergyEncoding:
    """Reusable geometry tokens and their authoritative physical query frame."""

    tokens: torch.Tensor
    frame: QueryFrame

    def __post_init__(self) -> None:
        width = (
            self.tokens.shape[-1]
            if isinstance(self.tokens, torch.Tensor) and self.tokens.ndim > 0
            else 0
        )
        validate_field(self.tokens, "tokens", trailing=width)
        _validate_frame(self.frame, self.tokens)


def _validate_frame(frame: QueryFrame, points: torch.Tensor) -> None:
    if type(frame) is not QueryFrame:
        raise TypeError("frame must be a QueryFrame")
    if frame.origin.shape[0] != points.shape[0]:
        raise ValueError("frame and points must share a batch dimension")
    require_same_context(points, frame.origin, "points", "frame")


class BoundaryEnergyOperator(nn.Module):
    """Weighted point-set geometry encoder with a constitutive energy decoder."""

    def __init__(self, config: BoundaryEnergyConfig) -> None:
        super().__init__()
        if type(config) is not BoundaryEnergyConfig:
            raise TypeError("config must be a BoundaryEnergyConfig")
        self.config = config
        width = config.energy.hidden_dim
        self.boundary_projection = nn.Sequential(
            nn.Linear(4 + config.boundary_feature_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.geometry_queries = nn.Parameter(
            torch.randn(config.energy.latent_token_count, width) / math.sqrt(width)
        )
        self.boundary_attention = _WeightedSetCrossAttention(
            width, config.energy.num_attention_heads
        )
        self.boundary_norm = nn.LayerNorm(width)
        self.decoder = EnergyFormOperator(config.energy)

    @property
    def architecture_id(self) -> str:
        return BOUNDARY_ENERGY_ARCHITECTURE_ID

    def encode_boundary(
        self,
        *,
        boundary_xy: torch.Tensor,
        boundary_normals: torch.Tensor,
        boundary_features: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        frame: QueryFrame,
    ) -> BoundaryEnergyEncoding:
        """Encode physical coordinates, unit normals, and generic BC features.

        ``weights`` supplies the nonnegative boundary measure, and ``mask``
        selects real points. Every sample needs positive unmasked measure.
        Padded entries must remain finite but their normals need not be unit.
        Reuse the encoding for query chunks within the current model pass.
        Training encodings belong to the current autograd graph; rebuild
        after backward consumes it. Inference encodings remain valid only
        while weights, boundary inputs, frame, dtype and device stay fixed.
        """

        validate_field(boundary_xy, "boundary_xy", trailing=2)
        validate_field(boundary_normals, "boundary_normals", trailing=2)
        validate_field(
            boundary_features,
            "boundary_features",
            trailing=self.config.boundary_feature_dim,
        )
        for name, field in (
            ("boundary_normals", boundary_normals),
            ("boundary_features", boundary_features),
        ):
            if field.shape[:2] != boundary_xy.shape[:2]:
                raise ValueError(f"{name} must align with boundary_xy")
            require_same_context(boundary_xy, field, "boundary_xy", name)
        _validate_frame(frame, boundary_xy)
        require_same_context(boundary_xy, self.geometry_queries, "boundary_xy", "model")
        usable = _validate_measure(weights, mask, boundary_xy)
        norm = torch.linalg.vector_norm(boundary_normals, dim=-1)
        tolerance = 1e-5 if boundary_xy.dtype is torch.float32 else 1e-10
        if bool((torch.abs(norm[usable] - 1.0) > tolerance).any().item()):
            raise ValueError("unmasked positive-measure boundary_normals must be unit")
        active = usable.unsqueeze(-1)
        features = torch.cat(
            (
                frame.normalize(
                    torch.where(active, boundary_xy, frame.origin.unsqueeze(1))
                ),
                torch.where(active, boundary_normals, 0.0),
                torch.where(active, boundary_features, 0.0),
            ),
            dim=-1,
        )
        source = self.boundary_projection(features)
        measure = torch.where(usable, weights, 0.0)
        measure = measure / measure.amax(dim=1, keepdim=True)
        queries = self.geometry_queries.unsqueeze(0).expand(
            boundary_xy.shape[0], -1, -1
        )
        tokens = self.boundary_norm(
            queries + self.boundary_attention(queries, source, measure, mask)
        )
        return BoundaryEnergyEncoding(tokens, frame)

    def displacement(
        self,
        *,
        encoding: BoundaryEnergyEncoding,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        lifting: DirichletLifting,
    ) -> torch.Tensor:
        """Apply the existing point-local decoder and exact affine lifting."""

        if type(encoding) is not BoundaryEnergyEncoding:
            raise TypeError("encoding must be a BoundaryEnergyEncoding")
        validate_field(query_xy, "query_xy", trailing=2)
        validate_matrix(
            parameters, "parameters", trailing=self.config.energy.parameter_dim
        )
        if (
            encoding.tokens.shape[0] != query_xy.shape[0]
            or parameters.shape[0] != query_xy.shape[0]
        ):
            raise ValueError(
                "encoding, parameters and query_xy must share a batch dimension"
            )
        expected = (
            self.config.energy.latent_token_count,
            self.config.energy.hidden_dim,
        )
        if tuple(encoding.tokens.shape[1:]) != expected:
            raise ValueError(
                "encoding tokens must match the configured token count and width"
            )
        for name, value in (
            ("encoding", encoding.tokens),
            ("parameters", parameters),
            ("frame", encoding.frame.origin),
            ("model", self.geometry_queries),
        ):
            require_same_context(query_xy, value, "query_xy", name)
        tokens = torch.cat(
            (encoding.tokens, self.decoder.encode_parameters(parameters)), dim=1
        )
        phi = self.decoder.correction(tokens, encoding.frame.normalize(query_xy))
        affine = lifting(query_xy)
        if type(affine) is not AffineDirichletLifting:
            raise TypeError("lifting must return an AffineDirichletLifting")
        if affine.offset.shape != phi.shape:
            raise ValueError("lifting must align with the query points")
        require_same_context(query_xy, affine.offset, "query_xy", "lifting")
        return affine.offset + affine.scale * phi

    def displacement_and_gradient(
        self,
        *,
        encoding: BoundaryEnergyEncoding,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        lifting: DirichletLifting,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return displacement and its point-local physical coordinate gradient."""

        query = (
            query_xy
            if query_xy.requires_grad
            else query_xy.detach().clone().requires_grad_(True)
        )
        with torch.enable_grad():
            displacement = self.displacement(
                encoding=encoding,
                query_xy=query,
                parameters=parameters,
                lifting=lifting,
            )
            rows = [
                torch.autograd.grad(
                    displacement[..., component].sum(),
                    query,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]
                for component in range(2)
            ]
        gradient = torch.stack(rows, dim=-2)
        if not create_graph:
            return displacement.detach(), gradient.detach()
        return displacement, gradient

    def forward(
        self,
        *,
        encoding: BoundaryEnergyEncoding,
        query_xy: torch.Tensor,
        parameters: torch.Tensor,
        lifting: DirichletLifting,
        materials: Sequence[MaterialSpec],
    ) -> OperatorOutput:
        """Serve constitutive plane-stress first Piola tensors for each material."""

        if len(materials) != query_xy.shape[0]:
            raise ValueError("materials must provide one MaterialSpec per sample")
        displacement, gradient = self.displacement_and_gradient(
            encoding=encoding,
            query_xy=query_xy,
            parameters=parameters,
            lifting=lifting,
            create_graph=False,
        )
        stresses = tuple(
            constitutive_in_plane_stress(gradient[index], material)
            for index, material in enumerate(materials)
        )
        return OperatorOutput(displacement, torch.stack(stresses, dim=0))


def _validate_measure(
    weights: torch.Tensor, mask: torch.Tensor, boundary_xy: torch.Tensor
) -> torch.Tensor:
    validate_matrix(weights, "weights", trailing=boundary_xy.shape[1])
    if weights.shape[0] != boundary_xy.shape[0]:
        raise ValueError("weights must align with boundary_xy")
    require_same_context(boundary_xy, weights, "boundary_xy", "weights")
    if bool((weights < 0.0).any().item()):
        raise ValueError("weights must be nonnegative")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if (
        mask.layout is not torch.strided
        or mask.dtype is not torch.bool
        or mask.shape != weights.shape
        or mask.device != boundary_xy.device
    ):
        raise ValueError("mask must be a bool tensor aligned with boundary_xy")
    usable = mask & (weights > 0.0)
    if not bool(usable.any(dim=1).all().item()):
        raise ValueError("every sample needs positive unmasked boundary measure")
    return usable


def boundary_energy_module_digest(module: BoundaryEnergyOperator) -> str:
    """Bind geometry/decoder semantics, configuration and schema, never values."""

    if type(module) is not BoundaryEnergyOperator:
        raise TypeError("module must be an exact BoundaryEnergyOperator")
    config = asdict(module.config)
    config["boundary_feature_names"] = list(module.config.boundary_feature_names)
    config["energy"]["parameter_names"] = list(module.config.energy.parameter_names)
    payload: dict[str, object] = {
        "architecture_id": module.architecture_id,
        "decoder_architecture_id": ENERGY_ARCHITECTURE_ID,
        "config": config,
        "boundary_encoding": "explicit-measure-weighted-set-attention-unit-normals",
        "coordinate_frame": "authoritative-origin-extent-affine-minus-one-to-one",
        "conditioning": "concatenated-geometry-and-parameter-tokens",
        "displacement": "exact-affine-dirichlet-lifting-point-local-query-decoder",
        "served_stress": "registered-material-constitutive-plane-stress",
        "state_schema": [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in sorted(module.state_dict().items())
        ],
    }
    digest = hashlib.blake2b(canonical_json_bytes(payload), digest_size=32)
    return f"blake2b256:{digest.hexdigest()}"


__all__ = (
    "BOUNDARY_ENERGY_ARCHITECTURE_ID",
    "BoundaryEnergyConfig",
    "BoundaryEnergyEncoding",
    "BoundaryEnergyOperator",
    "boundary_energy_module_digest",
)
