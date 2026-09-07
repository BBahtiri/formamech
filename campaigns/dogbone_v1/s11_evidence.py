"""Deterministic S11 architecture evidence on the ratified DogBone reference."""

from __future__ import annotations

import pickle
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import torch

from campaigns.dogbone_v1.family import compiled_family
from piginot.contracts import CanonicalMesh
from piginot.operators import (
    ARCHITECTURE_ID,
    OperatorConfig,
    OperatorOutput,
    PiGinotOperator,
    S11GateEvaluation,
    S11GateMeasurements,
    evaluate_s11_gate,
    load_operator_state_dict,
    module_digest,
)

DOGBONE_S11_PARAMETER_NAMES = (
    "L_total",
    "W_grip",
    "W_gauge",
    "R_fillet",
    "E",
    "nu",
    "nominal_axial_strain",
)
_MODEL_SEED = 11011
_PERMUTATION_SEED = 11012
_QUERY_COUNT = 23


class _OperatorInputs(TypedDict):
    boundary_xy: torch.Tensor
    boundary_tags: torch.Tensor
    boundary_normals: torch.Tensor
    boundary_weights: torch.Tensor
    boundary_mask: torch.Tensor
    query_xy: torch.Tensor
    parameters: torch.Tensor


@dataclass(frozen=True, slots=True)
class DogboneS11Evidence:
    """Scalar-only evidence suitable for deterministic gate serialization."""

    architecture_id: str
    module_digest: str
    reference_mesh_id: str
    model_seed: int
    permutation_seed: int
    measurements: S11GateMeasurements
    evaluation: S11GateEvaluation
    mask_control_relative_change: float
    mask_mutant_relative_change: float


class _DeserializationMarker:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return self.marker.write_text, ("DESERIALIZED",)


def build_dogbone_s11_operator() -> PiGinotOperator:
    """Build the fixed S11 adaptation using upstream PI-GINOT dimensions."""

    boundary_tags = tuple(compiled_family().reference_mesh.semantic_tags)
    config = OperatorConfig(
        parameter_names=DOGBONE_S11_PARAMETER_NAMES,
        boundary_tag_names=boundary_tags,
        hidden_dim=64,
        latent_token_count=32,
        num_attention_heads=4,
        num_fourier_frequencies=6,
        encoder_self_attention_layers=3,
        decoder_cross_attention_layers=6,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_MODEL_SEED)
        return PiGinotOperator(config)


@lru_cache(maxsize=2)
def run_dogbone_s11_evidence(
    *, code_coverage_percent: float | None
) -> DogboneS11Evidence:
    """Execute all structural G11 measurements without training a checkpoint."""

    family = compiled_family()
    mesh = family.reference_mesh
    model = build_dogbone_s11_operator().to(dtype=torch.float64).eval()
    low = _operator_inputs(mesh, 320)
    high = _operator_inputs(mesh, 480)
    baseline = model(**low)

    permutation_max = _permutation_metric(model, low, baseline.flat())
    resampling_change = _relative_change(baseline.flat(), model(**high).flat())
    node_order_change = _node_order_metric(model, low, baseline)
    query_batch_change = _query_batch_metric(model, low, baseline)
    reachable = _reachable_parameter_count(model, low)
    mask_control, mask_mutant = _mask_mutant_metrics(model, low, baseline.flat())
    pickle_refused = _pickle_refused_before_deserialization(model)
    measurements = S11GateMeasurements(
        permutation_count=100,
        permutation_max_relative_change=permutation_max,
        resampling_low_count=320,
        resampling_high_count=480,
        resampling_relative_change=resampling_change,
        node_order_relative_change=node_order_change,
        query_batch_relative_change=query_batch_change,
        reachable_parameter_count=reachable,
        required_parameter_count=len(DOGBONE_S11_PARAMETER_NAMES),
        mask_mutant_fired=mask_control < 1.0e-10 and mask_mutant > 1.0e-3,
        pickle_refused_before_deserialization=pickle_refused,
        u_shape_verified=baseline.u.shape == (1, _QUERY_COUNT, 2),
        p_shape_verified=baseline.P.shape == (1, _QUERY_COUNT, 2, 2),
        served_stress_is_predicted=baseline.served_stress is baseline.P,
        code_coverage_percent=code_coverage_percent,
    )
    return DogboneS11Evidence(
        architecture_id=ARCHITECTURE_ID,
        module_digest=module_digest(model),
        reference_mesh_id=mesh.mesh_id,
        model_seed=_MODEL_SEED,
        permutation_seed=_PERMUTATION_SEED,
        measurements=measurements,
        evaluation=evaluate_s11_gate(measurements),
        mask_control_relative_change=mask_control,
        mask_mutant_relative_change=mask_mutant,
    )


def evidence_payload(evidence: DogboneS11Evidence) -> dict[str, object]:
    """Convert measured evidence to a stable JSON-compatible payload."""

    if type(evidence) is not DogboneS11Evidence:
        raise TypeError("evidence must be exact DogboneS11Evidence")
    checks = [
        {
            "id": check.check_id,
            "status": check.status.value,
            "measured": check.measured,
            "threshold": check.threshold,
            "comparison": check.comparison,
            "reason": check.reason,
        }
        for check in evidence.evaluation.checks
    ]
    return {
        "architecture_id": evidence.architecture_id,
        "module_digest": evidence.module_digest,
        "reference_mesh_id": evidence.reference_mesh_id,
        "model_seed": evidence.model_seed,
        "permutation_seed": evidence.permutation_seed,
        "parameter_names": list(DOGBONE_S11_PARAMETER_NAMES),
        "measurements": asdict(evidence.measurements),
        "diagnostics": {
            "mask_control_relative_change": evidence.mask_control_relative_change,
            "mask_mutant_relative_change": evidence.mask_mutant_relative_change,
        },
        "gate": {
            "verdict": evidence.evaluation.verdict.value,
            "checks": checks,
        },
    }


def _operator_inputs(mesh: CanonicalMesh, boundary_count: int) -> _OperatorInputs:
    boundary_xy, tags, normals, weights = _sample_boundary(mesh, boundary_count)
    query_indices = (
        torch.linspace(0, len(mesh.nodes) - 1, _QUERY_COUNT, dtype=torch.float64)
        .round()
        .to(torch.long)
    )
    query_xy = torch.tensor(mesh.nodes, dtype=torch.float64)[query_indices].unsqueeze(0)
    parameters = torch.ones(1, len(DOGBONE_S11_PARAMETER_NAMES), dtype=torch.float64)
    return {
        "boundary_xy": boundary_xy,
        "boundary_tags": tags,
        "boundary_normals": normals,
        "boundary_weights": weights,
        "boundary_mask": torch.ones(1, boundary_count, dtype=torch.bool),
        "query_xy": query_xy,
        "parameters": parameters,
    }


def _sample_boundary(
    mesh: CanonicalMesh, count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = torch.tensor(mesh.nodes, dtype=torch.float64)
    facets = torch.tensor(mesh.boundary_facets, dtype=torch.long)
    starts = nodes[facets[:, 0]]
    ends = nodes[facets[:, 1]]
    tangents = ends - starts
    lengths = torch.linalg.vector_norm(tangents, dim=1)
    cumulative = torch.cumsum(lengths, dim=0)
    total = cumulative[-1]
    locations = (torch.arange(count, dtype=torch.float64) + 0.5) * total / count
    facet_ids = torch.searchsorted(cumulative, locations, right=False)
    previous = torch.cat((torch.zeros(1, dtype=torch.float64), cumulative[:-1]))
    fractions = (locations - previous[facet_ids]) / lengths[facet_ids]
    points = starts[facet_ids] + fractions.unsqueeze(1) * tangents[facet_ids]

    unit_tangents = tangents[facet_ids] / lengths[facet_ids].unsqueeze(1)
    normals = torch.stack((unit_tangents[:, 1], -unit_tangents[:, 0]), dim=1)
    centroid = nodes.mean(dim=0)
    outward = torch.sum(normals * (points - centroid), dim=1) >= 0.0
    normals = torch.where(outward.unsqueeze(1), normals, -normals)

    tag_names = tuple(mesh.semantic_tags)
    tag_by_facet = torch.empty(len(mesh.boundary_facets), dtype=torch.long)
    for tag_id, tag_name in enumerate(tag_names):
        tag_by_facet[list(mesh.semantic_tags[tag_name])] = tag_id
    tags = torch.nn.functional.one_hot(
        tag_by_facet[facet_ids], num_classes=len(tag_names)
    ).to(torch.float64)
    weights = torch.full((1, count), 1.0 / count, dtype=torch.float64)
    return points.unsqueeze(0), tags.unsqueeze(0), normals.unsqueeze(0), weights


def _permutation_metric(
    model: PiGinotOperator,
    inputs: _OperatorInputs,
    baseline: torch.Tensor,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(_PERMUTATION_SEED)
    maximum = 0.0
    with torch.no_grad():
        for _ in range(100):
            order = torch.randperm(inputs["boundary_xy"].shape[1], generator=generator)
            permuted = _permute_boundary(inputs, order)
            maximum = max(
                maximum,
                _relative_change(baseline, model(**permuted).flat()),
            )
    return maximum


def _permute_boundary(inputs: _OperatorInputs, order: torch.Tensor) -> _OperatorInputs:
    result = inputs.copy()
    result["boundary_xy"] = inputs["boundary_xy"][:, order]
    result["boundary_tags"] = inputs["boundary_tags"][:, order]
    result["boundary_normals"] = inputs["boundary_normals"][:, order]
    result["boundary_weights"] = inputs["boundary_weights"][:, order]
    result["boundary_mask"] = inputs["boundary_mask"][:, order]
    return result


def _node_order_metric(
    model: PiGinotOperator,
    inputs: _OperatorInputs,
    baseline: OperatorOutput,
) -> float:
    order = torch.arange(_QUERY_COUNT - 1, -1, -1)
    permuted_inputs = inputs.copy()
    permuted_inputs["query_xy"] = inputs["query_xy"][:, order]
    permuted = model(**permuted_inputs)
    return max(
        _relative_change(baseline.u, permuted.u[:, order]),
        _relative_change(baseline.P, permuted.P[:, order]),
    )


def _query_batch_metric(
    model: PiGinotOperator,
    inputs: _OperatorInputs,
    baseline: OperatorOutput,
) -> float:
    encoding = model.encode_boundary(
        boundary_xy=inputs["boundary_xy"],
        boundary_tags=inputs["boundary_tags"],
        boundary_normals=inputs["boundary_normals"],
        boundary_weights=inputs["boundary_weights"],
        boundary_mask=inputs["boundary_mask"],
    )
    pieces = [
        model.decode_queries(
            encoding=encoding,
            query_xy=query,
            parameters=inputs["parameters"],
        )
        for query in torch.split(inputs["query_xy"], [7, 8, 8], dim=1)
    ]
    return max(
        _relative_change(baseline.u, torch.cat([part.u for part in pieces], dim=1)),
        _relative_change(baseline.P, torch.cat([part.P for part in pieces], dim=1)),
    )


def _reachable_parameter_count(model: PiGinotOperator, inputs: _OperatorInputs) -> int:
    parameters = inputs["parameters"].clone().requires_grad_(True)
    active = inputs.copy()
    active["parameters"] = parameters
    output = model(**active)
    u_gradient = torch.autograd.grad(
        output.u.square().sum(), parameters, retain_graph=True
    )[0]
    p_gradient = torch.autograd.grad(output.P.square().sum(), parameters)[0]
    reachable = (torch.abs(u_gradient) > 1.0e-12) & (torch.abs(p_gradient) > 1.0e-12)
    return int(torch.count_nonzero(reachable).item())


def _mask_mutant_metrics(
    model: PiGinotOperator,
    inputs: _OperatorInputs,
    baseline: torch.Tensor,
) -> tuple[float, float]:
    pad_count = 16
    padded = inputs.copy()
    padded["boundary_xy"] = torch.cat(
        (
            inputs["boundary_xy"],
            torch.full_like(inputs["boundary_xy"][:, :pad_count], 1000.0),
        ),
        dim=1,
    )
    padded["boundary_tags"] = torch.cat(
        (
            inputs["boundary_tags"],
            torch.full_like(inputs["boundary_tags"][:, :pad_count], 1000.0),
        ),
        dim=1,
    )
    padded["boundary_normals"] = torch.cat(
        (
            inputs["boundary_normals"],
            torch.full_like(inputs["boundary_normals"][:, :pad_count], 1000.0),
        ),
        dim=1,
    )
    padded["boundary_weights"] = torch.cat(
        (
            inputs["boundary_weights"],
            torch.ones_like(inputs["boundary_weights"][:, :pad_count]),
        ),
        dim=1,
    )
    padded["boundary_mask"] = torch.cat(
        (
            inputs["boundary_mask"],
            torch.zeros(1, pad_count, dtype=torch.bool),
        ),
        dim=1,
    )
    control = _relative_change(baseline, model(**padded).flat())
    mutant = padded.copy()
    mutant["boundary_mask"] = torch.ones_like(padded["boundary_mask"])
    return control, _relative_change(baseline, model(**mutant).flat())


def _pickle_refused_before_deserialization(model: PiGinotOperator) -> bool:
    with tempfile.TemporaryDirectory(prefix="piginot-s11-") as directory:
        root = Path(directory)
        marker = root / "executed.txt"
        malicious = root / "malicious.safetensors"
        malicious.write_bytes(pickle.dumps(_DeserializationMarker(marker)))
        try:
            load_operator_state_dict(malicious, model=model)
        except ValueError:
            return not marker.exists()
        return False


def _relative_change(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(left - right)
    denominator = torch.clamp(torch.linalg.vector_norm(left), min=1.0e-12)
    return float((numerator / denominator).detach().item())


__all__ = (
    "DOGBONE_S11_PARAMETER_NAMES",
    "DogboneS11Evidence",
    "build_dogbone_s11_operator",
    "evidence_payload",
    "run_dogbone_s11_evidence",
)
