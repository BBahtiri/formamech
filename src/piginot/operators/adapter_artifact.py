"""Tamper-checked ``safetensors`` artifact persistence for P2 multi-CAD operators & LoRA.

Provides portable, pickle-free save/load routines for:
* ``BoundaryEnergyOperator`` base checkpoints (``weights.safetensors`` + ``model.json``),
* ``AdaptedBoundaryEnergyOperator`` low-rank deltas (``adapter.safetensors`` + ``adapter.json``),
* ``GeometryOODDetector`` envelopes (``ood-detector.json``).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, cast

import torch

from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.operators.boundary_energy_operator import (
    BoundaryEnergyConfig,
    BoundaryEnergyOperator,
    boundary_energy_module_digest,
)
from piginot.operators.energy_operator import EnergyOperatorConfig
from piginot.operators.geometry_tokenizer import (
    GeometryOODDetector,
    GeometryOODReport,
)
from piginot.operators.low_rank_adapter import (
    AdaptationResult,
    AdaptedBoundaryEnergyOperator,
    LowRankAdapterConfig,
)

BOUNDARY_CHECKPOINT_SCHEMA: Final = "piginot.boundary-operator-checkpoint-v1"
GEOMETRY_ADAPTER_ARTIFACT_SCHEMA: Final = "piginot.geometry-adapter-artifact-v1"
GEOMETRY_OOD_ARTIFACT_SCHEMA: Final = "piginot.geometry-ood-detector-v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_boundary_operator_checkpoint(
    directory: Path,
    operator: BoundaryEnergyOperator,
) -> dict[str, Any]:
    """Save a ``BoundaryEnergyOperator`` as ``safetensors`` + canonical manifest."""
    from safetensors.torch import save as save_tensors

    if type(operator) is not BoundaryEnergyOperator:
        raise TypeError("operator must be an exact BoundaryEnergyOperator")
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in operator.state_dict().items()}
    weights_bytes = save_tensors(state)
    weights_path = directory / "weights.safetensors"
    weights_path.write_bytes(weights_bytes)

    energy_cfg = asdict(operator.config.energy)
    energy_cfg["parameter_names"] = list(operator.config.energy.parameter_names)
    manifest: dict[str, Any] = {
        "schema": BOUNDARY_CHECKPOINT_SCHEMA,
        "base_digest": boundary_energy_module_digest(operator),
        "weights_file": "weights.safetensors",
        "weights_sha256": _sha256_bytes(weights_bytes),
        "config": {
            "energy": energy_cfg,
            "boundary_feature_names": list(operator.config.boundary_feature_names),
        },
        "dtype": str(next(operator.parameters()).dtype),
    }
    (directory / "model.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_boundary_operator_checkpoint(
    directory: Path,
) -> BoundaryEnergyOperator:
    """Verify SHA-256 + module digest and reload a ``BoundaryEnergyOperator``."""
    from safetensors.torch import load as load_tensors

    parsed = strict_json_loads((directory / "model.json").read_bytes())
    if (
        not isinstance(parsed, dict)
        or parsed.get("schema") != BOUNDARY_CHECKPOINT_SCHEMA
    ):
        raise ValueError("invalid boundary operator checkpoint manifest")
    manifest = cast(dict[str, Any], parsed)
    weights_path = directory / str(manifest["weights_file"])
    raw = weights_path.read_bytes()
    if _sha256_bytes(raw) != manifest["weights_sha256"]:
        raise ValueError("boundary operator weights SHA-256 mismatch")

    cfg_doc = cast(dict[str, Any], manifest["config"])
    e_doc = dict(cast(dict[str, Any], cfg_doc["energy"]))
    e_doc["parameter_names"] = tuple(str(v) for v in e_doc["parameter_names"])
    b_cfg = BoundaryEnergyConfig(
        energy=EnergyOperatorConfig(**e_doc),
        boundary_feature_names=tuple(str(v) for v in cfg_doc["boundary_feature_names"]),
    )
    dtype = torch.float32 if manifest.get("dtype") == "torch.float32" else torch.float64
    op = BoundaryEnergyOperator(b_cfg).to(dtype)
    tensors = load_tensors(raw)
    op.load_state_dict(dict(tensors), strict=True)
    if boundary_energy_module_digest(op) != manifest["base_digest"]:
        raise ValueError("reloaded boundary operator module digest mismatch")
    return op


def save_geometry_ood_detector(
    path: Path,
    detector: GeometryOODDetector,
) -> dict[str, Any]:
    """Persist a fitted ``GeometryOODDetector`` as canonical JSON with SHA-256."""
    body: dict[str, Any] = {
        "schema": GEOMETRY_OOD_ARTIFACT_SCHEMA,
        "descriptor_mean": list(detector.descriptor_mean),
        "descriptor_inv_cov": [list(row) for row in detector.descriptor_inv_cov],
        "descriptor_threshold": detector.descriptor_threshold,
        "supported_hole_counts": list(detector.supported_hole_counts),
        "token_mean": None
        if detector.token_mean is None
        else list(detector.token_mean),
        "token_inv_var": (
            None if detector.token_inv_var is None else list(detector.token_inv_var)
        ),
        "token_threshold": detector.token_threshold,
    }
    doc = {**body, "detector_sha256": _sha256_bytes(canonical_json_bytes(body))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(doc) + b"\n")
    return doc


def load_geometry_ood_detector(path: Path) -> GeometryOODDetector:
    """Verify SHA-256 and load a persisted ``GeometryOODDetector``."""
    parsed = strict_json_loads(path.read_bytes())
    if (
        not isinstance(parsed, dict)
        or parsed.get("schema") != GEOMETRY_OOD_ARTIFACT_SCHEMA
    ):
        raise ValueError("invalid GeometryOODDetector artifact")
    raw = cast(dict[str, Any], parsed)
    expected_sha = raw.get("detector_sha256")
    body: dict[str, Any] = {k: v for k, v in raw.items() if k != "detector_sha256"}
    if _sha256_bytes(canonical_json_bytes(body)) != expected_sha:
        raise ValueError("GeometryOODDetector artifact SHA-256 mismatch")
    return GeometryOODDetector(
        descriptor_mean=tuple(float(v) for v in body["descriptor_mean"]),
        descriptor_inv_cov=tuple(
            tuple(float(v) for v in row) for row in body["descriptor_inv_cov"]
        ),
        descriptor_threshold=float(body["descriptor_threshold"]),
        supported_hole_counts=tuple(int(v) for v in body["supported_hole_counts"]),
        token_mean=(
            None
            if body["token_mean"] is None
            else tuple(float(v) for v in body["token_mean"])
        ),
        token_inv_var=(
            None
            if body["token_inv_var"] is None
            else tuple(float(v) for v in body["token_inv_var"])
        ),
        token_threshold=(
            None if body["token_threshold"] is None else float(body["token_threshold"])
        ),
    )


def save_geometry_adapter(
    directory: Path,
    *,
    adapted_operator: AdaptedBoundaryEnergyOperator,
    result: AdaptationResult,
    ood_report: GeometryOODReport | None = None,
) -> dict[str, Any]:
    """Write a tamper-checked low-rank adapter delta sidecar (``safetensors`` + JSON)."""
    from safetensors.torch import save as save_tensors

    directory.mkdir(parents=True, exist_ok=True)
    adapter_tensors = {
        k: v.detach().cpu().contiguous()
        for k, v in adapted_operator.adapter_state_dict().items()
    }
    raw_tensors = save_tensors(adapter_tensors)
    (directory / "adapter.safetensors").write_bytes(raw_tensors)

    body: dict[str, Any] = {
        "schema": GEOMETRY_ADAPTER_ARTIFACT_SCHEMA,
        "base_digest": adapted_operator.base_digest,
        "adapter_digest": adapted_operator.adapter_digest(),
        "adapter_file": "adapter.safetensors",
        "adapter_sha256": _sha256_bytes(raw_tensors),
        "adapter_config": adapted_operator.adapter_config.document(),
        "adapted_layers": list(adapted_operator.adapted_layers),
        "result": result.document(),
        "ood_report": None if ood_report is None else ood_report.document(),
        "claim_level": result.claim_level,
    }
    manifest = {**body, "manifest_sha256": _sha256_bytes(canonical_json_bytes(body))}
    (directory / "adapter.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_geometry_adapter(
    directory: Path,
    base_operator: BoundaryEnergyOperator,
) -> tuple[AdaptedBoundaryEnergyOperator, dict[str, Any]]:
    """Verify integrity + base-model binding and attach a low-rank adapter delta."""
    from safetensors.torch import load as load_tensors

    parsed = strict_json_loads((directory / "adapter.json").read_bytes())
    if (
        not isinstance(parsed, dict)
        or parsed.get("schema") != GEOMETRY_ADAPTER_ARTIFACT_SCHEMA
    ):
        raise ValueError("invalid geometry adapter manifest")
    raw_manifest = cast(dict[str, Any], parsed)
    expected_manifest_sha = raw_manifest.get("manifest_sha256")
    body: dict[str, Any] = {
        k: v for k, v in raw_manifest.items() if k != "manifest_sha256"
    }
    if _sha256_bytes(canonical_json_bytes(body)) != expected_manifest_sha:
        raise ValueError("geometry adapter manifest SHA-256 mismatch")

    actual_base = boundary_energy_module_digest(base_operator)
    if body["base_digest"] != actual_base:
        raise ValueError("geometry adapter belongs to a different base operator digest")

    raw_tensors = (directory / str(body["adapter_file"])).read_bytes()
    if _sha256_bytes(raw_tensors) != body["adapter_sha256"]:
        raise ValueError("geometry adapter safetensors SHA-256 mismatch")

    cfg_doc = dict(body["adapter_config"])
    cfg_doc.pop("schema", None)
    cfg_doc["target_substrings"] = tuple(cfg_doc["target_substrings"])
    adapter_cfg = LowRankAdapterConfig(**cfg_doc)
    adapted = AdaptedBoundaryEnergyOperator(base_operator, adapter_cfg)
    adapted.load_adapter_state_dict(dict(load_tensors(raw_tensors)))
    if adapted.adapter_digest() != body["adapter_digest"]:
        raise ValueError("reloaded geometry adapter digest mismatch")
    return adapted, raw_manifest


__all__ = (
    "BOUNDARY_CHECKPOINT_SCHEMA",
    "GEOMETRY_ADAPTER_ARTIFACT_SCHEMA",
    "GEOMETRY_OOD_ARTIFACT_SCHEMA",
    "load_boundary_operator_checkpoint",
    "load_geometry_adapter",
    "load_geometry_ood_detector",
    "save_boundary_operator_checkpoint",
    "save_geometry_adapter",
    "save_geometry_ood_detector",
)
