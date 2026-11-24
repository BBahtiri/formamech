"""Tamper-checked 3D structural foundation checkpoint registry (V3-2).

Stores and verifies pretrained multi-CAD ``Surface3DEnergyOperator`` weights +
``GeometryOODDetector3D`` envelopes with explicit epistemic claim ceilings and
one-call zero-shot or budgeted 3D LoRA adaptation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from piginot import __version__
from piginot.contracts import canonical_json_bytes
from piginot.operators.geometry_3d import GeometryOODDetector3D
from piginot.operators.surface_operator_3d import (
    SURFACE_3D_ARCHITECTURE_ID,
    Surface3DConfig,
    Surface3DEnergyOperator,
)

FOUNDATION_CHECKPOINT_SCHEMA: Final = "formamech.foundation-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class FoundationCheckpointSpec:
    """Metadata descriptor for one pretrained 3D structural foundation model."""

    checkpoint_id: str
    family_domain: str
    architecture_id: str
    operator_config: Surface3DConfig
    training_mode: str
    claim_level: str
    weights_sha256: str
    ood_detector_sha256: str | None
    dtype: str
    formamech_version: str = __version__

    def document(self) -> dict[str, Any]:
        return {
            "schema": FOUNDATION_CHECKPOINT_SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "family_domain": self.family_domain,
            "architecture_id": self.architecture_id,
            "operator_config": asdict(self.operator_config),
            "training_mode": self.training_mode,
            "claim_level": self.claim_level,
            "weights_sha256": self.weights_sha256,
            "ood_detector_sha256": self.ood_detector_sha256,
            "dtype": self.dtype,
            "formamech_version": self.formamech_version,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> FoundationCheckpointSpec:
        if doc.get("schema") != FOUNDATION_CHECKPOINT_SCHEMA:
            raise ValueError(
                f"unexpected foundation checkpoint schema {doc.get('schema')!r}"
            )
        op_cfg = Surface3DConfig(**dict(doc["operator_config"]))
        return cls(
            checkpoint_id=str(doc["checkpoint_id"]),
            family_domain=str(doc["family_domain"]),
            architecture_id=str(doc["architecture_id"]),
            operator_config=op_cfg,
            training_mode=str(doc["training_mode"]),
            claim_level=str(doc["claim_level"]),
            weights_sha256=str(doc["weights_sha256"]),
            ood_detector_sha256=(
                None
                if doc.get("ood_detector_sha256") is None
                else str(doc["ood_detector_sha256"])
            ),
            dtype=str(doc.get("dtype", "float64")),
            formamech_version=str(doc.get("formamech_version", __version__)),
        )


@dataclass(frozen=True, slots=True)
class LoadedFoundation3D:
    """Verified in-memory handle to a pretrained 3D structural foundation model."""

    spec: FoundationCheckpointSpec
    operator: Surface3DEnergyOperator
    ood_detector: GeometryOODDetector3D | None


def save_foundation_3d_checkpoint(
    destination_dir: Path | str,
    *,
    checkpoint_id: str,
    family_domain: str,
    operator: Surface3DEnergyOperator,
    ood_detector: GeometryOODDetector3D | None = None,
    training_mode: str = "variational-energy-label-free",
    claim_level: str = "NOT_VERIFIED",
) -> FoundationCheckpointSpec:
    """Persist a tamper-checked 3D foundation checkpoint directory."""
    if claim_level not in ("NOT_VERIFIED", "DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"):
        raise ValueError(f"illegal claim_level {claim_level!r}")
    out = Path(destination_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    state = {k: v.detach().contiguous().cpu() for k, v in operator.state_dict().items()}
    dtype_str = str(next(iter(state.values())).dtype).removeprefix("torch.")
    w_bytes = save_safetensors(
        state,
        metadata={
            "schema": FOUNDATION_CHECKPOINT_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "architecture_id": SURFACE_3D_ARCHITECTURE_ID,
        },
    )
    (out / "weights.safetensors").write_bytes(w_bytes)
    w_sha = hashlib.sha256(w_bytes).hexdigest()

    ood_sha: str | None = None
    if ood_detector is not None:
        ood_bytes = canonical_json_bytes(ood_detector.document()) + b"\n"
        (out / "ood-detector-3d.json").write_bytes(ood_bytes)
        ood_sha = hashlib.sha256(ood_bytes).hexdigest()

    spec = FoundationCheckpointSpec(
        checkpoint_id=checkpoint_id,
        family_domain=family_domain,
        architecture_id=operator.architecture_id,
        operator_config=operator.config,
        training_mode=training_mode,
        claim_level=claim_level,
        weights_sha256=w_sha,
        ood_detector_sha256=ood_sha,
        dtype=dtype_str,
    )
    doc = spec.document()
    doc["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(doc)).hexdigest()
    (out / "foundation.json").write_bytes(canonical_json_bytes(doc) + b"\n")
    return spec


def load_foundation_3d_checkpoint(
    checkpoint_dir: Path | str,
) -> LoadedFoundation3D:
    """Verify SHA-256 integrity and load a pretrained 3D foundation checkpoint."""
    root = Path(checkpoint_dir).resolve()
    raw_doc = json.loads((root / "foundation.json").read_text(encoding="utf-8"))
    recorded_m = raw_doc.get("manifest_sha256")
    without_m = {k: v for k, v in raw_doc.items() if k != "manifest_sha256"}
    if hashlib.sha256(canonical_json_bytes(without_m)).hexdigest() != recorded_m:
        raise ValueError("foundation manifest SHA-256 mismatch (tampered checkpoint)")
    spec = FoundationCheckpointSpec.from_document(without_m)

    w_bytes = (root / "weights.safetensors").read_bytes()
    if hashlib.sha256(w_bytes).hexdigest() != spec.weights_sha256:
        raise ValueError("foundation weights SHA-256 mismatch (tampered safetensors)")

    dtype = torch.float32 if spec.dtype == "float32" else torch.float64
    op = Surface3DEnergyOperator(spec.operator_config).to(dtype=dtype)
    tensors = load_safetensors(w_bytes)
    op.load_state_dict({k: v.to(dtype=dtype) for k, v in tensors.items()}, strict=True)

    ood: GeometryOODDetector3D | None = None
    if spec.ood_detector_sha256 is not None:
        ood_bytes = (root / "ood-detector-3d.json").read_bytes()
        if hashlib.sha256(ood_bytes).hexdigest() != spec.ood_detector_sha256:
            raise ValueError("foundation OOD detector SHA-256 mismatch")
        ood = GeometryOODDetector3D.from_document(json.loads(ood_bytes.decode("utf-8")))

    return LoadedFoundation3D(spec=spec, operator=op, ood_detector=ood)


class FoundationRegistry:
    """Local directory index of tamper-checked 3D structural foundation checkpoints."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir).resolve()

    def list_checkpoints(self) -> tuple[FoundationCheckpointSpec, ...]:
        if not self.root_dir.is_dir():
            return ()
        found: list[FoundationCheckpointSpec] = []
        for sub in sorted(self.root_dir.iterdir()):
            manifest_path = sub / "foundation.json"
            if sub.is_dir() and manifest_path.is_file():
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                without_m = {k: v for k, v in raw.items() if k != "manifest_sha256"}
                found.append(FoundationCheckpointSpec.from_document(without_m))
        return tuple(found)

    def load(self, checkpoint_id: str) -> LoadedFoundation3D:
        for sub in sorted(self.root_dir.iterdir()):
            manifest_path = sub / "foundation.json"
            if sub.is_dir() and manifest_path.is_file():
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                if raw.get("checkpoint_id") == checkpoint_id:
                    return load_foundation_3d_checkpoint(sub)
        raise LookupError(
            f"foundation checkpoint {checkpoint_id!r} not found in {self.root_dir}"
        )


__all__ = (
    "FOUNDATION_CHECKPOINT_SCHEMA",
    "FoundationCheckpointSpec",
    "FoundationRegistry",
    "LoadedFoundation3D",
    "load_foundation_3d_checkpoint",
    "save_foundation_3d_checkpoint",
)
