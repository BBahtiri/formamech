"""ONNX export wrapper and numerical parity verification for FormaMech bundles.

Wraps a verified bundle's frozen ``EnergyFormOperator``, conditioning
normalization, ``QueryFrame``, and exact ``DirichletLifting`` into a pure
tensor-in / tensor-out ``nn.Module`` suitable for ``torch.onnx.export`` and
ONNX Runtime parity checking.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn

from piginot.contracts import canonical_json_bytes
from piginot.operators.energy_operator import EnergyFormOperator, QueryFrame

ONNX_EXPORT_SCHEMA: Final = "formamech.onnx-export-v1"


@dataclass(frozen=True, slots=True)
class OnnxExportResult:
    """Auditable record of one exported ONNX model and its parity verification."""

    onnx_path: Path
    metadata_path: Path
    bundle_manifest_sha256: str
    onnx_sha256: str
    opset_version: int
    input_names: tuple[str, ...]
    parity_checked: bool
    parity_max_abs_error: float | None

    def document(self) -> dict[str, Any]:
        return {
            "schema": ONNX_EXPORT_SCHEMA,
            "onnx_path": str(self.onnx_path),
            "metadata_path": str(self.metadata_path),
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "onnx_sha256": self.onnx_sha256,
            "opset_version": self.opset_version,
            "input_names": list(self.input_names),
            "parity_checked": self.parity_checked,
            "parity_max_abs_error": self.parity_max_abs_error,
        }


class OnnxExportModule(nn.Module):
    """Frozen PyTorch module mapping ``(query_xy, normalized_params) -> u``."""

    def __init__(
        self,
        operator: EnergyFormOperator,
        *,
        origin: torch.Tensor,
        extent: torch.Tensor,
        lifting: Any,
        input_names: tuple[str, ...],
        cond_means: tuple[float, ...],
        cond_stds: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.operator = operator
        self.lifting = lifting
        self.input_names = input_names
        self.register_buffer("origin", origin.detach().clone())
        self.register_buffer("extent", extent.detach().clone())
        self.register_buffer("cond_means", torch.tensor(cond_means, dtype=origin.dtype))
        self.register_buffer("cond_stds", torch.tensor(cond_stds, dtype=origin.dtype))

    def forward(
        self, query_xy: torch.Tensor, normalized_params: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate exact-lifting displacement ``u`` at ``query_xy`` shape ``[N, 2]``."""
        origin = self.origin
        extent = self.extent
        assert isinstance(origin, torch.Tensor)
        assert isinstance(extent, torch.Tensor)
        frame = QueryFrame(origin=origin, extent=extent)
        pts = query_xy.unsqueeze(0)
        params = normalized_params.unsqueeze(0)
        u = self.operator.displacement(
            query_xy=pts,
            parameters=params,
            frame=frame,
            lifting=self.lifting,
        )
        return u[0]


def build_onnx_export_module(
    bundle_path: Path | str,
    *,
    reference_inputs: Mapping[str, Any] | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[OnnxExportModule, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build a frozen ``OnnxExportModule`` plus canonical sample tensors for a bundle."""
    import copy

    from piginot.bundle import load_bundle
    from piginot.bundle.inference import _Request
    from piginot.cli import parse_request

    bundle = load_bundle(Path(bundle_path))
    base_op = getattr(bundle.serving, "_model", None)
    if not isinstance(base_op, EnergyFormOperator):
        raise ValueError(
            "ONNX export currently supports EnergyFormOperator bundles only"
        )

    if reference_inputs is None:
        raw_inputs: dict[str, Any] = {
            b.name: {
                "value": 0.5
                * (float(b.lower_bound.value) + float(b.upper_bound.value)),
                "unit": b.lower_bound.unit.value,
            }
            for b in bundle.envelope.primitive_bounds
        }
        parsed_inputs, _ = parse_request({"inputs": raw_inputs})
    else:
        parsed_inputs, _ = parse_request({"inputs": dict(reference_inputs)})

    geom = bundle.geometry(parsed_inputs)
    mesh = geom.mesh if geom.mesh is not None else bundle.reference_mesh
    req = _Request(bundle, parsed_inputs, mesh)

    op_copy = copy.deepcopy(base_op).to(dtype=dtype).eval()
    for p in op_copy.parameters():
        p.requires_grad_(False)

    # Convert lifting tensors to target dtype (supporting frozen dataclasses)
    lifting = copy.deepcopy(req.lifting)
    for attr in ("extent", "displacement", "origin", "offset", "scale"):
        val = getattr(lifting, attr, None)
        if isinstance(val, torch.Tensor):
            object.__setattr__(lifting, attr, val.to(dtype=dtype))

    cond_means = tuple(float(c.mean) for c in bundle.model.conditioning)
    cond_stds = tuple(float(c.standard_deviation) for c in bundle.model.conditioning)
    export_mod = OnnxExportModule(
        op_copy,
        origin=req.frame.origin.to(dtype=dtype),
        extent=req.frame.extent.to(dtype=dtype),
        lifting=lifting,
        input_names=tuple(bundle.model.inputs),
        cond_means=cond_means,
        cond_stds=cond_stds,
    ).eval()

    sample_xy = torch.tensor(mesh.nodes[: min(16, len(mesh.nodes))], dtype=dtype)
    sample_params = req.parameters[0].to(dtype=dtype)
    meta = {
        "bundle_name": bundle.manifest.name,
        "bundle_manifest_sha256": bundle.manifest.manifest_sha256,
        "weights_sha256": bundle.identities["weights_sha256"],
        "claim_level": bundle.claim_level.value,
        "input_names": list(bundle.model.inputs),
        "conditioning": [
            {
                "source": c.source,
                "mean": float(c.mean),
                "standard_deviation": float(c.standard_deviation),
            }
            for c in bundle.model.conditioning
        ],
        "envelope_bounds": {
            b.name: {
                "min": float(b.lower_bound.value),
                "max": float(b.upper_bound.value),
                "unit": b.lower_bound.unit.value,
            }
            for b in bundle.envelope.primitive_bounds
        },
        "limitations": str(bundle.limitations.get("paragraph", "")),
    }
    return export_mod, sample_xy, sample_params, meta


def export_bundle_to_onnx(
    bundle_path: Path | str,
    output_path: Path | str,
    *,
    opset_version: int = 17,
    verify_parity: bool = True,
    max_abs_tol: float = 1.0e-5,
    reference_inputs: Mapping[str, Any] | None = None,
) -> OnnxExportResult:
    """Export a verified FormaMech bundle to ONNX and verify ONNX Runtime parity."""
    try:
        import onnx as _onnx  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        from piginot.serving.hybrid import MissingExtraError

        raise MissingExtraError(
            "ONNX export requires formamech[export] "
            "(install with `pip install 'formamech[export]'`)."
        ) from exc

    export_mod, sample_xy, sample_params, meta = build_onnx_export_module(
        bundle_path, reference_inputs=reference_inputs, dtype=torch.float32
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_mod,
        (sample_xy, sample_params),
        str(out),
        opset_version=opset_version,
        input_names=["query_xy", "normalized_params"],
        output_names=["displacement_u"],
        dynamic_axes={
            "query_xy": {0: "num_points"},
            "displacement_u": {0: "num_points"},
        },
    )

    onnx_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    parity_checked = False
    parity_err: float | None = None
    if verify_parity:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError:
            parity_checked = False
        else:
            with torch.no_grad():
                ref = export_mod(sample_xy, sample_params).cpu().numpy()
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            ort_out = sess.run(
                None,
                {
                    "query_xy": sample_xy.cpu().numpy(),
                    "normalized_params": sample_params.cpu().numpy(),
                },
            )[0]
            parity_err = float(abs(ref - ort_out).max())
            parity_checked = True
            if parity_err > max_abs_tol:
                raise ValueError(
                    f"ONNX Runtime parity failed: max_abs_error={parity_err:.3e} > {max_abs_tol:.3e}"
                )

    meta_path = out.with_suffix(out.suffix + ".json")
    full_meta = {
        "schema": ONNX_EXPORT_SCHEMA,
        **meta,
        "onnx_sha256": onnx_sha,
        "opset_version": opset_version,
        "parity_checked": parity_checked,
        "parity_max_abs_error": parity_err,
    }
    meta_path.write_bytes(canonical_json_bytes(full_meta) + b"\n")
    return OnnxExportResult(
        onnx_path=out,
        metadata_path=meta_path,
        bundle_manifest_sha256=str(meta["bundle_manifest_sha256"]),
        onnx_sha256=onnx_sha,
        opset_version=opset_version,
        input_names=tuple(meta["input_names"]),
        parity_checked=parity_checked,
        parity_max_abs_error=parity_err,
    )
