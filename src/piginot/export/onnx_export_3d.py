"""3D neural operator & input-sensitivity ONNX / TorchScript exporter (V6-R4).

Packages a trained :class:`~piginot.operators.surface_operator_3d.Surface3DEnergyOperator`
(with frozen pre-encoded surface slices) into a pure tensor-in / tensor-out
real-time module outputting both 3D displacements and parameter sensitivities
``du / dp`` suitable for ``torch.onnx.export``, TorchScript, and sub-5 ms
robotics control runtimes.
"""

from __future__ import annotations

import hashlib
import importlib
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from piginot.contracts.canonical import canonical_json_bytes
from piginot.export.onnx_export import OnnxParityStatus
from piginot.operators.surface_operator_3d import (
    Surface3DEncoding,
    Surface3DEnergyOperator,
)

ONNX_3D_EXPORT_SCHEMA: Final = "formamech.onnx-export-3d-v1"


class FrozenSurface3DRuntimeModule(nn.Module):
    """Self-contained tensor-in/tensor-out 3D operator with frozen surface slices."""

    def __init__(
        self,
        operator: Surface3DEnergyOperator,
        encoding: Surface3DEncoding,
        query_xyz: torch.Tensor,  # (1, Q, 3)
        *,
        lifting_offset: torch.Tensor | None = None,
        lifting_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.operator = operator
        self.register_buffer("slice_tokens", encoding.slice_tokens.detach().clone())
        self.register_buffer("origin", encoding.origin.detach().clone())
        self.register_buffer("extent", encoding.extent.detach().clone())
        self.register_buffer("query_xyz", query_xyz.detach().clone())
        if lifting_offset is not None:
            self.register_buffer("lifting_offset", lifting_offset.detach().clone())
        else:
            self.lifting_offset: torch.Tensor | None = None
        if lifting_scale is not None:
            self.register_buffer("lifting_scale", lifting_scale.detach().clone())
        else:
            self.lifting_scale: torch.Tensor | None = None

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Predict 3D nodal displacements ``(B, Q, 3)`` from actuator parameters ``(B, P)``."""
        b = parameters.shape[0]
        sl = self.slice_tokens
        orig = self.origin
        ext = self.extent
        qxyz = self.query_xyz
        assert isinstance(sl, torch.Tensor)
        assert isinstance(orig, torch.Tensor)
        assert isinstance(ext, torch.Tensor)
        assert isinstance(qxyz, torch.Tensor)
        enc = Surface3DEncoding(
            slice_tokens=sl.expand(b, -1, -1),
            origin=orig.expand(b, -1),
            extent=ext.expand(b, -1),
        )
        lo = (
            self.lifting_offset.expand(b, -1, -1)
            if isinstance(self.lifting_offset, torch.Tensor)
            else None
        )
        ls = (
            self.lifting_scale.expand(b, -1, -1)
            if isinstance(self.lifting_scale, torch.Tensor)
            else None
        )
        return self.operator.displacement(
            encoding=enc,
            query_xyz=qxyz.expand(b, -1, -1),
            parameters=parameters,
            lifting_offset=lo,
            lifting_scale=ls,
        )

    def predict_and_parameter_jacobian(
        self, parameters: torch.Tensor, *, fd_eps: float = 1.0e-4
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u (Q, 3), du_dp (Q, 3, P))`` in one batched forward pass."""
        p0 = parameters.reshape(1, -1)
        p_dim = int(p0.shape[1])
        batch_list = [p0]
        for j in range(p_dim):
            dp = torch.zeros_like(p0)
            dp[0, j] = float(fd_eps)
            batch_list.append(p0 + dp)
            batch_list.append(p0 - dp)
        p_batch = torch.cat(batch_list, dim=0)
        with torch.no_grad():
            u_batch = self.forward(p_batch)
        u0 = u_batch[0]
        jacs: list[torch.Tensor] = []
        for j in range(p_dim):
            up = u_batch[1 + 2 * j]
            um = u_batch[2 + 2 * j]
            jacs.append((up - um) / (2.0 * float(fd_eps)))
        return u0, torch.stack(jacs, dim=-1)


@dataclass(frozen=True, slots=True)
class Export3DArtifactReport:
    """Verified 3D runtime export report (TorchScript and optional ONNX)."""

    artifact_path: Path
    format_kind: str  # "onnx" | "torchscript"
    artifact_sha256: str
    parity_status: OnnxParityStatus
    max_abs_diff: float
    mean_latency_ms: float
    query_count: int
    parameter_dim: int

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": ONNX_3D_EXPORT_SCHEMA,
            "artifact_path": str(self.artifact_path),
            "format_kind": self.format_kind,
            "artifact_sha256": self.artifact_sha256,
            "parity_status": self.parity_status.value,
            "max_abs_diff": float(self.max_abs_diff),
            "mean_latency_ms": float(self.mean_latency_ms),
            "query_count": self.query_count,
            "parameter_dim": self.parameter_dim,
        }
        doc["report_sha256"] = hashlib.sha256(canonical_json_bytes(doc)).hexdigest()
        return doc


def export_surface_3d_runtime(
    operator: Surface3DEnergyOperator,
    encoding: Surface3DEncoding,
    query_xyz: torch.Tensor,
    output_path: str | Path,
    *,
    sample_parameters: torch.Tensor,
    lifting_offset: torch.Tensor | None = None,
    lifting_scale: torch.Tensor | None = None,
    prefer_onnx: bool = True,
    parity_tol: float = 1.0e-5,
) -> Export3DArtifactReport:
    """Export frozen 3D operator to ONNX (if installed) or TorchScript with parity check."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    operator.eval()
    wrapper = FrozenSurface3DRuntimeModule(
        operator,
        encoding,
        query_xyz,
        lifting_offset=lifting_offset,
        lifting_scale=lifting_scale,
    )
    wrapper.eval()
    p_sample = sample_parameters.detach().clone()
    with torch.no_grad():
        ref_out = wrapper(p_sample).numpy().astype(np.float64)

    onnx_ok = False
    if prefer_onnx and out.suffix.lower() == ".onnx":
        try:
            importlib.import_module("onnx")
            ort: Any = importlib.import_module("onnxruntime")

            torch.onnx.export(
                wrapper,
                (p_sample,),
                str(out),
                input_names=["parameters"],
                output_names=["displacement_mm"],
                dynamic_axes={
                    "parameters": {0: "batch"},
                    "displacement_mm": {0: "batch"},
                },
                opset_version=17,
            )
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            t0 = time.perf_counter()
            for _ in range(10):
                ort_out = sess.run(None, {"parameters": p_sample.numpy()})[0]
            lat_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0
            max_diff = float(
                np.max(np.abs(ref_out - np.asarray(ort_out, dtype=np.float64)))
            )
            status = (
                OnnxParityStatus.VERIFIED_PASS
                if max_diff <= parity_tol
                else OnnxParityStatus.VERIFIED_FAIL
            )
            onnx_ok = True
            return Export3DArtifactReport(
                artifact_path=out,
                format_kind="onnx",
                artifact_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                parity_status=status,
                max_abs_diff=max_diff,
                mean_latency_ms=lat_ms,
                query_count=int(query_xyz.shape[1]),
                parameter_dim=int(p_sample.shape[1]),
            )
        except ImportError:
            onnx_ok = False

    if not onnx_ok:
        ts_path = out if out.suffix.lower() == ".pt" else out.with_suffix(".pt")
        jit_mod: Any = torch.jit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            traced = jit_mod.trace(wrapper, (p_sample,))
            traced.save(str(ts_path))
            loaded = jit_mod.load(str(ts_path))
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                ts_out = loaded(p_sample)
        lat_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0
        max_diff = float(
            np.max(np.abs(ref_out - np.asarray(ts_out.numpy(), dtype=np.float64)))
        )
        status = (
            OnnxParityStatus.VERIFIED_PASS
            if max_diff <= parity_tol
            else OnnxParityStatus.VERIFIED_FAIL
        )
        return Export3DArtifactReport(
            artifact_path=ts_path,
            format_kind="torchscript",
            artifact_sha256=hashlib.sha256(ts_path.read_bytes()).hexdigest(),
            parity_status=status,
            max_abs_diff=max_diff,
            mean_latency_ms=lat_ms,
            query_count=int(query_xyz.shape[1]),
            parameter_dim=int(p_sample.shape[1]),
        )

    raise RuntimeError("unreachable export branch")


__all__ = (
    "ONNX_3D_EXPORT_SCHEMA",
    "Export3DArtifactReport",
    "FrozenSurface3DRuntimeModule",
    "export_surface_3d_runtime",
)
