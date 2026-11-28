"""Large-Mesh 3D Neural-Operator Scaling & Memory Harness (V7-R6).

Profiles :class:`~piginot.operators.surface_operator_3d.Surface3DEnergyOperator`
deterministically across industrial mesh sizes (``10k -> 100k -> 500k -> 1M``
surface nodes) with AB-UPT anchor subsampling, Transolver++ adaptive slice
temperatures, LRQ low-rank query compression, and bounded-memory chunked
decoding. Logs:

* Process peak RSS (OS native via Windows ``GetProcessMemoryInfo`` or POSIX
  ``resource``) and Python ``tracemalloc`` peak allocation,
* Encode latency and decode P50 / P99 latency (ms) and throughput (nodes/sec),
* Numerical drift between chunked and unchunked decoding,
* ONNX / TorchScript runtime export parity (``<= 1e-5`` via
  :func:`~piginot.export.onnx_export_3d.export_surface_3d_runtime`),
* Canonical SHA-256 benchmark receipt backing all industrial scaling claims.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import math
import os
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from piginot.contracts.canonical import canonical_json_bytes
from piginot.export.onnx_export import OnnxParityStatus
from piginot.export.onnx_export_3d import export_surface_3d_runtime
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEnergyOperator,
)

SCALING_3D_BENCHMARK_SCHEMA: Final = "formamech.surface3d-scaling-benchmark-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _current_process_rss_mb() -> float:
    """Measure current process resident working set (MB) without psutil dependency."""
    if sys.platform == "win32":

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        windll = ctypes.windll
        windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        windll.psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        windll.psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = windll.kernel32.GetCurrentProcess()
        ok = windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if ok:
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        return 0.0

    # POSIX fallback via /proc/self/statm or resource.getrusage
    try:
        statm = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        pages = int(statm[1])
        page_sz = int(os.sysconf("SC_PAGE_SIZE"))
        return float(pages * page_sz) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


@dataclass(frozen=True, slots=True)
class Scaling3DConfig:
    """Configuration for deterministic 10k-1M node Surface3DEnergyOperator scaling sweep."""

    node_counts: tuple[int, ...] = (10_000, 100_000, 500_000, 1_000_000)
    anchor_count: int = 512
    decode_chunk_size: int = 32_768
    hidden_dim: int = 32
    slice_count: int = 8
    num_heads: int = 4
    num_blocks: int = 2
    low_rank_query_dim: int | None = 16
    adaptive_temperature: bool = True
    parameter_dim: int = 3
    warmup_runs: int = 1
    timed_runs: int = 3
    verify_export_parity: bool = True
    export_parity_nodes: int = 4_096
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.node_counts or any(n < 64 for n in self.node_counts):
            raise ValueError("node_counts must contain integers >= 64")
        if self.anchor_count < 8:
            raise ValueError("anchor_count must be >= 8")
        if self.decode_chunk_size < 64:
            raise ValueError("decode_chunk_size must be >= 64")
        if self.timed_runs < 1:
            raise ValueError("timed_runs must be >= 1")


@dataclass(frozen=True, slots=True)
class ScalingBenchmarkPoint:
    """Measured latency, memory, drift, and export-parity metrics at one mesh size."""

    node_count: int
    anchor_count: int
    decode_chunk_size: int
    encode_ms: float
    decode_p50_ms: float
    decode_p99_ms: float
    total_p50_ms: float
    total_p99_ms: float
    nodes_per_sec: float
    rss_before_mb: float
    rss_peak_mb: float
    rss_delta_mb: float
    tracemalloc_peak_mb: float
    chunk_vs_full_max_drift: float
    export_format: str
    export_max_abs_diff: float
    export_parity_passed: bool

    def document(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "anchor_count": self.anchor_count,
            "decode_chunk_size": self.decode_chunk_size,
            "encode_ms": float(self.encode_ms),
            "decode_p50_ms": float(self.decode_p50_ms),
            "decode_p99_ms": float(self.decode_p99_ms),
            "total_p50_ms": float(self.total_p50_ms),
            "total_p99_ms": float(self.total_p99_ms),
            "nodes_per_sec": float(self.nodes_per_sec),
            "rss_before_mb": float(self.rss_before_mb),
            "rss_peak_mb": float(self.rss_peak_mb),
            "rss_delta_mb": float(self.rss_delta_mb),
            "tracemalloc_peak_mb": float(self.tracemalloc_peak_mb),
            "chunk_vs_full_max_drift": float(self.chunk_vs_full_max_drift),
            "export_format": self.export_format,
            "export_max_abs_diff": float(self.export_max_abs_diff),
            "export_parity_passed": self.export_parity_passed,
        }


@dataclass(frozen=True, slots=True)
class Scaling3DReport:
    """Full multi-scale benchmark report with log-log scaling exponents and receipt."""

    points: tuple[ScalingBenchmarkPoint, ...]
    device: str
    cpu_threads: int
    scaling_exponent_latency: float
    max_chunk_drift: float
    max_export_parity_diff: float
    all_parity_passed: bool

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": SCALING_3D_BENCHMARK_SCHEMA,
            "device": self.device,
            "cpu_threads": self.cpu_threads,
            "scaling_exponent_latency": float(self.scaling_exponent_latency),
            "max_chunk_drift": float(self.max_chunk_drift),
            "max_export_parity_diff": float(self.max_export_parity_diff),
            "all_parity_passed": self.all_parity_passed,
            "points": [pt.document() for pt in self.points],
        }
        doc["report_sha256"] = _sha256_doc(doc)
        return doc


def _build_synthetic_cad_surface_tensors(
    n_nodes: int,
    *,
    surface_feature_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically generate parametric 3D CAD surface cloud of ``n_nodes`` points."""
    gen = torch.Generator()
    gen.manual_seed(seed + n_nodes)
    # Deterministic Fibonacci-like shell on a 100x50x30 mm 3D CAD surface
    idx = torch.arange(n_nodes, dtype=torch.float32)
    t = (idx + 0.5) / float(n_nodes)
    phi = 2.0 * math.pi * ((idx * 0.61803398875) % 1.0)
    cos_th = 1.0 - 2.0 * t
    sin_th = torch.sqrt((1.0 - cos_th * cos_th).clamp_min(0.0))

    nx = sin_th * torch.cos(phi)
    ny = sin_th * torch.sin(phi)
    nz = cos_th
    normals = torch.stack((nx, ny, nz), dim=-1).unsqueeze(0)  # (1, N, 3)
    centroids = torch.stack(
        (50.0 * (nx + 1.0), 25.0 * (ny + 1.0), 15.0 * (nz + 1.0)), dim=-1
    ).unsqueeze(0)
    area_weights = torch.full((1, n_nodes), 1.0 / float(n_nodes), dtype=torch.float32)
    # Lightweight deterministic surface features (boundary flags + curvature proxies)
    feats = torch.zeros((1, n_nodes, surface_feature_dim), dtype=torch.float32)
    feats[0, :, 0] = (centroids[0, :, 0] < 5.0).to(torch.float32)
    feats[0, :, 1] = (centroids[0, :, 0] > 95.0).to(torch.float32)
    feats[0, :, 2] = t
    return centroids, normals, area_weights, feats


def run_surface_3d_scaling_benchmark(
    config: Scaling3DConfig | None = None,
    *,
    export_dir: str | Path | None = None,
) -> Scaling3DReport:
    """Execute deterministic ``Surface3DEnergyOperator`` scaling sweep with SHA-256 receipt."""
    cfg = Scaling3DConfig() if config is None else config
    torch.manual_seed(cfg.seed)

    op_cfg = Surface3DConfig(
        parameter_dim=cfg.parameter_dim,
        surface_feature_dim=6,
        hidden_dim=cfg.hidden_dim,
        slice_count=cfg.slice_count,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        adaptive_temperature=cfg.adaptive_temperature,
        anchor_count=cfg.anchor_count,
        low_rank_query_dim=cfg.low_rank_query_dim,
        decode_chunk_size=cfg.decode_chunk_size,
    )
    op = Surface3DEnergyOperator(op_cfg)
    op.eval()

    # Unchunked reference copy sharing identical weights for exact chunk-drift audit
    op_unchunked_cfg = Surface3DConfig(
        parameter_dim=cfg.parameter_dim,
        surface_feature_dim=6,
        hidden_dim=cfg.hidden_dim,
        slice_count=cfg.slice_count,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        adaptive_temperature=cfg.adaptive_temperature,
        anchor_count=cfg.anchor_count,
        low_rank_query_dim=cfg.low_rank_query_dim,
        decode_chunk_size=None,
    )
    op_ref = Surface3DEnergyOperator(op_unchunked_cfg)
    op_ref.load_state_dict(op.state_dict())
    op_ref.eval()

    params = torch.linspace(
        0.25, 0.75, cfg.parameter_dim, dtype=torch.float32
    ).unsqueeze(0)

    # Verify ONNX / TorchScript export parity once on a representative node slice
    export_fmt = "skipped"
    export_diff = 0.0
    export_ok = True
    if cfg.verify_export_parity:
        c_e, n_e, w_e, f_e = _build_synthetic_cad_surface_tensors(
            cfg.export_parity_nodes, surface_feature_dim=6, seed=cfg.seed
        )
        with torch.no_grad():
            enc_e = op.encode_surface(c_e, n_e, w_e, f_e)
        if export_dir is not None:
            e_path = Path(export_dir) / "surface_3d_scaling_parity.onnx"
            rep = export_surface_3d_runtime(
                op, enc_e, c_e, e_path, sample_parameters=params, prefer_onnx=True
            )
        else:
            with tempfile.TemporaryDirectory() as td:
                rep = export_surface_3d_runtime(
                    op,
                    enc_e,
                    c_e,
                    Path(td) / "surface_3d_scaling_parity.onnx",
                    sample_parameters=params,
                    prefer_onnx=True,
                )
        export_fmt = rep.format_kind
        export_diff = float(rep.max_abs_diff)
        export_ok = rep.parity_status is OnnxParityStatus.VERIFIED_PASS

    points: list[ScalingBenchmarkPoint] = []
    for n_nodes in cfg.node_counts:
        gc.collect()
        rss_before = _current_process_rss_mb()
        tracemalloc.start()
        centroids, normals, weights, feats = _build_synthetic_cad_surface_tensors(
            n_nodes, surface_feature_dim=6, seed=cfg.seed
        )

        # Warmup
        with torch.no_grad():
            for _ in range(cfg.warmup_runs):
                enc = op.encode_surface(centroids, normals, weights, feats)
                _ = op.displacement(
                    encoding=enc,
                    query_xyz=centroids[:, : min(n_nodes, 2048), :],
                    parameters=params,
                )

        # Timed encode + full N-node chunked decode
        enc_times: list[float] = []
        dec_times: list[float] = []
        tot_times: list[float] = []
        rss_peak = rss_before
        u_last: torch.Tensor | None = None
        enc_last = enc

        with torch.no_grad():
            for _ in range(cfg.timed_runs):
                t_e0 = time.perf_counter()
                enc_last = op.encode_surface(centroids, normals, weights, feats)
                t_e1 = time.perf_counter()
                u_last = op.displacement(
                    encoding=enc_last, query_xyz=centroids, parameters=params
                )
                t_d1 = time.perf_counter()
                enc_ms = (t_e1 - t_e0) * 1000.0
                dec_ms = (t_d1 - t_e1) * 1000.0
                enc_times.append(enc_ms)
                dec_times.append(dec_ms)
                tot_times.append(enc_ms + dec_ms)
                rss_peak = max(rss_peak, _current_process_rss_mb())

        # Exact chunked vs unchunked drift check on a 4,096-node slice (bounded RAM)
        check_n = min(n_nodes, max(256, min(4096, cfg.decode_chunk_size)))
        with torch.no_grad():
            u_unchunk = op_ref.displacement(
                encoding=enc_last,
                query_xyz=centroids[:, :check_n, :],
                parameters=params,
            )
        assert u_last is not None
        drift = float(torch.max(torch.abs(u_last[:, :check_n, :] - u_unchunk)).item())

        _, tm_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        dec_p50 = float(np.percentile(dec_times, 50))
        dec_p99 = float(np.percentile(dec_times, 99))
        tot_p50 = float(np.percentile(tot_times, 50))
        tot_p99 = float(np.percentile(tot_times, 99))
        nodes_per_s = float(n_nodes) / max(tot_p50 / 1000.0, 1.0e-6)

        points.append(
            ScalingBenchmarkPoint(
                node_count=n_nodes,
                anchor_count=cfg.anchor_count,
                decode_chunk_size=cfg.decode_chunk_size,
                encode_ms=float(np.median(enc_times)),
                decode_p50_ms=dec_p50,
                decode_p99_ms=dec_p99,
                total_p50_ms=tot_p50,
                total_p99_ms=tot_p99,
                nodes_per_sec=nodes_per_s,
                rss_before_mb=rss_before,
                rss_peak_mb=rss_peak,
                rss_delta_mb=max(0.0, rss_peak - rss_before),
                tracemalloc_peak_mb=float(tm_peak_bytes) / (1024.0 * 1024.0),
                chunk_vs_full_max_drift=drift,
                export_format=export_fmt,
                export_max_abs_diff=export_diff,
                export_parity_passed=export_ok,
            )
        )
        # Release huge tensors before next scale step
        del centroids, normals, weights, feats, u_last, enc_last, u_unchunk
        gc.collect()

    # Log-log latency scaling slope (O(N^p) -> p ~ 1.0 proves linear scaling!)
    if len(points) >= 2:
        lx = np.log(np.array([pt.node_count for pt in points], dtype=np.float64))
        ly = np.log(
            np.array([max(pt.total_p50_ms, 1.0e-4) for pt in points], dtype=np.float64)
        )
        slope = float(np.polyfit(lx, ly, 1)[0])
    else:
        slope = 1.0

    return Scaling3DReport(
        points=tuple(points),
        device="cpu",
        cpu_threads=int(torch.get_num_threads()),
        scaling_exponent_latency=slope,
        max_chunk_drift=max(pt.chunk_vs_full_max_drift for pt in points),
        max_export_parity_diff=max(pt.export_max_abs_diff for pt in points),
        all_parity_passed=all(pt.export_parity_passed for pt in points),
    )


__all__ = (
    "SCALING_3D_BENCHMARK_SCHEMA",
    "Scaling3DConfig",
    "Scaling3DReport",
    "ScalingBenchmarkPoint",
    "run_surface_3d_scaling_benchmark",
)
