"""Reproducible prototype benchmark for spectrally safe warm-start FEM (Phase 2).

Compares:
1. Cold load-stepped Newton-Raphson FEM,
2. Direct neural warm-start Newton (with and without synthetic spectral-bias
   perturbations that break local ``det(F)`` / tangent positive definiteness),
3. Spectrally repaired warm-start Newton (``spectrally_safe_repair``),

reporting exact Newton iteration counts, iteration reduction ratio
``R = M_cold / M_warm``, wall-clock speedup, and ``min det(F)`` / inertia checks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from piginot.bundle import load_bundle
from piginot.cli import parse_request
from piginot.fem import ScikitFemBackend
from piginot.serving import (
    ServingMode,
    load_recipe,
    operator_displacement,
    serve,
    spectrally_safe_repair,
)


def run_bundle_benchmark(
    bundle_dir: Path,
    request_path: Path,
    *,
    perturbation_scale: float = 0.0,
) -> dict[str, Any]:
    bundle = load_bundle(bundle_dir)
    inputs, queries = parse_request(json.loads(request_path.read_text("utf-8")))
    recipe = load_recipe(bundle, bundle_dir)
    backend = ScikitFemBackend()

    # 1. Cold FEM baseline
    fast = serve(bundle, inputs, queries, mode=ServingMode.FAST)
    assert fast.prediction.mesh is not None
    req = recipe.build_request(inputs, fast.prediction.mesh, "warm-start-bench")
    t0 = time.perf_counter()
    cold_attempt = backend.solve_many((req,))[0]
    cold_wall = time.perf_counter() - t0
    assert cold_attempt.success and cold_attempt.result is not None
    cold_iters = len(cold_attempt.result.convergence_history)

    # 2. Neural prediction (optionally perturbed to induce spectral-bias defects)
    linearization = backend.linearize(req)
    u_pred = operator_displacement(
        bundle, inputs, fast.prediction.mesh, linearization.dof_coordinates
    )
    if perturbation_scale > 0.0:
        rng = np.random.default_rng(42)
        u_pred = u_pred + perturbation_scale * rng.standard_normal(u_pred.shape)
    start_vec = linearization.with_exact_dirichlet(linearization.vector(u_pred))

    # 3. Direct warm Newton
    t1 = time.perf_counter()
    raw_warm = backend.solve_from(req, start_vec, (1.0,))
    raw_wall = time.perf_counter() - t1
    raw_iters = (
        len(raw_warm.result.convergence_history)
        if raw_warm.success and raw_warm.result is not None
        else None
    )

    # 4. Spectrally repaired warm Newton
    t2 = time.perf_counter()
    repaired_vec, repair_report = spectrally_safe_repair(linearization, start_vec)
    rep_warm = backend.solve_from(req, repaired_vec, (1.0,))
    rep_wall = time.perf_counter() - t2
    rep_iters = (
        len(rep_warm.result.convergence_history)
        if rep_warm.success and rep_warm.result is not None
        else None
    )

    return {
        "bundle": bundle_dir.name,
        "request": request_path.name,
        "dof_count": linearization.dof_count,
        "perturbation_scale": perturbation_scale,
        "cold_fem": {
            "converged": cold_attempt.success,
            "newton_iterations": cold_iters,
            "wall_seconds": cold_wall,
        },
        "raw_warm_newton": {
            "converged": raw_warm.success,
            "newton_iterations": raw_iters,
            "wall_seconds": raw_wall,
            "iteration_reduction_ratio": (
                None if not raw_iters else cold_iters / raw_iters
            ),
        },
        "spectrally_repaired_warm_newton": {
            "converged": rep_warm.success,
            "newton_iterations": rep_iters,
            "wall_seconds": rep_wall,
            "iteration_reduction_ratio": (
                None if not rep_iters else cold_iters / rep_iters
            ),
            "repair_report": repair_report.document(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("bundles/notched-v1-cp-freeze")
    )
    parser.add_argument(
        "--request", type=Path, default=Path("bundles/requests/notched-in-domain.json")
    )
    parser.add_argument("--perturbation", type=float, default=0.0)
    args = parser.parse_args()
    result = run_bundle_benchmark(
        args.bundle, args.request, perturbation_scale=args.perturbation
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
