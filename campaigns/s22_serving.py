"""S22 evaluation: verified serving on DEVELOPMENT (ADR 0103).

Per DEVELOPMENT case (protocol ``artifacts/serving/S22-PROTOCOL.json``):
cold solve of the recipe-rebuilt request (timed), ``exact`` served end to end
(timed), equivalence of exact and cold, and peak-stress errors of the
operator's and the corrected displacement against the cold solution::

    python -m campaigns.s22_serving --campaign dogbone-v1 --workers 3
    python -m campaigns.s22_serving --campaign notched-v1 --workers 3
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from campaigns import s21_certificate as s21
from campaigns import s22_recipes

REPOSITORY = s21.REPOSITORY
OUTPUT = Path("artifacts/serving")
PROTOCOL_PATH = OUTPUT / "S22-PROTOCOL.json"
FAMILIES = s21.FAMILIES


def results_path(campaign: str) -> Path:
    return OUTPUT / f"S22-{campaign}-DEVELOPMENT.json"


@functools.cache
def recipe_for(campaign: str) -> Any:
    from piginot.serving.recipe import FemRecipe

    name = s21.runtime(campaign).bundle
    bundle = s21._bundle(name)
    return FemRecipe.load(
        REPOSITORY / s22_recipes.recipe_path(name),
        bundle=name,
        manifest_sha256=bundle.manifest.manifest_sha256,
    )


def _peak(linearization: Any, vector: np.ndarray[Any, Any]) -> float:
    stress = linearization.recovered_stress(linearization.state(vector))
    return float(np.max(np.linalg.norm(stress, axis=(-2, -1))))


def evaluate_case(campaign: str, case_id: str) -> dict[str, Any]:
    """Every S22 measurement for one DEVELOPMENT case."""

    import torch

    from piginot.bundle import predict
    from piginot.certify import field_difference_l2, field_l2
    from piginot.fem import ScikitFemBackend
    from piginot.fem.linearization import FemLinearization
    from piginot.serving import operator_displacement, serve

    torch.set_num_threads(1)
    bundle = s21._bundle(s21.runtime(campaign).bundle)
    recipe = recipe_for(campaign)
    _, inputs = s22_recipes.case_inputs(campaign, case_id)
    prediction = predict(bundle, inputs, confidence=False)
    request = recipe.build_request(inputs, prediction.mesh, case_id)

    start = time.perf_counter()
    cold = ScikitFemBackend().solve_many((request,))[0]
    cold_seconds = time.perf_counter() - start
    if not cold.success or cold.result is None:
        return {"case_id": case_id, "cold_success": False, "cold_message": cold.message}

    start = time.perf_counter()
    exact = serve(bundle, inputs, mode="exact", recipe=recipe, request_id=case_id)
    exact_seconds = time.perf_counter() - start
    certified = serve(
        bundle,
        inputs,
        mode="certified",
        recipe=recipe,
        corrected=True,
        request_id=case_id,
    )

    linearization = FemLinearization(request)
    cold_vector = linearization.vector(np.asarray(cold.result.displacement))
    cold_state = linearization.state(cold_vector)
    cold_peak = _peak(linearization, cold_vector)
    surrogate = operator_displacement(
        bundle, inputs, prediction.mesh, linearization.dof_coordinates
    )
    operator_vector = linearization.with_exact_dirichlet(
        linearization.vector(surrogate)
    )
    record: dict[str, Any] = {
        "case_id": case_id,
        "cold_success": True,
        "path": exact.path.value,
        "attempts": [attempt.document() for attempt in exact.attempts],
        "certificate_verdict": None
        if certified.certificate is None
        else certified.certificate["verdict"],
        "seconds": {"cold": cold_seconds, "exact": exact_seconds},
        "operator_peak_error": abs(_peak(linearization, operator_vector) - cold_peak)
        / cold_peak,
        "served_peak_error": None,
        "corrected_peak_error": None,
        "equivalence": None,
        "exact_result_hash": exact.exact_result_hash,
        "certified_document_sha256": hashlib.sha256(
            json.dumps(certified.document(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    vertices = len(request.mesh.nodes)
    served = prediction.vertices.P_served.detach().numpy()
    served_peak = float(np.max(np.linalg.norm(served, axis=(-2, -1))))
    cold_vertex_peak = float(
        np.max(
            np.linalg.norm(
                linearization.recovered_stress(cold_state)[:vertices, :2, :2],
                axis=(-2, -1),
            )
        )
    )
    record["served_peak_error"] = abs(served_peak - cold_vertex_peak) / cold_vertex_peak
    if certified.corrected_displacement is not None:
        corrected_vector = linearization.vector(certified.corrected_displacement)
        record["corrected_peak_error"] = (
            abs(_peak(linearization, corrected_vector) - cold_peak) / cold_peak
        )
    if exact.exact is not None:
        exact_vector = linearization.vector(np.asarray(exact.exact.displacement))
        exact_state = linearization.state(exact_vector)
        du, dp = field_difference_l2(linearization, cold_state, exact_state)
        _, p_norm = field_l2(linearization, cold_state)
        record["equivalence"] = {
            "u_max_relative": float(
                np.max(np.abs(exact_vector - cold_vector)) / np.max(np.abs(cold_vector))
            ),
            "energy_relative": abs(
                exact.exact.total_potential_energy - cold.result.total_potential_energy
            )
            / abs(cold.result.total_potential_energy),
            "p_relative_l2": dp / p_norm,
        }
    return record


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The G22 statistics of one family, from its per-case records."""

    solved = [r for r in records if r["cold_success"]]
    paths: dict[str, int] = {}
    for record in solved:
        paths[record["path"]] = paths.get(record["path"], 0) + 1
    silent = sum(
        1 for r in solved if r["equivalence"] is None and r["path"] != "solve_failed"
    )
    equivalence = [r["equivalence"] for r in solved if r["equivalence"] is not None]
    worst = max(
        (max(e.values()) for e in equivalence),
        default=float("inf"),
    )
    speedups = [r["seconds"]["cold"] / r["seconds"]["exact"] for r in solved]
    operator = [r["operator_peak_error"] for r in solved]
    corrected = [
        r["corrected_peak_error"]
        for r in solved
        if r["corrected_peak_error"] is not None
    ]
    return {
        "cases": len(records),
        "cold_solved": len(solved),
        "paths": dict(sorted(paths.items())),
        "silent_results": silent,
        "equivalence_cases": len(equivalence),
        "equivalence_max_relative": worst,
        "median_speedup": statistics.median(speedups) if speedups else 0.0,
        "min_speedup": min(speedups) if speedups else 0.0,
        "median_operator_peak_error": statistics.median(operator) if operator else None,
        "median_corrected_peak_error": statistics.median(corrected)
        if corrected
        else None,
        "corrected_cases": len(corrected),
        "median_served_peak_error": statistics.median(
            r["served_peak_error"] for r in solved
        )
        if solved
        else None,
        "certificates_ok": sum(1 for r in solved if r["certificate_verdict"] == "OK"),
    }


def run(campaign: str, workers: int) -> dict[str, Any]:
    from campaigns.g19_gate import execution_profile
    from piginot.training.platform_fingerprint import platform_fingerprint

    protocol = json.loads((REPOSITORY / PROTOCOL_PATH).read_text("utf-8"))
    ids = protocol["families"][campaign]["case_ids"]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(evaluate_case, campaign, case_id) for case_id in ids]
        records = []
        for future in futures:
            records.append(future.result())
            print(f"S22 {campaign} {len(records)}/{len(ids)}", flush=True)
    return {
        "schema_version": "1.0",
        "campaign_id": campaign,
        "protocol_sha256": hashlib.sha256(
            (REPOSITORY / PROTOCOL_PATH).read_bytes()
        ).hexdigest(),
        "workers": workers,
        "platform_fingerprint": platform_fingerprint(),
        "execution_profile": execution_profile(),
        "records": records,
        "summary": summarize(records),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S22 verified-serving evaluation.")
    parser.add_argument("--campaign", choices=FAMILIES, required=True)
    parser.add_argument("--workers", type=int, default=3)
    options = parser.parse_args(arguments)
    document = run(options.campaign, options.workers)
    path = REPOSITORY / results_path(options.campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(document["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
