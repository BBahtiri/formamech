"""S21 evaluation: certificates of the v0.1.0 bundles on DEVELOPMENT (ADR 0102).

For every DEVELOPMENT case of a family:

1. the S10 P2 request is rebuilt from the published coordinates and solved
   cold (the reference solution, and the cold-solve time);
2. the shipped bundle predicts at the request's displacement nodes (the S10
   mesh vertices, where S15 proved bit-identity with the trained model, and
   the P2 mid-edge nodes as queries), with its S14 confidence bound;
3. the certificate is issued (timed with its ``FemLinearization`` setup);
4. the true discrete errors are measured against the reference: energy norm
   in the reference tangent, relative L2 errors of displacement and first
   Piola stress at the quadrature points;
5. for the protocol's reproduction subset the coarse level is solved too and
   the S14 coarse terms must reproduce.

Runs on the reference machine (ADR 0101 decision 3)::

    python -m campaigns.s21_certificate --campaign dogbone-v1 --workers 4
    python -m campaigns.s21_certificate --campaign notched-v1 --workers 4
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/certify")
PROTOCOL_PATH = OUTPUT / "S21-PROTOCOL.json"
FAMILIES = ("dogbone-v1", "notched-v1")
REPRODUCTION_TOLERANCE = 1e-6


def results_path(campaign_id: str) -> Path:
    return OUTPUT / f"S21-{campaign_id}-DEVELOPMENT.json"


@dataclass(frozen=True, slots=True)
class FamilyRuntime:
    """The family-specific pieces of the S21 evaluation."""

    campaign_id: str
    bundle: str
    s10_request: Callable[..., Any]
    level_request: Callable[..., Any]
    reference_terms: Callable[..., Any]
    theta: Callable[..., dict[str, object]]
    profile: Callable[[], Any]


def runtime(campaign_id: str) -> FamilyRuntime:
    if campaign_id == "dogbone-v1":
        from campaigns.dogbone_v1 import s14_envelope, s14_evaluation, s14_reference
        from campaigns.dogbone_v1.s10_production import default_s10_execution_profile

        return FamilyRuntime(
            campaign_id,
            "dogbone-v1-s13c-cp-freeze",
            s14_evaluation.s10_request,
            s14_reference.level_request,
            s14_reference.reference_terms,
            s14_envelope.theta,
            default_s10_execution_profile,
        )
    if campaign_id == "notched-v1":
        from campaigns.notched_v1 import pilot, s14, s14_envelope

        return FamilyRuntime(
            campaign_id,
            "notched-v1-cp-freeze",
            s14.s10_request,
            s14.level_request,
            s14.reference_terms,
            s14_envelope.theta,
            pilot.execution_profile,
        )
    raise ValueError(f"unknown campaign {campaign_id!r}")


def _artifacts(campaign_id: str) -> Path:
    return REPOSITORY / "artifacts" / "campaigns" / campaign_id


def development_records(campaign_id: str) -> list[Mapping[str, Any]]:
    path = _artifacts(campaign_id) / "S14-DEVELOPMENT.json"
    cases: list[Mapping[str, Any]] = json.loads(path.read_text("utf-8"))["cases"]
    return cases


@functools.cache
def _published(campaign_id: str) -> Mapping[str, Any]:
    from piginot.doe.publication import PublishedCaseManifest

    manifest = PublishedCaseManifest.from_bytes(
        (_artifacts(campaign_id) / "public-case-manifest.json").read_bytes()
    )
    return {case.case_id: case for case in manifest.cases}


@functools.cache
def _bundle(name: str) -> Any:
    from piginot.bundle import load_bundle

    return load_bundle(REPOSITORY / "bundles" / name)


def _median_time(function: Callable[[], Any], repeats: int) -> tuple[Any, float]:
    times, value = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        times.append(time.perf_counter() - start)
    return value, statistics.median(times)


def evaluate_case(
    campaign_id: str, case_id: str, reproduce: bool, repeats: int = 3
) -> dict[str, Any]:
    """Every S21 measurement for one DEVELOPMENT case."""

    import torch

    from campaigns.dogbone_v1.s14_candidate import S14Case
    from campaigns.dogbone_v1.s14_rehearsal import _solve
    from piginot.bundle import predict
    from piginot.certify import certify, field_difference_l2, field_l2
    from piginot.fem.linearization import FemLinearization

    torch.set_num_threads(1)
    family = runtime(campaign_id)
    case = S14Case.from_published(_published(campaign_id)[case_id])
    request = family.s10_request(case, family.profile())
    result, cold_seconds = _solve(request)
    reference = np.asarray(result.displacement, dtype=np.float64)

    values = {k: float(v) for k, v in case.primitive_coordinates.items()}
    linearization = FemLinearization(request)
    nodes = linearization.dof_coordinates
    prediction = predict(
        _bundle(family.bundle),
        family.theta(
            values,
            elastic_modulus=values["elastic_modulus"],
            poisson_ratio=values["poisson_ratio"],
        ),
        nodes.tolist(),
        confidence=True,
    )
    if prediction.queries is None or prediction.vertices is None:
        raise RuntimeError(f"{case_id}: the bundle refused {prediction.reason_codes}")
    surrogate = prediction.queries.u.detach().numpy().astype(np.float64)
    vertex_count = len(request.mesh.nodes)
    vertex_gap = float(
        np.max(
            np.abs(surrogate[:vertex_count] - prediction.vertices.u.detach().numpy())
        )
    )

    def issue() -> Any:
        return certify(FemLinearization(request), surrogate)

    certificate, certificate_seconds = _median_time(issue, repeats)
    floor = certify(linearization, reference)
    reference_state = linearization.state(linearization.vector(reference))
    surrogate_vector = linearization.with_exact_dirichlet(
        linearization.vector(surrogate)
    )
    surrogate_state = linearization.state(surrogate_vector)
    energy_error = linearization.energy_norm(
        surrogate_vector - reference_state.displacement, reference_state
    )
    du, dp = field_difference_l2(linearization, reference_state, surrogate_state)
    nu, npiola = field_l2(linearization, reference_state)
    bound = prediction.vertices.confidence_bound
    record: dict[str, Any] = {
        "case_id": case_id,
        "mesh_id_equal": prediction.mesh is not None
        and prediction.mesh.mesh_id == request.mesh.mesh_id,
        "vertex_gap": vertex_gap,
        "certificate": certificate.document(),
        "floor_eta": floor.eta_energy,
        "true": {
            "energy": energy_error,
            "u_relative_l2": du / nu,
            "p_relative_l2": dp / npiola,
        },
        "s14_indicator_median": None
        if bound is None
        else float(torch.median(bound).item()),
        "seconds": {"cold_fem": cold_seconds, "certificate": certificate_seconds},
    }
    if reproduce:
        coarse = family.level_request(case, request, 0.5)
        coarse_result, _ = _solve(coarse)
        terms = family.reference_terms(case_id, request, result, coarse, coarse_result)
        record["reproduction"] = {
            name: getattr(terms, name)
            for name in (
                "coarse_u_difference",
                "coarse_p_difference",
                "coarse_peak_difference",
            )
        }
    return record


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def summarize(
    records: Sequence[Mapping[str, Any]],
    recorded_terms: Mapping[str, Mapping[str, float]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """The G21 statistics of one family, from its per-case records."""

    rules = protocol["thresholds"]
    ok = [r for r in records if r["certificate"]["verdict"] == "OK"]
    eta = [r["certificate"]["eta_energy"] for r in ok]
    energy = [r["true"]["energy"] for r in ok]
    p_change = [r["certificate"]["indicators"]["p_l2_change"] for r in ok]
    p_error = [r["true"]["p_relative_l2"] for r in ok]
    floor_factor = protocol["effectivity_floor_factor"]
    above = [
        r for r in ok if r["true"]["energy"] > floor_factor * (r["floor_eta"] or 0.0)
    ]
    effectivity = [r["certificate"]["eta_energy"] / r["true"]["energy"] for r in above]
    low, high = rules["effectivity_band"]
    within = sum(low <= value <= high for value in effectivity)
    s14 = [r["s14_indicator_median"] for r in ok]
    ratio = statistics.median(
        r["seconds"]["certificate"] for r in records
    ) / statistics.median(r["seconds"]["cold_fem"] for r in records)
    worst = 0.0
    for r in records:
        for name, value in (r.get("reproduction") or {}).items():
            recorded = float(recorded_terms[r["case_id"]][name])
            worst = max(worst, abs(value - recorded) / max(abs(recorded), 1e-300))
    energy_ratio = [r["certificate"]["indicators"]["energy_ratio"] for r in ok]
    return {
        "cases": len(records),
        "certificates_ok": len(ok),
        "spearman_eta_energy": _spearman(eta, energy),
        "effectivity_cases": len(effectivity),
        "effectivity_within_band": within / len(effectivity) if effectivity else 0.0,
        "effectivity_median": statistics.median(effectivity) if effectivity else None,
        "spearman_stress": _spearman(p_change, p_error),
        "spearman_s14_stress": _spearman(s14, p_error)
        if all(v is not None for v in s14)
        else None,
        "cost_ratio": ratio,
        "reproduction_max_relative": worst,
        "reproduction_cases": sum(1 for r in records if r.get("reproduction")),
        "mesh_identity": all(r["mesh_id_equal"] for r in records),
        "max_vertex_gap": max(r["vertex_gap"] for r in records),
        "max_dirichlet_deviation": max(
            r["certificate"]["dirichlet_deviation"] for r in records
        ),
        "energy_ratio_median": statistics.median(energy_ratio)
        if energy_ratio
        else None,
    }


def run(campaign_id: str, workers: int) -> dict[str, Any]:
    from campaigns.g19_gate import execution_profile
    from piginot.training.platform_fingerprint import platform_fingerprint

    protocol = json.loads((REPOSITORY / PROTOCOL_PATH).read_text("utf-8"))
    family = protocol["families"][campaign_id]
    ids = family["case_ids"]
    reproduce = set(family["reproduction_case_ids"])
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(evaluate_case, campaign_id, i, i in reproduce) for i in ids
        ]
        records = []
        for future in futures:
            records.append(future.result())
            print(f"S21 {campaign_id} {len(records)}/{len(ids)}", flush=True)
    terms = {c["case_id"]: c["terms"] for c in development_records(campaign_id)}
    return {
        "schema_version": "1.0",
        "campaign_id": campaign_id,
        "protocol_sha256": hashlib.sha256(
            (REPOSITORY / PROTOCOL_PATH).read_bytes()
        ).hexdigest(),
        "workers": workers,
        "platform_fingerprint": platform_fingerprint(),
        "execution_profile": execution_profile(),
        "records": records,
        "summary": summarize(records, terms, protocol),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S21 certificate evaluation.")
    parser.add_argument("--campaign", choices=FAMILIES, required=True)
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args(arguments)
    document = run(options.campaign, options.workers)
    path = REPOSITORY / results_path(options.campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(document["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
