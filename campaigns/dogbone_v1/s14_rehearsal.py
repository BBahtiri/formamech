"""S14 step 2: TRAIN-only rehearsal of the verification pipeline (ADR 0019).

The frozen candidate is scored on the 64 TRAIN-screen cases exactly as S13c
scored them. For every case the rehearsal adds the FEM reference error terms
(one coarse solve each), the pointwise stress error and the three
equilibrium-residual indicator variants. It then evaluates the error budget
per slice, runs the confidence pilot that selects the indicator variant, the
refusal fixtures and the single-thread latency harness. DEVELOPMENT and
HOLDOUT stay closed; the S10 source records any DEVELOPMENT access durably.

Usage (one BLAS thread for the FEM timings)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m campaigns.dogbone_v1.s14_rehearsal \
        --s10-run-root RUN --checkpoint-path CKPT --run-root OUT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from campaigns.dogbone_v1.s11_contingency_source import (
    VerifiedS10Case,
    open_selective_s10_source,
)
from campaigns.dogbone_v1.s13c_data import DogboneS13cGeometryCase
from campaigns.dogbone_v1.s13c_protocol import load_s13c_normalization
from campaigns.dogbone_v1.s14_candidate import (
    INDICATOR_LENGTH_SCALE_MM,
    ScoredCase,
    load_frozen_candidate,
    score_case,
    served_geometry,
    slice_ids,
)
from campaigns.dogbone_v1.s14_envelope import (
    FIXTURE_COUNTS,
    check_fixture,
    check_in_domain,
    fixture_set_hash,
    in_domain_cases,
    refusal_fixtures,
)
from campaigns.dogbone_v1.s14_reference import (
    FemReferenceTerms,
    level_request,
    observed_orders,
    reference_terms,
    refined_error_proxy,
)
from piginot.doe import EvidenceRole
from piginot.fem import FemRequest, FemResult, ScikitFemBackend
from piginot.verification import (
    INDICATOR_VARIANTS,
    BudgetTolerance,
    CaseBudgetEntry,
    CheckStatus,
    evaluate_budget,
    fit_quantile_curve,
    measure_latency,
    spearman,
    speedup,
)

GATED_AXES = ("L_total", "W_grip", "W_gauge", "R_fillet", "nominal_axial_strain")
REPORTED_AXES = (
    "arc_feasibility_ratio",
    "fillet_to_gauge_ratio",
    "grip_to_gauge_ratio",
    "length_to_gauge_ratio",
)
AXIS_STRATA = {
    **{axis: 4 for axis in GATED_AXES},
    "arc_feasibility_ratio": 2,
    "fillet_to_gauge_ratio": 4,
    "grip_to_gauge_ratio": 4,
    "length_to_gauge_ratio": 4,
}
TOLERANCES = (
    BudgetTolerance("p_relative_l2", 0.05),
    BudgetTolerance("absolute_peak_error", 0.05),
    BudgetTolerance("u_relative_l2", 0.02),
)
CONFIDENCE_THRESHOLD = 0.6
CALIBRATION_QUANTILE = 0.9
CALIBRATION_BINS = 20
SCORING_THREADS = 2
REHEARSAL_ARTIFACT = "S14-TRAIN-REHEARSAL.json"
_TRAIN = frozenset({EvidenceRole.TRAIN})
_DEVELOPMENT_RECORD = "development-access.json"


def slice_catalogue(axes: Sequence[str]) -> tuple[str, ...]:
    """``all`` plus every ``axis:bin`` of the S9 strata of ``axes``."""

    return ("all",) + tuple(
        f"{axis}:{index}" for axis in axes for index in range(AXIS_STRATA[axis])
    )


GATED_SLICES = slice_catalogue(GATED_AXES)
REPORTED_SLICES = slice_catalogue(REPORTED_AXES)


def surrogate_components(scored: ScoredCase) -> dict[str, float]:
    """The surrogate-to-reference discrepancy of each budget quantity."""

    diagnostics = scored.diagnostics
    return {
        "p_relative_l2": diagnostics.p_relative_l2,
        "absolute_peak_error": abs(diagnostics.signed_peak_error),
        "u_relative_l2": diagnostics.u_relative_l2,
    }


def budget_entry(
    case_id: str,
    slices: tuple[str, ...],
    surrogate: Mapping[str, float],
    terms: FemReferenceTerms | None,
) -> CaseBudgetEntry:
    """One case of the budget; missing FEM terms stay ``None`` (``NOT_RUN``)."""

    reference = None if terms is None else terms.components()
    components: dict[str, dict[str, float | None]] = {}
    for tolerance in TOLERANCES:
        quantity = tolerance.quantity
        parts = None if reference is None else reference[quantity]
        components[quantity] = {
            "surrogate": surrogate[quantity],
            "discretisation": None if parts is None else parts["discretisation"],
            "projection": None if parts is None else parts["projection"],
        }
    return CaseBudgetEntry(case_id, slices, components)


def pooled_spearman(
    errors: Sequence[torch.Tensor],
    indicators: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, float]:
    """Spearman correlation of each variant with the error, all vertices pooled."""

    pooled_error = torch.cat(tuple(errors))
    return {
        variant: spearman(
            torch.cat(tuple(case[variant] for case in indicators)), pooled_error
        )
        for variant in INDICATOR_VARIANTS
    }


def select_variant(correlations: Mapping[str, float]) -> str:
    """Largest correlation; ties keep the preregistered order."""

    best = INDICATOR_VARIANTS[0]
    for variant in INDICATOR_VARIANTS[1:]:
        if correlations[variant] > correlations[best]:
            best = variant
    return best


@dataclass(frozen=True, slots=True)
class CaseRecord:
    """Everything the rehearsal keeps about one TRAIN-screen case."""

    case_id: str
    slices: tuple[str, ...]
    reported_slices: tuple[str, ...]
    surrogate: dict[str, float]
    terms: dict[str, float | int | str]
    components: dict[str, dict[str, float]]
    coarse_solve_seconds: float
    minimum_vertex_determinant: float


def _solve(request: FemRequest) -> tuple[FemResult, float]:
    """Solve on one torch thread (the S10 execution profile) and time it."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        start = time.perf_counter()
        (attempt,) = ScikitFemBackend().solve_many((request,))
        elapsed = time.perf_counter() - start
    finally:
        torch.set_num_threads(previous)
    if not attempt.success or attempt.result is None:
        raise RuntimeError(f"{request.request_id} FEM failed: {attempt.message}")
    return attempt.result, elapsed


def _case_files(run_root: Path, case_id: str) -> tuple[Path, Path]:
    directory = run_root / "cases"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{case_id}.json", directory / f"{case_id}.npz"


def rehearse_case(
    model: torch.nn.Module,
    verified: VerifiedS10Case,
    normalization: object,
    run_root: Path,
) -> tuple[CaseRecord, torch.Tensor, dict[str, torch.Tensor]]:
    """Score, solve the coarse level and measure the reference terms (cached)."""

    case = verified.case
    record_path, arrays_path = _case_files(run_root, case.case_id)
    if record_path.exists() and arrays_path.exists():
        document = json.loads(record_path.read_text(encoding="utf-8"))
        document["slices"] = tuple(document["slices"])
        document["reported_slices"] = tuple(document["reported_slices"])
        arrays = np.load(arrays_path)
        return (
            CaseRecord(**document),
            torch.from_numpy(arrays["error"]),
            {
                variant: torch.from_numpy(arrays[variant])
                for variant in INDICATOR_VARIANTS
            },
        )
    torch.set_num_threads(SCORING_THREADS)
    scored = score_case(
        model,  # type: ignore[arg-type]
        case,
        verified.request,
        verified.result,
        normalization,  # type: ignore[arg-type]
        allowed_roles=_TRAIN,
    )
    coarse = level_request(case, verified.request, 0.5)
    coarse_result, seconds = _solve(coarse)
    terms = reference_terms(
        case.case_id, verified.request, verified.result, coarse, coarse_result
    )
    record = CaseRecord(
        case_id=case.case_id,
        slices=slice_ids(case, GATED_AXES),
        reported_slices=slice_ids(case, REPORTED_AXES)[1:],
        surrogate=surrogate_components(scored),
        terms=asdict(terms),
        components=terms.components(),
        coarse_solve_seconds=seconds,
        minimum_vertex_determinant=scored.diagnostics.minimum_vertex_determinant,
    )
    record_path.write_text(json.dumps(asdict(record), sort_keys=True), encoding="utf-8")
    np.savez(
        arrays_path,
        error=scored.pointwise_stress_error.numpy(),
        **{
            variant: scored.indicators[variant].numpy()
            for variant in INDICATOR_VARIANTS
        },
    )
    return record, scored.pointwise_stress_error, dict(scored.indicators)


def _terms(record: CaseRecord) -> FemReferenceTerms:
    return FemReferenceTerms(**record.terms)  # type: ignore[arg-type]


def budget_summary(records: Sequence[CaseRecord]) -> dict[str, object]:
    """Gated and reported slice results of the budget on ``records``."""

    gated = evaluate_budget(
        tuple(
            budget_entry(r.case_id, r.slices, r.surrogate, _terms(r)) for r in records
        ),
        TOLERANCES,
        GATED_SLICES,
    )
    reported = evaluate_budget(
        tuple(
            budget_entry(r.case_id, ("all", *r.reported_slices), r.surrogate, _terms(r))
            for r in records
        ),
        TOLERANCES,
        REPORTED_SLICES,
    )

    def rows(results: tuple[object, ...]) -> list[dict[str, object]]:
        return [
            {
                "slice": item.slice_id,  # type: ignore[attr-defined]
                "quantity": item.quantity,  # type: ignore[attr-defined]
                "cases": item.case_count,  # type: ignore[attr-defined]
                "statistic": item.statistic,  # type: ignore[attr-defined]
                "tolerance": item.tolerance,  # type: ignore[attr-defined]
                "status": item.status.value,  # type: ignore[attr-defined]
            }
            for item in results
        ]

    statuses = [item.status for item in gated]
    return {
        "gated_status": (
            CheckStatus.PASS.value
            if all(status is CheckStatus.PASS for status in statuses)
            else next(s.value for s in statuses if s is not CheckStatus.PASS)
        ),
        "gated": rows(gated),
        "reported": [row for row in rows(reported) if row["slice"] != "all"],
    }


def component_summary(records: Sequence[CaseRecord]) -> dict[str, dict[str, float]]:
    """Median and maximum of every budget component over the cases."""

    summary: dict[str, dict[str, float]] = {}
    for tolerance in TOLERANCES:
        quantity = tolerance.quantity
        for name, values in (
            ("surrogate", [r.surrogate[quantity] for r in records]),
            (
                "discretisation",
                [r.components[quantity]["discretisation"] for r in records],
            ),
            ("projection", [r.components[quantity]["projection"] for r in records]),
        ):
            summary[f"{quantity}.{name}"] = {
                "median": statistics.median(values),
                "maximum": max(values),
            }
    return summary


def confidence_pilot(
    errors: Sequence[torch.Tensor],
    indicators: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, object]:
    """S14-D4 pilot: pooled Spearman per variant, the selection and diagnostics."""

    correlations = pooled_spearman(errors, indicators)
    chosen = select_variant(correlations)
    per_case = {
        variant: statistics.median(
            spearman(case[variant], error)
            for case, error in zip(indicators, errors, strict=True)
        )
        for variant in INDICATOR_VARIANTS
    }
    pooled_indicator = torch.cat(tuple(case[chosen] for case in indicators))
    pooled_error = torch.cat(tuple(errors))
    curve = fit_quantile_curve(
        pooled_indicator,
        pooled_error,
        quantile=CALIBRATION_QUANTILE,
        bins=CALIBRATION_BINS,
    )
    return {
        "population": "TRAIN-screen vertices, all cases pooled",
        "vertex_count": int(pooled_error.numel()),
        "error_definition": "Frobenius norm of the in-plane first-Piola error "
        "at the vertex, divided by 760 MPa",
        "pooled_spearman": correlations,
        "median_per_case_spearman": per_case,
        "selected_variant": chosen,
        "threshold_on_development": CONFIDENCE_THRESHOLD,
        "train_meets_threshold": correlations[chosen] >= CONFIDENCE_THRESHOLD,
        "train_calibration_diagnostic": {
            "quantile": CALIBRATION_QUANTILE,
            "bins": CALIBRATION_BINS,
            "in_sample_coverage": curve.coverage(pooled_indicator, pooled_error),
            "upper_edges": list(curve.upper_edges),
            "values": list(curve.values),
        },
    }


def refusal_run() -> dict[str, object]:
    """All 240 fixtures and the 844 in-domain cases, twice for code stability."""

    fixtures = refusal_fixtures()
    start = time.perf_counter()
    mismatches: list[str] = []
    unstable: list[str] = []
    for fixture in fixtures:
        first = check_fixture(fixture)
        if first.accepted or fixture.expected_code not in first.reason_codes:
            mismatches.append(fixture.fixture_id)
        if check_fixture(fixture) != first:
            unstable.append(fixture.fixture_id)
    cases = in_domain_cases()
    refused = [case.case_id for case in cases if not check_in_domain(case).accepted]
    return {
        "fixture_count": len(fixtures),
        "fixture_counts": FIXTURE_COUNTS,
        "fixture_set_hash": fixture_set_hash(fixtures),
        "refused_with_expected_code": len(fixtures) - len(mismatches),
        "mismatched_fixtures": mismatches,
        "unstable_reason_codes": unstable,
        "in_domain_count": len(cases),
        "false_refusals": refused,
        "false_refusal_rate": len(refused) / len(cases),
        "seconds": time.perf_counter() - start,
    }


def _single_thread_environment() -> bool:
    return all(
        os.environ.get(name) == "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )


def _model_calls(
    model: torch.nn.Module, geometry: DogboneS13cGeometryCase
) -> tuple[Callable[[], object], Callable[[], object]]:
    """The two timed operator calls at every vertex of one case."""

    inputs = {
        "query_xy": geometry.domain.vertices.unsqueeze(0),
        "parameters": geometry.parameters(),
        "frame": geometry.frame(),
        "lifting": geometry.lifting(),
    }

    def displacement_only() -> object:
        with torch.no_grad():
            return model.displacement(**inputs)  # type: ignore[operator]

    def displacement_and_stress() -> object:
        return model(**inputs, materials=(geometry.material,))

    return displacement_only, displacement_and_stress


def latency_run(
    model: torch.nn.Module,
    cases: Sequence[VerifiedS10Case],
    normalization: object,
    *,
    repeats: int = 20,
) -> dict[str, object]:
    """Single-thread operator latency at the vertices against re-solved FEM."""

    if not _single_thread_environment():
        raise RuntimeError("latency needs OMP/OPENBLAS/MKL_NUM_THREADS=1")
    model_u: list[float] = []
    model_up: list[float] = []
    p99: list[float] = []
    fem: list[float] = []
    reproduced: list[bool] = []
    for verified in cases:
        geometry = served_geometry(
            verified.case,
            verified.request,
            normalization,  # type: ignore[arg-type]
            allowed_roles=_TRAIN,
        )
        displacement_only, displacement_and_stress = _model_calls(model, geometry)
        for call, sink in (
            (displacement_only, model_u),
            (displacement_and_stress, model_up),
        ):
            result = measure_latency(call, repeats=repeats, warmup=3, threads=1)
            sink.append(result.median_seconds)
            if sink is model_up:
                p99.append(result.p99_seconds)
        result, seconds = _solve(verified.request)
        fem.append(seconds)
        reproduced.append(
            result.total_strain_energy == verified.result.total_strain_energy
        )
    fem_median = statistics.median(fem)
    up_median = statistics.median(model_up)
    return {
        "case_count": len(cases),
        "query_set": "all mesh vertices of each case (969 points)",
        "threads": 1,
        "model_u_median_seconds": statistics.median(model_u),
        "model_u_and_p_median_seconds": up_median,
        "model_u_and_p_p99_seconds": max(p99),
        "fem_median_seconds": fem_median,
        "fem_seconds": fem,
        "fem_reproduces_stored_energy": reproduced,
        "speedup_u_and_p": speedup(fem_median, up_median),
        "speedup_target": 100.0,
    }


def refinement_check(
    cases: Sequence[VerifiedS10Case], records: Mapping[str, CaseRecord]
) -> list[dict[str, object]]:
    """Refined (2x) solves: extrapolated S10 error against the budget terms."""

    rows: list[dict[str, object]] = []
    for verified in cases:
        fine = level_request(verified.case, verified.request, 2.0)
        fine_result, seconds = _solve(fine)
        proxy = refined_error_proxy(
            verified.request, verified.result, fine, fine_result
        )
        record = records[verified.case.case_id]
        components = record.components
        rows.append(
            {
                "case_id": verified.case.case_id,
                "refined_solve_seconds": seconds,
                "observed_orders": observed_orders(
                    verified.case,
                    verified.request,
                    verified.result,
                    fine,
                    fine_result,
                    _terms(record),
                ),
                "extrapolated_reference_error": proxy,
                "budget_reference_terms": {
                    quantity: parts["discretisation"] + parts["projection"]
                    for quantity, parts in components.items()
                },
                "covered": {
                    quantity: proxy[quantity]
                    <= components[quantity]["discretisation"]
                    + components[quantity]["projection"]
                    for quantity in proxy
                },
            }
        )
    return rows


def run_rehearsal(
    *,
    s10_run_root: Path,
    checkpoint_path: Path,
    run_root: Path,
    latency_cases: int = 4,
    refinement_cases: int = 0,
    with_refusal: bool = True,
    max_cases: int | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run every step-2 measurement on TRAIN evidence and return the report."""

    run_root.mkdir(parents=True, exist_ok=True)
    source = open_selective_s10_source(
        s10_run_root, development_access_record=run_root / _DEVELOPMENT_RECORD
    )
    model = load_frozen_candidate(checkpoint_path)
    normalization = load_s13c_normalization()
    records: list[CaseRecord] = []
    errors: list[torch.Tensor] = []
    indicators: list[dict[str, torch.Tensor]] = []
    verified_cases: list[VerifiedS10Case] = []
    for verified in source.iter_train_screen():
        if max_cases is not None and len(records) >= max_cases:
            break
        if verified.case.role is not EvidenceRole.TRAIN:
            raise PermissionError("non-TRAIN evidence reached the rehearsal")
        record, error, variants = rehearse_case(
            model, verified, normalization, run_root
        )
        records.append(record)
        errors.append(error)
        indicators.append(variants)
        verified_cases.append(verified)
        progress(f"{len(records):2d}/64 {record.case_id}")
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during the TRAIN rehearsal")
    report: dict[str, object] = {
        "step": "S14 step 2 (TRAIN rehearsal)",
        "evidence_role": "TRAIN",
        "development_opened": False,
        "holdout_opened": False,
        "case_count": len(records),
        "tolerances": {t.quantity: t.tolerance for t in TOLERANCES},
        "indicator_length_scale_mm": INDICATOR_LENGTH_SCALE_MM,
        "budget": budget_summary(records),
        "components": component_summary(records),
        "confidence_pilot": confidence_pilot(errors, indicators),
        "cases": [asdict(record) for record in records],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "scoring_threads": SCORING_THREADS,
        },
    }
    if with_refusal:
        progress("refusal fixtures")
        report["refusal"] = refusal_run()
    if latency_cases:
        progress("latency")
        report["latency"] = latency_run(
            model, verified_cases[:latency_cases], normalization
        )
    if refinement_cases:
        progress("refinement check")
        by_id = {record.case_id: record for record in records}
        report["refinement_check"] = refinement_check(
            verified_cases[:refinement_cases], by_id
        )
    return report


def _json_default(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value in the rehearsal report")
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--latency-cases", type=int, default=4)
    parser.add_argument("--refinement-cases", type=int, default=0)
    parser.add_argument("--skip-refusal", action="store_true")
    parser.add_argument("--max-cases", type=int, help="smoke runs only")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    report = run_rehearsal(
        s10_run_root=options.s10_run_root,
        checkpoint_path=options.checkpoint_path,
        run_root=options.run_root,
        latency_cases=options.latency_cases,
        refinement_cases=options.refinement_cases,
        with_refusal=not options.skip_refusal,
        max_cases=options.max_cases,
    )
    output = options.output or options.run_root / REHEARSAL_ARTIFACT
    output.write_text(
        json.dumps(report, indent=1, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "AXIS_STRATA",
    "CONFIDENCE_THRESHOLD",
    "GATED_AXES",
    "GATED_SLICES",
    "REPORTED_AXES",
    "REPORTED_SLICES",
    "TOLERANCES",
    "CaseRecord",
    "budget_entry",
    "budget_summary",
    "confidence_pilot",
    "pooled_spearman",
    "refusal_run",
    "run_rehearsal",
    "select_variant",
    "slice_catalogue",
    "surrogate_components",
)
