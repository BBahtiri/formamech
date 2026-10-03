"""S14 step 3: DEVELOPMENT, then HOLDOUT once, then the G14 report.

Every phase first requires the signed, still-frozen ``S14-PROTOCOL.json``.

- ``development`` writes the DEVELOPMENT access record, scores the 64 cases,
  evaluates the budget per slice, the pooled Spearman correlation of the
  frozen indicator, fits the calibration curve and measures latency.
- ``holdout`` requires the DEVELOPMENT results, writes the consumption record,
  reads and verifies the private reveal, solves the 64 HOLDOUT cases with the
  S10 recipe and scores them once with the same code.
- ``gate`` assembles the G14 checks into the sectioned report.

Usage::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m campaigns.dogbone_v1.s14_validate development \
        --s10-run-root RUN --checkpoint-path CKPT --run-root OUT
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from campaigns.dogbone_v1.s10_closeout import _load_run_execution_profile
from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13c_protocol import load_s13c_normalization
from campaigns.dogbone_v1.s14_candidate import (
    CP_FREEZE_PATH,
    frozen_candidate_record,
    load_frozen_candidate,
)
from campaigns.dogbone_v1.s14_evaluation import (
    DEVELOPMENT_ACCESS_NAME,
    HOLDOUT_CONSUMPTION_NAME,
    REVEAL_RELATIVE_PATH,
    DevelopmentAccess,
    consume_and_reveal_holdout,
    holdout_cases,
    holdout_commitment,
    open_development,
    parse_reveal,
    s10_request,
)
from campaigns.dogbone_v1.s14_gate import (
    SPEARMAN_MINIMUM,
    G14Measurements,
    g14_report,
)
from campaigns.dogbone_v1.s14_protocol import (
    LATENCY_CASES,
    LATENCY_REPEATS,
    REPOSITORY_ROOT,
    SIGNOFF_PATH,
    load_signed_protocol,
    text_sha256,
)
from campaigns.dogbone_v1.s14_rehearsal import (
    CALIBRATION_BINS,
    CALIBRATION_QUANTILE,
    CaseRecord,
    SolvedCase,
    _solve,
    budget_summary,
    component_summary,
    latency_run,
    refusal_run,
    rehearse_case,
)
from piginot.doe import EvidenceRole
from piginot.verification import (
    CheckStatus,
    ClaimLevel,
    QuantileCurve,
    fit_quantile_curve,
    inject_failures,
    read_registered,
    reduce_registered,
    render_report,
    spearman,
)
from piginot.verification.holdout import (
    ConsumptionRecord,
    reveal_after_consumption,
)

DEVELOPMENT_RESULT = "S14-DEVELOPMENT.json"
HOLDOUT_RESULT = "S14-HOLDOUT.json"
GATE_RESULT = "S14-GATE.json"
_DEVELOPMENT = frozenset({EvidenceRole.DEVELOPMENT})
_HOLDOUT = frozenset({EvidenceRole.HOLDOUT})


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: Path, document: object) -> None:
    text = json.dumps(document, indent=1, sort_keys=True, default=list) + "\n"
    path.write_text(text, encoding="utf-8")


def _resume_or_open(
    source: object, record_path: Path, access: DevelopmentAccess
) -> Iterator[object]:
    if not record_path.exists():
        return open_development(source, record_path, access)
    existing = json.loads(record_path.read_text(encoding="utf-8"))
    if existing["protocol_sha256"] != access.protocol_sha256 or tuple(
        existing["case_ids"]
    ) != tuple(access.case_ids):
        raise PermissionError("DEVELOPMENT was opened under another protocol")
    ids = tuple(existing["case_ids"])
    return (
        source._load_case(case_id, EvidenceRole.DEVELOPMENT)  # type: ignore[attr-defined]
        for case_id in ids
    )


def run_development(
    *, s10_run_root: Path, checkpoint_path: Path, run_root: Path
) -> dict[str, object]:
    """Open DEVELOPMENT under the signed protocol and measure it."""

    protocol, digest = load_signed_protocol()
    signoff = json.loads(SIGNOFF_PATH.read_text(encoding="utf-8"))
    variant = protocol["confidence"]["indicator"]  # type: ignore[index]
    run_root.mkdir(parents=True, exist_ok=True)
    source = open_selective_s10_source(
        s10_run_root, development_access_record=run_root / "s11-access-unused.json"
    )
    model = load_frozen_candidate(checkpoint_path)
    normalization = load_s13c_normalization()
    access = DevelopmentAccess(
        protocol_sha256=digest,
        signed_off_by=signoff["signed_off_by"],
        signed_off_utc=signoff["signed_off_utc"],
        opened_utc=_utc(),
        case_ids=tuple(source._protocol.development_case_ids),
    )
    records: list[CaseRecord] = []
    errors: list[torch.Tensor] = []
    indicators: list[dict[str, torch.Tensor]] = []
    solved_cases: list[SolvedCase] = []
    for verified in _resume_or_open(source, run_root / DEVELOPMENT_ACCESS_NAME, access):
        solved = SolvedCase.from_s10(verified)  # type: ignore[arg-type]
        record, error, variants = rehearse_case(
            model,
            solved,
            normalization,
            run_root / "development",
            allowed_roles=_DEVELOPMENT,
        )
        records.append(record)
        errors.append(error)
        indicators.append(variants)
        solved_cases.append(solved)
        print(f"DEVELOPMENT {len(records):2d}/64 {record.case_id}", flush=True)
    pooled_indicator = torch.cat(tuple(case[variant] for case in indicators))
    pooled_error = torch.cat(tuple(errors))
    curve = fit_quantile_curve(
        pooled_indicator,
        pooled_error,
        quantile=CALIBRATION_QUANTILE,
        bins=CALIBRATION_BINS,
    )
    latency = latency_run(
        model,
        solved_cases[:LATENCY_CASES],
        normalization,
        allowed_roles=_DEVELOPMENT,
        repeats=LATENCY_REPEATS,
    )
    report: dict[str, object] = {
        "evidence_role": "DEVELOPMENT",
        "protocol_sha256": digest,
        "case_count": len(records),
        "budget": budget_summary(records),
        "components": component_summary(records),
        "confidence": {
            "indicator": variant,
            "pooled_spearman": spearman(pooled_indicator, pooled_error),
            "vertex_count": int(pooled_error.numel()),
            "calibration": {
                "quantile": curve.quantile,
                "upper_edges": list(curve.upper_edges),
                "values": list(curve.values),
                "in_sample_coverage": curve.coverage(pooled_indicator, pooled_error),
            },
        },
        "latency": latency,
        "cases": [record_document(record) for record in records],
    }
    _write(run_root / DEVELOPMENT_RESULT, report)
    return report


def record_document(record: CaseRecord) -> dict[str, object]:
    return {
        "case_id": record.case_id,
        "slices": list(record.slices),
        "reported_slices": list(record.reported_slices),
        "surrogate": record.surrogate,
        "terms": record.terms,
        "components": record.components,
        "minimum_vertex_determinant": record.minimum_vertex_determinant,
    }


def failing_budget_rows(development: dict[str, object]) -> tuple[tuple[str, str], ...]:
    """``(slice, quantity)`` of every gated budget row that did not pass."""

    rows = development["budget"]["gated"]  # type: ignore[index]
    return tuple(
        sorted(
            (row["slice"], row["quantity"])
            for row in rows  # type: ignore[attr-defined]
            if row["status"] != CheckStatus.PASS.value
        )
    )


def development_waiver(
    protocol: dict[str, object], development_sha256: str
) -> dict[str, object] | None:
    """The decision owner's signed waiver of the DEVELOPMENT budget, if any.

    A waiver lives in a signed protocol amendment and names the exact
    DEVELOPMENT result (LF-normalised SHA-256) it accepts.
    """

    for amendment in protocol.get("amendments", ()):  # type: ignore[attr-defined]
        for waiver in amendment.get("waivers", ()):
            if (
                waiver.get("check") == "G14.budget"
                and waiver.get("population") == "DEVELOPMENT"
                and waiver.get("development_result_sha256") == development_sha256
            ):
                return dict(waiver)
    return None


def waiver_covers(
    waiver: dict[str, object] | None, development: dict[str, object]
) -> bool:
    """True only if the waiver names exactly the measured failing rows.

    Rows that were not measured (``NOT_RUN`` or ``ERROR``) are never waived.
    """

    if waiver is None:
        return False
    rows = development["budget"]["gated"]  # type: ignore[index]
    if any(
        row["status"] not in (CheckStatus.PASS.value, CheckStatus.FAIL.value)
        for row in rows  # type: ignore[attr-defined]
    ):
        return False
    named = tuple(
        sorted(
            (item["slice"], item["quantity"])
            for item in waiver.get("failing", ())  # type: ignore[attr-defined]
        )
    )
    return bool(named) and named == failing_budget_rows(development)


def holdout_blockers(
    development: dict[str, object], waiver: dict[str, object] | None = None
) -> tuple[str, ...]:
    """DEVELOPMENT results that keep HOLDOUT closed (ADR 0020, decision 4).

    HOLDOUT is consumed only for a candidate whose DEVELOPMENT budget passes
    on every gated slice, or whose measured budget failures the decision
    owner accepted in a signed waiver naming exactly those rows (ADR 0021),
    and whose pooled Spearman correlation meets the threshold.
    """

    blockers: list[str] = []
    if development["budget"]["gated_status"] != CheckStatus.PASS.value and not (  # type: ignore[index]
        waiver_covers(waiver, development)
    ):
        blockers.append("G14.budget")
    if development["confidence"]["pooled_spearman"] < SPEARMAN_MINIMUM:  # type: ignore[index,operator]
        blockers.append("G14.confidence")
    return tuple(blockers)


def _consumption_record(digest: str) -> ConsumptionRecord:
    return ConsumptionRecord(
        campaign_id="dogbone-v1",
        candidate_run_hash=str(frozen_candidate_record()["run_hash"]),
        freeze_record_sha256=text_sha256(CP_FREEZE_PATH),
        consumed_utc=_utc(),
        purpose=f"S14 HOLDOUT evaluation under protocol {digest}",
    )


def run_holdout(
    *,
    s10_run_root: Path,
    checkpoint_path: Path,
    run_root: Path,
    reveal_root: Path,
) -> dict[str, object]:
    """Consume HOLDOUT once, solve and score it with the frozen code."""

    protocol, digest = load_signed_protocol()
    development_path = run_root / DEVELOPMENT_RESULT
    development = json.loads(development_path.read_text("utf-8"))
    if development["protocol_sha256"] != digest:
        raise PermissionError("DEVELOPMENT results belong to another protocol")
    waiver = development_waiver(protocol, text_sha256(development_path))
    failed = holdout_blockers(development, waiver)
    if failed:
        raise PermissionError(
            "HOLDOUT stays closed because DEVELOPMENT failed: " + ", ".join(failed)
        )
    calibration = development["confidence"]["calibration"]
    curve = QuantileCurve(
        calibration["quantile"],
        tuple(calibration["upper_edges"]),
        tuple(calibration["values"]),
    )
    variant = development["confidence"]["indicator"]
    record_path = run_root / HOLDOUT_CONSUMPTION_NAME
    reveal_path = reveal_root / REVEAL_RELATIVE_PATH
    commitment = holdout_commitment()
    if record_path.exists():
        coordinates = reveal_after_consumption(
            record_path,
            candidate_run_hash=str(frozen_candidate_record()["run_hash"]),
            reader=lambda: parse_reveal(reveal_path.read_bytes(), commitment),
        )
    else:
        coordinates = consume_and_reveal_holdout(
            record_path=record_path,
            record=_consumption_record(digest),
            reveal_path=reveal_path,
            commitment=commitment,
        )
    model = load_frozen_candidate(checkpoint_path)
    normalization = load_s13c_normalization()
    profile = _load_run_execution_profile(s10_run_root)
    records: list[CaseRecord] = []
    errors: list[torch.Tensor] = []
    indicators: list[dict[str, torch.Tensor]] = []
    solve_seconds: list[float] = []
    for case in holdout_cases(coordinates):
        directory = run_root / "holdout"
        cached = (directory / "cases" / f"{case.case_id}.json").exists()
        request = s10_request(case, profile)
        if cached:
            solved = SolvedCase(case, request, None)  # type: ignore[arg-type]
        else:
            result, seconds = _solve(request)
            solve_seconds.append(seconds)
            solved = SolvedCase(case, request, result)
        record, error, variants = rehearse_case(
            model, solved, normalization, directory, allowed_roles=_HOLDOUT
        )
        records.append(record)
        errors.append(error)
        indicators.append(variants)
        print(f"HOLDOUT {len(records):2d}/64 {record.case_id}", flush=True)
    pooled_indicator = torch.cat(tuple(case[variant] for case in indicators))
    pooled_error = torch.cat(tuple(errors))
    report: dict[str, object] = {
        "evidence_role": "HOLDOUT",
        "protocol_sha256": digest,
        "case_count": len(records),
        "budget": budget_summary(records),
        "components": component_summary(records),
        "confidence": {
            "indicator": variant,
            "pooled_spearman": spearman(pooled_indicator, pooled_error),
            "calibration_coverage": curve.coverage(pooled_indicator, pooled_error),
        },
        "fem_solve_seconds": solve_seconds,
        "cases": [record_document(record) for record in records],
    }
    _write(run_root / HOLDOUT_RESULT, report)
    return report


def _status(document: dict[str, object] | None) -> CheckStatus | None:
    if document is None:
        return None
    return CheckStatus(document["budget"]["gated_status"])  # type: ignore[index]


def family_reduction(root: Path) -> tuple[CheckStatus, dict[str, object]]:
    """``G14.reducer``: the injected-case test and the twelve recorded families.

    The check passes only if every injected failure gives ``NOT_VERIFIED``
    and every registered family (ADR 0022) is ``PASS``.
    """

    evidence = read_registered(root)
    injection = inject_failures(evidence, evidence_level=ClaimLevel.HOLDOUT_VERIFIED)
    reduction = reduce_registered(evidence, evidence_level=ClaimLevel.HOLDOUT_VERIFIED)
    families = {
        f"{item.family.family_id} {item.family.name} ({item.family.gate_id})": (
            outcome.status.value
        )
        for item, outcome in zip(evidence, reduction.families, strict=True)
    }
    passed = injection.passed and reduction.verified
    return (
        CheckStatus.PASS if passed else CheckStatus.FAIL,
        {
            "injected_cases": injection.cases,
            "injected_not_verified": injection.not_verified,
            "control_claim": injection.control_level.value,
            "families": families,
        },
    )


def run_gate(*, run_root: Path, coverage_percent: float | None) -> dict[str, object]:
    """Assemble every G14 check from the recorded phases."""

    protocol, digest = load_signed_protocol()

    def load(name: str) -> dict[str, object] | None:
        path = run_root / name
        return json.loads(path.read_text("utf-8")) if path.exists() else None

    development = load(DEVELOPMENT_RESULT)
    reducer_status, reducer_detail = family_reduction(REPOSITORY_ROOT)
    waiver_reason = None
    if development is not None:
        waiver = development_waiver(
            protocol, text_sha256(run_root / DEVELOPMENT_RESULT)
        )
        if development["budget"]["gated_status"] == CheckStatus.FAIL.value and (  # type: ignore[index]
            waiver_covers(waiver, development)
        ):
            assert waiver is not None
            waiver_reason = (
                f"accepted by the decision owner ({waiver['decision_record']}); "
                "computed FAIL on "
                + ", ".join(
                    f"{item['slice']} {item['quantity']} {item['statistic']:.5f}"
                    for item in waiver["failing"]  # type: ignore[attr-defined]
                )
            )
    holdout = load(HOLDOUT_RESULT)
    first = refusal_run()
    second = refusal_run()
    stable = (
        first["mismatched_fixtures"] == second["mismatched_fixtures"]
        and first["false_refusals"] == second["false_refusals"]
        and not first["unstable_reason_codes"]
    )
    latency = None
    spearman_value = None
    if development is not None:
        timing = development["latency"]
        latency = {
            "median_seconds": timing["model_u_and_p_median_seconds"],  # type: ignore[index]
            "p99_seconds": timing["model_u_and_p_p99_seconds"],  # type: ignore[index]
            "speedup": timing["speedup_u_and_p"],  # type: ignore[index]
        }
        spearman_value = development["confidence"]["pooled_spearman"]  # type: ignore[index]
    measurements = G14Measurements(
        development_budget=_status(development),
        development_budget_waiver=waiver_reason,
        holdout_budget=_status(holdout),
        reducer=reducer_status,
        reducer_detail=reducer_detail,
        report_sections=True,
        fixture_count=first["fixture_count"],  # type: ignore[arg-type]
        refused_with_expected_code=first["refused_with_expected_code"],  # type: ignore[arg-type]
        false_refusal_rate=first["false_refusal_rate"],  # type: ignore[arg-type]
        reason_codes_stable=stable,
        development_spearman=spearman_value,  # type: ignore[arg-type]
        latency=latency,
        holdout_order=(run_root / HOLDOUT_CONSUMPTION_NAME).exists()
        if holdout is not None
        else None,
        no_certified=True,
        coverage_percent=coverage_percent,
        holdout_calibration_coverage=None
        if holdout is None
        else holdout["confidence"]["calibration_coverage"],  # type: ignore[index]
    )
    report = g14_report(measurements)
    document = {
        "gate_id": "G14",
        "step": "S14",
        "protocol_sha256": digest,
        "claim_level": report.claim_level.value,
        "verdict": "PASS"
        if all(
            check.status is CheckStatus.PASS
            for check in report.checks
            if check.required
        )
        else "FAIL",
        "report": json.loads(render_report(report)),
    }
    _write(run_root / GATE_RESULT, document)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("development", "holdout", "gate"))
    parser.add_argument("--s10-run-root", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--reveal-root", type=Path)
    parser.add_argument("--coverage-percent", type=float)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    if options.phase == "development":
        run_development(
            s10_run_root=options.s10_run_root,
            checkpoint_path=options.checkpoint_path,
            run_root=options.run_root,
        )
    elif options.phase == "holdout":
        run_holdout(
            s10_run_root=options.s10_run_root,
            checkpoint_path=options.checkpoint_path,
            run_root=options.run_root,
            reveal_root=options.reveal_root,
        )
    else:
        run_gate(
            run_root=options.run_root,
            coverage_percent=options.coverage_percent,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "DEVELOPMENT_RESULT",
    "GATE_RESULT",
    "HOLDOUT_RESULT",
    "build_parser",
    "development_waiver",
    "family_reduction",
    "failing_budget_rows",
    "holdout_blockers",
    "main",
    "record_document",
    "run_development",
    "run_gate",
    "run_holdout",
    "waiver_covers",
)
