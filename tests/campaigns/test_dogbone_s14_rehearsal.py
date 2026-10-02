from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from campaigns.dogbone_v1.s10_requests import _load_public_sources
from campaigns.dogbone_v1.s14_candidate import (
    frozen_candidate_record,
    served_geometry,
    slice_ids,
)
from campaigns.dogbone_v1.s14_reference import FemReferenceTerms
from campaigns.dogbone_v1.s14_rehearsal import (
    GATED_AXES,
    GATED_SLICES,
    REPORTED_AXES,
    REPORTED_SLICES,
    CaseRecord,
    budget_entry,
    budget_summary,
    confidence_pilot,
    latency_run,
    pooled_spearman,
    select_variant,
    surrogate_components,
)
from piginot.doe import EvidenceRole
from piginot.verification import INDICATOR_VARIANTS, CheckStatus, evaluate_budget
from piginot.verification.budget import BudgetTolerance


def test_slice_catalogues_match_adr_0019() -> None:
    assert len(GATED_SLICES) == 21
    assert GATED_SLICES[0] == "all"
    assert {s.split(":")[0] for s in GATED_SLICES[1:]} == set(GATED_AXES)
    assert len(REPORTED_SLICES) == 1 + 2 + 4 + 4 + 4
    assert {s.split(":")[0] for s in REPORTED_SLICES[1:]} == set(REPORTED_AXES)


def test_public_case_slices_use_the_s9_strata() -> None:
    _, manifest = _load_public_sources()
    case = manifest.cases[0]
    slices = slice_ids(case, GATED_AXES)

    assert slices[0] == "all" and len(slices) == 6
    assert all(s in GATED_SLICES for s in slices)
    assert slices[1] == f"L_total:{case.strata['L_total']}"


def _terms(value: float = 0.001) -> FemReferenceTerms:
    return FemReferenceTerms("c", value, 2 * value, value, 3 * value, value, 261)


def _surrogate(field: float = 0.01) -> dict[str, float]:
    return {
        "p_relative_l2": field,
        "absolute_peak_error": field,
        "u_relative_l2": field / 10,
    }


def test_budget_entry_sums_components_and_missing_terms_are_not_run() -> None:
    entry = budget_entry("c", ("all",), _surrogate(), _terms())
    parts = _terms().components()
    assert entry.total("p_relative_l2") == pytest.approx(
        0.01 + parts["p_relative_l2"]["discretisation"] + 0.001
    )
    missing = budget_entry("c", ("all",), _surrogate(), None)
    assert missing.total("u_relative_l2") is None
    tolerances = (BudgetTolerance("u_relative_l2", 0.02),)
    (result,) = evaluate_budget((missing,), tolerances, ("all",))
    assert result.status is CheckStatus.NOT_RUN


def _record(case_id: str, field: float, slices: tuple[str, ...]) -> CaseRecord:
    terms = _terms()
    from dataclasses import asdict

    return CaseRecord(
        case_id=case_id,
        slices=slices,
        reported_slices=("fillet_to_gauge_ratio:0",),
        surrogate=_surrogate(field),
        terms=asdict(terms),
        components=terms.components(),
        coarse_solve_seconds=1.0,
        minimum_vertex_determinant=0.99,
    )


def test_budget_summary_passes_and_fails_on_the_statistic() -> None:
    slices = ("all", *(f"{axis}:0" for axis in GATED_AXES))
    passing = [_record(f"c{i}", 0.01, slices) for i in range(12)]
    summary = budget_summary(passing)
    gated = {(row["slice"], row["quantity"]): row for row in summary["gated"]}  # type: ignore[union-attr]

    assert gated[("all", "p_relative_l2")]["status"] == "PASS"
    assert gated[("L_total:1", "p_relative_l2")]["status"] == "NOT_RUN"
    assert summary["gated_status"] == "NOT_RUN"

    failing = [_record(f"c{i}", 0.2, slices) for i in range(12)]
    rows = budget_summary(failing)["gated"]
    assert any(r["status"] == "FAIL" for r in rows)  # type: ignore[union-attr,index]


def test_variant_selection_and_pooled_spearman() -> None:
    assert (
        select_variant({"pointwise": 0.5, "patch-mean": 0.7, "patch-mean-times-h": 0.6})
        == "patch-mean"
    )
    assert select_variant({v: 0.4 for v in INDICATOR_VARIANTS}) == INDICATOR_VARIANTS[0]

    generator = torch.Generator().manual_seed(1)
    errors = [
        torch.rand(50, generator=generator, dtype=torch.float64) for _ in range(3)
    ]
    indicators = [
        {
            "pointwise": e
            + 0.5 * torch.rand(50, generator=generator, dtype=torch.float64),
            "patch-mean": e,
            "patch-mean-times-h": -e,
        }
        for e in errors
    ]
    rho = pooled_spearman(errors, indicators)
    assert rho["patch-mean"] == pytest.approx(1.0)
    assert rho["patch-mean-times-h"] == pytest.approx(-1.0)
    pilot = confidence_pilot(errors, indicators)
    assert pilot["selected_variant"] == "patch-mean"
    assert pilot["vertex_count"] == 150
    assert pilot["train_meets_threshold"] is True


def test_surrogate_components_use_the_absolute_peak_error() -> None:
    scored = SimpleNamespace(
        diagnostics=SimpleNamespace(
            p_relative_l2=0.01, signed_peak_error=-0.02, u_relative_l2=0.001
        )
    )
    assert surrogate_components(scored) == {  # type: ignore[arg-type]
        "p_relative_l2": 0.01,
        "absolute_peak_error": 0.02,
        "u_relative_l2": 0.001,
    }


def test_latency_requires_single_thread_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(RuntimeError):
        latency_run(torch.nn.Identity(), [], None)


def test_frozen_record_and_role_gate() -> None:
    record = frozen_candidate_record()
    assert record["parameter_count"] == 573_474
    _, manifest = _load_public_sources()
    development = next(
        case for case in manifest.cases if case.role is EvidenceRole.DEVELOPMENT
    )
    with pytest.raises(PermissionError):
        served_geometry(
            development,
            SimpleNamespace(request_id=development.case_id),  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            allowed_roles=frozenset({EvidenceRole.TRAIN}),
        )
    train = next(case for case in manifest.cases if case.role is EvidenceRole.TRAIN)
    with pytest.raises(ValueError):
        served_geometry(
            train,
            SimpleNamespace(request_id="other"),  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            allowed_roles=frozenset({EvidenceRole.TRAIN}),
        )
    assert math.isfinite(float(record["checkpoint_step"]))  # type: ignore[arg-type]
