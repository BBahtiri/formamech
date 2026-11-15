"""The compiler exposes field convergence alongside the legacy integral delta."""

from dataclasses import replace

import pytest

from piginot.compiler import reference_convergence, reference_jobs
from tests.compiler.test_cad_geometry_execution import prepared_cad as prepared_cad
from tests.compiler.test_execution_references import Backend, prediction


def test_cad_reference_report_contains_measured_adjacent_field_changes(
    prepared_cad,
    monkeypatch,
):
    plan, adapter = prepared_cad
    backend = Backend()
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)
    calls = []

    def measured(
        spec, coarse_request, coarse_result, fine_request, fine_result, *, tolerances
    ):
        assert tolerances is None
        assert coarse_request.mesh.mesh_id == coarse_result.mesh_id
        assert fine_request.mesh.mesh_id == fine_result.mesh_id
        assert spec == plan.problem
        calls.append((coarse_request.mesh.mesh_id, fine_request.mesh.mesh_id))
        return {"status": "MEASURED_CHANGES_ONLY", "error_bound": "NOT_ESTABLISHED"}

    monkeypatch.setattr(reference_convergence, "compare_fem_pair", measured)
    case_id = plan.config.source.screen_case_ids[0]
    result = reference_jobs.evaluate_case(
        plan,
        adapter,
        case_id,
        "train-screen",
        {},
        predict=prediction,
    )
    diagnostics = result["reference_diagnostics"]
    assert diagnostics["status"] == "UNCALIBRATED"
    assert (
        diagnostics["same_polygon_field_changes"][0]["status"]
        == "MEASURED_CHANGES_ONLY"
    )
    assert len(calls) == len(plan.config.reference.refinement_levels) - 1
    assert all(a != b for a, b in calls)
    assert result["claim_level"] == "NOT_VERIFIED"


def test_failed_cad_field_validation_is_not_silently_replaced(
    prepared_cad, monkeypatch
):
    plan, adapter = prepared_cad
    monkeypatch.setattr(reference_jobs, "_backend", lambda: Backend())

    def refused(*args, **kwargs):
        raise ValueError("stored FEM layout differs")

    monkeypatch.setattr(reference_convergence, "compare_fem_pair", refused)
    with pytest.raises(ValueError, match="stored FEM layout differs"):
        reference_jobs.evaluate_case(
            plan,
            adapter,
            plan.config.source.screen_case_ids[0],
            "train-screen",
            {},
            predict=prediction,
        )


def test_cad_report_supports_a_ladder_without_the_middle_level(
    prepared_cad, monkeypatch
):
    plan, adapter = prepared_cad
    policy = replace(plan.config.reference, refinement_levels=(1, 4))
    plan = replace(plan, config=replace(plan.config, reference=policy))
    backend = Backend()
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)
    result = reference_jobs.evaluate_case(
        plan,
        adapter,
        plan.config.source.screen_case_ids[0],
        "train-screen",
        {},
        predict=prediction,
    )
    pairs = result["reference_diagnostics"]["same_polygon_field_changes"]
    assert len(pairs) == 1
    assert pairs[0]["status"] == "MEASURED_CHANGES_ONLY"
    assert len(backend.calls) == 2
