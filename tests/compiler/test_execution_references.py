"""Synthetic reference outcomes and comparisons; no FEM solves."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from piginot.compiler import reference_jobs, reference_metrics
from piginot.compiler.planning import build_plan
from piginot.contracts import CanonicalQuantity, hash_spec
from piginot.fem.backend import _build_basis
from piginot.fem.models import FemAttempt, FemFailureCode, FemResult
from piginot.operators.architecture import OperatorOutput
from piginot.problem.requests import fem_request
from tests.compiler.plan_fixtures import inputs
from tests.materials.helpers import BUILTIN_VALUES, material
from tests.problem_helpers import plate_mesh


class Adapter:
    def geometry(self, spec, case, *, level=1):
        values = case.primitive_coordinates
        return plate_mesh(
            width=values["width"],
            height=values["height"],
            columns=2 * level,
            rows=level,
        )


def result_for(request):
    _, basis, scalar = _build_basis(request)
    xy = scalar.doflocs.T
    displacement = np.column_stack((0.07 * xy[:, 0], -0.02 * xy[:, 1]))
    points = np.moveaxis(np.asarray(basis.global_coordinates()), 0, -1)
    stress = np.zeros((*basis.dx.shape, 3, 3))
    stress[..., 0, 0], stress[..., 1, 1] = 2.0, 1.0
    recovered = np.broadcast_to(np.diag([2.0, 1.0, 0.0]), (len(xy), 3, 3))
    return FemResult(
        request.request_id,
        request.mesh.mesh_id,
        request.config.element_order,
        tuple(map(tuple, xy)),
        tuple(map(tuple, displacement)),
        points.tolist(),
        basis.dx.tolist(),
        stress.tolist(),
        stress.tolist(),
        recovered.tolist(),
        {},
        (),
        1.0,
        0.0,
        1.0,
        (),
        {},
    )


def prediction(case_id, points):
    u = torch.stack((0.07 * points[:, 0], -0.02 * points[:, 1]), -1)[None]
    stress = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    return OperatorOutput(u, stress.expand(1, len(points), 2, 2))


def fixture(tmp_path):
    spec_path, config_path, _, descriptor = inputs(tmp_path)
    plan = build_plan(spec_path, config_path, tmp_path / "run", descriptor=descriptor)
    case = next(case for case in plan.manifest.cases if case.case_id == "screen-001")
    quantities = {
        p.name: CanonicalQuantity(case.primitive_coordinates[p.name], p.unit)
        for p in plan.problem.parameters
    }
    request = fem_request(
        plan.problem,
        quantities,
        Adapter().geometry(plan.problem, case),
        "synthetic",
        plan.config.execution.reference,
    )
    return plan, request


class Backend:
    def __init__(self, *, interrupt=(), fail=()):
        self.calls = []
        self.interrupt, self.fail = interrupt, fail

    def solve_many(self, requests):
        return tuple(self.solve(request) for request in requests)

    def solve_from(self, request, vector, factors):
        assert factors == (1.0,)
        assert np.isfinite(vector).all()
        return self.solve(request)

    def solve(self, request):
        self.calls.append(request)
        if len(self.calls) in self.interrupt:
            raise KeyboardInterrupt
        if len(self.calls) in self.fail:
            return FemAttempt(
                request,
                request.recipe_hash,
                False,
                None,
                None,
                None,
                FemFailureCode.NEWTON_NON_CONVERGENCE,
                "fixed numerical failure",
            )
        result = result_for(request)
        return FemAttempt(
            request,
            request.recipe_hash,
            True,
            result,
            hash_spec(result, "content"),
            None,
            None,
            "succeeded",
        )


def test_exact_synthetic_field_has_zero_weighted_and_peak_errors(tmp_path):
    plan, request = fixture(tmp_path)
    errors = reference_metrics.compare_prediction(
        plan.problem,
        request,
        result_for(request),
        lambda points: prediction("x", points),
    )
    for name in ("u_relative_l2", "p_relative_l2", "absolute_peak_error"):
        assert errors[name] == pytest.approx(0.0, abs=1e-14)
    assert errors["qois"]["peak"]["reference"] == pytest.approx(5**0.5)


def test_cauchy_qoi_uses_deformation_and_declared_stress_field(tmp_path):
    from piginot.fem.backend import _kernel_material
    from piginot.physics import plane_stress_closure

    plan, request = fixture(tmp_path)
    spec = replace(
        plan.problem,
        qois=(
            plan.problem.qois[0],
            replace(plan.problem.qois[0], name="sigma_peak", field="sigma"),
        ),
    )
    result = result_for(request)
    deformation = torch.diag(torch.tensor([1.07, 0.98], dtype=torch.float64))
    f33 = plane_stress_closure(deformation[None], _kernel_material(request.material))[0]
    embedded = np.diag([1.07, 0.98, float(f33)])
    result = replace(
        result,
        deformation_gradient=np.broadcast_to(
            embedded, np.asarray(result.deformation_gradient).shape
        ).tolist(),
    )

    def predict(points):
        output = prediction("x", points)
        return SimpleNamespace(
            u=output.u,
            P=output.P,
            gradient=(deformation - torch.eye(2)).expand(1, len(points), 2, 2),
        )

    errors = reference_metrics.compare_prediction(spec, request, result, predict)
    expected = np.linalg.norm(np.diag([2.14, 0.98]) / np.linalg.det(embedded))
    assert errors["qois"]["sigma_peak"]["reference"] == pytest.approx(expected)
    assert errors["qois"]["sigma_peak"]["predicted"] == pytest.approx(expected)
    assert errors["qois"]["peak"]["reference"] == pytest.approx(5**0.5)
    assert errors["absolute_peak_error"] == pytest.approx(0.0, abs=1e-14)


def test_cauchy_qoi_refuses_missing_operator_gradient(tmp_path):
    plan, request = fixture(tmp_path)
    spec = replace(plan.problem, qois=(replace(plan.problem.qois[0], field="sigma"),))
    result = result_for(request)
    result = replace(
        result,
        deformation_gradient=np.broadcast_to(
            np.eye(3), np.asarray(result.deformation_gradient).shape
        ).tolist(),
    )
    with pytest.raises(ValueError, match="gradient"):
        reference_metrics.compare_prediction(
            spec, request, result, lambda points: prediction("x", points)
        )


@pytest.mark.parametrize(
    "invalid", ["reference", "operator", "dtype", "device", "shape"]
)
def test_cauchy_qoi_refuses_invalid_deformation_or_gradient(tmp_path, invalid):
    plan, request = fixture(tmp_path)
    spec = replace(plan.problem, qois=(replace(plan.problem.qois[0], field="sigma"),))
    result = result_for(request)
    deformation = np.diag([-1.0, 1.0, 1.0]) if invalid == "reference" else np.eye(3)
    result = replace(
        result,
        deformation_gradient=np.broadcast_to(
            deformation, np.asarray(result.deformation_gradient).shape
        ).tolist(),
    )

    def predict(points):
        output = prediction("x", points)
        gradient = torch.zeros((1, len(points), 2, 2), dtype=torch.float64)
        if invalid == "operator":
            gradient[..., 0, 0] = -2.0
        elif invalid == "dtype":
            gradient = gradient.to(torch.float32)
        elif invalid == "device":
            gradient = gradient.to("meta")
        elif invalid == "shape":
            gradient = gradient[:, :-1]
        return SimpleNamespace(u=output.u, P=output.P, gradient=gradient)

    with pytest.raises(ValueError, match="gradient|determinant"):
        reference_metrics.compare_prediction(spec, request, result, predict)


def test_reference_resume_preserves_failure_and_runs_pending_tail(
    tmp_path, monkeypatch
):
    plan, _ = fixture(tmp_path)
    backend = Backend(interrupt=(2,), fail=(1,))
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)
    with pytest.raises(KeyboardInterrupt):
        reference_jobs.evaluate_case(
            plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
        )
    saved = {p: p.read_bytes() for p in plan.output_path.rglob("outcome-*.json")}
    outcome = reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    assert outcome["status"] == "INCOMPLETE"
    assert len(backend.calls) == 3
    assert all(path.read_bytes() == payload for path, payload in saved.items())
    assert len(outcome["references"]) == 2
    reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    assert len(backend.calls) == 3


def test_complete_reference_is_discrete_only_and_cached(tmp_path, monkeypatch):
    plan, _ = fixture(tmp_path)
    backend = Backend()
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)
    outcome = reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    assert outcome["status"] == "PASS"
    assert outcome["scientific_status"] == "NOT_VERIFIED"
    assert outcome["reference_diagnostics"]["status"] == "UNCALIBRATED"
    assert outcome["discrete_fem_errors"]["p_relative_l2"] == 0.0
    second = reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    assert outcome == second
    assert len(backend.calls) == 2


@pytest.mark.parametrize("fail", [(), (1,)])
def test_case_report_references_keep_bound_raw_evidence_without_payload_copies(
    tmp_path, monkeypatch, fail
):
    import hashlib

    from piginot.compiler.execution_records import load_bound
    from piginot.contracts import canonical_json_bytes

    plan, _ = fixture(tmp_path)
    backend = Backend(fail=fail)
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)
    outcome = reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    raw_bytes = 0
    for reference in outcome["references"]:
        assert "request_utf8" not in reference
        assert "result_utf8" not in reference
        evidence = reference["evidence"]
        path = plan.output_path / evidence["path"]
        raw = path.read_bytes()
        raw_bytes += len(raw)
        assert hashlib.sha256(raw).hexdigest() == evidence["sha256"]
        saved = load_bound(path, evidence["binding_sha256"])
        assert saved["request_utf8"]
        assert saved["status"] == reference["status"]
        assert saved["request_sha256"] == reference["request_sha256"]
        if reference["status"] == "PASS":
            assert saved["result_utf8"]
            assert saved["result_sha256"] == reference["result_sha256"]
    assert len(canonical_json_bytes(outcome["references"])) < raw_bytes / 2
    assert (
        reference_jobs.validate_case_cache(
            plan, Adapter(), "screen-001", "train-screen", {}
        )
        == outcome["references"]
    )
    assert len(backend.calls) == 2

    from piginot.compiler.execution import _Run
    from piginot.contracts import strict_json_loads

    run = _Run(plan, object(), False)
    cases = [{**outcome, "case_id": f"synthetic-{i}"} for i in range(16)]
    run.stages["train-screen"] = {"status": "PASS", "cases": cases}
    run.report()
    report = plan.output_path / "report.json"
    markdown = plan.output_path / "report.md"
    assert report.stat().st_size < 100_000
    assert markdown.stat().st_size < 200_000
    assert (
        strict_json_loads(report.read_bytes())["stages"]["train-screen"]["cases"]
        == cases
    )
    assert '"result_utf8"' not in markdown.read_text()


@pytest.mark.parametrize(
    "partition,case",
    [
        ("H2", "screen-001"),
        ("train-screen", "fit-001"),
        ("public-development", "development-001"),
    ],
)
def test_unauthorized_roster_refuses_before_geometry(tmp_path, partition, case):
    plan, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="authorized"):
        reference_jobs.evaluate_case(
            plan, SimpleNamespace(), case, partition, {}, predict=prediction
        )


def test_warm_initializer_and_final_survive_two_interruptions(tmp_path, monkeypatch):
    plan, _ = fixture(tmp_path)
    policy = replace(plan.config.reference, initialization="neo-hookean-full-load-hgo")
    plan = replace(plan, config=replace(plan.config, reference=policy))
    original = reference_jobs.fem_request

    def hgo_request(*args):
        return replace(original(*args), material=material("hgo", BUILTIN_VALUES["hgo"]))

    monkeypatch.setattr(reference_jobs, "fem_request", hgo_request)
    backend = Backend(interrupt=(2, 4))
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)
    snapshots = {}
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            reference_jobs.evaluate_case(
                plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
            )
        assert all(path.read_bytes() == value for path, value in snapshots.items())
        snapshots = {
            p: p.read_bytes() for p in plan.output_path.rglob("outcome-*.json")
        }
    outcome = reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    assert outcome["status"] == "PASS"
    assert len(backend.calls) == 6
    assert len(outcome["references"]) == 4
    assert all(path.read_bytes() == value for path, value in snapshots.items())
    assert backend.calls[0].material.model == "compressible_neo_hookean"
    assert backend.calls[1].material.model == "hgo"
    reference_jobs.validate_case_cache(
        plan, Adapter(), "screen-001", "train-screen", {}
    )
    assert len(backend.calls) == 6


def test_initializer_failure_does_not_run_final_or_retry(tmp_path, monkeypatch):
    plan, request = fixture(tmp_path)
    hgo = replace(request, material=material("hgo", BUILTIN_VALUES["hgo"]))
    initial = reference_jobs.initializer_request(hgo)
    mu, bulk = 300.0, 500.0
    assert initial.material.properties["elastic_modulus"].value == pytest.approx(
        9 * mu * bulk / (3 * bulk + mu)
    )
    assert initial.config == hgo.config
    assert (
        reference_jobs._solve(
            hgo, initializer={"status": "FAIL"}, initial_request=initial
        )["status"]
        == "INCOMPLETE"
    )
    with pytest.raises(ValueError, match="requires HGO"):
        reference_jobs.initializer_request(request)


def test_geometry_failure_does_not_hide_successful_tail(tmp_path, monkeypatch):
    plan, _ = fixture(tmp_path)
    backend = Backend()
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)

    class Missing(Adapter):
        def geometry(self, spec, case, *, level=1):
            if level == 1:
                raise ValueError("geometry failed")
            return super().geometry(spec, case, level=level)

    outcome = reference_jobs.evaluate_case(
        plan, Missing(), "screen-001", "train-screen", {}, predict=prediction
    )
    assert outcome["status"] == "INCOMPLETE"
    assert [row["status"] for row in outcome["references"]] == ["INCOMPLETE", "PASS"]
    assert len(backend.calls) == 1
    reference_jobs.validate_case_cache(
        plan, Missing(), "screen-001", "train-screen", {}
    )


def test_unexpected_geometry_error_is_terminal_error_and_keeps_reference_tail(
    tmp_path, monkeypatch
):
    plan, _ = fixture(tmp_path)
    backend = Backend()
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)

    class BrokenAdapter(Adapter):
        def geometry(self, spec, case, *, level=1):
            if level == 1:
                raise RuntimeError("geometry implementation failed")
            return super().geometry(spec, case, level=level)

    outcome = reference_jobs.evaluate_case(
        plan, BrokenAdapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    assert outcome["status"] == "ERROR"
    assert outcome["references"][0]["error"] == {
        "category": "RuntimeError",
        "message": "geometry implementation failed",
    }
    assert outcome["references"][1]["status"] == "PASS"
    assert len(backend.calls) == 1
    reference_jobs.validate_case_cache(
        plan, BrokenAdapter(), "screen-001", "train-screen", {}
    )


def test_unexpected_solver_error_remains_terminal_and_changed_binding_refuses(
    tmp_path, monkeypatch
):
    plan, _ = fixture(tmp_path)

    def failed_backend():
        raise RuntimeError("solver unavailable")

    monkeypatch.setattr(reference_jobs, "_backend", failed_backend)
    outcome = reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    assert outcome["status"] == "ERROR"
    reference_jobs.validate_case_cache(
        plan, Adapter(), "screen-001", "train-screen", {}
    )
    with pytest.raises(ValueError, match="binding differs"):
        reference_jobs.evaluate_case(
            plan,
            Adapter(),
            "screen-001",
            "train-screen",
            {"checkpoint": "changed"},
            predict=prediction,
        )


def test_solver_result_and_cache_integrity_are_checked(tmp_path, monkeypatch):
    _, request = fixture(tmp_path)
    backend = Backend()
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)
    outcome = reference_jobs._solve(request)
    with pytest.raises(ValueError, match="bytes digest"):
        reference_jobs._decode_result({**outcome, "result_sha256": "0" * 64}, request)
    for payload, message in ((b"[]", "must be an object"), (b"{}", "schema differs")):
        import hashlib

        with pytest.raises(ValueError, match=message):
            reference_jobs._decode_result(
                {
                    **outcome,
                    "result_utf8": payload.decode(),
                    "result_sha256": hashlib.sha256(payload).hexdigest(),
                },
                request,
            )
    with pytest.raises(ValueError, match="identity differs"):
        reference_jobs._decode_result({**outcome, "result_hash": "0" * 64}, request)
    valid = backend.solve(request)
    for attempt, message in (
        (replace(valid, recipe_hash="0" * 64), "different recipe"),
        (replace(valid, result=None), "matching result"),
    ):
        monkeypatch.setattr(
            backend, "solve_many", lambda requests, attempt=attempt: (attempt,)
        )
        with pytest.raises(ValueError, match=message):
            reference_jobs._solve(request)


def test_default_backend_is_loaded_only_when_requested(monkeypatch):
    import piginot.fem

    sentinel = object()
    monkeypatch.setattr(piginot.fem, "ScikitFemBackend", lambda: sentinel)
    assert reference_jobs._backend() is sentinel


@pytest.mark.parametrize(
    "change,message",
    [
        ("identity", "identity differs"),
        ("dof", "DOF coordinates"),
        ("quadrature", "quadrature differs"),
        ("nonfinite", "nonfinite"),
    ],
)
def test_bad_reference_layout_is_refused(tmp_path, change, message):
    plan, request = fixture(tmp_path)
    result = result_for(request)
    if change == "identity":
        result = replace(result, request_id="other")
    elif change == "dof":
        values = np.asarray(result.displacement_dof_coordinates).copy()
        values[0, 0] += 0.01
        result = replace(result, displacement_dof_coordinates=values)
    elif change == "quadrature":
        result = replace(
            result, quadrature_weights=np.zeros_like(result.quadrature_weights)
        )
    else:
        result = replace(result, displacement=np.full_like(result.displacement, np.nan))
    with pytest.raises(ValueError, match=message):
        reference_metrics.compare_prediction(
            plan.problem, request, result, lambda points: prediction("x", points)
        )


def test_zero_norm_and_empty_qoi_refusals(tmp_path):
    assert (
        reference_metrics._relative(np.zeros((1, 2)), np.zeros((1, 2)), np.ones(1))
        == 0.0
    )
    with pytest.raises(ValueError, match="zero reference"):
        reference_metrics._relative(np.ones((1, 2)), np.zeros((1, 2)), np.ones(1))
    plan, request = fixture(tmp_path)
    qoi = replace(plan.problem.qois[0], exclude_regions=("top",))
    with pytest.raises(ValueError, match="no vertices"):
        reference_metrics.compare_prediction(
            replace(plan.problem, qois=(qoi,)),
            request,
            result_for(request),
            lambda points: prediction("x", points),
        )


def test_comparison_refuses_an_inference_dtype_change(tmp_path):
    plan, request = fixture(tmp_path)

    def changed(points):
        output = prediction("x", points)
        return OperatorOutput(output.u.float(), output.P.float())

    with pytest.raises(ValueError, match="float64 CPU"):
        reference_metrics.compare_prediction(
            plan.problem, request, result_for(request), changed
        )


def test_known_scale_error_is_reported_without_a_bound(tmp_path):
    plan, request = fixture(tmp_path)

    def scaled(points):
        output = prediction("x", points)
        return OperatorOutput(output.u * 1.1, output.P * 1.1)

    result = result_for(request)
    errors = reference_metrics.compare_prediction(plan.problem, request, result, scaled)
    for name in ("u_relative_l2", "p_relative_l2", "absolute_peak_error"):
        assert errors[name] == pytest.approx(0.1)
    diagnostics = reference_metrics.refinement_diagnostics((result, result))
    assert diagnostics["successive_integral_deltas"] == [0.0]
    assert diagnostics["total_error_bound"] is None


def test_completed_case_validation_reads_without_solving_or_writing(
    tmp_path, monkeypatch
):
    plan, _ = fixture(tmp_path)
    backend = Backend(fail=(1,))
    monkeypatch.setattr(reference_jobs, "_backend", lambda: backend)
    reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    snapshot = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in plan.output_path.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        reference_jobs, "_backend", lambda: pytest.fail("cached case solved")
    )
    reference_jobs.validate_case_cache(
        plan, Adapter(), "screen-001", "train-screen", {}
    )
    assert snapshot == {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in plan.output_path.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("change", ["missing", "identity", "bytes", "status"])
def test_completed_case_validation_refuses_invalid_inner_cache(
    tmp_path, monkeypatch, change
):
    from piginot.compiler.execution_records import load_bound, save_bound

    plan, _ = fixture(tmp_path)
    monkeypatch.setattr(reference_jobs, "_backend", lambda: Backend())
    reference_jobs.evaluate_case(
        plan, Adapter(), "screen-001", "train-screen", {}, predict=prediction
    )
    store, prepared = reference_jobs._case_references(
        plan, Adapter(), "screen-001", "train-screen", {}
    )
    job = prepared[0][0][0]
    path = store._path(job)
    if change == "missing":
        path.unlink()
    else:
        row = load_bound(path, store._binding(job))
        if change == "identity":
            row["request_id"] = "foreign"
        elif change == "bytes":
            row["request_utf8"] = "{}"
        else:
            row["status"] = "RUNNING"
        save_bound(path, store._binding(job), row)
    with pytest.raises(ValueError, match="reference cache"):
        reference_jobs.validate_case_cache(
            plan, Adapter(), "screen-001", "train-screen", {}
        )
