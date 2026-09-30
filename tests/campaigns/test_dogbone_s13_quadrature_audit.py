from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rfc8785
import torch

from campaigns.dogbone_v1 import s13_quadrature_audit as audit
from campaigns.dogbone_v1.s11_contingency_models import model_state_hash
from campaigns.dogbone_v1.s13_objective import _COORDINATE_CONVERSION, _scales
from campaigns.dogbone_v1.s13_stress_capacity import (
    StressCapacityPlan,
    build_stress_arm,
    frozen_state_hash,
    trainable_names,
)
from piginot.contracts import canonical_json_bytes, strict_json_loads
from piginot.doe import EvidenceRole
from piginot.objectives import displacement_gradient
from piginot.operators import OperatorOutput, module_digest
from tests.campaigns.test_dogbone_s13_objective import _model
from tests.campaigns.test_dogbone_s13_screen import _case
from tests.fem.test_verification_norms import _request, _synthetic_exact_result


def test_comparison_plan_is_fixed_train_only_and_canonical() -> None:
    plan = audit.QuadratureAuditPlan()
    assert plan.query_chunk_size == 512
    assert plan.case_id == "candidate-002485"
    assert plan.role is EvidenceRole.TRAIN
    assert plan.non_gate is True
    assert len(plan.content_hash) == 64
    assert canonical_json_bytes(plan) == canonical_json_bytes(
        audit.QuadratureAuditPlan()
    )
    with pytest.raises(FrozenInstanceError):
        plan.query_chunk_size = 1
    for changes in (
        {"query_chunk_size": 513},
        {"query_chunk_size": True},
        {"case_id": "other"},
        {"role": EvidenceRole.DEVELOPMENT},
        {"non_gate": False},
    ):
        with pytest.raises(ValueError, match="frozen"):
            replace(plan, **changes)


def test_rebuilt_p2_quadrature_retains_cell_sided_units_and_dx_weights() -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    fields = audit.rebuild_fem_quadrature(request, result, (1,))
    assert fields.xy_mm.shape == (2, 7, 2)
    assert fields.u_mm.shape == (2, 7, 2)
    assert fields.gradient.shape == fields.raw_p_mpa.shape == (2, 7, 2, 2)
    assert fields.area_weights.shape == (2, 7)
    assert fields.area_weights.sum().item() == pytest.approx(1.0)
    assert fields.root_cell_mask.tolist() == [True, False]
    assert fields.gradient_max_absolute_difference < 1.0e-14
    torch.testing.assert_close(
        fields.xy_mm,
        torch.tensor(np.asarray(result.quadrature_points), dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(fields.u_mm[..., 0], 0.2 * fields.xy_mm[..., 0])
    torch.testing.assert_close(
        fields.gradient[..., 0, 0], torch.full((2, 7), 0.2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        fields.raw_p_mpa,
        torch.tensor(np.asarray(result.first_piola_stress)[..., :2, :2]),
        rtol=0,
        atol=0,
    )
    for value in (fields.xy_mm, fields.u_mm, fields.gradient, fields.raw_p_mpa):
        assert value.dtype is torch.float64 and not value.requires_grad


def test_reconstruction_uses_p2_edge_displacements_not_vertex_interpolation() -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    dofs = np.asarray(result.displacement_dof_coordinates)
    u = np.asarray(result.displacement).copy()
    u[:, 0] += 0.1 * dofs[:, 0] * dofs[:, 1]
    points = np.asarray(result.quadrature_points)
    deformation = np.asarray(result.deformation_gradient).copy()
    deformation[..., 0, 0] += 0.1 * points[..., 1]
    deformation[..., 0, 1] += 0.1 * points[..., 0]
    fields = audit.rebuild_fem_quadrature(
        request, replace(result, displacement=u, deformation_gradient=deformation), (1,)
    )
    torch.testing.assert_close(fields.gradient[..., 0, 1], 0.1 * fields.xy_mm[..., 0])
    torch.testing.assert_close(
        fields.u_mm[..., 0],
        0.2 * fields.xy_mm[..., 0] + 0.1 * fields.xy_mm[..., 0] * fields.xy_mm[..., 1],
    )


@pytest.mark.parametrize("qoi", ((), (1, 1), (3, 1), (-1,), (4,), (0,)))
def test_reconstruction_refuses_invalid_qoi_and_empty_cell_complement(
    qoi: tuple[int, ...],
) -> None:
    request = _request(element_order=2)
    with pytest.raises(ValueError, match="QoI|complement"):
        audit.rebuild_fem_quadrature(request, _synthetic_exact_result(request), qoi)


@pytest.mark.parametrize(
    "invalid",
    (
        "request_id",
        "mesh_id",
        "element_order",
        "coordinates",
        "gradient",
        "nonfinite",
        "weight",
    ),
)
def test_reconstruction_refuses_identity_layout_and_field_defects(invalid: str) -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    if invalid == "request_id":
        result = replace(result, request_id="wrong")
    elif invalid == "mesh_id":
        result = replace(result, mesh_id="wrong")
    elif invalid == "element_order":
        result = replace(result, element_order=1)
    elif invalid == "coordinates":
        result = replace(
            result,
            displacement_dof_coordinates=tuple(
                reversed(result.displacement_dof_coordinates)
            ),
        )
    elif invalid == "gradient":
        f = np.asarray(result.deformation_gradient).copy()
        f[0, 0, 0, 0] += 1.0e-6
        result = replace(result, deformation_gradient=f)
    elif invalid == "nonfinite":
        p = np.asarray(result.first_piola_stress).copy()
        p[0, 0, 0, 0] = np.nan
        result = replace(result, first_piola_stress=p)
    else:
        weights = np.asarray(result.quadrature_weights).copy()
        weights[0, 0] = 0.0
        result = replace(result, quadrature_weights=weights)
    with pytest.raises(ValueError):
        audit.rebuild_fem_quadrature(request, result, (1,))


def test_quadrature_prediction_matches_direct_forward_with_physical_units() -> None:
    model, case = _model(), _case("candidate-002485")
    xy = torch.tensor(
        [[[0.13, 0.28], [0.9, 0.07]], [[0.43, 0.81], [0.6, 0.52]]], dtype=torch.float64
    )
    model.train()
    prediction = audit.predict_quadrature(model, case, xy, chunk_size=4)
    assert model.training
    coordinates = xy.reshape(1, -1, 2).clone().requires_grad_(True)
    tensors = case.tensors
    model.eval()
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = model(
            boundary_xy=tensors.boundary_xy.unsqueeze(0),
            boundary_tags=tensors.boundary_tags.unsqueeze(0),
            boundary_normals=tensors.boundary_normals.unsqueeze(0),
            boundary_weights=tensors.boundary_weights.unsqueeze(0),
            boundary_mask=tensors.boundary_mask.unsqueeze(0),
            query_xy=coordinates,
            parameters=tensors.parameters_for(model.config.parameter_names).unsqueeze(
                0
            ),
        )
        expected_gradient = displacement_gradient(
            output.u * _scales(case.request).displacement_mm,
            coordinates,
            _COORDINATE_CONVERSION,
        )
    torch.testing.assert_close(
        prediction.u_mm, output.u.reshape_as(xy) * 0.1, rtol=1e-12, atol=1e-14
    )
    torch.testing.assert_close(
        prediction.gradient,
        expected_gradient.reshape(2, 2, 2, 2),
        rtol=1e-12,
        atol=1e-14,
    )
    torch.testing.assert_close(
        prediction.p_mpa, output.P.reshape(2, 2, 2, 2) * 760.0, rtol=1e-12, atol=1e-14
    )
    for value in (prediction.u_mm, prediction.gradient, prediction.p_mpa):
        assert not value.requires_grad


def test_chunked_query_has_same_frozen_u_and_grad_across_neutral_stress_arms() -> None:
    source, case = _model(), _case("candidate-002485")
    xy = torch.linspace(0.1, 0.9, 20, dtype=torch.float64).reshape(2, 5, 2)
    expected = audit.predict_quadrature(source, case, xy, chunk_size=3)
    full = audit.predict_quadrature(source, case, xy, chunk_size=10)
    torch.testing.assert_close(full.u_mm, expected.u_mm, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(full.gradient, expected.gradient, rtol=1e-12, atol=1e-14)
    for enriched in (False, True):
        model = build_stress_arm(source, StressCapacityPlan(), enriched=enriched)
        actual = audit.predict_quadrature(model, case, xy, chunk_size=3)
        assert torch.equal(actual.u_mm, expected.u_mm)
        assert torch.equal(actual.gradient, expected.gradient)


def test_query_gradients_are_physical_and_mode_restores_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, case = _model(), _case("candidate-002485")

    def affine(**kwargs: torch.Tensor) -> OperatorOutput:
        xy = kwargs["query_xy"]
        return OperatorOutput(
            u=torch.stack(
                (
                    2.0 * xy[..., 0] + 3.0 * xy[..., 1],
                    -4.0 * xy[..., 0] + 5.0 * xy[..., 1],
                ),
                dim=-1,
            ),
            P=torch.ones((*xy.shape[:-1], 2, 2), dtype=torch.float64),
        )

    monkeypatch.setattr(model, "decode_queries", affine)
    model.eval()
    xy = torch.ones((1, 2, 2), dtype=torch.float64)
    actual = audit.predict_quadrature(model, case, xy, chunk_size=1)
    assert not model.training
    torch.testing.assert_close(
        actual.gradient,
        torch.tensor([[[[0.2, 0.3], [-0.4, 0.5]]] * 2], dtype=torch.float64),
    )
    torch.testing.assert_close(
        actual.p_mpa, torch.full((1, 2, 2, 2), 760.0, dtype=torch.float64)
    )

    def failing(**kwargs: torch.Tensor) -> OperatorOutput:
        raise RuntimeError("query failed")

    monkeypatch.setattr(model, "decode_queries", failing)
    model.train()
    with pytest.raises(RuntimeError, match="query failed"):
        audit.predict_quadrature(model, case, xy)
    assert model.training


@pytest.mark.parametrize(
    "invalid", ("float32", "rank", "empty", "nonfinite", "zero_chunk", "bool_chunk")
)
def test_prediction_refuses_invalid_queries(invalid: str) -> None:
    xy = torch.ones((1, 2, 2), dtype=torch.float64)
    chunk = 2
    if invalid == "float32":
        xy = xy.float()
    elif invalid == "rank":
        xy = xy.reshape(-1, 2)
    elif invalid == "empty":
        xy = xy[:, :0]
    elif invalid == "nonfinite":
        xy[0, 0, 0] = float("nan")
    elif invalid == "zero_chunk":
        chunk = 0
    else:
        chunk = True
    with pytest.raises((ValueError, TypeError)):
        audit.predict_quadrature(
            _model(), _case("candidate-002485"), xy, chunk_size=chunk
        )


def test_pinned_report_requires_exact_hash_canonical_bytes_and_public_role(
    tmp_path: Path,
) -> None:
    report = {
        "$type": "piginot.campaigns.StressCapacityReport",
        "role": "TRAIN",
        "non_gate": True,
        "plan": {"case_id": "candidate-002485"},
    }
    payload = rfc8785.dumps(report) + b"\n"
    path = tmp_path / "report.json"
    path.write_bytes(payload)
    actual = audit.load_pinned_report(
        path, hashlib.sha256(payload).hexdigest(), report["$type"]
    )
    assert actual == report


@pytest.mark.parametrize(
    "invalid", ("hash", "canonical", "newline", "type", "role", "gate", "list")
)
def test_report_loader_fails_closed_on_mutated_or_noncanonical_input(
    tmp_path: Path, invalid: str
) -> None:
    report = {"$type": "expected", "role": "TRAIN", "non_gate": True}
    if invalid == "type":
        report["$type"] = "other"
    elif invalid == "role":
        report["role"] = "DEVELOPMENT"
    elif invalid == "gate":
        report["non_gate"] = False
    payload = rfc8785.dumps([] if invalid == "list" else report) + b"\n"
    if invalid == "canonical":
        payload = b'{ "role": "TRAIN", "non_gate": true, "$type": "expected" }\n'
    elif invalid == "newline":
        payload = payload[:-1]
    path = tmp_path / "report.json"
    path.write_bytes(payload)
    digest = "0" * 64 if invalid == "hash" else hashlib.sha256(payload).hexdigest()
    with pytest.raises((ValueError, RuntimeError, PermissionError)):
        audit.load_pinned_report(path, digest, "expected")


def test_binding_checks_are_exact_and_fail_on_missing_keys() -> None:
    audit.require_bindings({"a": "value", "extra": 1}, {"a": "value"}, "source")
    for report in ({}, {"a": "other"}):
        with pytest.raises(ValueError, match="source"):
            audit.require_bindings(report, {"a": "value"}, "source")


def test_replay_requires_loss_trace_and_final_tensor_hash() -> None:
    model = _model()
    state = model_state_hash(model)
    audit.verify_replay(model, (0.3, 0.2), [0.3, 0.2], state, "v5")
    for actual, expected, expected_hash in (
        ((0.3,), [0.3, 0.2], state),
        ((0.3, 0.21), [0.3, 0.2], state),
        ((0.3, 0.2), [0.3, 0.2], "0" * 64),
    ):
        with pytest.raises(ValueError, match="v5"):
            audit.verify_replay(model, actual, expected, expected_hash, "v5")


def test_arm_identity_binds_schema_trainable_scope_and_neutral_state() -> None:
    warmstart = _model()
    model = build_stress_arm(warmstart, StressCapacityPlan(), enriched=True)
    bindings = audit.arm_bindings(model, warmstart)
    assert bindings["config"] == model.config
    assert bindings["operator_module_digest"] == module_digest(model)
    assert bindings["initial_state_hash"] == model_state_hash(model)
    assert bindings["frozen_state_hash"] == frozen_state_hash(model)
    assert bindings["trainable_parameter_names"] == trainable_names(model)
    assert set(bindings["copied_state_names"]) == set(warmstart.state_dict())
    assert set(bindings["new_state_names"]) == set(model.state_dict()) - set(
        warmstart.state_dict()
    )
    assert bindings["trainable_parameter_count"] == sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )


def _runner_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    s10, s11, out = (tmp_path / name for name in ("s10", "s11", "out"))
    for path in (s10, s11, out):
        path.mkdir()
    case, model = _case("candidate-002485"), _model()
    case = replace(case, tensors=replace(case.tensors, qoi_indices=torch.tensor([1])))
    plan = audit.QuadratureAuditPlan()
    fixture_plan = SimpleNamespace(
        **{
            name: getattr(plan, name)
            for name in plan.__dataclass_fields__
            if not name.startswith("__")
        },
        content_hash=plan.content_hash,
    )
    fixture_plan.expected_vertices = 4
    fixture_plan.expected_cells = 2
    fixture_plan.expected_qoi_vertices = 1
    source = SimpleNamespace(
        development_opened=False, bindings=SimpleNamespace(content_hash="a" * 64)
    )
    initialization = SimpleNamespace(
        model=model,
        evidence_file_blake2b256="b" * 64,
        checkpoint_file_blake2b256="c" * 64,
    )
    verified = SimpleNamespace(
        case=SimpleNamespace(case_id=plan.case_id, role=EvidenceRole.TRAIN),
        request=case.request,
        result=_synthetic_exact_result(case.request),
    )
    loaded: list[str] = []

    def load(case_id: str) -> tuple[SimpleNamespace, None]:
        loaded.append(case_id)
        return verified, None

    source.load_fit_with_collocation = load
    capacity_plan = StressCapacityPlan()
    sampler = audit._build_vertex_sampler_plan(capacity_plan.fine_tune_plan, 4)
    v5 = {
        "case_id": plan.case_id,
        "plan_hash": capacity_plan.fine_tune_plan.content_hash,
        "source_bindings_hash": source.bindings.content_hash,
        "s11_evidence_file_hash": initialization.evidence_file_blake2b256,
        "initial_checkpoint_file_hash": initialization.checkpoint_file_blake2b256,
        "initial_model_state_hash": model_state_hash(model),
        "final_model_state_hash": model_state_hash(model),
        "vertex_sampler_plan_hash": sampler.plan_hash,
        "optimizer_loss_trace": [],
    }
    records = []
    for enriched in (False, True):
        arm = build_stress_arm(model, capacity_plan, enriched=enriched)
        records.append(
            {
                "arm_id": "enriched" if enriched else "control",
                **strict_json_loads(
                    canonical_json_bytes(audit.arm_bindings(arm, model))
                ),
                "final_state_hash": model_state_hash(arm),
                "optimizer_loss_trace": [],
            }
        )
    capacity = {
        "plan": strict_json_loads(canonical_json_bytes(capacity_plan)),
        "plan_hash": capacity_plan.content_hash,
        "implementation_sha256": strict_json_loads(
            canonical_json_bytes(audit._implementation_hashes())
        ),
        "fixed_displacement_verified": True,
        "source_bindings_hash": source.bindings.content_hash,
        "s11_evidence_file_hash": initialization.evidence_file_blake2b256,
        "s11_checkpoint_file_hash": initialization.checkpoint_file_blake2b256,
        "sampler_hash": sampler.plan_hash,
        "warmstart_state_hash": model_state_hash(model),
        "warmstart_frozen_state_hash": frozen_state_hash(model),
        "arms": records,
    }
    monkeypatch.setattr(audit, "QuadratureAuditPlan", lambda: fixture_plan)
    monkeypatch.setattr(
        audit, "load_selected_s11_initialization", lambda *args: initialization
    )
    monkeypatch.setattr(
        audit, "open_selective_s10_source", lambda *args, **kwargs: source
    )
    monkeypatch.setattr(
        audit, "materialize_dogbone_s13_training_case", lambda *args: case
    )
    monkeypatch.setattr(audit, "qoi_vertex_indices", lambda *args, **kwargs: (1,))
    monkeypatch.setattr(
        audit,
        "load_pinned_report",
        lambda path, *args: v5 if path.name == "v5.json" else capacity,
    )
    trained: list[object] = []

    def train(model: object, *args: object) -> tuple[float, ...]:
        trained.append(model)
        return ()

    monkeypatch.setattr(audit, "_train_one_case", train)
    arguments = {
        "s10_run_root": s10,
        "s11_checkpoint_root": s11,
        "development_access_record": out / "access.json",
        "v5_report_path": out / "v5.json",
        "capacity_report_path": out / "capacity.json",
        "output_path": out / "result.json",
    }
    return SimpleNamespace(
        arguments=arguments,
        source=source,
        verified=verified,
        case=case,
        model=model,
        v5=v5,
        capacity=capacity,
        plan=fixture_plan,
        loaded=loaded,
        trained=trained,
    )


def test_runner_compares_three_exactly_bound_train_states_without_source_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    report = audit.run_quadrature_audit(**fixture.arguments)
    assert fixture.loaded == ["candidate-002485", "candidate-002485"]
    assert len(fixture.trained) == 3
    assert tuple(arm.arm_id for arm in report.arms) == ("v5", "control", "enriched")
    assert report.role is EvidenceRole.TRAIN and report.non_gate is True
    assert report.fixed_displacement_verified is True
    assert report.gradient_max_absolute_difference < 1.0e-14
    assert report.source_bindings_hash == fixture.source.bindings.content_hash
    assert report.arms[0].metrics.all_points.point_count == 14
    assert report.arms[0].metrics.root_incident.cell_count == 1
    assert not fixture.arguments["development_access_record"].exists()
    assert not fixture.arguments["output_path"].exists()


@pytest.mark.parametrize(
    "failure",
    (
        "development_before",
        "access_record",
        "development_during",
        "role",
        "case_id",
        "population",
        "quadrature_count",
        "v5_binding",
        "capacity_code",
        "v5_replay",
        "warmstart",
        "arm_records",
        "migration",
        "arm_replay",
        "frozen",
        "prediction",
        "code_changed",
    ),
)
def test_runner_fails_closed_on_identity_scope_replay_or_immutable_state_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    if failure == "development_before":
        fixture.source.development_opened = True
    elif failure == "access_record":
        fixture.arguments["development_access_record"].write_bytes(b"opened")
    elif failure == "development_during":
        original = audit.materialize_dogbone_s13_training_case

        def materialize(*args: object) -> object:
            fixture.source.development_opened = True
            return original(*args)

        monkeypatch.setattr(audit, "materialize_dogbone_s13_training_case", materialize)
    elif failure == "role":
        fixture.verified.case.role = EvidenceRole.DEVELOPMENT
    elif failure == "case_id":
        fixture.verified.case.case_id = "other"
    elif failure == "population":
        fixture.plan.expected_vertices = 969
    elif failure == "quadrature_count":
        fixture.plan.expected_points_per_cell = 8
    elif failure == "v5_binding":
        fixture.v5["source_bindings_hash"] = "e" * 64
    elif failure == "capacity_code":
        fixture.capacity["implementation_sha256"] = []
    elif failure == "v5_replay":
        fixture.v5["optimizer_loss_trace"] = [0.1]
    elif failure == "warmstart":
        fixture.capacity["warmstart_state_hash"] = "e" * 64
    elif failure == "arm_records":
        fixture.capacity["arms"] = []
    elif failure == "migration":
        fixture.capacity["arms"][0]["trainable_parameter_count"] = 0
    elif failure == "arm_replay":
        fixture.capacity["arms"][0]["final_state_hash"] = "e" * 64
    elif failure == "frozen":
        original_hash = audit.frozen_state_hash

        def changed_hash(model: object) -> str:
            if len(fixture.trained) > 1 and model is not fixture.model:
                return "e" * 64
            return original_hash(model)

        monkeypatch.setattr(audit, "frozen_state_hash", changed_hash)
    elif failure == "prediction":
        original_predict = audit.predict_quadrature

        def changed_predict(model: object, *args: object, **kwargs: object) -> object:
            prediction = original_predict(model, *args, **kwargs)
            return replace(
                prediction,
                gradient=prediction.gradient
                + (1e-10 if model is not fixture.model else 0.0),
            )

        monkeypatch.setattr(audit, "predict_quadrature", changed_predict)
    else:
        original_hashes = audit.audit_implementation_hashes
        calls = 0

        def changed_code() -> object:
            nonlocal calls
            calls += 1
            return original_hashes() if calls == 1 else ()

        monkeypatch.setattr(audit, "audit_implementation_hashes", changed_code)
    with pytest.raises((ValueError, RuntimeError, PermissionError, FileExistsError)):
        audit.run_quadrature_audit(**fixture.arguments)
    assert not fixture.arguments["output_path"].exists()


@pytest.mark.parametrize(
    "invalid", ("relative", "source_output", "same_paths", "missing_parent")
)
def test_path_guards_run_before_source_or_evidence_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    arguments = fixture.arguments
    if invalid == "relative":
        arguments["output_path"] = Path("relative.json")
    elif invalid == "source_output":
        arguments["output_path"] = arguments["s10_run_root"] / "result.json"
    elif invalid == "same_paths":
        arguments["output_path"] = arguments["development_access_record"]
    else:
        arguments["output_path"] = tmp_path / "missing" / "result.json"
    monkeypatch.setattr(
        audit,
        "load_pinned_report",
        lambda *args: pytest.fail("loaded evidence before path guards"),
    )
    with pytest.raises(ValueError):
        audit.run_quadrature_audit(**arguments)


def test_cli_writes_canonical_report_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"role": EvidenceRole.TRAIN, "non_gate": True}
    monkeypatch.setattr(audit, "run_quadrature_audit", lambda **kwargs: report)
    output = tmp_path / "report.json"
    arguments = (
        "--s10-run-root",
        str(tmp_path / "s10"),
        "--s11-checkpoint-root",
        str(tmp_path / "s11"),
        "--development-access-record",
        str(tmp_path / "access.json"),
        "--v5-report",
        str(tmp_path / "v5.json"),
        "--capacity-report",
        str(tmp_path / "capacity.json"),
        "--output",
        str(output),
    )
    assert audit.main(arguments) == 0
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    with pytest.raises(FileExistsError):
        audit.main(arguments)


def test_runner_refuses_a_changed_source_result_before_returning_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    original_load = fixture.source.load_fit_with_collocation

    def changed_result(case_id: str) -> tuple[object, None]:
        verified, _ = original_load(case_id)
        if len(fixture.loaded) == 2:
            return SimpleNamespace(
                request=verified.request,
                result=replace(verified.result, total_strain_energy=0.01),
            ), None
        return verified, None

    fixture.source.load_fit_with_collocation = changed_result
    with pytest.raises(RuntimeError, match="request/result changed"):
        audit.run_quadrature_audit(**fixture.arguments)
    assert not fixture.arguments["output_path"].exists()


def test_runner_refuses_geometric_and_tensor_qoi_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _runner_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "qoi_vertex_indices", lambda *args, **kwargs: (2,))
    with pytest.raises(ValueError, match="geometric QoI"):
        audit.run_quadrature_audit(**fixture.arguments)
    assert not fixture.trained


def test_quadrature_prediction_refuses_a_non_training_case_object() -> None:
    with pytest.raises(PermissionError, match="TRAIN"):
        audit.predict_quadrature(
            _model(), object(), torch.zeros((2, 7, 2), dtype=torch.float64)
        )


def test_pinned_report_refuses_missing_or_relative_path(tmp_path: Path) -> None:
    for path in (tmp_path / "missing.json", Path("relative.json")):
        with pytest.raises(ValueError, match="absolute nonsymlink file"):
            audit.load_pinned_report(path, "a" * 64, "type")
