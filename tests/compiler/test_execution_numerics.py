"""Default stage bindings are verified using mocked numerical primitives."""

import sys
from types import SimpleNamespace

import pytest

from piginot.compiler import execution_numerics
from piginot.compiler.planning import build_plan
from piginot.physics.errors import PhysicsError
from piginot.problem.verification import LiftingEvidence
from tests.compiler.plan_fixtures import inputs
from tests.problem_helpers import plate_mesh


@pytest.fixture
def operations(tmp_path):
    spec, config, _, descriptor = inputs(tmp_path)
    plan = build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    return execution_numerics.NumericalOperations(plan, SimpleNamespace(), "c" * 40)


def admission_document(*, missing=False, passed=True):
    names = (
        "reference",
        "objectivity",
        "symmetry",
        "consistency",
        "moduli",
        "closure",
        "ellipticity",
    )
    return {
        "points": [{"modulus": 1.0}],
        "reports": [{"checks": [{"name": name, "passed": passed} for name in names]}],
        "fem": [] if missing else [{"passed": passed}],
        "passed": passed,
    }


@pytest.mark.parametrize(
    "missing,passed,status",
    [(False, True, "PASS"), (True, True, "INCOMPLETE"), (False, False, "FAIL")],
)
def test_all_eight_material_checks_are_required(
    operations, monkeypatch, missing, passed, status
):
    calls = []
    document = admission_document(missing=missing, passed=passed)
    monkeypatch.setattr(
        execution_numerics,
        "admit_material",
        lambda spec, **kwargs: (
            calls.append(kwargs) or SimpleNamespace(document=lambda: document)
        ),
    )
    result = operations.admit_material()
    assert calls == [{"fem": True}]
    assert result["status"] == status
    assert result["evidence"] == document


def test_full_parameter_box_lifting_evidence_and_refusals_are_retained(
    operations, monkeypatch
):
    points = [{"width": 12.0}, {"width": 13.0}]
    calls = []
    monkeypatch.setattr(execution_numerics, "box_points", lambda spec, **kwargs: points)
    operations.adapter.geometry_values = lambda spec, values, level: (
        (_ for _ in ()).throw(ValueError("bad geometry"))
        if values["width"] == 13.0
        else plate_mesh()
    )

    def verify(spec, mesh_for, actual, **kwargs):
        calls.extend(mesh_for(point) for point in actual)
        return LiftingEvidence(
            spec.problem_hash, "quarter", 1, 1, (320, 4800), 0.0, None, None, {}, 0.1
        )

    monkeypatch.setattr(execution_numerics, "verify_lifting", verify)
    result = operations.verify_geometry()
    assert len(calls) == 2 and calls[1] is None
    assert result["status"] == "INCOMPLETE"
    assert result["refused_geometries"][0]["error"]["message"] == "bad geometry"
    assert result["evidence"]["parameter_vectors"] == 1


def test_successful_lifting_stores_proof_and_full_evidence(operations, monkeypatch):
    evidence = LiftingEvidence(
        operations.plan.problem.problem_hash,
        "quarter",
        3,
        0,
        (320, 4800),
        0.0,
        None,
        None,
        {},
        0.1,
    )
    monkeypatch.setattr(execution_numerics, "box_points", lambda spec, **kwargs: [{}])
    monkeypatch.setattr(
        execution_numerics, "verify_lifting", lambda *args, **kwargs: evidence
    )
    result = operations.verify_geometry()
    assert result["status"] == "PASS"
    assert (
        result["verification"]["verification_hash"]
        == evidence.verification(operations.plan.problem).verification_hash
    )
    assert result["evidence"] == evidence.document()


def test_failed_lifting_cannot_produce_verification(operations, monkeypatch):
    evidence = LiftingEvidence(
        operations.plan.problem.problem_hash,
        "quarter",
        0,
        0,
        (320, 4800),
        1.0,
        None,
        None,
        {},
        0.0,
    )
    monkeypatch.setattr(execution_numerics, "box_points", lambda spec, **kwargs: [{}])
    monkeypatch.setattr(
        execution_numerics, "verify_lifting", lambda *args, **kwargs: evidence
    )
    result = operations.verify_geometry()
    assert result["status"] == "FAIL"
    assert "verification" not in result


def test_nonfinite_failed_evidence_is_reported_without_dropping_check(
    operations, monkeypatch
):
    document = admission_document(passed=False)
    document["reports"][0]["checks"][0]["measured"] = float("inf")
    monkeypatch.setattr(
        execution_numerics,
        "admit_material",
        lambda *args, **kwargs: SimpleNamespace(document=lambda: document),
    )
    result = operations.admit_material()
    assert result["evidence"]["reports"][0]["checks"][0]["measured"] == "Infinity"


def test_runtime_guard_is_closed(operations, monkeypatch):
    calls = []
    operations.guard = SimpleNamespace(
        probe=lambda *profiles: {"status": "PASS"}, close=lambda: calls.append("close")
    )
    assert operations.probe_runtime()["status"] == "PASS"
    operations.close()
    assert calls == ["close"]


def test_default_training_reference_freeze_and_export_bindings(operations, monkeypatch):
    calls = []
    binding = SimpleNamespace(
        run_hash="a" * 64,
        normalization_hash="b" * 64,
        objective_registry_hash="c" * 64,
        architecture_module_digest="d" * 64,
        dataset_snapshot_hash="e" * 64,
    )
    context = SimpleNamespace(
        prepare=lambda: {"status": "PASS"},
        train=lambda **kwargs: {"status": "PASS", "resume": kwargs["resume"]},
        load_final=lambda training: calls.append(("load", training)),
        predict=lambda *args: None,
        binding=binding,
        normalization={},
        model=object(),
        checkpoint_path=lambda step: operations.plan.output_path / str(step),
    )
    monkeypatch.setattr(execution_numerics, "EnergyTraining", lambda *args: context)
    with pytest.raises(ValueError, match="preparation"):
        operations._training()
    assert operations.prepare_training({}, {})["status"] == "PASS"
    assert operations.train(resume=True)["resume"] is True
    import piginot.compiler.reference_jobs as reference_jobs

    monkeypatch.setattr(
        reference_jobs,
        "evaluate_case",
        lambda *args, **kwargs: {"status": "PASS", "case": args[2]},
    )
    training = {"model_state_hash": "f" * 64, "checkpoint_sha256": "0" * 64}
    validated = []
    monkeypatch.setattr(
        reference_jobs,
        "validate_case_cache",
        lambda *args: validated.append(args),
        raising=False,
    )
    operations.validate_case_cache("screen-001", "train-screen", training)
    assert validated[0][2] == "screen-001"
    assert (
        operations.evaluate("screen-001", "train-screen", training)["case"]
        == "screen-001"
    )
    frozen = operations.freeze(training, {"status": "PASS"})
    assert frozen["status"] == "PASS"
    assert frozen["identities"]["model_state_hash"] == "f" * 64
    exports = []
    monkeypatch.setitem(
        sys.modules,
        "piginot.compiler.exporting",
        SimpleNamespace(
            export_candidate=lambda *args, **kwargs: (
                exports.append(kwargs) or {"status": "PASS"}
            )
        ),
    )
    evidence = LiftingEvidence(
        operations.plan.problem.problem_hash,
        "quarter",
        3,
        0,
        (320, 4800),
        0.0,
        None,
        None,
        {},
        0.1,
    )
    from dataclasses import asdict

    lifting = {
        "evidence": evidence.document(),
        "verification": asdict(evidence.verification(operations.plan.problem)),
    }
    assert operations.export(training, {"status": "PASS"}, lifting)["status"] == "PASS"
    assert exports[0]["lifting_evidence"] == evidence
    assert exports[0]["run_binding"] is binding
    assert len(calls) == 4


@pytest.mark.parametrize(
    "error", [PhysicsError("inadmissible state"), FloatingPointError("nonfinite")]
)
def test_numerical_training_failure_retains_last_checkpoint(operations, error):
    folder = operations.plan.output_path / "checkpoints" / "step-000000001"
    folder.mkdir(parents=True)
    operations.training = SimpleNamespace(
        last_checkpoint=folder,
        train=lambda **kwargs: (_ for _ in ()).throw(error),
    )
    result = operations.train(resume=True)
    assert result["status"] == "FAIL"
    assert result["last_checkpoint"] == str(folder)
    assert result["error"]["category"] == type(error).__name__
    assert operations._last_checkpoint() == str(folder)


def test_missing_checkpoint_and_nonfinite_evidence_are_explicit(operations):
    assert operations._last_checkpoint() is None
    result = execution_numerics._finite_evidence([float("nan"), float("-inf"), 1.0])
    assert result == ["NaN", "-Infinity", 1.0]


@pytest.mark.parametrize(
    "error", [RuntimeError("training bug"), ValueError("bad cache")]
)
@pytest.mark.parametrize("published", [False, True])
def test_unexpected_training_error_reports_only_the_trusted_checkpoint(
    operations, error, published
):
    folder = operations.plan.output_path / "checkpoints" / "step-000000001"
    folder.mkdir(parents=True)
    (folder.parent / "step-999999999").write_text("corrupt unvalidated path")
    operations.training = SimpleNamespace(
        last_checkpoint=folder if published else None,
        train=lambda **kwargs: (_ for _ in ()).throw(error),
    )
    result = operations.train(resume=True)
    assert result == {
        "status": "ERROR",
        "error": {"category": type(error).__name__, "message": str(error)},
        "last_checkpoint": str(folder) if published else None,
    }


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(130)])
def test_training_control_flow_exceptions_propagate(operations, error):
    operations.training = SimpleNamespace(
        train=lambda **kwargs: (_ for _ in ()).throw(error)
    )
    with pytest.raises(type(error)):
        operations.train(resume=True)
