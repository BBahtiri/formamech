"""Stage guards use only synthetic metadata and fake operations."""

from types import SimpleNamespace

import pytest

from piginot.compiler import execution
from piginot.compiler.planning import build_plan
from piginot.contracts import strict_json_loads
from tests.compiler.plan_fixtures import inputs


class Operations:
    calls: list[str]
    fail_stage: str | None = None
    error_stage: str | None = None

    def __init__(self, plan, adapter, revision):
        self.calls = []
        self.plan = plan

    def result(self, stage):
        self.calls.append(stage)
        if stage == self.error_stage:
            raise RuntimeError("synthetic execution error")
        return {
            "status": "FAIL" if stage == self.fail_stage else "PASS",
            "stage": stage,
        }

    def probe_runtime(self):
        return self.result("runtime")

    def admit_material(self):
        return self.result("material")

    def verify_geometry(self):
        return self.result("geometry-lifting")

    def prepare_training(self, material, lifting):
        return self.result("train-preparation")

    def train(self, *, resume):
        return self.result("train")

    def evaluate(self, case_id, partition, training):
        return {
            **self.result(partition),
            "scientific_status": "PASS",
            "discrete_fem_errors": {
                "u_relative_l2": 0.0,
                "p_relative_l2": 0.0,
                "absolute_peak_error": 0.0,
            },
        }

    def freeze(self, training, screen):
        return self.result("freeze")

    def validate_case_cache(self, case_id, partition, training):
        pass

    def export(self, training, material, lifting):
        return {**self.result("export-report"), "claim_level": "NOT_VERIFIED"}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "_require_numerical_dependencies", lambda: None)
    spec, config, _, descriptor = inputs(tmp_path, development=True)
    plan = build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    operations = Operations(plan, object(), "c" * 40)
    monkeypatch.setattr(execution, "NumericalOperations", lambda *args: operations)
    monkeypatch.setattr(
        execution,
        "require_clean_implementation_identity",
        lambda: SimpleNamespace(
            source_revision="c" * 40, package_sha256=plan.package_sha256
        ),
    )
    return plan, operations


def report(plan):
    return strict_json_loads((plan.output_path / "report.json").read_bytes())


def test_success_freezes_before_public_development_and_never_certifies(setup):
    plan, operations = setup
    assert execution.execute_campaign(plan, object()) == 0
    assert operations.calls.index("freeze") < operations.calls.index(
        "public-development"
    )
    document = report(plan)
    assert document["claim_level"] == "NOT_VERIFIED"
    assert document["scientific_status"] == "PASS"
    assert document["stages"]["public-development"]["status"] == "PASS"
    assert (plan.output_path / "report.md").is_file()


@pytest.mark.parametrize(
    "stage", ["runtime", "material", "geometry-lifting", "train-preparation", "train"]
)
def test_prerequisite_refusal_blocks_downstream_numerics(setup, stage):
    plan, operations = setup
    operations.fail_stage = stage
    assert execution.execute_campaign(plan, object()) == 3
    assert operations.calls[-1] == stage
    assert "public-development" not in operations.calls
    assert report(plan)["claim_level"] == "NOT_VERIFIED"


def test_failed_screen_retains_candidate_and_blocks_development(setup):
    plan, operations = setup
    operations.fail_stage = "train-screen"
    assert execution.execute_campaign(plan, object()) == 3
    assert "public-development" not in operations.calls
    assert "freeze" not in operations.calls
    assert "export-report" in operations.calls
    document = report(plan)
    assert document["stages"]["freeze"]["status"] == "BLOCKED"
    assert len(document["stages"]["train-screen"]["cases"]) == 1


def test_unexpected_error_records_category_and_blocks_descendants(setup):
    plan, operations = setup
    operations.error_stage = "train"
    assert execution.execute_campaign(plan, object()) == 1
    stage = report(plan)["stages"]["train"]
    assert stage["status"] == "ERROR"
    assert stage["error"]["category"] == "RuntimeError"
    assert "train-screen" not in operations.calls


def test_mixed_installed_implementation_refuses_before_numerics(setup, monkeypatch):
    plan, operations = setup
    monkeypatch.setattr(
        execution,
        "require_clean_implementation_identity",
        lambda: SimpleNamespace(source_revision="c" * 40, package_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="implementation"):
        execution.execute_campaign(plan, object())
    assert operations.calls == []


def test_completed_run_resume_reuses_terminal_evidence_and_reprobes_runtime(setup):
    plan, operations = setup
    assert execution.execute_campaign(plan, object()) == 0
    previous = {p.name: p.read_bytes() for p in plan.output_path.glob("stage-*.json")}
    operations.calls.clear()
    assert execution.execute_campaign(plan, object(), resume=True) == 0
    assert operations.calls == [
        "runtime",
        "train-preparation",
        "freeze",
        "export-report",
    ]
    assert all(
        (plan.output_path / name).read_bytes() == data
        for name, data in previous.items()
    )


@pytest.mark.parametrize("changed", [False, True])
def test_resume_compares_cached_case_reference_receipts_with_raw_evidence(
    setup, monkeypatch, changed
):
    plan, operations = setup
    references = [{"status": "PASS", "evidence": {"sha256": "a" * 64}}]
    evaluate = operations.evaluate
    monkeypatch.setattr(
        operations,
        "evaluate",
        lambda *args: {**evaluate(*args), "references": references},
    )
    assert execution.execute_campaign(plan, object()) == 0
    operations.calls.clear()
    monkeypatch.setattr(
        operations,
        "validate_case_cache",
        lambda *args: (
            [{"status": "PASS", "evidence": {"sha256": "b" * 64}}]
            if changed
            else references
        ),
    )
    assert execution.execute_campaign(plan, object(), resume=True) == (
        1 if changed else 0
    )
    assert "train" not in operations.calls
    if changed:
        error = report(plan)["stages"]["train-screen"]["error"]
        assert "raw evidence" in error["message"]


def test_optional_development_is_not_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "_require_numerical_dependencies", lambda: None)
    spec, config, _, descriptor = inputs(tmp_path)
    plan = build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    ops = Operations(plan, object(), "c" * 40)
    monkeypatch.setattr(execution, "NumericalOperations", lambda *args: ops)
    monkeypatch.setattr(
        execution,
        "require_clean_implementation_identity",
        lambda: SimpleNamespace(
            source_revision="c" * 40, package_sha256=plan.package_sha256
        ),
    )
    assert execution.execute_campaign(plan, object()) == 0
    assert "public-development" not in ops.calls
    assert report(plan)["stages"]["public-development"]["status"] == "NOT_REQUESTED"


def test_missing_numerical_extra_refuses_before_provenance_or_output(
    tmp_path, monkeypatch
):
    spec, config, _, descriptor = inputs(tmp_path)
    plan = build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    monkeypatch.setattr(
        execution, "find_spec", lambda name: None if name == "torch" else object()
    )
    monkeypatch.setattr(
        execution,
        "require_clean_implementation_identity",
        lambda: (_ for _ in ()).throw(AssertionError("provenance must not be queried")),
    )
    with pytest.raises(execution.AdapterError, match="verify"):
        execution.execute_campaign(plan, object())
    assert not plan.output_path.exists()
    monkeypatch.setattr(execution, "find_spec", lambda name: object())
    execution._require_numerical_dependencies()


def test_internal_provenance_and_default_operations_factory(tmp_path, monkeypatch):
    import piginot.compiler.execution_numerics as numerics
    import piginot.compiler.provenance as provenance

    identity = object()
    monkeypatch.setattr(
        provenance, "require_clean_implementation_identity", lambda: identity
    )
    assert execution.require_clean_implementation_identity() is identity
    monkeypatch.setattr(numerics, "NumericalOperations", lambda *args: identity)
    assert execution.NumericalOperations(None, None, "c" * 40) is identity


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("INCOMPLETE", 3),
        ("ERROR", 1),
        ("MISSING_METRICS", 3),
        ("UNQUALIFIED", 3),
        ("FAIL_METRICS", 3),
    ],
)
def test_complete_roster_and_reference_terms_gate_scientific_acceptance(
    setup, outcome, expected
):
    plan, operations = setup
    base = {
        "status": "PASS",
        "scientific_status": "PASS",
        "discrete_fem_errors": {
            "u_relative_l2": 0.0,
            "p_relative_l2": 0.0,
            "absolute_peak_error": 0.0,
        },
    }
    if outcome == "ERROR":
        operations.evaluate = lambda *args: (_ for _ in ()).throw(
            RuntimeError("broken case")
        )
    else:
        if outcome == "INCOMPLETE":
            base["status"] = "INCOMPLETE"
        elif outcome == "MISSING_METRICS":
            base["discrete_fem_errors"] = {}
        elif outcome == "UNQUALIFIED":
            base["scientific_status"] = "NOT_VERIFIED"
        else:
            base["discrete_fem_errors"]["p_relative_l2"] = 99.0
        operations.evaluate = lambda *args: base
    assert execution.execute_campaign(plan, object()) == expected
    assert "freeze" not in operations.calls
    assert "public-development" not in operations.calls
    assert "export-report" in operations.calls
    if outcome == "UNQUALIFIED":
        assert report(plan)["scientific_status"] == "NOT_VERIFIED"
        assert (
            report(plan)["stages"]["train-screen"]["discrete_fem_comparison"]["status"]
            == "PASS"
        )


def test_invalid_stage_result_and_rebuilt_preparation_are_errors(setup):
    plan, operations = setup
    operations.train = lambda **kwargs: {"status": "PENDING"}
    assert execution.execute_campaign(plan, object()) == 1
    assert report(plan)["stages"]["train"]["error"]["category"] == "ValueError"


def test_changed_preparation_refuses_completed_run_resume(setup):
    plan, operations = setup
    assert execution.execute_campaign(plan, object()) == 0
    operations.prepare_training = lambda *args: {"status": "PASS", "changed": True}
    assert execution.execute_campaign(plan, object(), resume=True) == 1
    assert "differs" in report(plan)["stages"]["train-preparation"]["error"]["message"]


def test_interrupted_train_reports_running_then_resumes_and_closes(setup):
    plan, operations = setup
    closed = []
    operations.close = lambda: closed.append(True)
    original = operations.train
    operations.train = lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        execution.execute_campaign(plan, object())
    assert report(plan)["execution_status"] == "INTERRUPTED"
    assert report(plan)["exit_code"] == 130
    assert report(plan)["scientific_status"] == "NOT_EVALUATED"
    assert report(plan)["stages"]["train"]["status"] == "RUNNING"
    operations.train = original
    assert execution.execute_campaign(plan, object(), resume=True) == 0
    assert closed == [True, True]


def test_unexpected_runtime_error_is_reported_without_numerical_stages(setup):
    plan, operations = setup
    operations.probe_runtime = lambda: (_ for _ in ()).throw(
        RuntimeError("probe failed")
    )
    assert execution.execute_campaign(plan, object()) == 1
    assert report(plan)["runtime"]["error"]["message"] == "probe failed"
    assert operations.calls == []


def test_resume_validates_cached_case_artifacts_before_using_summary(setup):
    plan, operations = setup
    assert execution.execute_campaign(plan, object()) == 0
    outcome = next((plan.output_path / "train-screen").glob("outcome-*.json"))
    outcome.write_bytes(outcome.read_bytes().replace(b'"PASS"', b'"FAIL"', 1))
    assert execution.execute_campaign(plan, object(), resume=True) == 1
    assert report(plan)["stages"]["train-screen"]["status"] == "ERROR"
    assert report(plan)["stages"]["train"]["status"] == "PASS"


def test_resume_refuses_missing_case_artifact_or_changed_stage_summary(setup):
    plan, operations = setup
    assert execution.execute_campaign(plan, object()) == 0
    outcome = next((plan.output_path / "train-screen").glob("outcome-*.json"))
    outcome.unlink()
    assert execution.execute_campaign(plan, object(), resume=True) == 1
    assert "missing" in report(plan)["stages"]["train-screen"]["error"]["message"]


def test_terminal_material_failure_is_never_retried(setup):
    plan, operations = setup
    operations.fail_stage = "material"
    assert execution.execute_campaign(plan, object()) == 3
    operations.calls.clear()
    assert execution.execute_campaign(plan, object(), resume=True) == 3
    assert operations.calls == ["runtime"]


def test_completed_resume_revalidates_candidate_before_reusing_export_pass(setup):
    plan, operations = setup
    assert execution.execute_campaign(plan, object()) == 0
    operations.export = lambda *args: (_ for _ in ()).throw(
        ValueError("candidate bundle manifest changed")
    )
    assert execution.execute_campaign(plan, object(), resume=True) == 1
    assert report(plan)["stages"]["export-report"]["error"]["message"] == (
        "candidate bundle manifest changed"
    )


def test_recomputed_case_summary_must_match_cached_stage(setup):
    plan, operations = setup
    assert execution.execute_campaign(plan, object()) == 0
    path = plan.output_path / "stage-train-screen.json"
    from piginot.compiler.execution_records import load_bound, save_bound

    stage = load_bound(path, plan.plan_sha256)
    stage["discrete_fem_comparison"]["quantities"]["p_relative_l2"] = 0.001
    save_bound(path, plan.plan_sha256, stage)
    assert execution.execute_campaign(plan, object(), resume=True) == 1
    assert (
        "summary differs" in report(plan)["stages"]["train-screen"]["error"]["message"]
    )
