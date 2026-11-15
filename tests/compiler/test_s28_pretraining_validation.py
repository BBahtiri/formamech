"""Metadata and sampling refusals precede geometry work and training."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from piginot.compiler import cli, execution, execution_numerics
from piginot.compiler.config import CampaignConfig, CompilerConfigError
from piginot.compiler.config_types import ReferencePolicy
from piginot.compiler.planning import PlanError, build_plan
from piginot.contracts import strict_json_loads
from tests.compiler.config_fixtures import config_document
from tests.compiler.plan_fixtures import inputs


@pytest.mark.parametrize("levels", [(1, 3), (1, 8), (1, 2, 3), (1, 2, 4, 8)])
def test_unsupported_refinement_refuses_in_typed_and_document_configuration(levels):
    with pytest.raises(CompilerConfigError, match="refinement"):
        ReferencePolicy("cold", levels, 1)
    document = config_document()
    document["reference"]["refinement_levels"] = list(levels)
    with pytest.raises(CompilerConfigError, match="refinement"):
        CampaignConfig.from_document(document)


@pytest.mark.parametrize("levels", [(1,), (1, 2), (1, 4), (1, 2, 4)])
def test_declared_supported_refinement_rosters_remain_available(levels):
    assert ReferencePolicy("cold", levels, 1).refinement_levels == levels


def test_invalid_refinement_plan_cannot_create_output_or_discover_adapter(
    tmp_path, monkeypatch
):
    spec_path, config_path, _, descriptor = inputs(tmp_path)
    document = json.loads(config_path.read_bytes())
    document["reference"]["refinement_levels"] = [1, 3]
    config_path.write_text(json.dumps(document))
    output = tmp_path / "output"
    with pytest.raises(PlanError, match="refinement"):
        build_plan(spec_path, config_path, output, descriptor=descriptor)
    monkeypatch.setattr(
        cli,
        "resolve_adapter",
        lambda *args: pytest.fail("invalid config discovered an adapter"),
    )
    assert (
        cli.compile_command(
            spec_path, config_path, output, plan_only=False, resume=False
        )
        == 2
    )
    assert not output.exists()


def test_parameter_sampling_refusal_blocks_lifting_and_train_preparation(
    tmp_path, monkeypatch
):
    spec_path, config_path, _, descriptor = inputs(tmp_path)
    plan = build_plan(
        spec_path, config_path, tmp_path / "output", descriptor=descriptor
    )
    plan.output_path.mkdir()
    operations = execution_numerics.NumericalOperations(plan, object(), "c" * 40)

    def sampling_refusal(*args, **kwargs):
        from piginot.problem.verification import ParameterSamplingError

        raise ParameterSamplingError(requested=64, accepted=0, attempts=64000)

    monkeypatch.setattr(execution_numerics, "box_points", sampling_refusal)
    monkeypatch.setattr(
        execution_numerics,
        "verify_lifting",
        lambda *args, **kwargs: pytest.fail(
            "sampling refusal entered lifting verification"
        ),
    )
    stages = SimpleNamespace(
        probe_runtime=lambda: {"status": "PASS"},
        admit_material=lambda: {"status": "PASS"},
        verify_geometry=operations.verify_geometry,
        prepare_training=lambda *args: pytest.fail(
            "sampling refusal entered training preparation"
        ),
    )
    run = execution._Run(plan, stages, resume=False)
    run.pipeline()
    assert run.report() == 3
    report = strict_json_loads((plan.output_path / "report.json").read_bytes())
    outcome = report["stages"]["geometry-lifting"]
    assert outcome["status"] == "FAIL"
    assert outcome["error"]["category"] == "ParameterSamplingError"
    assert outcome["sampling"] == {
        "requested": 64,
        "accepted": 0,
        "attempts": 64000,
        "seed": 2801,
    }
    assert report["stages"]["train-preparation"]["status"] == "BLOCKED"
    assert report["stages"]["train"]["status"] == "BLOCKED"
