"""The public compile command keeps planning separate from execution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from piginot import cli
from piginot.compiler import cli as compiler_cli
from tests.compiler.plan_fixtures import inputs


def arguments(spec, config, output):
    return [
        "compile",
        "--spec",
        str(spec),
        "--config",
        str(config),
        "--output",
        str(output),
    ]


def test_plan_only_creates_a_bound_plan_without_invoking_executor(
    tmp_path, monkeypatch, capsys
):
    spec, config, _, descriptor = inputs(tmp_path)
    adapter = SimpleNamespace(
        descriptor=descriptor,
        execute=lambda *a, **k: pytest.fail("numerical execution"),
    )
    monkeypatch.setattr(compiler_cli, "resolve_adapter", lambda name: adapter)
    output = tmp_path / "run"
    assert cli.main([*arguments(spec, config, output), "--plan-only"]) == 0
    document = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "plan.json").read_bytes())
    assert saved == document
    assert document["status"] == "PLANNED"
    assert document["stages"]["material"] == "PENDING"
    assert document["scientific_status"] == "NOT_EVALUATED"


def test_execution_capability_must_exist_before_output_is_created(
    tmp_path, monkeypatch, capsys
):
    spec, config, _, descriptor = inputs(tmp_path)
    monkeypatch.setattr(
        compiler_cli,
        "resolve_adapter",
        lambda name: SimpleNamespace(descriptor=descriptor),
    )
    output = tmp_path / "run"
    assert cli.main(arguments(spec, config, output)) == 2
    assert not output.exists()
    assert "execution" in capsys.readouterr().err.lower()


def test_registered_execution_receives_the_validated_plan_and_resume_flag(
    tmp_path, monkeypatch
):
    spec, config, _, descriptor = inputs(tmp_path)
    received = []

    def execute(plan, *, resume):
        received.append((plan.config.name, plan.problem.name, resume))
        return 3

    monkeypatch.setattr(
        compiler_cli,
        "resolve_adapter",
        lambda name: SimpleNamespace(descriptor=descriptor, execute=execute),
    )
    assert cli.main([*arguments(spec, config, tmp_path / "run"), "--resume"]) == 3
    assert received == [("tiny-fixture", "plate-v1", True)]


def test_input_errors_are_distinct_from_unexpected_execution_errors(
    tmp_path, monkeypatch, capsys
):
    spec, config, _, descriptor = inputs(tmp_path)
    output = tmp_path / "run"
    assert cli.main(arguments(tmp_path / "missing", config, output)) == 2
    config.write_text('{"schema": 1, "schema": 2}')
    assert cli.main(arguments(spec, config, output)) == 2
    spec, config, _, descriptor = inputs(tmp_path)

    def broken(plan, *, resume):
        raise RuntimeError("synthetic executor failure")

    monkeypatch.setattr(
        compiler_cli,
        "resolve_adapter",
        lambda name: SimpleNamespace(descriptor=descriptor, execute=broken),
    )
    assert cli.main(arguments(spec, config, output)) == 1
    assert "synthetic executor failure" in capsys.readouterr().err
    assert not output.exists()


def test_plan_only_and_execution_resume_are_unambiguous(tmp_path):
    spec, config, _, _ = inputs(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(
            [*arguments(spec, config, tmp_path / "run"), "--plan-only", "--resume"]
        )
    assert error.value.code == 2
