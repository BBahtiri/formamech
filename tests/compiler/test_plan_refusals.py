"""Planning and publication refuse inconsistent inputs before mutating output."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import formamech
import piginot
import piginot_campaigns
from piginot import cli
from piginot.compiler import cli as compiler_cli
from piginot.compiler import planning, records
from piginot.compiler.adapters import GeometryIdentity
from piginot.compiler.config import CompilerConfigError
from piginot.compiler.public_source import select_public_cases
from piginot.doe import PublishedCaseManifest
from tests.compiler.plan_fixtures import inputs
from tests.compiler.test_cli import arguments


def symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"symlink capability unavailable: {error}")


def test_unregistered_geometry_and_development_budget_refuse(tmp_path):
    spec, config, _, descriptor = inputs(tmp_path, development=True)
    output = tmp_path / "output"
    wrong = replace(descriptor, geometries=(GeometryIdentity("f" * 64, "e" * 64),))
    with pytest.raises(planning.PlanError, match="unsupported reference"):
        planning.build_plan(spec, config, output, descriptor=wrong)
    document = json.loads(spec.read_bytes())
    document["split"]["development"] = 2
    spec.write_text(json.dumps(document))
    with pytest.raises(planning.PlanError, match="DEVELOPMENT case count"):
        planning.build_plan(spec, config, output, descriptor=descriptor)
    assert not output.exists()


def test_registry_resolution_and_absolute_source_produce_the_same_plan(
    tmp_path, monkeypatch
):
    spec, config, _, descriptor = inputs(tmp_path)
    document = json.loads(config.read_bytes())
    document["source"]["path"] = str(tmp_path / "manifest.json")
    config.write_text(json.dumps(document))
    calls = []

    def resolve(name):
        calls.append(name)
        return SimpleNamespace(descriptor=descriptor)

    monkeypatch.setattr(planning, "resolve_adapter", resolve)
    output = tmp_path / "output"
    actual = planning.build_plan(spec, config, output)
    expected = planning.build_plan(spec, config, output, descriptor=descriptor)
    assert actual.document() == expected.document()
    assert calls == [descriptor.adapter_id]
    assert not output.exists()


@pytest.mark.parametrize(
    "kind", ["symlink", "core-package", "adapter-package", "public-package", "file"]
)
def test_output_cannot_alias_other_data_or_modify_the_package(tmp_path, kind):
    spec, config, _, descriptor = inputs(tmp_path)
    output = tmp_path / "output"
    if kind == "symlink":
        target = tmp_path / "unrelated"
        target.mkdir()
        symlink(output, target, directory=True)
    elif kind.endswith("-package"):
        package = {
            "core-package": piginot,
            "adapter-package": piginot_campaigns,
            "public-package": formamech,
        }[kind]
        output = Path(package.__file__).resolve().parent / "compiler-test-output"
    else:
        output.write_bytes(b"preserve")
    with pytest.raises(planning.PlanError, match="output"):
        planning.build_plan(spec, config, output, descriptor=descriptor)
    if kind == "file":
        assert output.read_bytes() == b"preserve"
    elif kind.endswith("-package"):
        assert not output.exists()
    else:
        assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "package",
    [piginot, piginot_campaigns, formamech],
    ids=["core", "adapters", "public"],
)
def test_output_parent_symlink_cannot_alias_an_installed_package(tmp_path, package):
    root = Path(package.__file__).resolve().parent
    link = tmp_path / "package"
    symlink(link, root, directory=True)
    output = link / "compiler-test-output"
    with pytest.raises(planning.PlanError, match="output"):
        planning.check_output_path(output, ())
    assert not (root / output.name).exists()


def test_public_axes_must_match_the_declared_problem(tmp_path):
    spec, config, manifest, descriptor = inputs(tmp_path)
    plan = planning.build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    cases = tuple(
        replace(case, primitive_coordinates={"wrong": 1.0}, strata={"wrong": 0})
        for case in manifest.cases
    )
    foreign = PublishedCaseManifest(
        "foreign", ("wrong",), manifest.derived_axes, ("wrong",), cases
    )
    with pytest.raises(CompilerConfigError, match="parameter axes"):
        select_public_cases(plan.problem, plan.config, foreign)


@pytest.mark.parametrize("expression", ["width/height", "width/(height-height)"])
def test_public_coordinates_must_satisfy_evaluable_constraints(tmp_path, expression):
    spec, config, _, descriptor = inputs(tmp_path)
    document = json.loads(spec.read_bytes())
    document["constraints"][0].update(expression=expression, lo=100.0, hi=101.0)
    spec.write_text(json.dumps(document))
    with pytest.raises(planning.PlanError, match="constraint"):
        planning.build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", ["lock", "record"])
def test_publication_cannot_follow_an_existing_symlink(tmp_path, kind):
    target = tmp_path / "unrelated.json"
    target.write_bytes(b"preserve")
    output = tmp_path / "output"
    output.mkdir()
    link = output / (".compiler.lock" if kind == "lock" else "report.json")
    symlink(link, target)
    with pytest.raises(planning.PlanError, match="symlink"):
        if kind == "lock":
            with records.run_lock(output):
                pytest.fail("linked lock acquired")
        else:
            records.atomic_write(link, b"new data")
    assert target.read_bytes() == b"preserve"
    assert not tuple(output.glob("*.partial"))


def test_changed_implementation_refuses_before_directory_creation(
    tmp_path, monkeypatch
):
    spec, config, _, descriptor = inputs(tmp_path)
    plan = planning.build_plan(spec, config, tmp_path / "output", descriptor=descriptor)
    monkeypatch.setattr(
        records, "_implementation", lambda: (plan.package_version, "f" * 64)
    )
    with pytest.raises(planning.PlanError, match="implementation changed"):
        records.write_plan(plan)
    assert not plan.output_path.exists()


def test_output_race_cannot_publish_over_another_writer(tmp_path, monkeypatch):
    spec, config, _, descriptor = inputs(tmp_path)
    plan = planning.build_plan(spec, config, tmp_path / "output", descriptor=descriptor)

    @contextmanager
    def competing_writer(output):
        (output / "other-writer.json").write_bytes(b"preserve")
        yield

    monkeypatch.setattr(records, "run_lock", competing_writer)
    with pytest.raises(planning.PlanError, match="changed before"):
        records.write_plan(plan)
    assert (plan.output_path / "other-writer.json").read_bytes() == b"preserve"
    assert not (plan.output_path / "plan.json").exists()


def test_windows_lock_interface_locks_one_byte_and_releases_on_error(
    tmp_path, monkeypatch
):
    calls = []
    fake = SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=lambda fd, mode, count: calls.append((mode, count)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(records, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(RuntimeError):
        with records.run_lock(tmp_path):
            assert (tmp_path / ".compiler.lock").read_bytes() == b"\0"
            raise RuntimeError("owner interrupted")
    assert calls == [(1, 1), (2, 1)]


def test_atomic_replace_failure_preserves_old_file_and_cleans_partial(
    tmp_path, monkeypatch
):
    target = tmp_path / "report.json"
    target.write_bytes(b"preserve")

    def unavailable(*args):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(
        records,
        "os",
        SimpleNamespace(fdopen=os.fdopen, fsync=os.fsync, replace=unavailable),
    )
    with pytest.raises(OSError, match="publication failure"):
        records.atomic_write(target, b"new report")
    assert target.read_bytes() == b"preserve"
    assert not tuple(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("code", [True, None, 99])
def test_adapter_exit_contract_does_not_accept_unexpected_values(
    tmp_path, monkeypatch, capsys, code
):
    spec, config, _, descriptor = inputs(tmp_path)
    monkeypatch.setattr(
        compiler_cli,
        "resolve_adapter",
        lambda name: SimpleNamespace(
            descriptor=descriptor, execute=lambda plan, **kwargs: code
        ),
    )
    output = tmp_path / "output"
    assert cli.main(arguments(spec, config, output)) == 1
    assert "invalid compiler exit code" in capsys.readouterr().err
    assert not output.exists()
