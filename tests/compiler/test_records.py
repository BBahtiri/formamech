"""Plan publication and resume never replace a differently bound run."""

from __future__ import annotations

import json

import pytest

from piginot.compiler import records
from piginot.compiler.planning import PlanError, build_plan
from tests.compiler.plan_fixtures import inputs


def plan(tmp_path):
    spec, config, _, adapter = inputs(tmp_path)
    return build_plan(spec, config, tmp_path / "run", descriptor=adapter)


def test_atomic_plan_publication_and_exact_resume(tmp_path):
    value = plan(tmp_path)
    path = records.write_plan(value)
    assert path == tmp_path / "run/plan.json"
    assert json.loads(path.read_bytes()) == value.document()
    payload = path.read_bytes()
    assert records.write_plan(value, resume=True) == path
    assert path.read_bytes() == payload
    assert not tuple(path.parent.glob("*.partial"))
    with pytest.raises(PlanError, match="resume"):
        records.write_plan(value)


def test_unbound_directory_and_missing_resume_are_refused(tmp_path):
    value = plan(tmp_path)
    with pytest.raises(PlanError, match="resume"):
        records.write_plan(value, resume=True)
    value.output_path.mkdir()
    original = value.output_path / "user-data.txt"
    original.write_text("preserve")
    with pytest.raises(PlanError):
        records.write_plan(value)
    assert original.read_text() == "preserve"
    assert not (value.output_path / "plan.json").exists()


def test_changed_source_refuses_before_output_creation(tmp_path):
    value = plan(tmp_path)
    value.source_path.write_bytes(b"changed")
    with pytest.raises(PlanError, match="input"):
        records.write_plan(value)
    assert not value.output_path.exists()


def test_changed_saved_plan_cannot_be_adopted(tmp_path):
    value = plan(tmp_path)
    path = records.write_plan(value)
    path.write_bytes(b"{}")
    with pytest.raises(PlanError, match="binding"):
        records.write_plan(value, resume=True)
    assert path.read_bytes() == b"{}"


def test_process_lock_excludes_a_second_owner_and_releases_on_exception(tmp_path):
    value = plan(tmp_path)
    records.write_plan(value)
    with pytest.raises(RuntimeError):
        with records.run_lock(value.output_path):
            with pytest.raises(PlanError, match="locked"):
                with records.run_lock(value.output_path):
                    pytest.fail("second owner")
            raise RuntimeError("synthetic owner interruption")
    with records.run_lock(value.output_path):
        pass


def test_symlink_plan_is_refused_before_reading_other_files(tmp_path):
    value = plan(tmp_path)
    value.output_path.mkdir()
    secret = tmp_path / "unrelated.json"
    secret.write_bytes(b"unrelated")
    path = value.output_path / "plan.json"
    try:
        path.symlink_to(secret)
    except OSError as error:
        pytest.skip(f"symlink capability unavailable: {error}")
    with pytest.raises(PlanError, match="symlink"):
        records.write_plan(value, resume=True)
    assert secret.read_bytes() == b"unrelated"
