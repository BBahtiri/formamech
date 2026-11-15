"""The installed CAD surface routes lazily and reports explicit refusals."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from piginot import __version__
from piginot import cli as main_cli
from piginot.cad import cli
from piginot.contracts import Unit


@pytest.mark.parametrize("capture", [False, True])
def test_prepare_routes_user_settings_without_starting_training(
    monkeypatch, tmp_path, capsys, capture
):
    observed = []
    project, settings, output = (
        tmp_path / name for name in ("project", "physics.json", "inputs")
    )

    def prepare(source, physics, target, *, execution_profile):
        observed.append((source, physics, target, execution_profile))
        return {"status": "PREPARED", "training_started": False}

    monkeypatch.setattr(cli, "prepare_campaign", prepare, raising=False)
    arguments = [
        "cad",
        "prepare",
        "--project",
        str(project),
        "--settings",
        str(settings),
        "--output",
        str(output),
    ]
    if capture:
        arguments.append("--execution-profile")
    assert main_cli.main(arguments) == 0
    assert observed == [(project, settings, output, capture)]
    assert json.loads(capsys.readouterr().out)["training_started"] is False


def test_import_command_routes_paths_units_and_mesh_size(monkeypatch, tmp_path, capsys):
    source, output = tmp_path / "profile.dxf", tmp_path / "project"
    observed = []

    def prepare(path, target, *, unit, mesh_size_mm):
        observed.append((path, target, unit, mesh_size_mm))
        return {"status": "IMPORTED", "claim_level": "NOT_VERIFIED"}

    monkeypatch.setattr(cli, "prepare_geometry", prepare)
    assert (
        main_cli.main(
            [
                "cad",
                "import",
                str(source),
                "--unit",
                "cm",
                "--mesh-size-mm",
                "1.25",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert observed == [(source, output, Unit.CM, 1.25)]
    assert json.loads(capsys.readouterr().out)["status"] == "IMPORTED"


def test_inspect_command_routes_the_project(monkeypatch, tmp_path, capsys):
    observed = []

    def inspect(path):
        observed.append(path)
        return {"schema": "piginot.cad-preview-v1", "boundaries": []}

    monkeypatch.setattr(cli, "inspect_geometry", inspect)
    assert main_cli.main(["cad", "inspect", str(tmp_path)]) == 0
    assert observed == [tmp_path]
    assert json.loads(capsys.readouterr().out)["boundaries"] == []


@pytest.mark.parametrize(
    "error",
    (
        ValueError("bad input"),
        OSError("unreadable"),
        ImportError("optional backend unavailable"),
    ),
)
def test_import_errors_are_structured_refusals(monkeypatch, tmp_path, capsys, error):
    def refused(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "prepare_geometry", refused)
    assert (
        main_cli.main(
            [
                "cad",
                "import",
                "profile.step",
                "--unit",
                "mm",
                "--mesh-size-mm",
                "1",
                "--output",
                str(tmp_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert not captured.out
    document = json.loads(captured.err)
    assert document["status"] == "REFUSED"
    assert str(error) in document["reason"]


def test_inspect_failure_is_a_structured_refusal(monkeypatch, tmp_path, capsys):
    def refused(*args):
        raise ValueError("tampered mesh")

    monkeypatch.setattr(cli, "inspect_geometry", refused)
    assert main_cli.main(["cad", "inspect", str(tmp_path)]) == 2
    assert "tampered mesh" in json.loads(capsys.readouterr().err)["reason"]


def test_unknown_direct_action_refuses(capsys):
    assert cli.run(argparse.Namespace(cad_command="unknown")) == 2
    assert "unknown" in json.loads(capsys.readouterr().err)["reason"]


@pytest.mark.parametrize(
    "arguments",
    (
        ["cad"],
        ["cad", "import", "file.step"],
        [
            "cad",
            "import",
            "file.step",
            "--unit",
            "inch",
            "--mesh-size-mm",
            "1",
            "--output",
            "project",
        ],
    ),
)
def test_parser_rejects_missing_or_unknown_required_inputs(arguments):
    with pytest.raises(SystemExit) as caught:
        main_cli.main(arguments)
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    (
        ["--version"],
        ["compile", "--help"],
        ["cad", "--help"],
        ["cad", "import", "--help"],
    ),
)
def test_help_and_version_do_not_import_cad_or_numerical_modules(arguments, tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc, json, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('piginot.cad') or fullname.split('.')[0] in {'gmsh','ezdxf','numpy','torch','scipy','skfem','sympy','meshio'}:
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Deny())
from piginot.cli import main
try:
    main(json.loads(sys.argv[1]))
except SystemExit as error:
    assert error.code == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, json.dumps(arguments)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        f"piginot {__version__}" if arguments == ["--version"] else "usage:"
    ) in completed.stdout
