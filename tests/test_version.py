import runpy
import sys
from importlib import import_module
from importlib.metadata import version

import pytest


def test_package_version_matches_distribution_metadata() -> None:
    package = import_module("formamech")

    assert package.__version__ == version("formamech")


def test_package_public_api_contains_only_version() -> None:
    package = import_module("piginot")

    assert package.__all__ == ("__version__",)


def test_cli_reports_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = import_module("piginot.cli")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"piginot {version('formamech')}"


def test_module_entry_point_reports_version(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["piginot", "--version"])
    monkeypatch.delitem(sys.modules, "piginot.cli", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("piginot.cli", run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"piginot {version('formamech')}"
