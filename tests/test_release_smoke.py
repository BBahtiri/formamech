"""Installed release smoke checks distinguish base metadata from runtime checks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke():
    spec = importlib.util.spec_from_file_location(
        "release_smoke_under_test", ROOT / "scripts/release_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installation(smoke, monkeypatch, tmp_path):
    import piginot
    import piginot.compiler.adapters as adapters
    import piginot.compiler.provenance as provenance
    import piginot_campaigns
    from piginot_campaigns.adapters import DOGBONE, NOTCHED

    package_root = tmp_path / "site-packages"
    monkeypatch.setattr(piginot, "__file__", str(package_root / "piginot/__init__.py"))
    monkeypatch.setattr(
        piginot_campaigns,
        "__file__",
        str(package_root / "piginot_campaigns/__init__.py"),
    )
    entries = [
        SimpleNamespace(group="piginot.compiler.adapters", name=name, value=value)
        for name, value in {
            "dogbone-v1": "piginot_campaigns.adapters:DOGBONE",
            "notched-v1": "piginot_campaigns.adapters:NOTCHED",
            "cad-fixed-v1": "piginot.cad.adapter:FIXED_CAD",
        }.items()
    ] + [
        SimpleNamespace(
            group="console_scripts", name="piginot", value="piginot.cli:main"
        )
    ]
    distribution = SimpleNamespace(
        version=piginot.__version__,
        locate_file=lambda relative: package_root / relative,
        entry_points=entries,
    )
    monkeypatch.setattr(smoke, "distribution", lambda name: distribution, raising=False)
    identity = provenance.ImplementationIdentity(
        "a" * 40, "b" * 64, "c" * 64, piginot.__version__, False, "packaged"
    )
    monkeypatch.setattr(
        provenance, "require_clean_implementation_identity", lambda: identity
    )
    monkeypatch.setattr(
        adapters,
        "resolve_adapter",
        lambda name: {"dogbone-v1": DOGBONE, "notched-v1": NOTCHED}[name],
    )
    monkeypatch.setattr(smoke, "_loaded_modules", lambda prefixes: [], raising=False)
    monkeypatch.setattr(
        smoke.shutil, "which", lambda *args, **kwargs: "/venv/bin/piginot"
    )

    def console(command, **kwargs):
        text = (
            f"piginot {piginot.__version__}"
            if command[-1] == "--version"
            else "usage: piginot compile --spec --config --output --plan-only --resume"
        )
        return subprocess.CompletedProcess(command, 0, text, "")

    monkeypatch.setattr(smoke.subprocess, "run", console)
    return distribution, identity, provenance


def test_base_preflight_checks_installed_identity_and_both_adapters(
    smoke, installation
):
    import piginot

    result = smoke.run(piginot.__version__, base_only=True)
    assert result["passed"] is True
    assert result["scope"] == "base-metadata"
    assert result["implementation"]["source_revision"] == "a" * 40
    assert set(result["adapters"]) == {"dogbone-v1", "notched-v1"}
    assert result["resource_required_adapters"] == ["cad-fixed-v1"]
    assert result["not_requested"] == ["claim_ceiling", "missing_bundle_refused"]
    assert "claim_ceiling" not in result["checks"]


def test_runtime_mode_preserves_the_legacy_checks(smoke, installation, monkeypatch):
    import piginot

    monkeypatch.setattr(
        smoke,
        "_runtime_checks",
        lambda: (
            {"claim_ceiling": True, "missing_bundle_refused": True},
            ["BUNDLE_UNREADABLE"],
        ),
    )
    result = smoke.run(piginot.__version__)
    assert result["passed"] is True
    assert result["scope"] == "runtime-bundle-refusal"
    assert result["not_requested"] == []
    assert result["refusal_codes"] == ["BUNDLE_UNREADABLE"]


@pytest.mark.parametrize("wrong", ["0.0.0", "", "0.1.0"])
def test_wrong_expected_version_fails(smoke, installation, wrong):
    result = smoke.run(wrong, base_only=True)
    assert result["passed"] is False
    assert result["checks"]["version"] is False


def test_distribution_version_must_match_imported_package(smoke, installation):
    import piginot

    installation[0].version = "0.0.0"
    assert (
        smoke.run(piginot.__version__, base_only=True)["checks"]["metadata_version"]
        is False
    )


def test_imported_package_must_come_from_its_distribution(
    smoke, installation, monkeypatch
):
    import piginot

    monkeypatch.setattr(piginot, "__file__", "/checkout/src/piginot/__init__.py")
    result = smoke.run(piginot.__version__, base_only=True)
    assert result["checks"]["installed_not_source"] is False


def test_adapter_package_must_come_from_the_same_distribution(
    smoke, installation, monkeypatch
):
    import piginot
    import piginot_campaigns

    monkeypatch.setattr(
        piginot_campaigns, "__file__", "/checkout/piginot_campaigns/__init__.py"
    )
    assert (
        smoke.run(piginot.__version__, base_only=True)["checks"]["installed_not_source"]
        is False
    )


def test_source_identity_is_not_an_installed_release(smoke, installation, monkeypatch):
    from dataclasses import replace

    import piginot

    monkeypatch.setattr(
        installation[2],
        "require_clean_implementation_identity",
        lambda: replace(installation[1], origin="source"),
    )
    assert (
        smoke.run(piginot.__version__, base_only=True)["checks"]["packaged_identity"]
        is False
    )


def test_wrong_adapter_registration_fails(smoke, installation):
    import piginot

    installation[0].entry_points[0].value = "elsewhere:adapter"
    assert (
        smoke.run(piginot.__version__, base_only=True)["checks"]["adapter_entry_points"]
        is False
    )


def test_missing_console_does_not_fall_back_to_another_environment(
    smoke, installation, monkeypatch
):
    import piginot

    monkeypatch.setattr(smoke.shutil, "which", lambda *args, **kwargs: None)
    result = smoke.run(piginot.__version__, base_only=True)
    assert result["checks"]["console_script"] is False
    assert result["checks"]["compiler_help"] is False


@pytest.mark.parametrize("returncode,stdout", [(2, "error"), (0, "wrong help")])
def test_console_or_compiler_failure_is_not_a_pass(
    smoke, installation, monkeypatch, returncode, stdout
):
    import piginot

    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode, stdout, ""
        ),
    )
    result = smoke.run(piginot.__version__, base_only=True)
    assert result["passed"] is False
    assert result["checks"]["compiler_help"] is False


def test_base_scope_rejects_loaded_numerical_modules(smoke, installation, monkeypatch):
    import piginot

    monkeypatch.setattr(smoke, "_loaded_modules", lambda prefixes: ["torch"])
    result = smoke.run(piginot.__version__, base_only=True)
    assert result["passed"] is False
    assert result["checks"]["no_numerical_imports"] is False


@pytest.mark.parametrize("mode", [[], ["--base-only"]])
def test_main_reports_bad_provenance_as_json(
    smoke, installation, monkeypatch, capsys, mode
):
    def corrupt():
        raise installation[2].BuildIdentityError("changed installed bytes")

    monkeypatch.setattr(
        installation[2], "require_clean_implementation_identity", corrupt
    )
    assert smoke.main(["--expect-version", "0.2.0.dev0", *mode]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is False
    assert "changed installed bytes" in result["error"]


def test_main_writes_the_same_json_to_summary(
    smoke, installation, monkeypatch, tmp_path, capsys
):
    import piginot

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert smoke.main(["--expect-version", piginot.__version__, "--base-only"]) == 0
    line = capsys.readouterr().out.strip()
    assert summary.read_text() == f"```json\n{line}\n```\n"


def test_module_detection_matches_only_real_prefixes(smoke, monkeypatch):
    monkeypatch.setitem(smoke.sys.modules, "release_smoke_fake", SimpleNamespace())
    monkeypatch.setitem(
        smoke.sys.modules, "release_smoke_fake.child", SimpleNamespace()
    )
    monkeypatch.setitem(
        smoke.sys.modules, "release_smoke_fake_other", SimpleNamespace()
    )
    assert smoke._loaded_modules(("release_smoke_fake",)) == [
        "release_smoke_fake",
        "release_smoke_fake.child",
    ]


def test_runtime_missing_bundle_is_refused_with_the_real_loader(smoke):
    checks, reasons = smoke._runtime_checks()
    assert checks == {"claim_ceiling": True, "missing_bundle_refused": True}
    assert reasons


def test_runtime_checks_reject_a_missing_refusal_and_stronger_claim(smoke, monkeypatch):
    import piginot.bundle as bundle
    import piginot.verification as verification

    monkeypatch.setattr(bundle, "load_bundle", lambda path: object())
    monkeypatch.setattr(
        verification, "ClaimLevel", [SimpleNamespace(value="SEALED_RELEASE_VERIFIED")]
    )
    checks, reasons = smoke._runtime_checks()
    assert checks == {"claim_ceiling": False, "missing_bundle_refused": False}
    assert reasons == []


def test_preflight_workflow_is_manual_pinned_and_contains_no_numerical_run():
    import re

    import yaml

    path = ROOT / ".github/workflows/s29-preflight.yml"
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    matrix = workflow["jobs"]["install"]["strategy"]["matrix"]
    assert set(matrix["os"]) == {"ubuntu-latest", "macos-latest", "windows-latest"}
    assert matrix["python-version"] == ["3.11", "3.12", "3.13"]
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", step["uses"])
            command = step.get("run", "")
            assert "twine upload" not in command
            assert "--execution-profile" not in command
            for line in command.splitlines():
                if " compile " in line:
                    assert "--plan-only" in line


def test_command_requires_an_explicit_expected_version(monkeypatch):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["release_smoke.py"])
    with pytest.raises(SystemExit, match="2"):
        runpy.run_path(str(ROOT / "scripts/release_smoke.py"), run_name="__main__")
