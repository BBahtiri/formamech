"""FormaMech exposes stable classes and serialized identities during the rename."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_primary_package_uses_the_existing_version():
    public = importlib.import_module("formamech")
    legacy = importlib.import_module("piginot")
    assert public.__version__ == legacy.__version__
    assert public.__all__ == ("__version__",)


@pytest.mark.parametrize(
    "suffix,export",
    [
        ("contracts", "Quantity"),
        ("contracts.specs", "CanonicalMesh"),
        ("contracts.canonical", "canonical_json_bytes"),
        ("bundle", "load_bundle"),
        ("bundle.manifest", "BundleManifest"),
        ("compiler.adapters", "resolve_adapter"),
        ("compiler.config", "CampaignConfig"),
        ("materials.registry", "known_models"),
    ],
)
def test_deep_imports_share_module_and_class_identity(suffix, export):
    legacy = importlib.import_module("piginot." + suffix)
    original_spec = legacy.__spec__
    public = importlib.import_module("formamech." + suffix)
    assert public is legacy
    assert getattr(public, export) is getattr(legacy, export)
    assert public.__name__ == "piginot." + suffix
    assert public.__spec__ is original_spec


def test_contract_bytes_are_not_renamed():
    from piginot.contracts import Quantity, Unit, canonical_json_bytes

    original = Quantity(12.0, Unit.MM)
    before = canonical_json_bytes(original)
    public = importlib.import_module("formamech.contracts")
    assert public.Quantity is Quantity
    assert public.canonical_json_bytes(public.Quantity(12.0, public.Unit.MM)) == before
    assert canonical_json_bytes(original) == before


def test_primary_cli_and_subcommand_help_use_formamech(capsys):
    cli = importlib.import_module("formamech.cli")
    assert cli.build_parser().prog == "formamech"
    for arguments, expected in (
        (["--version"], "formamech 0.2.0.dev0"),
        (["compile", "--help"], "usage: formamech compile"),
        (["cad", "--help"], "usage: formamech cad"),
    ):
        with pytest.raises(SystemExit) as result:
            cli.main(arguments)
        assert result.value.code == 0
        assert expected in capsys.readouterr().out


def test_new_cli_reuses_request_parsing_without_duplicate_contracts():
    from piginot.cli import parse_request

    assert importlib.import_module("formamech.cli").parse_request is parse_request


def test_planning_and_execution_bind_the_same_implementation():
    from piginot.compiler.planning import _implementation
    from piginot.compiler.provenance import load_implementation_identity

    identity = load_implementation_identity()
    assert _implementation() == (identity.package_version, identity.package_sha256)


@pytest.mark.parametrize(
    "module,factory",
    [
        ("campaigns.dogbone_v1.s10_production", "default_s10_execution_profile"),
        ("campaigns.dogbone_v1.strain_pilot", "_execution_profile"),
        ("campaigns.notched_v1.pilot", "execution_profile"),
    ],
)
def test_legacy_profiles_use_new_metadata_and_keep_record_keys(module, factory):
    profile = getattr(importlib.import_module(module), factory)()
    assert profile.library_versions["piginot"] == version("formamech")
    assert "formamech" not in profile.library_versions


def test_branding_keeps_a_legacy_console_alias():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "formamech"
    assert project["scripts"]["formamech"] == "formamech.cli:main"
    assert project["scripts"]["piginot"] == "piginot.cli:main"


def test_unknown_public_module_raises_the_requested_module_name():
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module("formamech.no_such_module")
    assert error.value.name == "formamech.no_such_module"


def test_metadata_imports_do_not_load_numerical_libraries():
    program = """
import sys
import formamech
from formamech.compiler.adapters import ADAPTER_GROUP
assert ADAPTER_GROUP == 'piginot.compiler.adapters'
assert not any(n == p or n.startswith(p + '.') for n in sys.modules
    for p in ('torch', 'numpy', 'scipy', 'sympy', 'skfem', 'campaigns'))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_module_cli_executes_under_the_new_name():
    for name in ("formamech", "formamech.cli"):
        result = subprocess.run(
            [sys.executable, "-m", name, "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "formamech 0.2.0.dev0"


@pytest.mark.parametrize("first", ["formamech", "piginot"])
def test_import_order_and_reload_preserve_identity(first):
    program = f"""
import importlib
first = importlib.import_module('{first}.contracts.specs')
public = importlib.import_module('formamech.contracts.specs')
legacy = importlib.import_module('piginot.contracts.specs')
assert first is public is legacy
assert importlib.reload(public) is legacy
assert public.CanonicalMesh is legacy.CanonicalMesh
assert public.__spec__.name == 'piginot.contracts.specs'
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
