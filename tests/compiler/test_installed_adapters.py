"""Public adapter metadata is packaged and independent of a source checkout."""

from __future__ import annotations

import dataclasses
import hashlib
import sys
import tomllib
from pathlib import Path

import pytest

from piginot.compiler.adapters import AdapterDescriptor

ROOT = Path(__file__).resolve().parents[2]


def test_both_adapters_match_the_packaged_problem_resources():
    from piginot_campaigns.adapters import DOGBONE, NOTCHED, load_problem

    for adapter in (DOGBONE, NOTCHED):
        description = adapter.descriptor
        spec = load_problem(description.adapter_id)
        assert type(description) is AdapterDescriptor
        assert len(description.implementation_sha256) == 64
        geometry = description.geometries[0]
        assert geometry.reference_mesh_id == spec.geometry.reference_mesh_id
        assert geometry.morph_map_sha256 == spec.geometry.morph_map_sha256
        assert "hgo" in description.materials
        assert "formula-fibre" in description.materials
        assert description.liftings == ("quarter", "adf")
        with pytest.raises(dataclasses.FrozenInstanceError):
            adapter.adapter_id = "changed"
    with pytest.raises(ValueError, match="unsupported"):
        load_problem("module:attribute")


def test_adapter_digest_is_computed_from_current_code_and_resources(monkeypatch):
    from piginot_campaigns import adapters

    original = adapters.implementation_sha256()
    assert original == adapters.DOGBONE.descriptor.implementation_sha256
    original_read = adapters._read

    def changed(name):
        payload = original_read(name)
        return payload + b" " if name.endswith("notched-v1.json") else payload

    monkeypatch.setattr(adapters, "_read", changed)
    assert adapters.implementation_sha256() != original
    assert adapters.DOGBONE.descriptor.implementation_sha256 != original
    assert hashlib.sha256(b"caller-supplied-revision").hexdigest() != original


def test_wheel_and_source_distribution_include_public_adapter_package():
    document = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert (
        "piginot_campaigns"
        in document["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    )
    assert (
        "/piginot_campaigns"
        in document["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    )
    entries = document["project"]["entry-points"]["piginot.compiler.adapters"]
    assert set(entries) == {"dogbone-v1", "notched-v1", "cad-fixed-v1"}


def test_adapter_metadata_imports_no_numerical_or_private_campaign_modules():
    import subprocess

    program = (
        "from piginot_campaigns.adapters import DOGBONE; import sys; "
        "assert DOGBONE.descriptor.geometries; "
        "assert not any(n == p or n.startswith(p+'.') for n in sys.modules "
        "for p in ('torch','numpy','scipy','skfem','piginot.fem','piginot.problem','campaigns'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
