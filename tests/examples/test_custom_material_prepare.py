"""The released tutorial prepares bound public inputs without running numerics."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from examples.custom_material import prepare
from piginot.compiler import adapters
from piginot.compiler.planning import build_plan
from piginot.contracts import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "examples/custom_material"


@pytest.fixture(autouse=True)
def metadata_versions(monkeypatch):
    monkeypatch.setattr(prepare.metadata, "version", lambda name: "1.0")
    entry = metadata.EntryPoint(
        name="dogbone-v1",
        value="piginot_campaigns.adapters:DOGBONE",
        group=adapters.ADAPTER_GROUP,
    )
    monkeypatch.setattr(adapters, "entry_points", lambda **kwargs: (entry,))


def test_metadata_preparation_preserves_physical_inputs_and_binds_every_input(tmp_path):
    output = tmp_path / "inputs"
    record = prepare.prepare(output)
    config = json.loads((output / "config.json").read_bytes())
    assert record["profile_status"] == "METADATA_ONLY"
    assert record["claim_level"] == "NOT_VERIFIED"
    assert config["training"]["optimizer_steps"] == 1
    assert config["training"]["seed"] == 2901
    assert config["reference"] == {
        "initialization": "cold",
        "refinement_levels": [1, 2],
        "workers": 2,
    }
    assert config["selection"] == {"checkpoint_rule": "final", "development": False}
    for name in ("problem.json", "source.json"):
        assert (output / name).read_bytes() == (ASSETS / name).read_bytes()
    for name, value in record["input_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == value
    for name, value in record["template_sha256"].items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == value
    for name, value in record["profile_sha256"].items():
        assert (
            hashlib.sha256(canonical_json_bytes(config["execution"][name])).hexdigest()
            == value
        )
    assert json.loads((output / "preparation.json").read_bytes()) == record
    plan = build_plan(
        output / "problem.json", output / "config.json", tmp_path / "plan"
    )
    assert plan.config.source.fit_case_ids == ("fit-001", "fit-002")
    assert plan.config.source.screen_case_ids == ("screen-001",)
    assert not plan.config.source.development_case_ids


def test_metadata_preparation_never_probes_cpu(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("Numerical profile probe during metadata-only preparation")

    monkeypatch.setattr(prepare, "_cpu_capability", forbidden)
    prepare.prepare(tmp_path / "inputs")
    config = json.loads((tmp_path / "inputs/config.json").read_bytes())
    assert (
        config["execution"]["training"]["platform_fingerprint"]["cpu_capability"]
        == "NOT_PROBED"
    )


def test_execution_profile_captures_environment_without_running_a_campaign(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prepare, "_cpu_capability", lambda: "MOCK_CPU")
    record = prepare.prepare(tmp_path / "inputs", execution_profile=True)
    config = json.loads((tmp_path / "inputs/config.json").read_bytes())
    training = config["execution"]["training"]
    reference = config["execution"]["reference"]
    assert record["profile_status"] == "ENVIRONMENT_CAPTURED_NOT_EXECUTED"
    assert training["platform_fingerprint"]["cpu_capability"] == "MOCK_CPU"
    assert set(training["library_versions"].values()) == {"1.0"}
    assert training["dtype"] == "float32"
    assert reference == {**training, "dtype": "float64"}
    assert not list((tmp_path / "inputs").glob("stage-*.json"))


def test_cpu_capture_uses_only_the_existing_profile_probe(monkeypatch):
    class CPU:
        @staticmethod
        def get_cpu_capability():
            return "MOCK_CPU"

    class Torch:
        class backends:
            cpu = CPU()

    monkeypatch.setitem(sys.modules, "torch", Torch())
    assert prepare._cpu_capability() == "MOCK_CPU"


def missing_torch(name):
    if name == "torch":
        raise metadata.PackageNotFoundError(name)
    return "1.0"


def test_base_installation_can_prepare_without_numerical_distributions(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prepare.metadata, "version", missing_torch)
    prepare.prepare(tmp_path / "inputs")
    config = json.loads((tmp_path / "inputs/config.json").read_bytes())
    assert (
        config["execution"]["training"]["library_versions"]["torch"] == "NOT_INSTALLED"
    )


def test_execution_profile_requires_dependencies_before_creating_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prepare.metadata, "version", missing_torch)
    with pytest.raises(ValueError, match=r"formamech\[verify\].*torch"):
        prepare.prepare(tmp_path / "inputs", execution_profile=True)
    assert not (tmp_path / "inputs").exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_existing_output_is_retained(tmp_path, kind):
    output = tmp_path / "inputs"
    if kind == "directory":
        output.mkdir()
        (output / "retained").write_bytes(b"keep")
    else:
        output.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare.prepare(output)
    retained = output / "retained" if kind == "directory" else output
    assert retained.read_bytes() == b"keep"


@pytest.mark.parametrize("relative", [".", "..", "nested/new-inputs"])
def test_output_cannot_overlap_tutorial_assets(relative):
    with pytest.raises(ValueError, match="overlap"):
        prepare.prepare(ASSETS / relative)


def test_symlink_output_is_refused_without_following_it(tmp_path, monkeypatch):
    output = tmp_path / "link"
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: path == output or original(path)
    )
    with pytest.raises(ValueError, match="symlink"):
        prepare.prepare(output)
    assert not output.exists()


@pytest.fixture
def copied_assets(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    shutil.copytree(ASSETS, assets)
    monkeypatch.setattr(prepare, "ASSETS", assets)
    return assets


def test_linked_input_is_refused(copied_assets, tmp_path, monkeypatch):
    linked = copied_assets / "source.json"
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: path == linked or original(path)
    )
    with pytest.raises(ValueError, match="regular input"):
        prepare.prepare(tmp_path / "inputs")
    assert not (tmp_path / "inputs").exists()


def test_missing_input_is_refused_before_output(copied_assets, tmp_path):
    (copied_assets / "source.json").unlink()
    with pytest.raises(ValueError, match="regular input"):
        prepare.prepare(tmp_path / "inputs")
    assert not (tmp_path / "inputs").exists()


@pytest.mark.parametrize("payload", [b"[]", b"not json"])
def test_malformed_config_is_refused_before_output(copied_assets, tmp_path, payload):
    (copied_assets / "config-template.json").write_bytes(payload)
    with pytest.raises(ValueError):
        prepare.prepare(tmp_path / "inputs")
    assert not (tmp_path / "inputs").exists()


@pytest.mark.parametrize("key,value", [("optimizer_steps", 2), ("seed", 1)])
def test_tutorial_budget_is_fixed(copied_assets, tmp_path, key, value):
    path = copied_assets / "config-template.json"
    config = json.loads(path.read_bytes())
    config["training"][key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        prepare.prepare(tmp_path / "inputs")
    assert not (tmp_path / "inputs").exists()


def test_invalid_public_manifest_is_refused_before_output(copied_assets, tmp_path):
    (copied_assets / "source.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        prepare.prepare(tmp_path / "inputs")
    assert not (tmp_path / "inputs").exists()


def test_main_reports_preparation_and_never_compiles(tmp_path, capsys):
    assert prepare.main(["--output", str(tmp_path / "inputs")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PREPARED"
    assert result["profile_status"] == "METADATA_ONLY"


def test_main_reports_refusals_without_overwriting(tmp_path, capsys):
    output = tmp_path / "inputs"
    output.mkdir()
    assert prepare.main(["--output", str(output)]) == 2
    result = json.loads(capsys.readouterr().err)
    assert result["status"] == "REFUSED"
    assert "already exists" in result["reason"]


def test_relocated_standalone_assets_prepare_and_plan_without_numerical_imports(
    tmp_path,
):
    relocated = tmp_path / "released-tutorial"
    shutil.copytree(ASSETS, relocated)
    code = r"""
import importlib.abc, json, runpy, sys
from pathlib import Path
class NoNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy', 'sympy', 'skfem', 'meshio', 'safetensors', 'threadpoolctl'}:
            raise AssertionError('Numerical import: ' + fullname)
sys.meta_path.insert(0, NoNumerics())
from importlib.metadata import EntryPoint
from piginot.compiler import adapters
entry = EntryPoint(name='dogbone-v1', value='piginot_campaigns.adapters:DOGBONE', group=adapters.ADAPTER_GROUP)
adapters.entry_points = lambda **kwargs: (entry,)
script = Path(sys.argv[1])
output = Path(sys.argv[2])
ns = runpy.run_path(str(script))
assert ns['main'](['--output', str(output)]) == 0
from piginot.cli import main
assert main(['compile', '--spec', str(output / 'problem.json'), '--config', str(output / 'config.json'), '--output', str(output.parent / 'plan'), '--plan-only']) == 0
plan = json.loads((output.parent / 'plan/plan.json').read_bytes())
assert plan['claim_level'] == 'NOT_VERIFIED'
assert plan['execution_profile_status'] == 'DECLARED_NOT_NUMERICALLY_PROBED'
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(relocated / "prepare.py"),
            str(tmp_path / "inputs"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
