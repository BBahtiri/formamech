"""Installed-wheel qualification; set FORMAMECH_TEST_WHEEL to the built artifact.

These tests install only the wheel and its declared dependencies in a temporary
environment. They never import repository test helpers in that environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

MAKE_INPUTS = r"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import piginot
from piginot.compiler.adapters import resolve_adapter
from piginot.doe import EvidenceRole, PublishedCase, PublishedCaseManifest, PublishedCaseStatus
from piginot_campaigns.adapters import load_problem

root = Path(sys.argv[1])
family = sys.argv[2]
root.mkdir(parents=True)
document = load_problem(family).document()
document['split'] = {'pilot': 0, 'train': 3, 'development': 1, 'holdout': 1, 'reserve': 0}
(root / 'problem.json').write_text(json.dumps(document))
parameters = document['parameters']
axes = tuple(parameter['name'] for parameter in parameters)
cases = []
for index, case_id in enumerate(('fit-001', 'fit-002', 'screen-001')):
    values = {parameter['name']: (parameter['lo'] + parameter['hi']) / 2 for parameter in parameters}
    for parameter in parameters:
        if parameter['role'] == 'load':
            values[parameter['name']] = parameter['lo'] + (index + 1) / 4 * (parameter['hi'] - parameter['lo'])
    cases.append(PublishedCase(case_id, f'group-{case_id}', EvidenceRole.TRAIN,
        values, {'fixture_derived': 1.0}, {axes[0]: index}, PublishedCaseStatus.ACTIVE))
manifest = PublishedCaseManifest('installed-fixture', axes, ('fixture_derived',), (axes[0],), tuple(cases))
payload = manifest.to_bytes()
(root / 'public.json').write_bytes(payload)
profile = {
    'python_version': platform.python_version(),
    'library_versions': {name: 'NOT_INSTALLED' for name in
        ('numpy', 'torch', 'scipy', 'scikit-fem', 'sympy', 'safetensors', 'threadpoolctl')},
    'blas_threads': 1, 'device': 'cpu', 'dtype': 'float32', 'rng_algorithm': 'torch-manual-seed',
    'seed': 2801, 'platform_fingerprint': {'os': platform.system(), 'machine': platform.machine(), 'cpu_capability': 'NOT_PROBED'},
}
config = {
    'schema': 'piginot.campaign-config-v1', 'name': 'installed-fixture',
    'adapter': {'id': family, 'implementation_sha256': resolve_adapter(family).descriptor.implementation_sha256},
    'source': {'kind': 'public-manifest', 'path': 'public.json', 'sha256': hashlib.sha256(payload).hexdigest(),
        'fit_case_ids': ['fit-001', 'fit-002'], 'screen_case_ids': ['screen-001'], 'development_case_ids': []},
    'model': {'architecture_id': 'pi-ginot-energy-parametric-film-v1', 'parameter_names': list(axes)},
    'normalization': 'train-fit',
    'training': {'optimizer_steps': 4, 'case_batch_size': 1, 'checkpoint_interval': 1, 'learning_rate': 0.001,
        'final_learning_rate': 0.0001, 'constant_steps': 0, 'seed': 2801},
    'objective': {'points_per_cell': 3, 'barrier_weight': 1000.0, 'barrier_threshold': 0.05, 'admissibility_floor': 1e-6},
    'reference': {'initialization': 'cold', 'refinement_levels': [1, 2], 'workers': 2},
    'execution': {'training': profile, 'reference': {**profile, 'dtype': 'float64'}, 'inference_dtype': 'float64'},
    'selection': {'checkpoint_rule': 'final', 'development': False},
}
(root / 'campaign.json').write_text(json.dumps(config))
"""


def _run(arguments, *, cwd, env):
    result = subprocess.run(
        arguments, cwd=cwd, env=env, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    configured = os.environ.get("FORMAMECH_TEST_WHEEL") or os.environ.get(
        "PIGINOT_S28_TEST_WHEEL"
    )
    if configured is None:
        pytest.skip("installed-wheel job requires FORMAMECH_TEST_WHEEL")
    wheel = Path(configured).resolve(strict=True)
    root = tmp_path_factory.mktemp("s28-installed-wheel")
    environment = root / "environment"
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    _run([sys.executable, "-m", "venv", str(environment)], cwd=root, env=env)
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    command = scripts / ("formamech.exe" if os.name == "nt" else "formamech")
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(wheel),
        ],
        cwd=root,
        env=env,
    )
    empty = root / "empty"
    empty.mkdir()
    generator = root / "make_inputs.py"
    generator.write_text(textwrap.dedent(MAKE_INPUTS))
    inputs = {}
    for family in ("dogbone-v1", "notched-v1"):
        source = root / "inputs" / family
        _run(
            [str(python), "-I", str(generator), str(source), family], cwd=empty, env=env
        )
        inputs[family] = source
    return wheel, root, python, command, empty, env, inputs


def test_built_wheel_contains_public_compiler_and_adapter_assets(installed):
    wheel, *_ = installed
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for required in (
            "formamech/__init__.py",
            "formamech/__main__.py",
            "formamech/_compat.py",
            "formamech/cli.py",
            "piginot/compiler/config.py",
            "piginot/compiler/planning.py",
            "piginot_campaigns/adapters.py",
            "piginot_campaigns/resources/dogbone-v1.json",
            "piginot_campaigns/resources/geometry-templates.json",
            "piginot_campaigns/resources/notched-v1.json",
        ):
            assert required in names
        assert not any(name.startswith(("campaigns/", "artifacts/")) for name in names)
        entry_file = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entries = archive.read(entry_file).decode()
        assert "formamech = formamech.cli:main" in entries
        assert "piginot = piginot.cli:main" in entries
        assert "[piginot.compiler.adapters]" in entries
        assert "dogbone-v1 = piginot_campaigns.adapters:DOGBONE" in entries
        assert "notched-v1 = piginot_campaigns.adapters:NOTCHED" in entries


def test_built_wheel_attests_its_observed_installed_code_and_resources(installed):
    wheel, *_ = installed
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "piginot/_build_identity.json" in names
        identity = json.loads(archive.read("piginot/_build_identity.json"))
        assert identity["schema"] == "piginot.build-identity-v1"
        assert re.fullmatch("[0-9a-f]{40}", identity["source_revision"])
        assert type(identity["dirty"]) is bool
        metadata_file = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_file).decode())
        assert metadata["Name"] == "formamech"
        assert identity["package_version"] == metadata["Version"]
        assert identity["manifest_algorithm"] == "sha256-rfc8785-file-map-v1"
        assert set(identity["build_inputs_sha256"]) == {
            "pyproject.toml",
            "hatch_build.py",
        }
        assert all(
            re.fullmatch("[0-9a-f]{64}", value)
            for value in identity["build_inputs_sha256"].values()
        )
        expected = {
            name
            for name in names
            if (name.startswith(("piginot/", "formamech/")) and name.endswith(".py"))
            or (
                name.startswith("piginot_campaigns/")
                and name.endswith((".py", ".json"))
            )
        }
        assert set(identity["files"]) == expected
        for name, digest in identity["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        canonical = json.dumps(
            identity["files"], sort_keys=True, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == identity["manifest_sha256"]


def test_installed_identity_verifies_without_git_and_detects_file_changes(installed):
    _, _, python, _, empty, env, _ = installed
    program = textwrap.dedent("""
        import json
        import shutil
        from pathlib import Path
        import formamech
        import piginot
        from formamech.compiler.planning import _implementation
        from formamech.compiler.provenance import (
            BuildIdentityError, load_implementation_identity,
            require_clean_implementation_identity,
        )
        assert shutil.which("git") is None
        package = Path(piginot.__file__).resolve().parent
        recorded = json.loads((package / "_build_identity.json").read_bytes())
        identity = load_implementation_identity()
        assert identity.origin == "packaged"
        assert _implementation() == (identity.package_version, identity.package_sha256)
        for key in ("source_revision", "package_version", "dirty", "manifest_sha256"):
            assert getattr(identity, key) == recorded[key]
        if identity.dirty:
            try:
                require_clean_implementation_identity()
            except BuildIdentityError:
                pass
            else:
                raise AssertionError("dirty wheel authorized numerical execution")
        else:
            assert require_clean_implementation_identity() == identity
        for changed in (
            package / "compiler/config.py",
            Path(formamech.__file__).resolve().parent / "cli.py",
        ):
            original = changed.read_bytes()
            changed.write_bytes(original + b"\\n# installed provenance mutation\\n")
            try:
                try:
                    load_implementation_identity()
                except BuildIdentityError:
                    pass
                else:
                    raise AssertionError("modified installed source accepted")
            finally:
                changed.write_bytes(original)
        assert load_implementation_identity() == identity
    """)
    no_git = {**env, "PATH": str(python.parent)}
    _run([str(python), "-I", "-c", program], cwd=empty, env=no_git)


def test_installed_console_keeps_the_legacy_alias(installed):
    _, _, python, command, empty, env, _ = installed
    legacy = command.with_name("piginot.exe" if os.name == "nt" else "piginot")
    primary = _run([str(command), "--version"], cwd=empty, env=env)
    previous = _run([str(legacy), "--version"], cwd=empty, env=env)
    assert primary.stdout.startswith("formamech ")
    assert previous.stdout == primary.stdout.replace("formamech ", "piginot ", 1)
    _run(
        [
            str(python),
            "-I",
            "-c",
            "from formamech.contracts import Quantity; "
            "from piginot.contracts import Quantity as Legacy; assert Quantity is Legacy",
        ],
        cwd=empty,
        env=env,
    )


@pytest.mark.parametrize("family", ["dogbone-v1", "notched-v1"])
def test_installed_console_creates_bound_plan_from_empty_directory(installed, family):
    _, root, _, command, empty, env, inputs = installed
    source = inputs[family]
    output = root / "outputs" / family
    result = _run(
        [
            str(command),
            "compile",
            "--spec",
            str(source / "problem.json"),
            "--config",
            str(source / "campaign.json"),
            "--output",
            str(output),
            "--plan-only",
        ],
        cwd=empty,
        env=env,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLANNED" and plan["claim_level"] == "NOT_VERIFIED"
    assert plan["stages"]["material"] == "PENDING"
    assert plan["stages"]["train"] == "PENDING"
    assert plan["stages"]["public-development"] == "NOT_REQUESTED"
    assert len(plan["source"]["fit_case_ids"]) == 2
    assert list(empty.iterdir()) == []
    assert not (root / ".git").exists()
    assert (output / "plan.json").is_file()


def test_installed_plan_has_no_numerical_stack(installed):
    _, root, python, _, empty, env, inputs = installed
    source = inputs["dogbone-v1"]
    output = root / "outputs" / "import-audit"
    program = textwrap.dedent("""
        import importlib.util
        import sys
        from pathlib import Path
        import formamech
        import piginot
        import piginot_campaigns
        from formamech import cli
        assert "site-packages" in str(Path(formamech.__file__).resolve())
        assert "site-packages" in str(Path(piginot.__file__).resolve())
        assert "site-packages" in str(Path(piginot_campaigns.__file__).resolve())
        assert cli.main(sys.argv[1:]) == 0
        forbidden = ("torch", "numpy", "scipy", "skfem", "sympy", "piginot.fem", "piginot.problem", "campaigns")
        assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden)
        assert importlib.util.find_spec("torch") is None
        assert importlib.util.find_spec("numpy") is None
        assert importlib.util.find_spec("skfem") is None
    """)
    _run(
        [
            str(python),
            "-I",
            "-c",
            program,
            "compile",
            "--spec",
            str(source / "problem.json"),
            "--config",
            str(source / "campaign.json"),
            "--output",
            str(output),
            "--plan-only",
        ],
        cwd=empty,
        env=env,
    )


def test_installed_executor_refuses_missing_optional_dependencies(installed):
    _, root, python, command, empty, env, inputs = installed
    source = inputs["dogbone-v1"]
    _run(
        [
            str(python),
            "-I",
            "-c",
            "from formamech.compiler.adapters import resolve_adapter; "
            "assert callable(resolve_adapter('dogbone-v1').execute)",
        ],
        cwd=empty,
        env=env,
    )
    execution_output = root / "outputs" / "unavailable-execution"
    result = subprocess.run(
        [
            str(command),
            "compile",
            "--spec",
            str(source / "problem.json"),
            "--config",
            str(source / "campaign.json"),
            "--output",
            str(execution_output),
        ],
        cwd=empty,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert re.search(r"optional|dependenc|formamech\[", result.stderr, re.IGNORECASE)
    assert "ModuleNotFoundError" not in result.stderr
    assert "Traceback" not in result.stderr
    assert not execution_output.exists()
