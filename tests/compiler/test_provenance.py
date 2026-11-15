"""Implementation identity checks without training or numerical dependencies."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from piginot.compiler import provenance
from piginot.contracts import canonical_json_bytes
from tests.platform_support import requires_symlinks

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.2.0.dev0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _project(root: Path, *, git: bool = True) -> Path:
    core = root / "src/piginot"
    core.mkdir(parents=True)
    (core / "__init__.py").write_text(f'__version__ = "{VERSION}"\n')
    (core / "operation.py").write_text("VALUE = 1\n")
    public = root / "piginot_campaigns"
    (public / "resources").mkdir(parents=True)
    (public / "__init__.py").write_text('"""Public adapters."""\n')
    (public / "resources/fixture.json").write_text('{"public":true}\n')
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "piginot"\nversion = "{VERSION}"\n'
    )
    (root / "hatch_build.py").write_text('"""Build input fixture."""\n')
    if git:
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "test@example.invalid")
        _git(root, "config", "user.name", "Fixture")
        _git(root, "add", ".")
        _git(root, "commit", "-qm", "fixture")
    return core


def _files(core: Path, public: Path) -> dict[str, str]:
    return {
        **{
            f"piginot/{path.relative_to(core).as_posix()}": _sha(path)
            for path in core.rglob("*.py")
        },
        **{
            f"piginot_campaigns/{path.relative_to(public).as_posix()}": _sha(path)
            for path in public.rglob("*")
            if path.is_file() and path.suffix in (".py", ".json")
        },
    }


def _record(core: Path, public: Path, *, dirty: bool = False) -> dict[str, Any]:
    files = _files(core, public)
    return {
        "schema": "piginot.build-identity-v1",
        "source_revision": "1" * 40,
        "package_version": VERSION,
        "dirty": dirty,
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
        "manifest_algorithm": "sha256-rfc8785-file-map-v1",
        "build_inputs_sha256": {"pyproject.toml": "2" * 64, "hatch_build.py": "3" * 64},
    }


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    core = _project(source)
    destination = tmp_path / "site-packages"
    installed_core = destination / "piginot"
    installed_public = destination / "piginot_campaigns"
    shutil.copytree(core, installed_core)
    shutil.copytree(source / "piginot_campaigns", installed_public)
    record = _record(installed_core, installed_public)
    (installed_core / "_build_identity.json").write_text(json.dumps(record))
    monkeypatch.setattr(provenance, "_package_root", lambda: installed_core)
    return installed_core, installed_public, record


def test_installed_identity_observes_files_and_matches_plan_hash(installed) -> None:
    core, _, record = installed
    identity = provenance.require_clean_implementation_identity()
    expected_core = {
        path.relative_to(core).as_posix(): _sha(path)
        for path in sorted(core.rglob("*.py"))
    }
    assert identity.source_revision == "1" * 40
    assert identity.package_version == VERSION
    assert identity.origin == "packaged"
    assert identity.dirty is False
    assert identity.manifest_sha256 == record["manifest_sha256"]
    assert (
        identity.package_sha256
        == hashlib.sha256(canonical_json_bytes(expected_core)).hexdigest()
    )


def test_packaged_identity_never_uses_ambient_git(
    installed, tmp_path, monkeypatch
) -> None:
    _project(tmp_path / "unrelated")
    monkeypatch.chdir(tmp_path / "unrelated")
    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: pytest.fail("ambient Git probe")
    )
    assert provenance.load_implementation_identity().origin == "packaged"


def test_missing_packaged_identity_does_not_adopt_ambient_git(
    installed, monkeypatch
) -> None:
    core, _, _ = installed
    (core / "_build_identity.json").unlink()
    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: pytest.fail("ambient Git probe")
    )
    with pytest.raises(provenance.BuildIdentityError):
        provenance.load_implementation_identity()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "future"),
        ("source_revision", "bad"),
        ("package_version", "999.0"),
        ("dirty", "false"),
        ("manifest_sha256", "0" * 64),
        ("manifest_algorithm", "unchecked"),
        ("files", {}),
        ("files", {"../outside.py": "0" * 64}),
        ("files", {"piginot/operation.py": 1}),
        ("build_inputs_sha256", {}),
        ("build_inputs_sha256", {"unexpected": "0" * 64}),
        ("unexpected", True),
    ],
)
def test_malformed_or_differently_bound_attestation_refuses(
    installed, key, value
) -> None:
    core, _, record = installed
    (core / "_build_identity.json").write_text(json.dumps({**record, key: value}))
    with pytest.raises(provenance.BuildIdentityError):
        provenance.load_implementation_identity()


@pytest.mark.parametrize(
    "change", ["modify", "extra-core", "extra-resource", "missing"]
)
def test_observed_code_and_resources_must_equal_attested_manifest(
    installed, change
) -> None:
    core, public, _ = installed
    if change == "modify":
        (core / "operation.py").write_text("VALUE = 2\n")
    elif change == "extra-core":
        (core / "extra.py").write_text("VALUE = 2\n")
    elif change == "extra-resource":
        (public / "resources/new.json").write_text("{}")
    elif change == "missing":
        (public / "resources/fixture.json").unlink()
    with pytest.raises(provenance.BuildIdentityError):
        provenance.load_implementation_identity()


@requires_symlinks
def test_linked_implementation_refuses(installed) -> None:
    core, _, _ = installed
    (core / "operation.py").unlink()
    (core / "operation.py").symlink_to(core / "__init__.py")
    with pytest.raises(provenance.BuildIdentityError):
        provenance.load_implementation_identity()


def test_package_location_and_ignored_caches(installed) -> None:
    core, public, record = installed
    for directory in (core, public):
        (directory / "__pycache__").mkdir()
        (directory / "__pycache__/ignored.pyc").write_bytes(b"cache")
    (public / "resources/notes.txt").write_text("not implementation data")
    assert (
        provenance.load_implementation_identity().manifest_sha256
        == record["manifest_sha256"]
    )


def test_missing_package_or_initializer_refuses(installed) -> None:
    _, public, _ = installed
    (public / "__init__.py").unlink()
    with pytest.raises(provenance.BuildIdentityError, match="incomplete"):
        provenance.load_implementation_identity()
    public.rename(public.with_name("absent-package"))
    with pytest.raises(provenance.BuildIdentityError, match="missing"):
        provenance.load_implementation_identity()


@pytest.mark.parametrize("payload", ["{", '{"schema":1,"schema":2}', "[]"])
def test_invalid_attestation_json_refuses(installed, payload) -> None:
    core, _, _ = installed
    (core / "_build_identity.json").write_text(payload)
    with pytest.raises(provenance.BuildIdentityError):
        provenance.load_implementation_identity()


def test_unreadable_package_data_refuses(installed, monkeypatch) -> None:
    def unreadable(path):
        raise PermissionError("fixture unreadable")

    monkeypatch.setattr(provenance, "_sha", unreadable)
    with pytest.raises(provenance.BuildIdentityError, match="unreadable"):
        provenance.load_implementation_identity()


def test_dirty_identity_is_visible_but_refused_for_numerical_execution(
    installed,
) -> None:
    core, _, record = installed
    (core / "_build_identity.json").write_text(json.dumps({**record, "dirty": True}))
    assert provenance.load_implementation_identity().dirty is True
    with pytest.raises(provenance.BuildIdentityError, match="dirty"):
        provenance.require_clean_implementation_identity()


def test_source_checkout_derives_its_actual_root_and_tracks_dirty_state(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "project"
    core = _project(root)
    unrelated = tmp_path / "unrelated"
    _project(unrelated)
    monkeypatch.chdir(unrelated)
    monkeypatch.setattr(provenance, "_package_root", lambda: core)
    identity = provenance.require_clean_implementation_identity()
    assert identity.origin == "source"
    assert identity.source_revision == _git(root, "rev-parse", "HEAD")
    assert identity.dirty is False
    (core / "operation.py").write_text("VALUE = 2\n")
    changed = provenance.load_implementation_identity()
    assert changed.dirty is True
    assert changed.package_sha256 != identity.package_sha256
    with pytest.raises(provenance.BuildIdentityError, match="dirty"):
        provenance.require_clean_implementation_identity()


def test_unproven_source_shape_does_not_use_git(tmp_path, monkeypatch) -> None:
    core = _project(tmp_path / "project")
    (core.parent.parent / "pyproject.toml").write_text('[project]\nname="other"\n')
    monkeypatch.setattr(provenance, "_package_root", lambda: core)
    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: pytest.fail("unproven Git probe")
    )
    with pytest.raises(provenance.BuildIdentityError):
        provenance.load_implementation_identity()


@pytest.mark.parametrize(
    "contents", ["not toml [", '[project]\nname="piginot"\nversion="other"', "other=1"]
)
def test_invalid_source_project_cannot_supply_git_revision(
    tmp_path, monkeypatch, contents
) -> None:
    core = _project(tmp_path / "project")
    (core.parent.parent / "pyproject.toml").write_text(contents)
    monkeypatch.setattr(provenance, "_package_root", lambda: core)
    with pytest.raises(provenance.BuildIdentityError):
        provenance.load_implementation_identity()


@pytest.mark.parametrize("mode", ["missing-git", "later-error", "failed-revision"])
def test_git_failures_cannot_yield_clean_execution(tmp_path, monkeypatch, mode) -> None:
    root = tmp_path / "project"
    core = _project(root)
    monkeypatch.setattr(provenance, "_package_root", lambda: core)
    original = provenance._git

    def failure(directory, *arguments):
        if mode == "missing-git" or (mode == "later-error" and arguments[-1] == "HEAD"):
            raise FileNotFoundError("Git unavailable")
        if mode == "failed-revision" and arguments[-1] == "HEAD":
            return subprocess.CompletedProcess(arguments, 1, "", "failed")
        return original(directory, *arguments)

    monkeypatch.setattr(provenance, "_git", failure)
    with pytest.raises(provenance.BuildIdentityError):
        provenance.require_clean_implementation_identity()


def test_ignored_new_source_is_still_dirty(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    core = _project(root)
    (root / ".gitignore").write_text("new.py\n")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-qm", "ignore fixture")
    (core / "new.py").write_text("new code\n")
    assert not _git(root, "status", "--porcelain")
    monkeypatch.setattr(provenance, "_package_root", lambda: core)
    assert provenance.load_implementation_identity().dirty is True


def test_gitless_source_attestation_checks_build_inputs(tmp_path, monkeypatch) -> None:
    root = tmp_path / "source"
    core = _project(root, git=False)
    record = {
        **_record(core, root / "piginot_campaigns"),
        "build_inputs_sha256": {
            name: _sha(root / name) for name in ("pyproject.toml", "hatch_build.py")
        },
    }
    (core / "_build_identity.json").write_text(json.dumps(record))
    monkeypatch.setattr(provenance, "_package_root", lambda: core)
    assert provenance.load_implementation_identity().origin == "packaged"
    (root / "hatch_build.py").write_text("# changed\n")
    with pytest.raises(provenance.BuildIdentityError, match="build inputs"):
        provenance.load_implementation_identity()


@pytest.fixture
def hook_module(monkeypatch):
    """Use the real hook with a minimal interface stand-in when Hatch is absent."""

    if importlib.util.find_spec("hatchling") is None:
        for name in (
            "hatchling",
            "hatchling.builders",
            "hatchling.builders.hooks",
            "hatchling.builders.hooks.plugin",
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
        interface = types.ModuleType("hatchling.builders.hooks.plugin.interface")

        class Interface:
            @classmethod
            def __class_getitem__(cls, item):
                return cls

            def __init__(
                self, root, config, build_config, metadata, directory, target_name, app
            ):
                self.root, self.metadata, self.target_name = root, metadata, target_name

        interface.BuildHookInterface = Interface
        monkeypatch.setitem(sys.modules, interface.__name__, interface)
    spec = importlib.util.spec_from_file_location(
        "hatch_build", ROOT / "hatch_build.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hatch_build", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_hook_force_includes_metadata_without_modifying_worktree(
    tmp_path, hook_module, target
) -> None:
    root = tmp_path / "project"
    core = _project(root)
    before = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    record = hook_module.build_identity(root, VERSION)
    assert record["dirty"] is False
    assert record["source_revision"] == _git(root, "rev-parse", "HEAD")
    assert record["files"] == _files(core, root / "piginot_campaigns")
    assert (
        record["manifest_sha256"]
        == hashlib.sha256(canonical_json_bytes(record["files"])).hexdigest()
    )
    assert record["build_inputs_sha256"] == {
        name: _sha(root / name) for name in ("hatch_build.py", "pyproject.toml")
    }
    hook = hook_module.CustomBuildHook(
        str(root),
        {},
        None,
        types.SimpleNamespace(version=VERSION),
        str(tmp_path),
        target,
        None,
    )
    data = {"force_include": {"existing": "existing"}, "artifacts": []}
    hook.initialize("standard", data)
    expected = (
        "piginot/_build_identity.json"
        if target == "wheel"
        else "src/piginot/_build_identity.json"
    )
    generated = next(
        Path(source)
        for source, destination in data["force_include"].items()
        if destination == expected
    )
    assert generated.is_file() and not generated.is_relative_to(root)
    assert json.loads(generated.read_bytes()) == record
    after = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    assert before == after
    assert _git(root, "status", "--porcelain") == ""
    hook.finalize("standard", data, "artifact")
    assert not generated.exists()


def test_gitless_sdist_rebuild_preserves_verified_revision(
    tmp_path, hook_module
) -> None:
    original = tmp_path / "original"
    core = _project(original)
    record = hook_module.build_identity(original, VERSION)
    extracted = tmp_path / "extracted"
    shutil.copytree(original, extracted, ignore=shutil.ignore_patterns(".git"))
    (extracted / "src/piginot/_build_identity.json").write_text(json.dumps(record))
    assert hook_module.build_identity(extracted, VERSION) == record
    (extracted / "src/piginot/operation.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError):
        hook_module.build_identity(extracted, VERSION)
    assert core.exists()


def test_sdist_rebuild_refuses_changed_build_instructions(
    tmp_path, hook_module
) -> None:
    root = tmp_path / "source"
    _project(root)
    record = hook_module.build_identity(root, VERSION)
    extracted = tmp_path / "extracted"
    shutil.copytree(root, extracted, ignore=shutil.ignore_patterns(".git"))
    (extracted / "src/piginot/_build_identity.json").write_text(json.dumps(record))
    (extracted / "hatch_build.py").write_text("# modified hook\n")
    with pytest.raises(ValueError):
        hook_module.build_identity(extracted, VERSION)


def test_unattested_gitless_source_cannot_invent_revision(
    tmp_path, hook_module
) -> None:
    _project(tmp_path / "source", git=False)
    with pytest.raises(ValueError):
        hook_module.build_identity(tmp_path / "source", VERSION)


@pytest.mark.parametrize("missing", ["package", "initializer"])
def test_hook_refuses_incomplete_source_packages(
    tmp_path, hook_module, missing
) -> None:
    root = tmp_path / "source"
    _project(root)
    public = root / "piginot_campaigns"
    if missing == "package":
        public.rename(root / "absent-package")
    else:
        (public / "__init__.py").unlink()
    with pytest.raises(ValueError, match="source package"):
        hook_module.build_identity(root, VERSION)


@requires_symlinks
def test_hook_refuses_linked_source_files(tmp_path, hook_module) -> None:
    root = tmp_path / "source"
    core = _project(root)
    (core / "operation.py").unlink()
    (core / "operation.py").symlink_to(core / "__init__.py")
    with pytest.raises(ValueError, match="linked"):
        hook_module.build_identity(root, VERSION)


@pytest.mark.parametrize("failure", ["unavailable", "revision"])
def test_hook_cannot_invent_identity_after_git_failure(
    tmp_path, hook_module, monkeypatch, failure
) -> None:
    root = tmp_path / "source"
    _project(root)
    original = hook_module._git

    def failed(directory, *arguments):
        if failure == "unavailable":
            raise FileNotFoundError("Git unavailable")
        if arguments[-1] == "HEAD":
            return subprocess.CompletedProcess(arguments, 1, "", "failed")
        return original(directory, *arguments)

    monkeypatch.setattr(hook_module, "_git", failed)
    with pytest.raises(ValueError):
        hook_module.build_identity(root, VERSION)


def test_sdist_duplicate_metadata_keys_refuse(tmp_path, hook_module) -> None:
    root = tmp_path / "source"
    core = _project(root, git=False)
    (core / "_build_identity.json").write_text('{"dirty":false,"dirty":true}')
    with pytest.raises(ValueError, match="duplicate"):
        hook_module.build_identity(root, VERSION)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "future"),
        ("source_revision", 1),
        ("dirty", "false"),
        ("manifest_algorithm", "unknown"),
        ("package_version", "other"),
        ("unexpected", 1),
        ("manifest_sha256", "0" * 64),
    ],
)
def test_sdist_invalid_metadata_refuses(tmp_path, hook_module, key, value) -> None:
    original = tmp_path / "original"
    _project(original)
    record = hook_module.build_identity(original, VERSION)
    extracted = tmp_path / "extracted"
    shutil.copytree(original, extracted, ignore=shutil.ignore_patterns(".git"))
    (extracted / "src/piginot/_build_identity.json").write_text(
        json.dumps({**record, key: value})
    )
    with pytest.raises(ValueError):
        hook_module.build_identity(extracted, VERSION)


def test_hook_manifest_matches_rfc8785_unicode_order_and_ignores_caches(
    tmp_path, hook_module
) -> None:
    root = tmp_path / "source"
    core = _project(root)
    public = root / "piginot_campaigns"
    for name in ("😀", "\ufb33"):
        (public / "resources" / f"{name}.json").write_text("{}")
    for directory in (core, public):
        (directory / "__pycache__").mkdir()
        (directory / "__pycache__/ignored.pyc").write_bytes(b"cache")
    record = hook_module.build_identity(root, VERSION)
    assert (
        record["manifest_sha256"]
        == hashlib.sha256(canonical_json_bytes(record["files"])).hexdigest()
    )


def test_hook_refuses_wrong_target_temp_location_and_project_version(
    tmp_path, hook_module, monkeypatch
) -> None:
    root = tmp_path / "source"
    _project(root)
    hook = hook_module.CustomBuildHook(
        str(root),
        {},
        None,
        types.SimpleNamespace(version=VERSION),
        str(tmp_path),
        "unsupported",
        None,
    )
    with pytest.raises(ValueError, match="target"):
        hook.initialize("standard", {})
    hook = hook_module.CustomBuildHook(
        str(root),
        {},
        None,
        types.SimpleNamespace(version=VERSION),
        str(tmp_path),
        "wheel",
        None,
    )
    monkeypatch.setattr(hook_module.tempfile, "gettempdir", lambda: str(root))
    with pytest.raises(ValueError, match="outside"):
        hook.initialize("standard", {})
    with pytest.raises(ValueError, match="version"):
        hook_module.build_identity(root, "other")


def test_actual_package_root_is_derived_from_module_file() -> None:
    assert provenance._package_root() == Path(provenance.__file__).resolve().parents[1]
