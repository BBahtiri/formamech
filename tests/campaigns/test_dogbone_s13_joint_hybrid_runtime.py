from __future__ import annotations

import hashlib
import importlib.metadata
import stat
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from campaigns.dogbone_v1 import s13_joint_hybrid_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {
    "numpy": "2.2.6",
    "scipy": "1.17.1",
    "scikit-fem": "12.0.2",
    "rfc8785": "0.1.4",
    "safetensors": "0.8.0",
}


def test_tree_snapshot_is_frozen_sorted_and_content_bound(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.bin").write_bytes(b"abc")
    (tmp_path / "nested" / "a.bin").write_bytes(b"defg")
    before = runtime.snapshot_runtime_tree(tmp_path)
    assert before.file_count == 2
    assert before.total_bytes == 7
    assert before == runtime.snapshot_runtime_tree(tmp_path)
    with pytest.raises(FrozenInstanceError):
        before.total_bytes = 0  # type: ignore[misc]
    (tmp_path / "z.bin").write_bytes(b"abd")
    assert runtime.snapshot_runtime_tree(tmp_path).tree_sha256 != before.tree_sha256


def test_tree_hash_format_binds_relative_names_and_streamed_bytes(
    tmp_path: Path,
) -> None:
    content = b"a" * (1024 * 1024 + 17)
    leaf = tmp_path / "test.bin"
    leaf.write_bytes(content)
    framed = b"S13-RUNTIME-TREE-v1\0F"
    framed += len(b"test.bin").to_bytes(8, "big") + b"test.bin"
    framed += len(content).to_bytes(8, "big") + hashlib.sha256(content).digest()
    before = runtime.snapshot_runtime_tree(tmp_path)
    assert before.tree_sha256 == hashlib.sha256(framed).hexdigest()
    leaf.rename(tmp_path / "renamed.bin")
    assert runtime.snapshot_runtime_tree(tmp_path).tree_sha256 != before.tree_sha256


def test_empty_tree_has_stable_versioned_hash(tmp_path: Path) -> None:
    snapshot = runtime.snapshot_runtime_tree(tmp_path)
    assert snapshot.file_count == 0
    assert snapshot.total_bytes == 0
    assert snapshot.tree_sha256 == hashlib.sha256(b"S13-RUNTIME-TREE-v1\0").hexdigest()


def test_tree_requires_absolute_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        runtime.snapshot_runtime_tree(Path("relative"))
    with pytest.raises(FileNotFoundError):
        runtime.snapshot_runtime_tree(tmp_path / "missing")
    leaf = tmp_path / "leaf"
    leaf.write_bytes(b"a")
    with pytest.raises(ValueError, match="directory"):
        runtime.snapshot_runtime_tree(leaf)


@pytest.mark.parametrize("directory_link", (False, True))
def test_tree_rejects_symlink_without_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_link: bool
) -> None:
    target = tmp_path / "target"
    if directory_link:
        target.mkdir()
    else:
        target.write_bytes(b"a")
    link = tmp_path / "link"
    original_lstat = Path.lstat

    def linked_stat(path: Path) -> Any:
        if path == link:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return original_lstat(path)

    if directory_link:
        link.mkdir()
    else:
        link.write_bytes(b"placeholder")
    monkeypatch.setattr(Path, "lstat", linked_stat)
    with pytest.raises(ValueError, match="link|reparse"):
        runtime.snapshot_runtime_tree(tmp_path)


def test_tree_rejects_windows_reparse_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_lstat = Path.lstat

    def reparse_stat(path: Path) -> Any:
        if path == tmp_path:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", reparse_stat)
    with pytest.raises(ValueError, match="reparse"):
        runtime.snapshot_runtime_tree(tmp_path)


def test_tree_rejects_nonregular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf = tmp_path / "fifo"
    leaf.write_bytes(b"placeholder")
    original_lstat = Path.lstat

    def fifo_stat(path: Path) -> Any:
        if path == leaf:
            return SimpleNamespace(st_mode=stat.S_IFIFO, st_file_attributes=0)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fifo_stat)
    with pytest.raises(ValueError, match="regular"):
        runtime.snapshot_runtime_tree(tmp_path)


def _mock_scientific_runtime(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.11.9")
    monkeypatch.setattr(runtime.metadata, "version", VERSIONS.__getitem__)
    modules = {}
    for name in ("torch", "numpy", "scipy", "skfem", "rfc8785", "safetensors"):
        directory = root / name
        directory.mkdir()
        source = directory / "__init__.py"
        source.write_bytes(b"# synthetic package\n")
        modules[name] = SimpleNamespace(
            __version__="2.13.0+cpu",
            version=SimpleNamespace(cuda=None),
            __file__=str(source),
            __path__=(str(directory),),
            __spec__=SimpleNamespace(
                origin=str(source), submodule_search_locations=None
            ),
        )
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(sys, "prefix", str(root))
    distribution = SimpleNamespace(locate_file=lambda name: root / name)
    monkeypatch.setattr(runtime.metadata, "distribution", lambda name: distribution)


def test_scientific_runtime_is_frozen_and_matches_exact_pins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_scientific_runtime(monkeypatch, tmp_path)
    snapshot = runtime.scientific_runtime()
    assert snapshot.python_version == "3.11.9"
    assert snapshot.torch_version == "2.13.0+cpu"
    assert snapshot.cuda_version is None
    assert snapshot.numpy_version == "2.2.6"
    assert snapshot.scipy_version == "1.17.1"
    assert snapshot.skfem_version == "12.0.2"
    assert snapshot.rfc8785_version == "0.1.4"
    assert snapshot.safetensors_version == "0.8.0"
    with pytest.raises(FrozenInstanceError):
        snapshot.python_version = "other"  # type: ignore[misc]


@pytest.mark.parametrize("drift", ("python", "torch", "cuda", "numpy"))
def test_scientific_runtime_rejects_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drift: str
) -> None:
    _mock_scientific_runtime(monkeypatch, tmp_path)
    if drift == "python":
        monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.11.10")
    elif drift == "numpy":
        monkeypatch.setattr(runtime.metadata, "version", lambda name: "other")
    else:
        imported = runtime.importlib.import_module
        original = imported("torch")
        torch = SimpleNamespace(
            **(
                vars(original)
                | {
                    "__version__": "other" if drift == "torch" else "2.13.0+cpu",
                    "version": SimpleNamespace(
                        cuda="12.8" if drift == "cuda" else None
                    ),
                }
            )
        )
        monkeypatch.setattr(
            runtime.importlib,
            "import_module",
            lambda name: torch if name == "torch" else imported(name),
        )
    with pytest.raises(ValueError, match="scientific runtime"):
        runtime.scientific_runtime()


def test_scientific_runtime_rejects_shadow_import_with_matching_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolated = tmp_path / "runtime"
    isolated.mkdir()
    _mock_scientific_runtime(monkeypatch, isolated)
    imported = runtime.importlib.import_module
    foreign = tmp_path / "numpy.py"
    foreign.write_bytes(b"# synthetic shadow\n")
    shadow = SimpleNamespace(**(vars(imported("numpy")) | {"__file__": str(foreign)}))
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: shadow if name == "numpy" else imported(name),
    )
    with pytest.raises(ValueError, match="scientific import"):
        runtime.scientific_runtime()


def test_project_origins_accept_this_checkout() -> None:
    runtime.assert_project_origins(ROOT)


@pytest.mark.parametrize("module_name", ("piginot", "campaigns", "piginot.mixed"))
def test_project_origins_reject_mixed_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    (tmp_path / "foreign.py").write_bytes(b"# synthetic source\n")
    foreign = ModuleType(module_name)
    foreign.__file__ = str(tmp_path / "foreign.py")
    monkeypatch.setitem(sys.modules, module_name, foreign)
    with pytest.raises(ValueError, match="different checkout"):
        runtime.assert_project_origins(ROOT)


def test_project_origins_reject_missing_source_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "piginot.unknown", ModuleType("piginot.unknown"))
    with pytest.raises(ValueError, match="source file"):
        runtime.assert_project_origins(ROOT)


def test_project_origins_reject_conflicting_import_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SimpleNamespace(origin=str(tmp_path / "foreign.py"))
    monkeypatch.setattr(sys.modules["piginot"], "__spec__", spec)
    with pytest.raises(ValueError, match="import spec"):
        runtime.assert_project_origins(ROOT)


def test_project_origins_reject_foreign_package_search_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys.modules["piginot"], "__path__", (str(tmp_path),))
    with pytest.raises(ValueError, match="package search path"):
        runtime.assert_project_origins(ROOT)


def _mock_distributions(
    monkeypatch: pytest.MonkeyPatch, root: Path, files: tuple[str, ...]
) -> None:
    distribution = SimpleNamespace(
        files=tuple(importlib.metadata.PackagePath(name) for name in files),
        locate_file=lambda name: root / name,
    )
    monkeypatch.setattr(runtime.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(sys, "prefix", str(root))


def test_scientific_package_hashes_include_binaries_and_ignore_missing_pyc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "core.pyd").write_bytes(b"binary")
    (tmp_path / "library.dll").write_bytes(b"library")
    _mock_distributions(monkeypatch, tmp_path, ("core.pyd", "library.dll", "cache.pyc"))
    snapshots = runtime.scientific_package_hashes()
    assert tuple(name for name, _ in snapshots) == (
        "torch",
        "numpy",
        "scipy",
        "scikit-fem",
        "rfc8785",
        "safetensors",
    )
    assert all(snapshot.file_count == 2 for _, snapshot in snapshots)
    before = snapshots[0][1]
    (tmp_path / "core.pyd").write_bytes(b"changed")
    assert runtime.scientific_package_hashes()[0][1].tree_sha256 != before.tree_sha256


def test_scientific_package_hashes_fail_on_missing_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_distributions(monkeypatch, tmp_path, ("missing.pyd",))
    with pytest.raises(FileNotFoundError):
        runtime.scientific_package_hashes()


def test_scientific_package_hashes_fail_on_missing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_distributions(monkeypatch, tmp_path, ())
    with pytest.raises(ValueError, match="manifest"):
        runtime.scientific_package_hashes()


def test_scientific_package_hashes_reject_outside_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated = tmp_path / "runtime"
    isolated.mkdir()
    (tmp_path / "outside.dll").write_bytes(b"binary")
    _mock_distributions(monkeypatch, isolated, ("../outside.dll",))
    with pytest.raises(ValueError, match="outside runtime"):
        runtime.scientific_package_hashes()


def test_scientific_package_hashes_reject_duplicate_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "core.pyd").write_bytes(b"binary")
    _mock_distributions(monkeypatch, tmp_path, ("core.pyd", "core.pyd"))
    with pytest.raises(ValueError, match="duplicate"):
        runtime.scientific_package_hashes()


def test_tree_rejects_file_changed_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf = tmp_path / "core.pyd"
    leaf.write_bytes(b"binary")
    original_fstat = runtime.os.fstat
    calls = 0

    def changed_fstat(handle: int) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            leaf.write_bytes(b"binary changed")
        return original_fstat(handle)

    monkeypatch.setattr(runtime.os, "fstat", changed_fstat)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        runtime.snapshot_runtime_tree(tmp_path)


def test_scientific_package_hashes_do_not_ignore_broken_pyc_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_distributions(monkeypatch, tmp_path, ("broken.pyc",))
    original_lstat = Path.lstat

    def linked_stat(path: Path) -> Any:
        if path == tmp_path / "broken.pyc":
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", linked_stat)
    with pytest.raises(ValueError, match="link|reparse"):
        runtime.scientific_package_hashes()
