"""Read-only runtime and imported-source proofs for the approved S13 contrast."""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import stat
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import ClassVar, Protocol, cast

_CHUNK_BYTES = 1024 * 1024
_TREE_MAGIC = b"S13-RUNTIME-TREE-v1\0"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SCIENTIFIC_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "scipy",
    "scikit-fem",
    "rfc8785",
    "safetensors",
)


@dataclass(frozen=True, slots=True)
class RuntimeTreeSnapshot:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    file_count: int
    total_bytes: int
    tree_sha256: str


@dataclass(frozen=True, slots=True)
class ScientificRuntime:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    python_version: str
    torch_version: str
    cuda_version: str | None
    numpy_version: str
    scipy_version: str
    skfem_version: str
    rfc8785_version: str
    safetensors_version: str


_EXPECTED_SCIENTIFIC_RUNTIME = ScientificRuntime(
    "3.11.9", "2.13.0+cpu", None, "2.2.6", "1.17.1", "12.0.2", "0.1.4", "0.8.0"
)


def _unlinked_stat(path: Path) -> os.stat_result:
    details = path.lstat()
    attributes = getattr(details, "st_file_attributes", 0)
    if stat.S_ISLNK(details.st_mode) or attributes & _REPARSE_POINT:
        raise ValueError("runtime paths must not be links or reparse points")
    return details


def _require_unlinked_path(path: Path) -> None:
    for parent in (*reversed(path.parents), path):
        _unlinked_stat(parent)


def _absolute_directory(root: str | Path) -> Path:
    path = Path(root)
    if not path.is_absolute():
        raise ValueError("runtime root must be absolute")
    _require_unlinked_path(path)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError("runtime root must be a directory")
    return path.resolve(strict=True)


def _file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _file_digest(path: Path) -> tuple[int, bytes]:
    before = _unlinked_stat(path)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("runtime entries must be regular files or directories")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if _file_identity(opened) != _file_identity(before):
            raise RuntimeError("runtime file changed while hashing")
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if _file_identity(before) != _file_identity(after) or _file_identity(
        before
    ) != _file_identity(_unlinked_stat(path)):
        raise RuntimeError("runtime file changed while hashing")
    return before.st_size, digest.digest()


def _raise_walk_error(error: OSError) -> None:
    raise error


def _tree_files(root: Path) -> Iterator[tuple[str, Path]]:
    for directory, names, filenames in os.walk(
        root, followlinks=False, onerror=_raise_walk_error
    ):
        parent = Path(directory)
        _unlinked_stat(parent)
        for name in names:
            _unlinked_stat(parent / name)
        for name in filenames:
            yield (parent / name).relative_to(root).as_posix(), parent / name


def _snapshot_files(records: Iterable[tuple[str, Path]]) -> RuntimeTreeSnapshot:
    ordered = tuple(sorted(records))
    if len({os.path.normcase(name) for name, _ in ordered}) != len(ordered):
        raise ValueError("runtime file manifest contains duplicate paths")
    digest = hashlib.sha256(_TREE_MAGIC)
    total_bytes = 0
    for name, path in ordered:
        size, file_digest = _file_digest(path)
        encoded_path = name.encode("utf-8")
        digest.update(b"F" + len(encoded_path).to_bytes(8, "big") + encoded_path)
        digest.update(size.to_bytes(8, "big") + file_digest)
        total_bytes += size
    return RuntimeTreeSnapshot(len(ordered), total_bytes, digest.hexdigest())


def snapshot_runtime_tree(root: str | Path) -> RuntimeTreeSnapshot:
    """Hash sorted relative file names, sizes, and bytes; never follow links.

    The v1 format is a magic prefix followed by file records: ``F``, uint64
    UTF-8 path length, POSIX relative path, uint64 byte size, raw SHA-256.
    Integers are big-endian. Empty directories and timestamps are not hashed.
    """
    path = _absolute_directory(root)
    return _snapshot_files(_tree_files(path))


class _TorchVersion(Protocol):
    cuda: str | None


class _TorchRuntime(Protocol):
    __version__: str
    version: _TorchVersion


def scientific_runtime() -> ScientificRuntime:
    """Capture and require the approved scientific versions without changing threads."""
    root = _absolute_directory(sys.prefix)
    torch = cast(_TorchRuntime, importlib.import_module("torch"))
    for name in _SCIENTIFIC_DISTRIBUTIONS:
        module_name = "skfem" if name == "scikit-fem" else name
        module = importlib.import_module(module_name)
        expected = Path(
            os.path.abspath(
                str(
                    metadata.distribution(name).locate_file(
                        f"{module_name}/__init__.py"
                    )
                )
            )
        )
        if not expected.is_relative_to(root):
            raise ValueError("scientific import artifact is outside runtime")
        source = _module_origin(module, expected.parent, "scientific import")
        if source != expected:
            raise ValueError("scientific import differs from its installed artifact")
    captured = ScientificRuntime(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        numpy_version=metadata.version("numpy"),
        scipy_version=metadata.version("scipy"),
        skfem_version=metadata.version("scikit-fem"),
        rfc8785_version=metadata.version("rfc8785"),
        safetensors_version=metadata.version("safetensors"),
    )
    if captured != _EXPECTED_SCIENTIFIC_RUNTIME:
        raise ValueError("scientific runtime differs from the approved exact versions")
    return captured


def scientific_package_hashes() -> tuple[tuple[str, RuntimeTreeSnapshot], ...]:
    """Hash all installed scientific artifacts, including binaries and metadata.

    Only missing generated ``.pyc`` records are ignored. All recorded artifacts
    must remain within the current interpreter's runtime root.
    """
    root = _absolute_directory(sys.prefix)
    return tuple(
        (name, _snapshot_files(_distribution_files(name, root)))
        for name in _SCIENTIFIC_DISTRIBUTIONS
    )


def _distribution_files(name: str, root: Path) -> Iterator[tuple[str, Path]]:
    distribution = metadata.distribution(name)
    if not distribution.files:
        raise ValueError("scientific distribution has no installed file manifest")
    for record in distribution.files:
        path = Path(os.path.abspath(str(distribution.locate_file(record))))
        if not path.is_relative_to(root):
            raise ValueError("scientific artifact is outside runtime")
        try:
            _require_unlinked_path(path)
        except FileNotFoundError:
            if path.suffix == ".pyc":
                continue
            raise
        yield path.relative_to(root).as_posix(), path


def _module_origin(module: object, expected: Path, label: str) -> Path:
    source = getattr(module, "__file__", None)
    if source is None:
        raise ValueError(f"{label} module has no source file")
    path = Path(source)
    _require_unlinked_path(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(expected):
        raise ValueError(f"{label} module was imported from a different checkout")
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin is None or Path(origin).resolve() != resolved:
        raise ValueError(f"{label} module has an inconsistent import spec")
    search_paths = (
        *getattr(module, "__path__", ()),
        *(getattr(spec, "submodule_search_locations", None) or ()),
    )
    for location in search_paths:
        search_path = Path(location)
        _require_unlinked_path(search_path)
        if not search_path.resolve(strict=True).is_relative_to(expected):
            raise ValueError(f"{label} has a foreign package search path")
    return resolved


def assert_project_origins(repository_root: str | Path) -> None:
    """Require every already-imported piginot/campaigns module in this checkout.

    This checks loaded source origins, not future imports or editable .pth files.
    The environment preparation step must also correct its editable source path.
    """
    root = _absolute_directory(repository_root)
    importlib.import_module("piginot")
    importlib.import_module("campaigns")
    for name, module in tuple(sys.modules.items()):
        package = name.partition(".")[0]
        if package not in ("piginot", "campaigns"):
            continue
        expected = root / ("src/piginot" if package == "piginot" else "campaigns")
        _module_origin(module, expected, "project")


__all__ = [
    "RuntimeTreeSnapshot",
    "ScientificRuntime",
    "assert_project_origins",
    "scientific_package_hashes",
    "scientific_runtime",
    "snapshot_runtime_tree",
]
