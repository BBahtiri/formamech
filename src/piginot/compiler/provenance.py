"""Observed implementation identity, independent of the caller's directory.

This verifies package integrity and build attribution, not publisher authenticity.
Numerical execution requires a clean identity; planning may report it as pending.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import piginot
from piginot.contracts import canonical_json_bytes, strict_json_loads

SCHEMA = "piginot.build-identity-v1"
MANIFEST_ALGORITHM = "sha256-rfc8785-file-map-v1"
IDENTITY_FILENAME = "_build_identity.json"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_BUILD_INPUTS = frozenset({"pyproject.toml", "hatch_build.py"})
_FIELDS = frozenset(
    {
        "schema",
        "source_revision",
        "package_version",
        "dirty",
        "files",
        "manifest_sha256",
        "manifest_algorithm",
        "build_inputs_sha256",
    }
)


class BuildIdentityError(ValueError):
    """The implementation lacks a verifiable or numerically eligible identity."""


@dataclass(frozen=True, slots=True)
class ImplementationIdentity:
    source_revision: str
    package_sha256: str
    manifest_sha256: str
    package_version: str
    dirty: bool
    origin: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root(package: Path) -> Path | None:
    if package.parent.name != "src":
        return None
    root = package.parent.parent
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(project, dict) or project.get("name") not in {
        "piginot",
        "formamech",
    }:
        return None
    if project.get("version") != piginot.__version__:
        raise BuildIdentityError("source project and package versions differ")
    return root if package == root / "src/piginot" else None


def _collect(directory: Path, prefix: str, suffixes: tuple[str, ...]) -> dict[str, str]:
    if not directory.is_dir() or directory.is_symlink():
        raise BuildIdentityError(f"missing or linked package directory: {directory}")
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if "__pycache__" in path.relative_to(directory).parts:
            continue
        if path.is_symlink():
            raise BuildIdentityError("implementation resources cannot be symlinks")
        if path.is_file() and path.suffix in suffixes:
            files[f"{prefix}/{path.relative_to(directory).as_posix()}"] = _sha(path)
    return files


def _observed_files(package: Path, public: Path) -> dict[str, str]:
    files = {
        **_collect(package, "piginot", (".py",)),
        **_collect(public, "piginot_campaigns", (".py", ".json")),
    }
    facade = package.parent / "formamech"
    if facade.exists() or facade.is_symlink():
        files.update(_collect(facade, "formamech", (".py",)))
    if not {"piginot/__init__.py", "piginot_campaigns/__init__.py"} <= files.keys():
        raise BuildIdentityError("implementation package manifests are incomplete")
    return files


def implementation_package_sha256(package: Path) -> str:
    """Use one package digest for planning and execution, including public aliases."""

    files = {
        path.relative_to(package).as_posix(): _sha(path)
        for path in sorted(package.rglob("*.py"))
    }
    facade = package.parent / "formamech"
    if facade.exists() or facade.is_symlink():
        files.update(_collect(facade, "formamech", (".py",)))
    return hashlib.sha256(canonical_json_bytes(files)).hexdigest()


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _source_state(root: Path, files: dict[str, str]) -> tuple[str, bool] | None:
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if top.returncode or Path(top.stdout.strip()).resolve() != root:
        return None
    try:
        revision = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain", "--untracked-files=all")
        tracked = _git(root, "ls-files", "-z")
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BuildIdentityError("cannot establish source Git state") from error
    value = revision.stdout.strip()
    if (
        revision.returncode
        or status.returncode
        or tracked.returncode
        or not _REVISION.fullmatch(value)
    ):
        raise BuildIdentityError("cannot establish source revision and cleanliness")
    required = {
        f"src/{name}" if name.startswith(("piginot/", "formamech/")) else name
        for name in files
    } | _BUILD_INPUTS
    dirty = bool(status.stdout) or not required <= set(tracked.stdout.split("\0"))
    return value, dirty


def _digest_map(value: object, label: str) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or not value
        or any(
            not isinstance(key, str)
            or not isinstance(digest, str)
            or _HEX.fullmatch(digest) is None
            for key, digest in value.items()
        )
    ):
        raise BuildIdentityError(f"invalid {label} digest map")
    return value


def _attested(
    package: Path, files: dict[str, str], root: Path | None
) -> dict[str, Any]:
    path = package / IDENTITY_FILENAME
    if not path.is_file() or path.is_symlink():
        raise BuildIdentityError("no packaged implementation attestation")
    try:
        record = strict_json_loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise BuildIdentityError("invalid implementation attestation JSON") from error
    if not isinstance(record, dict) or set(record) != _FIELDS:
        raise BuildIdentityError("invalid implementation attestation fields")
    if record["schema"] != SCHEMA or record["manifest_algorithm"] != MANIFEST_ALGORITHM:
        raise BuildIdentityError(
            "unsupported implementation attestation schema/algorithm"
        )
    if not isinstance(record["source_revision"], str) or not _REVISION.fullmatch(
        record["source_revision"]
    ):
        raise BuildIdentityError("invalid attested source revision")
    if (
        record["package_version"] != piginot.__version__
        or type(record["dirty"]) is not bool
    ):
        raise BuildIdentityError("attested version or dirty flag differs")
    attested_files = _digest_map(record["files"], "package")
    inputs = _digest_map(record["build_inputs_sha256"], "build input")
    if set(inputs) != _BUILD_INPUTS:
        raise BuildIdentityError("incomplete attested build inputs")
    observed_digest = hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    if attested_files != files or record["manifest_sha256"] != observed_digest:
        raise BuildIdentityError(
            "observed implementation files differ from attestation"
        )
    if root is not None and any(
        not (root / name).is_file() or _sha(root / name) != digest
        for name, digest in inputs.items()
    ):
        raise BuildIdentityError("source-distribution build inputs differ")
    return record


def load_implementation_identity() -> ImplementationIdentity:
    """Verify installed bytes, or attribute a proven source checkout to its Git root."""

    package = _package_root().resolve()
    root = _project_root(package)
    public = (root if root is not None else package.parent) / "piginot_campaigns"
    try:
        files = _observed_files(package, public)
        state = _source_state(root, files) if root is not None else None
        if state is None:
            record = _attested(package, files, root)
            revision, dirty, origin = (
                record["source_revision"],
                record["dirty"],
                "packaged",
            )
        else:
            revision, dirty = state
            origin = "source"
        return ImplementationIdentity(
            source_revision=revision,
            package_sha256=implementation_package_sha256(package),
            manifest_sha256=hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
            package_version=piginot.__version__,
            dirty=dirty,
            origin=origin,
        )
    except OSError as error:
        raise BuildIdentityError("implementation files are unreadable") from error


def require_clean_implementation_identity() -> ImplementationIdentity:
    """Refuse missing, changed or dirty provenance before numerical execution."""

    identity = load_implementation_identity()
    if identity.dirty:
        raise BuildIdentityError("numerical execution refuses a dirty implementation")
    return identity


__all__ = (
    "BuildIdentityError",
    "ImplementationIdentity",
    "load_implementation_identity",
    "require_clean_implementation_identity",
)
