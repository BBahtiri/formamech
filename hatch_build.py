"""Build-time source attribution for installed compiler execution.

The file-map digest is RFC 8785 over a string-to-string object. Sorting keys by
UTF-16 code units and using compact UTF-8 JSON implements that restricted case
without importing the package or adding runtime dependencies to the backend.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

SCHEMA = "piginot.build-identity-v1"
ALGORITHM = "sha256-rfc8785-file-map-v1"
IDENTITY = "_build_identity.json"
BUILD_INPUTS = frozenset({"hatch_build.py", "pyproject.toml"})
FIELDS = frozenset(
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    packages = [
        ("src/piginot", "piginot", (".py",)),
        ("piginot_campaigns", "piginot_campaigns", (".py", ".json")),
    ]
    facade = root / "src/formamech"
    if facade.exists() or facade.is_symlink():
        packages.append(("src/formamech", "formamech", (".py",)))
    for relative, prefix, suffixes in packages:
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"missing or linked source package: {relative}")
        for path in sorted(directory.rglob("*")):
            if "__pycache__" in path.relative_to(directory).parts:
                continue
            if path.is_symlink():
                raise ValueError("build identity refuses linked implementation files")
            if path.is_file() and path.suffix in suffixes:
                result[f"{prefix}/{path.relative_to(directory).as_posix()}"] = _sha(
                    path
                )
    if not {"piginot/__init__.py", "piginot_campaigns/__init__.py"} <= result.keys():
        raise ValueError("source package manifests are incomplete")
    return result


def _manifest_sha(files: dict[str, str]) -> str:
    ordered = {
        key: files[key]
        for key in sorted(files, key=lambda key: key.encode("utf-16-be"))
    }
    encoded = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _git_state(root: Path, files: dict[str, str]) -> tuple[str, bool] | None:
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if top.returncode or Path(top.stdout.strip()).resolve() != root:
        return None
    revision = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    tracked = _git(root, "ls-files", "-z")
    value = revision.stdout.strip()
    if (
        revision.returncode
        or status.returncode
        or tracked.returncode
        or not re.fullmatch("[0-9a-f]{40}", value)
    ):
        raise ValueError("cannot establish source revision and cleanliness")
    required = {
        f"src/{key}" if key.startswith(("piginot/", "formamech/")) else key
        for key in files
    } | BUILD_INPUTS
    dirty = bool(status.stdout) or not required <= set(tracked.stdout.split("\0"))
    return value, dirty


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate attestation key: {key}")
        result[key] = value
    return result


def _prior_record(
    root: Path, version: str, files: dict[str, str], inputs: dict[str, str]
) -> dict[str, Any]:
    path = root / "src/piginot" / IDENTITY
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            "Gitless source needs a verified source-distribution attestation"
        )
    record = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("invalid source-distribution attestation fields")
    if record["schema"] != SCHEMA or record["manifest_algorithm"] != ALGORITHM:
        raise ValueError("unsupported source-distribution attestation schema/algorithm")
    if not isinstance(record["source_revision"], str) or not re.fullmatch(
        "[0-9a-f]{40}", record["source_revision"]
    ):
        raise ValueError("invalid source-distribution revision")
    if record["package_version"] != version or type(record["dirty"]) is not bool:
        raise ValueError("source-distribution version or dirty flag differs")
    if (
        record["files"] != files
        or record["manifest_sha256"] != _manifest_sha(files)
        or record["build_inputs_sha256"] != inputs
    ):
        raise ValueError("source-distribution implementation/build inputs changed")
    return record


def build_identity(root: Path, version: str) -> dict[str, Any]:
    root = root.resolve()
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    if (
        project.get("name") not in {"piginot", "formamech"}
        or project.get("version") != version
    ):
        raise ValueError("source project does not match build package/version")
    files = _manifest(root)
    if project["name"] == "formamech" and "formamech/__init__.py" not in files:
        raise ValueError("FormaMech public package is missing")
    inputs = {name: _sha(root / name) for name in sorted(BUILD_INPUTS)}
    state = _git_state(root, files)
    if state is None:
        return _prior_record(root, version, files, inputs)
    revision, dirty = state
    return {
        "schema": SCHEMA,
        "source_revision": revision,
        "package_version": version,
        "dirty": dirty,
        "files": files,
        "manifest_sha256": _manifest_sha(files),
        "manifest_algorithm": ALGORITHM,
        "build_inputs_sha256": inputs,
    }


class CustomBuildHook(BuildHookInterface[Any]):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name not in ("wheel", "sdist"):
            raise ValueError("unsupported build-identity target")
        root = Path(self.root).resolve()
        record = build_identity(root, self.metadata.version)
        if Path(tempfile.gettempdir()).resolve().is_relative_to(root):
            raise ValueError(
                "build identity temporary directory must be outside source"
            )
        self._identity_directory = tempfile.TemporaryDirectory(
            prefix="piginot-build-identity-"
        )
        generated = Path(self._identity_directory.name) / IDENTITY
        generated.write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        destination = (
            f"piginot/{IDENTITY}"
            if self.target_name == "wheel"
            else f"src/piginot/{IDENTITY}"
        )
        build_data.setdefault("force_include", {})[str(generated)] = destination

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        self._identity_directory.cleanup()
