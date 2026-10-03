"""Bundle manifest: every contained file is hashed, and the manifest hashes itself.

A bundle is a directory with ``bundle.json`` at its root. The manifest lists the
SHA-256 and size of every other file and carries ``manifest_sha256``, the SHA-256
of the canonical JSON of the manifest without that field. Verification fails
closed with a stable reason code when a listed file is missing, changed, or
resized, when an unlisted file is present, when a path escapes the bundle, or
when any manifest byte changed (``G15.manifest``).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from piginot.contracts import canonical_json_bytes, strict_json_loads

MANIFEST_NAME = "bundle.json"
BUNDLE_FORMAT = "piginot-bundle-v1"


class BundleReasonCode(StrEnum):
    """Stable reasons a bundle is refused."""

    MANIFEST_MISSING = "BUNDLE_MANIFEST_MISSING"
    MANIFEST_INVALID = "BUNDLE_MANIFEST_INVALID"
    MANIFEST_HASH_MISMATCH = "BUNDLE_MANIFEST_HASH_MISMATCH"
    FILE_MISSING = "BUNDLE_FILE_MISSING"
    FILE_HASH_MISMATCH = "BUNDLE_FILE_HASH_MISMATCH"
    FILE_SIZE_MISMATCH = "BUNDLE_FILE_SIZE_MISMATCH"
    UNLISTED_FILE = "BUNDLE_UNLISTED_FILE"
    UNSAFE_PATH = "BUNDLE_UNSAFE_PATH"
    FILE_SET_INVALID = "BUNDLE_FILE_SET_INVALID"
    SPEC_INVALID = "BUNDLE_SPEC_INVALID"
    SPEC_INCONSISTENT = "BUNDLE_SPEC_INCONSISTENT"
    WEIGHTS_INVALID = "BUNDLE_WEIGHTS_INVALID"
    STRESS_AUTHORITY_MISMATCH = "BUNDLE_STRESS_AUTHORITY_MISMATCH"
    REPORT_INVALID = "BUNDLE_REPORT_INVALID"
    LIMITATIONS_MISSING = "BUNDLE_LIMITATIONS_MISSING"


class BundleIntegrityError(ValueError):
    """The bundle was refused; ``reason_codes`` are sorted and stable."""

    def __init__(self, reason_codes: tuple[BundleReasonCode, ...], detail: str):
        self.reason_codes = tuple(sorted(set(reason_codes)))
        super().__init__(
            f"{', '.join(code.value for code in self.reason_codes)}: {detail}"
        )


def _safe_relative(path: str) -> str:
    candidate = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or candidate.is_absolute()
        or any(part in ("", ".", "..") for part in candidate.parts)
        or str(candidate) != path
        or path == MANIFEST_NAME
    ):
        raise BundleIntegrityError((BundleReasonCode.UNSAFE_PATH,), path)
    return path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class FileEntry:
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """The verified manifest of one bundle."""

    format: str
    name: str
    created_utc: str
    claim_level: str
    identities: Mapping[str, str]
    files: Mapping[str, FileEntry]
    manifest_sha256: str = field(default="")

    def document(self) -> dict[str, object]:
        """The manifest as JSON, without ``manifest_sha256``."""

        return {
            "format": self.format,
            "name": self.name,
            "created_utc": self.created_utc,
            "claim_level": self.claim_level,
            "identities": dict(sorted(self.identities.items())),
            "files": {
                path: {"sha256": entry.sha256, "bytes": entry.bytes}
                for path, entry in sorted(self.files.items())
            },
        }


def manifest_hash(manifest: BundleManifest) -> str:
    """SHA-256 of the canonical manifest document without its own hash."""

    return sha256_bytes(canonical_json_bytes(manifest.document()))


def write_bundle(
    directory: Path,
    files: Mapping[str, bytes],
    *,
    name: str,
    created_utc: str,
    claim_level: str,
    identities: Mapping[str, str],
) -> BundleManifest:
    """Create a new bundle directory; refuses to overwrite an existing path."""

    if directory.exists():
        raise FileExistsError(f"bundle path already exists: {directory}")
    entries = {
        _safe_relative(path): FileEntry(sha256_bytes(payload), len(payload))
        for path, payload in files.items()
    }
    manifest = BundleManifest(
        format=BUNDLE_FORMAT,
        name=name,
        created_utc=created_utc,
        claim_level=claim_level,
        identities=MappingProxyType(dict(identities)),
        files=MappingProxyType(entries),
    )
    directory.mkdir(parents=True)
    for path, payload in sorted(files.items()):
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    digest = manifest_hash(manifest)
    document = manifest.document()
    document["manifest_sha256"] = digest
    (directory / MANIFEST_NAME).write_bytes(canonical_json_bytes(document))
    return replace(manifest, manifest_sha256=digest)


def _parse(payload: bytes) -> BundleManifest:
    try:
        parsed = strict_json_loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("manifest must be an object")
        document: dict[str, Any] = parsed
        expected = {
            "format",
            "name",
            "created_utc",
            "claim_level",
            "identities",
            "files",
            "manifest_sha256",
        }
        if set(document) != expected:
            raise ValueError("manifest fields differ from the format")
        if document["format"] != BUNDLE_FORMAT:
            raise ValueError(f"unsupported bundle format {document['format']!r}")
        files: dict[str, FileEntry] = {}
        for path, entry in document["files"].items():
            if set(entry) != {"sha256", "bytes"}:
                raise ValueError(f"file entry {path} is malformed")
            digest, size = entry["sha256"], entry["bytes"]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"file entry {path} has no SHA-256")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"file entry {path} has no size")
            files[_safe_relative(path)] = FileEntry(digest, size)
        identities = document["identities"]
        if not isinstance(identities, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in identities.items()
        ):
            raise ValueError("identities must map text to text")
        for key in ("name", "created_utc", "claim_level", "manifest_sha256"):
            if not isinstance(document[key], str) or not document[key]:
                raise ValueError(f"{key} must be non-empty text")
    except BundleIntegrityError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise BundleIntegrityError(
            (BundleReasonCode.MANIFEST_INVALID,), str(exc)
        ) from exc
    return BundleManifest(
        format=document["format"],
        name=document["name"],
        created_utc=document["created_utc"],
        claim_level=document["claim_level"],
        identities=MappingProxyType(dict(identities)),
        files=MappingProxyType(files),
        manifest_sha256=document["manifest_sha256"],
    )


def verify_bundle(directory: Path) -> tuple[BundleManifest, dict[str, bytes]]:
    """Verify every byte of a bundle and return its manifest and file contents."""

    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise BundleIntegrityError(
            (BundleReasonCode.MANIFEST_MISSING,), str(manifest_path)
        )
    raw = manifest_path.read_bytes()
    manifest = _parse(raw)
    if manifest_hash(manifest) != manifest.manifest_sha256:
        raise BundleIntegrityError(
            (BundleReasonCode.MANIFEST_HASH_MISMATCH,), "manifest content changed"
        )
    document = manifest.document()
    document["manifest_sha256"] = manifest.manifest_sha256
    if canonical_json_bytes(document) != raw:
        raise BundleIntegrityError(
            (BundleReasonCode.MANIFEST_HASH_MISMATCH,), "manifest is not canonical"
        )
    reasons: list[BundleReasonCode] = []
    details: list[str] = []
    contents: dict[str, bytes] = {}
    for path, entry in manifest.files.items():
        target = directory / path
        if not target.is_file() or target.is_symlink():
            reasons.append(BundleReasonCode.FILE_MISSING)
            details.append(path)
            continue
        payload = target.read_bytes()
        if len(payload) != entry.bytes:
            reasons.append(BundleReasonCode.FILE_SIZE_MISMATCH)
            details.append(path)
        if sha256_bytes(payload) != entry.sha256:
            reasons.append(BundleReasonCode.FILE_HASH_MISMATCH)
            details.append(path)
        contents[path] = payload
    listed = set(manifest.files) | {MANIFEST_NAME}
    for root, directories, names in os.walk(directory):
        for name in (*names, *directories):
            relative = Path(root, name).relative_to(directory).as_posix()
            full = Path(root, name)
            if full.is_symlink():
                reasons.append(BundleReasonCode.UNSAFE_PATH)
                details.append(relative)
            elif full.is_file() and relative not in listed:
                reasons.append(BundleReasonCode.UNLISTED_FILE)
                details.append(relative)
    if reasons:
        raise BundleIntegrityError(tuple(reasons), ", ".join(sorted(set(details))))
    return manifest, contents


def read_json(contents: Mapping[str, bytes], path: str) -> dict[str, Any]:
    """A verified JSON report of the bundle as an object, parsed strictly."""

    try:
        payload = contents[path]
    except KeyError as exc:
        raise BundleIntegrityError((BundleReasonCode.FILE_MISSING,), path) from exc
    try:
        value = strict_json_loads(payload)
    except ValueError as exc:
        raise BundleIntegrityError((BundleReasonCode.REPORT_INVALID,), path) from exc
    if not isinstance(value, dict):
        raise BundleIntegrityError((BundleReasonCode.REPORT_INVALID,), path)
    return value


__all__ = (
    "BUNDLE_FORMAT",
    "MANIFEST_NAME",
    "BundleIntegrityError",
    "BundleManifest",
    "BundleReasonCode",
    "FileEntry",
    "manifest_hash",
    "read_json",
    "sha256_bytes",
    "verify_bundle",
    "write_bundle",
)
