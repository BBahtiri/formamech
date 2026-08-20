"""Content-addressed object storage for PI-GINOT artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

from piginot.contracts.canonical import canonical_json_bytes

_HEX_256: Final = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_PREFIX: Final = b"piginot.content.v1\0"
_TREE_FORMAT: Final = "piginot-tree-v1"
_AVAILABLE: Final = "AVAILABLE"
_PRUNED: Final = "PRUNED"

JsonObject: TypeAlias = dict[str, Any]


class ObjectStoreError(ValueError):
    """Base class for content store failures."""


class MissingObjectError(ObjectStoreError):
    """An object hash is known or requested, but bytes are unavailable."""


class CorruptObjectError(ObjectStoreError):
    """Stored object bytes do not match the requested content hash."""


class DivergentRecipeError(ObjectStoreError):
    """One recipe hash was associated with different result contents."""


class RetentionTier(StrEnum):
    """Retention policy for stored object bytes."""

    PERMANENT = "PERMANENT"
    RETAINED = "RETAINED"
    PRUNABLE = "PRUNABLE"


@dataclass(frozen=True)
class PruneResult:
    """Summary of one prune operation."""

    pruned_hashes: tuple[str, ...]
    pruned_bytes: int


class ObjectStore:
    """Small filesystem-backed content-addressed store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(
        self,
        payload: bytes,
        *,
        recipe_hash: str | None = None,
        tier: RetentionTier = RetentionTier.RETAINED,
    ) -> str:
        """Store bytes and return their content hash."""

        if not isinstance(payload, bytes):
            raise ObjectStoreError("payload must be bytes")
        digest = content_hash(payload)
        if recipe_hash is not None:
            self._check_recipe(recipe_hash, digest)

        destination = self.object_path(digest)
        if destination.exists():
            self._read_verified(digest)
        else:
            _atomic_write(destination, payload)
        self._record_object(digest, len(payload), tier)

        if recipe_hash is not None:
            self._record_recipe(recipe_hash, digest)
        return digest

    def get(self, digest: str) -> bytes:
        """Read bytes by content hash, verifying the hash before returning."""

        path = self.object_path(digest)
        if not path.exists():
            metadata = self._object_metadata(digest)
            if metadata is not None and metadata.get("state") == _PRUNED:
                raise MissingObjectError(f"object {digest} has been pruned")
            raise MissingObjectError(f"object {digest} is missing")
        return self._read_verified(digest)

    def put_tree(
        self,
        path: str | Path,
        *,
        tier: RetentionTier = RetentionTier.RETAINED,
    ) -> str:
        """Store a deterministic manifest for a regular-file directory tree."""

        root = Path(path)
        if root.is_symlink():
            raise ObjectStoreError("tree root must not be a symbolic link")
        if not root.is_dir():
            raise ObjectStoreError("put_tree path must be a directory")
        if _is_relative_to(self.root.resolve(strict=False), root.resolve(strict=False)):
            raise ObjectStoreError("put_tree source must not contain the store root")

        files: list[JsonObject] = []
        for candidate in sorted(root.rglob("*"), key=_relative_sort_key(root)):
            if candidate.is_symlink():
                raise ObjectStoreError("put_tree refuses symbolic link entries")
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ObjectStoreError("put_tree accepts regular files only")
            payload = candidate.read_bytes()
            digest = self.put(payload, tier=tier)
            files.append(
                {
                    "hash": digest,
                    "path": candidate.relative_to(root).as_posix(),
                    "size": len(payload),
                }
            )

        manifest = {"files": files, "format": _TREE_FORMAT}
        return self.put(canonical_json_bytes(manifest), tier=tier)

    def has_object(self, digest: str) -> bool:
        """Return whether verified object bytes are visibly present."""

        return self.object_path(digest).exists()

    def object_path(self, digest: str) -> Path:
        """Return the validated object byte path for a content hash."""

        validated = _validate_hash(digest)
        return self.root / "objects" / validated[:2] / validated

    def contents_for_recipe(self, recipe_hash: str) -> tuple[str, ...]:
        """Return known content hashes for one recipe hash, even after pruning."""

        recipe = self._recipe_metadata(recipe_hash)
        if recipe is None:
            return ()
        contents = recipe.get("content_hashes", [])
        if not isinstance(contents, list):
            raise ObjectStoreError(f"recipe index for {recipe_hash} is malformed")
        return tuple(_validate_hash(str(item)) for item in contents)

    def prune(self) -> PruneResult:
        """Remove PRUNABLE object bytes while retaining metadata and lineage."""

        pruned: list[str] = []
        pruned_bytes = 0
        for metadata_path in sorted((self.root / "index" / "objects").rglob("*.json")):
            metadata = _read_json(metadata_path)
            digest = _validate_hash(str(metadata.get("hash", "")))
            tier = RetentionTier(str(metadata.get("tier", "")))
            state = str(metadata.get("state", ""))
            path = self.object_path(digest)
            if tier is not RetentionTier.PRUNABLE or state != _AVAILABLE:
                continue
            if path.exists():
                pruned_bytes += path.stat().st_size
                path.unlink()
            updated = {**metadata, "state": _PRUNED}
            _atomic_json_write(metadata_path, updated)
            pruned.append(digest)

        return PruneResult(tuple(pruned), pruned_bytes)

    def _read_verified(self, digest: str) -> bytes:
        path = self.object_path(digest)
        payload = path.read_bytes()
        actual = content_hash(payload)
        if actual != digest:
            raise CorruptObjectError(
                f"object {digest} is corrupt; bytes hash to {actual}"
            )
        return payload

    def _record_object(self, digest: str, size: int, tier: RetentionTier) -> None:
        path = self._object_metadata_path(digest)
        existing = self._object_metadata(digest)
        if existing is None:
            metadata: JsonObject = {
                "hash": digest,
                "size": size,
                "state": _AVAILABLE,
                "tier": tier.value,
            }
        else:
            metadata = {
                **existing,
                "size": size,
                "state": _AVAILABLE,
                "tier": _stronger_tier(
                    RetentionTier(str(existing.get("tier", ""))), tier
                ).value,
            }
        _atomic_json_write(path, metadata)

    def _object_metadata(self, digest: str) -> JsonObject | None:
        path = self._object_metadata_path(digest)
        if not path.exists():
            return None
        return _read_json(path)

    def _object_metadata_path(self, digest: str) -> Path:
        validated = _validate_hash(digest)
        return self.root / "index" / "objects" / validated[:2] / f"{validated}.json"

    def _check_recipe(self, recipe_hash: str, content_digest: str) -> None:
        existing = self.contents_for_recipe(recipe_hash)
        if existing and content_digest not in existing:
            raise DivergentRecipeError(
                f"recipe {recipe_hash} already produced {existing}; "
                f"refusing divergent content {content_digest}"
            )

    def _record_recipe(self, recipe_hash: str, content_digest: str) -> None:
        path = self._recipe_metadata_path(recipe_hash)
        contents = set(self.contents_for_recipe(recipe_hash))
        contents.add(content_digest)
        _atomic_json_write(
            path,
            {
                "content_hashes": sorted(contents),
                "recipe_hash": _validate_hash(recipe_hash),
            },
        )

    def _recipe_metadata(self, recipe_hash: str) -> JsonObject | None:
        path = self._recipe_metadata_path(recipe_hash)
        if not path.exists():
            return None
        return _read_json(path)

    def _recipe_metadata_path(self, recipe_hash: str) -> Path:
        validated = _validate_hash(recipe_hash)
        return self.root / "index" / "recipes" / validated[:2] / f"{validated}.json"


def content_hash(payload: bytes) -> str:
    """Hash raw bytes in the PI-GINOT content domain."""

    return hashlib.blake2b(
        _CONTENT_PREFIX + payload,
        digest_size=32,
    ).hexdigest()


def _validate_hash(digest: str) -> str:
    if not _HEX_256.fullmatch(digest):
        raise ObjectStoreError(f"invalid content hash {digest!r}")
    return digest


def _stronger_tier(left: RetentionTier, right: RetentionTier) -> RetentionTier:
    order = {
        RetentionTier.PRUNABLE: 0,
        RetentionTier.RETAINED: 1,
        RetentionTier.PERMANENT: 2,
    }
    return left if order[left] >= order[right] else right


def _relative_sort_key(root: Path) -> Callable[[Path], str]:
    def key(path: Path) -> str:
        return path.relative_to(root).as_posix()

    return key


def _is_relative_to(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _atomic_json_write(path: Path, payload: JsonObject) -> None:
    _atomic_write(path, canonical_json_bytes(payload))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)


def _read_json(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ObjectStoreError(f"metadata file {path} is malformed")
    return cast(JsonObject, payload)


__all__ = (
    "CorruptObjectError",
    "DivergentRecipeError",
    "MissingObjectError",
    "ObjectStore",
    "ObjectStoreError",
    "PruneResult",
    "RetentionTier",
    "content_hash",
)
