"""Filesystem-backed, single-writer content-addressed storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from piginot.contracts.canonical import canonical_json_bytes, strict_json_loads
from piginot.contracts.errors import ContractValidationError
from piginot.contracts.hashing import hash_spec
from piginot.store.errors import (
    ContentCorruptionError,
    ContentNotFoundError,
    ContentPrunedError,
    DivergentRecipeError,
    InvalidDigestError,
    InvalidTreeError,
    RecipeNotFoundError,
    RetentionPolicyError,
    StoreFormatError,
    StoreValidationError,
)
from piginot.store.models import ContentInfo, PruneResult, RetentionTier

_HASH_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_STORE_FORMAT: Final = "piginot.store.v1"
_CONTENT_FORMAT: Final = "piginot.raw-content.v1"
_TREE_FORMAT: Final = "piginot.tree.v1"
_SCHEMA_VERSION: Final = 1
_MARKER_KEYS: Final = frozenset(
    ("schema_version", "store_format", "content_hash_format")
)
_CONTENT_KEYS: Final = frozenset(
    ("schema_version", "content_hash", "size_bytes", "retention", "materialized")
)
_RECIPE_KEYS: Final = frozenset(("schema_version", "recipe_hash", "content_hashes"))
_TIER_STRENGTH: Final = {
    RetentionTier.PRUNABLE: 0,
    RetentionTier.RETAINED: 1,
    RetentionTier.PERMANENT: 2,
}

JsonObject = dict[str, object]


class ObjectStore:
    """Local immutable-byte store with retained identity metadata."""

    root: Path

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self._validate_root()
        self._validate_marker()

    @classmethod
    def create(cls, root: str | os.PathLike[str]) -> ObjectStore:
        """Create a new store, or reopen an existing valid store."""

        store = cls.__new__(cls)
        store.root = Path(root)
        marker = store._marker_path
        if store.root.exists() or store.root.is_symlink():
            store._validate_root()
            if marker.exists() or marker.is_symlink():
                store._validate_marker()
                return store
            if any(store.root.iterdir()):
                raise StoreFormatError(
                    f"store root {store.root} is non-empty but has no store marker"
                )
        else:
            store.root.mkdir(parents=True)
            store._validate_root()

        _atomic_json_write(marker, _store_marker())
        store._validate_marker()
        return store

    @classmethod
    def open(cls, root: str | os.PathLike[str]) -> ObjectStore:
        """Open an existing store only when its marker is valid."""

        return cls(root)

    def put(
        self,
        payload: bytes,
        *,
        recipe_hash: str | None = None,
        tier: RetentionTier = RetentionTier.RETAINED,
    ) -> str:
        """Store exact bytes and return their domain-separated content hash."""

        _require_bytes(payload)
        requested_tier = _require_tier(tier)
        digest = content_hash(payload)
        existing = self._content_info_optional(digest)
        destination = self.object_path(digest)

        if destination.exists() or destination.is_symlink():
            self._read_verified_file(destination, digest)
        else:
            self._ensure_safe_ancestors(destination)
            _atomic_write(destination, payload)

        retention = requested_tier
        if existing is not None:
            retention = _stronger_tier(existing.retention, requested_tier)
        committed = ContentInfo(
            content_hash=digest,
            size_bytes=len(payload),
            retention=retention,
            materialized=True,
        )
        self._write_content_info(committed)

        if recipe_hash is not None:
            self.record_recipe_result(recipe_hash, digest)
        return digest

    def get(self, content_hash_value: str) -> bytes:
        """Return committed bytes after verifying their content hash."""

        digest = _validate_digest(content_hash_value)
        info = self._content_info_optional(digest)
        if info is None:
            raise ContentNotFoundError(f"content {digest} is not committed")
        if not info.materialized:
            raise ContentPrunedError(f"content {digest} has been pruned")

        payload = self._read_verified_file(self.object_path(digest), digest)
        if len(payload) != info.size_bytes:
            raise ContentCorruptionError(
                f"content {digest} has size {len(payload)}, expected {info.size_bytes}"
            )
        return payload

    def info(self, content_hash_value: str) -> ContentInfo:
        """Return retained metadata without requiring materialized bytes."""

        digest = _validate_digest(content_hash_value)
        info = self._content_info_optional(digest)
        if info is None:
            raise ContentNotFoundError(f"content {digest} is not committed")
        return info

    def has_object(self, content_hash_value: str) -> bool:
        """Return whether a content hash has committed, verified bytes."""

        try:
            self.get(content_hash_value)
        except ContentNotFoundError:
            return False
        return True

    def object_path(self, content_hash_value: str) -> Path:
        """Return the addressed object path after validating the hash."""

        digest = _validate_digest(content_hash_value)
        return self.root / "objects" / digest[:2] / digest

    def put_tree(
        self,
        path: str | os.PathLike[str],
        *,
        tier: RetentionTier = RetentionTier.RETAINED,
    ) -> str:
        """Store a recursive canonical manifest for a directory tree."""

        requested_tier = _require_tier(tier)
        source = Path(path)
        source_kind = _tree_entry_kind(source)
        if source_kind != "tree":
            raise InvalidTreeError(f"put_tree source {source} must be a directory")

        source_resolved = source.resolve()
        store_resolved = self.root.resolve()
        if (
            source_resolved == store_resolved
            or source_resolved.is_relative_to(store_resolved)
            or store_resolved.is_relative_to(source_resolved)
        ):
            raise InvalidTreeError(
                "put_tree source and content store must not contain one another"
            )
        return self._put_directory(source, requested_tier)

    def set_retention(
        self,
        content_hash_value: str,
        tier: RetentionTier,
    ) -> ContentInfo:
        """Apply an explicit retention transition to committed content."""

        requested_tier = _require_tier(tier)
        current = self.info(content_hash_value)
        if (
            current.retention is RetentionTier.PERMANENT
            and requested_tier is not RetentionTier.PERMANENT
        ):
            raise RetentionPolicyError(
                f"PERMANENT content {current.content_hash} cannot be downgraded"
            )
        updated = ContentInfo(
            content_hash=current.content_hash,
            size_bytes=current.size_bytes,
            retention=requested_tier,
            materialized=current.materialized,
        )
        self._write_content_info(updated)
        return updated

    def record_recipe_result(
        self,
        recipe_hash: str,
        content_hash_value: str,
    ) -> None:
        """Persist one recipe result, preserving evidence before divergence raises."""

        recipe_digest = _validate_digest(recipe_hash)
        content_digest = _validate_digest(content_hash_value)
        self.info(content_digest)
        results = set(self.recipe_results(recipe_digest))
        results.add(content_digest)
        self._write_recipe_results(recipe_digest, results)
        if len(results) > 1:
            raise DivergentRecipeError(recipe_digest, results)

    def recipe_results(self, recipe_hash: str) -> frozenset[str]:
        """Return the complete result set for a recipe, including pruned results."""

        recipe_digest = _validate_digest(recipe_hash)
        document = self._recipe_document_optional(recipe_digest)
        if document is None:
            return frozenset()
        values = document["content_hashes"]
        if not isinstance(values, list):
            raise StoreFormatError(
                f"recipe metadata for {recipe_digest} has invalid content_hashes"
            )
        return frozenset(cast(list[str], values))

    def contents_for_recipe(self, recipe_hash: str) -> tuple[str, ...]:
        """Return sorted recipe results for callers that need stable iteration."""

        return tuple(sorted(self.recipe_results(recipe_hash)))

    def resolve_recipe(self, recipe_hash: str) -> str:
        """Resolve exactly one recipe result, never choosing through divergence."""

        recipe_digest = _validate_digest(recipe_hash)
        results = self.recipe_results(recipe_digest)
        if not results:
            raise RecipeNotFoundError(f"recipe {recipe_digest} has no results")
        if len(results) > 1:
            raise DivergentRecipeError(recipe_digest, results)
        return next(iter(results))

    def prune(self, *, dry_run: bool = False) -> PruneResult:
        """Remove only PRUNABLE bytes while retaining identity metadata."""

        selected: list[str] = []
        selected_bytes = 0
        metadata_root = self.root / "metadata" / "content"
        if not metadata_root.exists():
            return PruneResult((), 0, dry_run)

        for metadata_path in sorted(metadata_root.rglob("*.json")):
            digest = _validate_digest(metadata_path.stem)
            if metadata_path != self._content_metadata_path(digest):
                raise StoreFormatError(
                    f"content metadata path does not match address {digest}"
                )
            info = self.info(digest)
            if info.retention is not RetentionTier.PRUNABLE:
                continue

            object_path = self.object_path(digest)
            object_exists = object_path.exists() or object_path.is_symlink()
            if not info.materialized and not object_exists:
                continue
            object_size = 0
            if object_exists:
                object_size = _regular_file_size(object_path, digest)

            selected.append(digest)
            selected_bytes += object_size
            if dry_run:
                continue

            if info.materialized:
                self._write_content_info(
                    ContentInfo(
                        content_hash=digest,
                        size_bytes=info.size_bytes,
                        retention=info.retention,
                        materialized=False,
                    )
                )
            if object_exists:
                object_path.unlink()

        return PruneResult(tuple(selected), selected_bytes, dry_run)

    @property
    def _marker_path(self) -> Path:
        return self.root / "store.json"

    def _validate_root(self) -> None:
        try:
            details = self.root.lstat()
        except OSError as exc:
            raise StoreFormatError(f"store root {self.root} does not exist") from exc
        if stat.S_ISLNK(details.st_mode) or _is_reparse_point(details):
            raise StoreFormatError(f"store root {self.root} must not be a link")
        if not stat.S_ISDIR(details.st_mode):
            raise StoreFormatError(f"store root {self.root} must be a directory")

    def _validate_marker(self) -> None:
        document = _read_json_object(self._marker_path, "store marker")
        _require_keys(document, _MARKER_KEYS, "store marker")
        if document != _store_marker():
            raise StoreFormatError(
                f"store marker {self._marker_path} has an unsupported format"
            )

    def _put_directory(self, directory: Path, tier: RetentionTier) -> str:
        entries: list[JsonObject] = []
        try:
            children = sorted(directory.iterdir(), key=_tree_sort_key)
        except (OSError, UnicodeEncodeError) as exc:
            raise InvalidTreeError(
                f"cannot enumerate tree directory {directory}"
            ) from exc

        for child in children:
            kind = _tree_entry_kind(child)
            if kind == "tree":
                digest = self._put_directory(child, tier)
            elif kind == "blob":
                try:
                    payload = child.read_bytes()
                except OSError as exc:
                    raise InvalidTreeError(f"cannot read tree file {child}") from exc
                digest = self.put(payload, tier=tier)
            else:
                raise AssertionError(f"unexpected tree entry kind: {kind}")
            entries.append(
                {
                    "content_hash": digest,
                    "kind": kind,
                    "name": child.name,
                }
            )

        manifest = {"entries": entries, "format": _TREE_FORMAT}
        return self.put(canonical_json_bytes(manifest), tier=tier)

    def _read_verified_file(self, path: Path, expected_hash: str) -> bytes:
        self._ensure_safe_ancestors(path)
        if path.is_symlink():
            raise ContentCorruptionError(f"content {expected_hash} is a symbolic link")
        try:
            details = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ContentNotFoundError(
                f"content bytes for {expected_hash} are missing"
            ) from exc
        except OSError as exc:
            raise ContentCorruptionError(
                f"content bytes for {expected_hash} cannot be inspected"
            ) from exc
        if _is_reparse_point(details):
            raise ContentCorruptionError(
                f"content {expected_hash} is a filesystem reparse point"
            )
        if not stat.S_ISREG(details.st_mode):
            raise ContentCorruptionError(
                f"content {expected_hash} is not a regular file"
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ContentCorruptionError(
                f"content bytes for {expected_hash} cannot be read"
            ) from exc
        actual = content_hash(payload)
        if actual != expected_hash:
            raise ContentCorruptionError(
                f"content {expected_hash} is corrupt; bytes hash to {actual}"
            )
        return payload

    def _write_content_info(self, info: ContentInfo) -> None:
        path = self._content_metadata_path(info.content_hash)
        self._ensure_safe_ancestors(path)
        _atomic_json_write(
            path,
            {
                "content_hash": info.content_hash,
                "materialized": info.materialized,
                "retention": info.retention.value,
                "schema_version": _SCHEMA_VERSION,
                "size_bytes": info.size_bytes,
            },
        )

    def _content_info_optional(self, digest: str) -> ContentInfo | None:
        path = self._content_metadata_path(digest)
        self._ensure_safe_ancestors(path)
        if path.is_symlink():
            raise StoreFormatError(f"content metadata {path} must not be a link")
        if not path.exists():
            return None
        document = _read_json_object(path, "content metadata")
        _require_keys(document, _CONTENT_KEYS, "content metadata")

        recorded_hash = document["content_hash"]
        if recorded_hash != digest:
            raise StoreFormatError(
                f"content metadata address {digest} records {recorded_hash!r}"
            )
        if document["schema_version"] != _SCHEMA_VERSION:
            raise StoreFormatError(
                f"content metadata for {digest} has unsupported schema"
            )
        size = document["size_bytes"]
        if type(size) is not int or size < 0:
            raise StoreFormatError(
                f"content metadata for {digest} has invalid size_bytes"
            )
        materialized = document["materialized"]
        if type(materialized) is not bool:
            raise StoreFormatError(
                f"content metadata for {digest} has invalid materialized state"
            )
        retention_value = document["retention"]
        if not isinstance(retention_value, str):
            raise StoreFormatError(
                f"content metadata for {digest} has invalid retention"
            )
        try:
            retention = RetentionTier(retention_value)
        except ValueError as exc:
            raise StoreFormatError(
                f"content metadata for {digest} has invalid retention"
            ) from exc
        return ContentInfo(
            content_hash=digest,
            size_bytes=size,
            retention=retention,
            materialized=materialized,
        )

    def _content_metadata_path(self, content_hash_value: str) -> Path:
        digest = _validate_digest(content_hash_value)
        return self.root / "metadata" / "content" / digest[:2] / f"{digest}.json"

    def _write_recipe_results(
        self,
        recipe_hash: str,
        results: set[str],
    ) -> None:
        path = self._recipe_metadata_path(recipe_hash)
        self._ensure_safe_ancestors(path)
        _atomic_json_write(
            path,
            {
                "content_hashes": sorted(results),
                "recipe_hash": recipe_hash,
                "schema_version": _SCHEMA_VERSION,
            },
        )

    def _recipe_document_optional(self, recipe_hash: str) -> JsonObject | None:
        path = self._recipe_metadata_path(recipe_hash)
        self._ensure_safe_ancestors(path)
        if path.is_symlink():
            raise StoreFormatError(f"recipe metadata {path} must not be a link")
        if not path.exists():
            return None
        document = _read_json_object(path, "recipe metadata")
        _require_keys(document, _RECIPE_KEYS, "recipe metadata")
        if document["schema_version"] != _SCHEMA_VERSION:
            raise StoreFormatError(
                f"recipe metadata for {recipe_hash} has unsupported schema"
            )
        if document["recipe_hash"] != recipe_hash:
            raise StoreFormatError(
                f"recipe metadata address {recipe_hash} does not match its record"
            )
        values = document["content_hashes"]
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise StoreFormatError(
                f"recipe metadata for {recipe_hash} has invalid content_hashes"
            )
        try:
            validated = [_validate_digest(cast(str, value)) for value in values]
        except InvalidDigestError as exc:
            raise StoreFormatError(
                f"recipe metadata for {recipe_hash} contains an invalid hash"
            ) from exc
        if validated != sorted(set(validated)):
            raise StoreFormatError(
                f"recipe metadata for {recipe_hash} is not sorted and unique"
            )
        return document

    def _recipe_metadata_path(self, recipe_hash: str) -> Path:
        digest = _validate_digest(recipe_hash)
        return self.root / "metadata" / "recipes" / digest[:2] / f"{digest}.json"

    def _ensure_safe_ancestors(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise StoreFormatError(f"store path {path} escapes {self.root}") from exc
        current = self.root
        self._validate_root()
        for component in relative.parts[:-1]:
            current /= component
            if not current.exists() and not current.is_symlink():
                continue
            try:
                details = current.lstat()
            except OSError as exc:
                raise StoreFormatError(
                    f"store directory {current} cannot be inspected"
                ) from exc
            if (
                stat.S_ISLNK(details.st_mode)
                or _is_reparse_point(details)
                or not stat.S_ISDIR(details.st_mode)
            ):
                raise StoreFormatError(
                    f"store directory {current} must be a regular directory"
                )


def content_hash(payload: bytes) -> str:
    """Hash raw bytes through the canonical S2 content domain."""

    _require_bytes(payload)
    raw_digest = hashlib.blake2b(payload, digest_size=32).hexdigest()
    return hash_spec(
        {
            "blake2b_256": raw_digest,
            "format": _CONTENT_FORMAT,
            "size_bytes": len(payload),
        },
        "content",
    )


def _store_marker() -> JsonObject:
    return {
        "content_hash_format": _CONTENT_FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "store_format": _STORE_FORMAT,
    }


def _validate_digest(value: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise InvalidDigestError(f"invalid hash {value!r}")
    return value


def _require_bytes(payload: object) -> None:
    if not isinstance(payload, bytes):
        raise StoreValidationError("content payload must be bytes")


def _require_tier(value: object) -> RetentionTier:
    if not isinstance(value, RetentionTier):
        raise StoreValidationError("retention tier must be a RetentionTier")
    return value


def _stronger_tier(left: RetentionTier, right: RetentionTier) -> RetentionTier:
    return left if _TIER_STRENGTH[left] >= _TIER_STRENGTH[right] else right


def _tree_sort_key(path: Path) -> bytes:
    try:
        return path.name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidTreeError(f"tree entry name {path.name!r} is not UTF-8") from exc


def _tree_entry_kind(path: Path) -> str:
    try:
        details = path.lstat()
    except OSError as exc:
        raise InvalidTreeError(f"tree entry {path} cannot be inspected") from exc
    if stat.S_ISLNK(details.st_mode) or _is_reparse_point(details):
        raise InvalidTreeError(f"tree entry {path} must not be a link or reparse point")
    if stat.S_ISDIR(details.st_mode):
        return "tree"
    if stat.S_ISREG(details.st_mode):
        return "blob"
    raise InvalidTreeError(f"tree entry {path} must be a regular file or directory")


def _regular_file_size(path: Path, digest: str) -> int:
    if path.is_symlink():
        raise ContentCorruptionError(f"content {digest} is a symbolic link")
    try:
        details = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ContentCorruptionError(f"content {digest} cannot be inspected") from exc
    if _is_reparse_point(details) or not stat.S_ISREG(details.st_mode):
        raise ContentCorruptionError(f"content {digest} is not a regular file")
    return details.st_size


def _is_reparse_point(details: os.stat_result) -> bool:
    attributes = getattr(details, "st_file_attributes", 0)
    mask = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & mask)


def _read_json_object(path: Path, label: str) -> JsonObject:
    if path.is_symlink():
        raise StoreFormatError(f"{label} {path} must not be a link")
    try:
        details = path.stat(follow_symlinks=False)
        if _is_reparse_point(details) or not stat.S_ISREG(details.st_mode):
            raise StoreFormatError(f"{label} {path} must be a regular file")
        value = strict_json_loads(path.read_bytes())
    except (ContractValidationError, json.JSONDecodeError, OSError) as exc:
        raise StoreFormatError(f"{label} {path} is malformed") from exc
    if not isinstance(value, dict):
        raise StoreFormatError(f"{label} {path} must be a JSON object")
    return cast(JsonObject, value)


def _require_keys(
    document: Mapping[str, object], keys: frozenset[str], label: str
) -> None:
    if frozenset(document) != keys:
        raise StoreFormatError(f"{label} fields do not match schema")


def _atomic_json_write(path: Path, document: Mapping[str, object]) -> None:
    _atomic_write(path, canonical_json_bytes(document))


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
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)
