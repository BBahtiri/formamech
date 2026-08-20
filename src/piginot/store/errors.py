"""Typed failures for the local content-addressed store."""

from __future__ import annotations

from collections.abc import Iterable


class StoreError(ValueError):
    """Base class for content-store failures."""


class StoreValidationError(StoreError):
    """A public store argument was invalid."""


class InvalidDigestError(StoreValidationError):
    """A content or recipe hash was not lowercase BLAKE2b-256 hex."""


class InvalidTreeError(StoreValidationError):
    """A directory tree contained an unsupported entry."""


class StoreFormatError(StoreError):
    """Persisted store metadata did not match the S3 schema."""


class ContentNotFoundError(StoreError):
    """No committed metadata exists for a requested content hash."""


class ContentPrunedError(ContentNotFoundError):
    """The content identity exists, but its bytes were pruned."""


class ContentCorruptionError(StoreError):
    """Stored bytes or filesystem entries violate their recorded identity."""


class RecipeNotFoundError(StoreError):
    """No result set exists for a requested recipe hash."""


class DivergentRecipeError(StoreError):
    """One recipe produced more than one distinct content hash."""

    def __init__(self, recipe_hash: str, content_hashes: Iterable[str]) -> None:
        self.recipe_hash = recipe_hash
        self.content_hashes = tuple(sorted(content_hashes))
        rendered = ", ".join(self.content_hashes)
        super().__init__(f"recipe {recipe_hash} has divergent results: {rendered}")


class RetentionPolicyError(StoreError):
    """A requested retention transition violates the tier policy."""
