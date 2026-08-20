from __future__ import annotations

import json
from pathlib import Path

import pytest

from piginot.store import (
    ContentNotFoundError,
    DivergentRecipeError,
    ObjectStore,
    RecipeNotFoundError,
    RetentionTier,
    StoreFormatError,
)


def test_divergent_recipe_persists_both_results_then_raises(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    recipe_hash = "a" * 64
    first = store.put(b"first")
    second = store.put(b"second")

    store.record_recipe_result(recipe_hash, first)
    with pytest.raises(DivergentRecipeError) as error:
        store.record_recipe_result(recipe_hash, second)

    assert error.value.recipe_hash == recipe_hash
    assert error.value.content_hashes == tuple(sorted((first, second)))
    assert store.recipe_results(recipe_hash) == frozenset((first, second))
    with pytest.raises(DivergentRecipeError):
        store.resolve_recipe(recipe_hash)


def test_recipe_singleton_is_idempotent_and_resolves_after_prune(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    recipe_hash = "b" * 64
    digest = store.put(b"result", tier=RetentionTier.PRUNABLE)

    store.record_recipe_result(recipe_hash, digest)
    store.record_recipe_result(recipe_hash, digest)
    store.prune()

    assert store.recipe_results(recipe_hash) == frozenset((digest,))
    assert store.resolve_recipe(recipe_hash) == digest
    assert store.contents_for_recipe(recipe_hash) == (digest,)


def test_recipe_rejects_unknown_content_and_missing_recipe(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")

    with pytest.raises(ContentNotFoundError):
        store.record_recipe_result("c" * 64, "d" * 64)
    with pytest.raises(RecipeNotFoundError):
        store.resolve_recipe("e" * 64)


def test_put_recipe_convenience_preserves_divergence_evidence(tmp_path: Path) -> None:
    store = ObjectStore.create(tmp_path / "store")
    recipe_hash = "f" * 64
    first = store.put(b"first", recipe_hash=recipe_hash)

    with pytest.raises(DivergentRecipeError):
        store.put(b"second", recipe_hash=recipe_hash)

    second = store.put(b"second")
    assert store.recipe_results(recipe_hash) == frozenset((first, second))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("recipe_hash", "f" * 64),
        ("content_hashes", "not-a-list"),
        ("content_hashes", ["not-a-hash"]),
        ("content_hashes", ["a" * 64, "a" * 64]),
        ("content_hashes", ["b" * 64, "a" * 64]),
    ],
)
def test_recipe_metadata_is_strict(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    recipe_hash = "1" * 64
    digest = store.put(b"result")
    store.record_recipe_result(recipe_hash, digest)
    metadata = store._recipe_metadata_path(recipe_hash)
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document[field] = value
    metadata.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(StoreFormatError):
        store.recipe_results(recipe_hash)
