from __future__ import annotations

from pathlib import Path

import pytest

from piginot.store import (
    ContentPrunedError,
    ObjectStore,
    RetentionPolicyError,
    RetentionTier,
)


def test_prune_removes_only_prunable_bytes_and_keeps_metadata(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    permanent = store.put(b"forever", tier=RetentionTier.PERMANENT)
    retained = store.put(b"keep", tier=RetentionTier.RETAINED)
    prunable = store.put(b"drop", tier=RetentionTier.PRUNABLE)

    result = store.prune()

    assert result.pruned_hashes == (prunable,)
    assert result.pruned_bytes == 4
    assert result.dry_run is False
    assert store.get(permanent) == b"forever"
    assert store.get(retained) == b"keep"
    assert store.info(prunable).materialized is False
    with pytest.raises(ContentPrunedError, match=prunable):
        store.get(prunable)


def test_dry_run_reports_without_deleting_and_rehydration_keeps_identity(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"rehydrate", tier=RetentionTier.PRUNABLE)

    preview = store.prune(dry_run=True)
    assert preview.pruned_hashes == (digest,)
    assert preview.dry_run is True
    assert store.get(digest) == b"rehydrate"

    store.prune()
    assert store.put(b"rehydrate") == digest
    assert store.info(digest).materialized is True
    assert store.info(digest).retention is RetentionTier.RETAINED


def test_retention_is_promoted_on_put_and_permanent_is_irreversible(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"policy", tier=RetentionTier.PRUNABLE)
    store.put(b"policy", tier=RetentionTier.RETAINED)
    assert store.info(digest).retention is RetentionTier.RETAINED

    store.set_retention(digest, RetentionTier.PRUNABLE)
    assert store.info(digest).retention is RetentionTier.PRUNABLE
    store.set_retention(digest, RetentionTier.PERMANENT)
    with pytest.raises(RetentionPolicyError, match="PERMANENT"):
        store.set_retention(digest, RetentionTier.RETAINED)


def test_prune_cleans_residual_bytes_for_already_pruned_metadata(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"residue", tier=RetentionTier.PRUNABLE)
    metadata = store._content_metadata_path(digest)
    document = metadata.read_text(encoding="utf-8").replace(
        '"materialized":true', '"materialized":false'
    )
    metadata.write_text(document, encoding="utf-8")

    result = store.prune()

    assert result.pruned_hashes == (digest,)
    assert not store.object_path(digest).exists()


def test_prune_empty_store_and_already_absent_bytes_are_idempotent(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    assert store.prune().pruned_hashes == ()

    digest = store.put(b"gone", tier=RetentionTier.PRUNABLE)
    store.object_path(digest).unlink()
    assert store.prune().pruned_hashes == (digest,)
    assert store.info(digest).materialized is False
    assert store.prune().pruned_hashes == ()


def test_permanent_to_permanent_is_an_idempotent_explicit_transition(
    tmp_path: Path,
) -> None:
    store = ObjectStore.create(tmp_path / "store")
    digest = store.put(b"forever", tier=RetentionTier.PERMANENT)

    updated = store.set_retention(digest, RetentionTier.PERMANENT)

    assert updated.retention is RetentionTier.PERMANENT


def test_store_lifecycle_state_queries_and_paradoxical_guards(tmp_path: Path) -> None:
    from piginot.store import (
        ContentInfo,
        ContentLifecycleState,
        DivergentRecipeError,
        RecipeState,
        StoreFormatError,
    )

    store = ObjectStore.create(tmp_path / "store")
    missing = "0" * 64
    assert store.content_state(missing) is ContentLifecycleState.UNCOMMITTED
    assert store.recipe_state(missing) is RecipeState.UNRECORDED

    d_perm = store.put(b"perm", tier=RetentionTier.PERMANENT)
    d_ret = store.put(b"ret", tier=RetentionTier.RETAINED)
    d_prun = store.put(b"prun", tier=RetentionTier.PRUNABLE)
    assert store.content_state(d_perm) is ContentLifecycleState.PERMANENT_MATERIALIZED
    assert store.content_state(d_ret) is ContentLifecycleState.RETAINED_MATERIALIZED
    assert store.content_state(d_prun) is ContentLifecycleState.PRUNABLE_MATERIALIZED

    store.prune()
    assert store.content_state(d_prun) is ContentLifecycleState.PRUNABLE_PRUNED
    with pytest.raises(RetentionPolicyError, match="rehydration"):
        store.set_retention(d_prun, RetentionTier.PERMANENT)
    with pytest.raises(ValueError, match="unmaterialized"):
        ContentInfo(d_prun, 4, RetentionTier.RETAINED, materialized=False)

    r_hash = "1" * 64
    store.record_recipe_result(r_hash, d_perm)
    assert store.recipe_state(r_hash) is RecipeState.RESOLVED
    with pytest.raises(DivergentRecipeError):
        store.record_recipe_result(r_hash, d_ret)
    assert store.recipe_state(r_hash) is RecipeState.DIVERGENT

    # Corrupt paradoxical metadata rejected on read
    meta_path = store._content_metadata_path(d_prun)
    meta_path.write_text(
        meta_path.read_text(encoding="utf-8").replace('"PRUNABLE"', '"PERMANENT"'),
        encoding="utf-8",
    )
    with pytest.raises(StoreFormatError, match="inconsistent"):
        store.info(d_prun)
