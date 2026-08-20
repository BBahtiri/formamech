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
