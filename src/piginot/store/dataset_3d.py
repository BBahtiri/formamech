"""Out-of-core sharded 3D mesh/field dataset with split-leakage lock & SHA-256 integrity (V4-5).

Stores multi-geometry 3D structural corpora on disk in append-only chunked
``.npz`` shards governed by an RFC 8785 canonical ``manifest.json``:

* **Lazy streaming:** :meth:`ShardedMeshDataset3D.iter_records` and
  :meth:`ShardedMeshDataset3D.iter_batches` load at most one shard at a time.
* **Tamper-evident integrity:** every shard file is SHA-256 hashed in the
  manifest; any byte modification triggers :class:`ShardIntegrityError`.
* **Zero-leakage split lock:** every appended record is fingerprinted by its
  mesh content hash; attempting to insert a mesh already present in another
  split (e.g., ``HOLDOUT`` vs ``TRAIN``) raises :class:`SplitLeakageError`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.contracts import canonical_json_bytes

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DATASET_3D_SCHEMA: Final = "formamech.dataset-3d-v1"


class DatasetSplit(StrEnum):
    """Strict disjoint dataset splits for ASME V&V 10 qualification hygiene."""

    TRAIN = "TRAIN"
    DEV = "DEV"
    HOLDOUT = "HOLDOUT"


class SplitLeakageError(ValueError):
    """Fail-closed refusal when a geometry leaks across TRAIN / DEV / HOLDOUT splits."""


class ShardIntegrityError(ValueError):
    """Fail-closed refusal when a shard file's SHA-256 disagrees with manifest.json."""


@dataclass(frozen=True, slots=True)
class ShardedRecord3D:
    """One 3D geometry + solution snapshot stored inside a sharded corpus."""

    record_id: str
    mesh_id: str
    split: DatasetSplit
    parameters: tuple[float, ...]
    nodes_mm: FloatArray  # (N, 3)
    cells: IntArray  # (E, 4 or 10)
    displacement_mm: FloatArray  # (N, 3)
    metadata: Mapping[str, Any]

    def geometry_fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint of ``(nodes_mm, cells)``."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.nodes_mm, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.cells, dtype=np.int64).tobytes())
        return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


class ShardedMeshDataset3D:
    """Append-only, tamper-evident, out-of-core sharded 3D structural dataset."""

    def __init__(
        self, root_dir: Path | str, *, max_records_per_shard: int = 16
    ) -> None:
        if type(max_records_per_shard) is not int or max_records_per_shard < 1:
            raise ValueError("max_records_per_shard must be >= 1")
        self.root_dir = Path(root_dir).resolve()
        self.shards_dir = self.root_dir / "shards"
        self.manifest_path = self.root_dir / "manifest.json"
        self.max_records_per_shard = max_records_per_shard
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.is_file():
            self._write_manifest(
                {
                    "schema": DATASET_3D_SCHEMA,
                    "max_records_per_shard": self.max_records_per_shard,
                    "shards": [],
                    "mesh_splits": {},
                    "record_ids": [],
                }
            )

    def _read_manifest(self) -> dict[str, Any]:
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema") != DATASET_3D_SCHEMA:
            raise ShardIntegrityError("invalid or corrupted dataset manifest schema")
        return raw

    def _write_manifest(self, doc: Mapping[str, Any]) -> None:
        self.manifest_path.write_bytes(canonical_json_bytes(doc))

    @property
    def manifest_sha256(self) -> str:
        return _sha256_file(self.manifest_path)

    def summary(self) -> dict[str, Any]:
        m = self._read_manifest()
        counts = {s.value: 0 for s in DatasetSplit}
        for shard in m["shards"]:
            for sp, cnt in shard["split_counts"].items():
                counts[str(sp)] = counts.get(str(sp), 0) + int(cnt)
        return {
            "schema": DATASET_3D_SCHEMA,
            "root_dir": str(self.root_dir),
            "shard_count": len(m["shards"]),
            "total_records": len(m["record_ids"]),
            "split_counts": counts,
            "manifest_sha256": self.manifest_sha256,
        }

    def verify_integrity(self) -> dict[str, Any]:
        """Re-hash every shard on disk and refuse on any missing/tampered byte."""
        m = self._read_manifest()
        for shard in m["shards"]:
            rel = str(shard["relpath"])
            p = self.root_dir / rel
            if not p.is_file():
                raise ShardIntegrityError(f"missing shard file {rel!r}")
            actual = _sha256_file(p)
            expected = str(shard["sha256"])
            if actual != expected:
                raise ShardIntegrityError(
                    f"shard {rel!r} SHA-256 mismatch: expected {expected}, got {actual}"
                )
        return {
            "ok": True,
            "shard_count": len(m["shards"]),
            "total_records": len(m["record_ids"]),
            "manifest_sha256": self.manifest_sha256,
        }

    def append_records(self, records: Sequence[ShardedRecord3D]) -> list[str]:
        """Append records into new immutable ``.npz`` shards with zero split leakage."""
        if not records:
            return []
        manifest = self._read_manifest()
        mesh_splits: dict[str, str] = dict(manifest["mesh_splits"])
        seen_ids: set[str] = set(str(r) for r in manifest["record_ids"])

        # Validate entire batch fail-closed BEFORE writing any shard
        staged_splits = dict(mesh_splits)
        staged_ids = set(seen_ids)
        for rec in records:
            if rec.record_id in staged_ids:
                raise ValueError(f"duplicate record_id {rec.record_id!r}")
            staged_ids.add(rec.record_id)
            fp = rec.geometry_fingerprint()
            prior = staged_splits.get(fp)
            if prior is not None and prior != rec.split.value:
                raise SplitLeakageError(
                    f"split leakage refused: geometry {rec.mesh_id!r} (sha={fp[:12]}) "
                    f"already locked in split {prior!r}, cannot add to {rec.split.value!r}"
                )
            staged_splits[fp] = rec.split.value

        written_shards: list[str] = []
        next_idx = len(manifest["shards"])
        chunk_sz = int(
            manifest.get("max_records_per_shard", self.max_records_per_shard)
        )

        for start in range(0, len(records), chunk_sz):
            batch = records[start : start + chunk_sz]
            shard_name = f"shard-{next_idx:04d}.npz"
            shard_path = self.shards_dir / shard_name
            next_idx += 1

            arrays: dict[str, Any] = {}
            index_rows: list[dict[str, Any]] = []
            split_counts: dict[str, int] = {}
            for i, rec in enumerate(batch):
                arrays[f"r{i}_nodes"] = np.asarray(rec.nodes_mm, dtype=np.float64)
                arrays[f"r{i}_cells"] = np.asarray(rec.cells, dtype=np.int64)
                arrays[f"r{i}_disp"] = np.asarray(rec.displacement_mm, dtype=np.float64)
                arrays[f"r{i}_params"] = np.asarray(rec.parameters, dtype=np.float64)
                split_counts[rec.split.value] = split_counts.get(rec.split.value, 0) + 1
                index_rows.append(
                    {
                        "slot": i,
                        "record_id": rec.record_id,
                        "mesh_id": rec.mesh_id,
                        "split": rec.split.value,
                        "geometry_sha256": rec.geometry_fingerprint(),
                        "metadata": dict(rec.metadata),
                    }
                )
            arrays["index_json"] = np.asarray(
                canonical_json_bytes(index_rows).decode("utf-8")
            )
            np.savez_compressed(shard_path, **arrays)

            relpath = f"shards/{shard_name}"
            digest = _sha256_file(shard_path)
            manifest["shards"].append(
                {
                    "relpath": relpath,
                    "sha256": digest,
                    "record_count": len(batch),
                    "split_counts": split_counts,
                }
            )
            written_shards.append(relpath)

        manifest["mesh_splits"] = staged_splits
        manifest["record_ids"] = sorted(staged_ids)
        self._write_manifest(manifest)
        return written_shards

    def iter_records(
        self,
        *,
        split: DatasetSplit | None = None,
        verify: bool = True,
    ) -> Iterator[ShardedRecord3D]:
        """Lazily stream records one shard at a time (O(shard) peak memory)."""
        manifest = self._read_manifest()
        for shard in manifest["shards"]:
            if split is not None and shard["split_counts"].get(split.value, 0) == 0:
                continue
            p = self.root_dir / str(shard["relpath"])
            if not p.is_file():
                raise ShardIntegrityError(f"missing shard {shard['relpath']!r}")
            if verify:
                actual = _sha256_file(p)
                if actual != str(shard["sha256"]):
                    raise ShardIntegrityError(
                        f"tampered shard {shard['relpath']!r}: SHA-256 mismatch"
                    )
            with np.load(p, allow_pickle=False) as zf:
                rows = json.loads(str(zf["index_json"].item()))
                for row in rows:
                    sp = DatasetSplit(str(row["split"]))
                    if split is not None and sp is not split:
                        continue
                    slot = int(row["slot"])
                    yield ShardedRecord3D(
                        record_id=str(row["record_id"]),
                        mesh_id=str(row["mesh_id"]),
                        split=sp,
                        parameters=tuple(
                            float(v) for v in zf[f"r{slot}_params"].tolist()
                        ),
                        nodes_mm=np.asarray(zf[f"r{slot}_nodes"], dtype=np.float64),
                        cells=np.asarray(zf[f"r{slot}_cells"], dtype=np.int64),
                        displacement_mm=np.asarray(
                            zf[f"r{slot}_disp"], dtype=np.float64
                        ),
                        metadata=dict(row.get("metadata", {})),
                    )

    def iter_batches(
        self,
        batch_size: int,
        *,
        split: DatasetSplit | None = None,
        verify: bool = True,
    ) -> Iterator[tuple[ShardedRecord3D, ...]]:
        """Yield fixed-size streaming batches lazily across shards."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        buf: list[ShardedRecord3D] = []
        for rec in self.iter_records(split=split, verify=verify):
            buf.append(rec)
            if len(buf) == batch_size:
                yield tuple(buf)
                buf.clear()
        if buf:
            yield tuple(buf)


__all__ = (
    "DATASET_3D_SCHEMA",
    "DatasetSplit",
    "ShardIntegrityError",
    "ShardedMeshDataset3D",
    "ShardedRecord3D",
    "SplitLeakageError",
)
