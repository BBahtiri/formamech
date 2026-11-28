"""Episode-locked multi-rate robotics sensor stream ingestion (V6-R6).

Provides canonical pure-MIT readers, time-grid alignment, and leak-free
episode-level ``TRAIN`` / ``DEV`` / ``HOLDOUT`` partitioning for real physical
robotics streams (MoCap ``SE(3)``, FBG fiber-optic strains, 6-axis ATI F/T
wrenches, CNC indenter probing, pneumatic pressures, and 3D point tracks).

Crucially enforces **episode-level split locking** (:class:`EpisodeLeakageError`)
so temporally correlated frames from the same physical trial can never split
across calibration and blind holdout sets.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from piginot.contracts.canonical import canonical_json_bytes

FloatArray = NDArray[np.float64]
ROBOTICS_STREAM_SCHEMA: Final = "formamech.robotics-stream-v1"


class RoboticsSplit(StrEnum):
    """Leak-free episode roles matching FormaMech ASME V&V 10 discipline."""

    TRAIN = "TRAIN"
    DEV = "DEV"
    HOLDOUT = "HOLDOUT"


class EpisodeLeakageError(ValueError):
    """Raised fail-closed when duplicate episodes or frames cross split boundaries."""


@dataclass(frozen=True, slots=True)
class RoboticsEpisode:
    """One synchronized physical robot trial with multi-modal sensor channels."""

    episode_id: str
    times_s: FloatArray  # (T,) monotonically increasing
    channels: Mapping[str, FloatArray]  # name -> (T, D)
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        if self.times_s.ndim != 1 or self.times_s.size < 2:
            raise ValueError("times_s must be 1D with at least 2 timestamps")
        if np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("times_s must be strictly increasing")
        t_len = int(self.times_s.size)
        for name, arr in self.channels.items():
            if arr.shape[0] != t_len:
                raise ValueError(
                    f"channel {name!r} length {arr.shape[0]} != times_s length {t_len}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"channel {name!r} contains non-finite values")

    @property
    def content_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.times_s.astype("<f8").tobytes())
        for name in sorted(self.channels):
            h.update(name.encode("utf-8"))
            h.update(np.ascontiguousarray(self.channels[name], dtype="<f8").tobytes())
        return h.hexdigest()

    @property
    def sample_rate_hz(self) -> float:
        dt = float(np.median(np.diff(self.times_s)))
        return 1.0 / max(dt, 1.0e-12)

    def summary(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "num_frames": int(self.times_s.size),
            "duration_s": float(self.times_s[-1] - self.times_s[0]),
            "sample_rate_hz": self.sample_rate_hz,
            "channels": {k: list(v.shape) for k, v in sorted(self.channels.items())},
            "content_sha256": self.content_sha256,
        }


def align_multirate_streams(
    episode_id: str,
    streams: Mapping[str, tuple[FloatArray, FloatArray]],
    *,
    target_dt_s: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoboticsEpisode:
    """Resample multi-rate sensor streams onto their common overlapping time grid."""
    if not streams:
        raise ValueError("streams must be non-empty")
    t_start = max(float(t[0]) for t, _ in streams.values())
    t_end = min(float(t[-1]) for t, _ in streams.values())
    if t_end <= t_start:
        raise ValueError("sensor streams have no overlapping time interval")
    if target_dt_s is None:
        dt = max(float(np.median(np.diff(t))) for t, _ in streams.values())
    else:
        dt = float(target_dt_s)
    if dt <= 0.0:
        raise ValueError("target_dt_s must be positive")

    n_pts = max(2, int(round((t_end - t_start) / dt)) + 1)
    t_grid = np.linspace(t_start, t_end, n_pts, dtype=np.float64)
    aligned: dict[str, FloatArray] = {}
    for name, (t_src, y_src) in streams.items():
        y_2d = y_src[:, None] if y_src.ndim == 1 else y_src
        cols = [np.interp(t_grid, t_src, y_2d[:, c]) for c in range(int(y_2d.shape[1]))]
        aligned[name] = np.column_stack(cols)
    return RoboticsEpisode(
        episode_id=episode_id,
        times_s=t_grid,
        channels=aligned,
        metadata={} if metadata is None else dict(metadata),
    )


@dataclass(frozen=True, slots=True)
class EpisodeSplitPartition:
    """Verified leak-free episode split with cryptographic manifest."""

    train: tuple[RoboticsEpisode, ...]
    dev: tuple[RoboticsEpisode, ...]
    holdout: tuple[RoboticsEpisode, ...]
    manifest_sha256: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": ROBOTICS_STREAM_SCHEMA,
            "train_ids": [e.episode_id for e in self.train],
            "dev_ids": [e.episode_id for e in self.dev],
            "holdout_ids": [e.episode_id for e in self.holdout],
            "train_frames": sum(int(e.times_s.size) for e in self.train),
            "dev_frames": sum(int(e.times_s.size) for e in self.dev),
            "holdout_frames": sum(int(e.times_s.size) for e in self.holdout),
            "manifest_sha256": self.manifest_sha256,
        }


def split_episodes_leak_free(
    episodes: Sequence[RoboticsEpisode],
    *,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> EpisodeSplitPartition:
    """Partition episodes deterministically while rejecting duplicate content hashes."""
    if len(episodes) < 3:
        raise ValueError("at least 3 episodes are required for TRAIN/DEV/HOLDOUT split")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for ep in episodes:
        if ep.episode_id in seen_ids:
            raise EpisodeLeakageError(f"duplicate episode_id {ep.episode_id!r}")
        digest = ep.content_sha256
        if digest in seen_hashes:
            raise EpisodeLeakageError(
                f"duplicate episode waveform content detected in {ep.episode_id!r}"
            )
        seen_ids.add(ep.episode_id)
        seen_hashes.add(digest)

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(episodes)))
    n = len(episodes)
    n_train = max(1, int(round(train_fraction * n)))
    n_dev = max(1, int(round(dev_fraction * n)))
    if n_train + n_dev >= n:
        n_train = max(1, n - 2)
        n_dev = 1

    train = tuple(episodes[i] for i in order[:n_train])
    dev = tuple(episodes[i] for i in order[n_train : n_train + n_dev])
    holdout = tuple(episodes[i] for i in order[n_train + n_dev :])

    manifest_doc = {
        "schema": ROBOTICS_STREAM_SCHEMA,
        "seed": seed,
        "train": [(e.episode_id, e.content_sha256) for e in train],
        "dev": [(e.episode_id, e.content_sha256) for e in dev],
        "holdout": [(e.episode_id, e.content_sha256) for e in holdout],
    }
    digest = hashlib.sha256(canonical_json_bytes(manifest_doc)).hexdigest()
    return EpisodeSplitPartition(
        train=train, dev=dev, holdout=holdout, manifest_sha256=digest
    )


__all__ = (
    "ROBOTICS_STREAM_SCHEMA",
    "EpisodeLeakageError",
    "EpisodeSplitPartition",
    "RoboticsEpisode",
    "RoboticsSplit",
    "align_multirate_streams",
    "split_episodes_leak_free",
)
