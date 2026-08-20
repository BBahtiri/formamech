# ADR 0003: S3 content identity, retention, and crash visibility

- Status: Accepted
- Date: 2026-08-20
- Gate: G3

## Context

S3 needs a local content-addressed store for arbitrary bytes and deterministic
directory trees. S2 already owns the four public hash domains, including content,
but its canonical JSON encoder intentionally rejects raw bytes. The S3 brief also
requires process-kill-safe writes, recipe divergence detection, and pruning that
removes eligible bytes without erasing identity or lineage.

The brief does not define the raw-byte framing, whether an installed object or its
metadata is the visibility boundary, how empty directories participate in tree
identity, or how repeated writes interact with retention.

## Decision

Raw bytes use a two-stage, versioned identity. S3 computes BLAKE2b-256 over the
exact bytes, then hashes a canonical descriptor containing the format
piginot.raw-content.v1, raw digest, and byte count through S2's content domain.
Committed vectors freeze this framing. A store marker records the schema and hash
format, and opening an unmarked or incompatible directory fails closed.

Object bytes are installed before content metadata. The metadata record is the
visibility boundary: complete bytes left by a killed writer are an invisible
orphan until a later put verifies and adopts them. Every returned byte string is
rehashed. Metadata is strict JSON with exact fields, and its recorded identity must
match its addressed path.

All final files are written to a unique temporary file in the destination
directory, flushed, and installed with os.replace. The guarantee covers a process
killed before replacement or after object installation but before metadata commit.
It does not claim sudden-power-loss durability or concurrent-writer arbitration.

Tree identity is recursive. Each directory, including an empty directory, is a
canonical manifest of UTF-8-byte-sorted entries containing the exact name, kind,
and child content hash. Root names, timestamps, permissions, ownership, and
platform separators are excluded. Symlinks, reparse points, and non-regular
entries are rejected.

Recipe metadata stores a sorted set of result hashes. Adding a second distinct
result persists the union first and then raises DivergentRecipeError; resolution
never chooses one result from a divergent set.

Retention defaults to RETAINED. Repeated put calls can only promote a tier. An
explicit transition may move RETAINED to PRUNABLE, while PERMANENT is irreversible.
Pruning atomically records materialized=false before unlinking bytes. Recipe lineage
and content metadata survive, and a later put can rehydrate the same identity.

## Consequences

The domain-wrapped descriptor is more expensive than a direct raw digest, but it
keeps all public identities inside S2's declared domains and makes the byte format
explicitly versioned.

A killed process can leave temporary files or complete orphan bytes. They are not
API-visible and a retry is safe; S3 deliberately adds no quarantine, repair
service, background collector, lock manager, narrative cache, or cross-profile
compatibility policy.

Pruned tree-manifest bytes are unavailable like any other pruned content. S3
preserves recipe lineage, not an additional permanent tree traversal index.

## Verification

G3 binds the decision through:

- 1,000 deterministic random-byte round trips;
- fixed raw-byte hash vectors;
- byte-flip corruption detection on every read;
- child processes killed on both sides of object replacement, proving metadata
  controls visibility;
- persisted divergent recipe sets and fail-closed resolution;
- recursive tree tests including empty directories and forbidden entry types;
- pruning tests for all tiers, dry runs, residual cleanup, lineage, and
  rehydration;
- strict metadata, path, marker, CLI, lint, type, security, and coverage checks.

