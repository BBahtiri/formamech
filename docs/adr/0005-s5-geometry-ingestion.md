# ADR 0005: S5 geometry ingestion contract

- Status: Accepted
- Date: 2026-08-22
- Step: S5

## Context

S5 needs a deterministic geometry boundary before FEM or training can start. The
boundary must admit meshes from external tools without allowing hidden repair,
orientation changes, unit inference, or provenance-dependent identity.

## Decision

S5 accepts explicitly unitized, planar Gmsh v4.1 meshes through `meshio` and a typed
`MeshPayload` boundary. The canonical representation is limited to homogeneous
linear 2D triangles plus line boundary facets. Coordinates are converted to canonical
millimetres using the supplied unit, with signed-zero normalization only.

Canonical identity includes nodes, triangle cells, directed boundary facets,
semantic boundary tags, region IDs, canonical unit, and counterclockwise orientation.
It excludes raw Gmsh IDs, source paths, adapter provenance, and derived quality
metrics. Boundary semantics are name-based and every canonical boundary facet must
belong to exactly one semantic tag. Region IDs must be positive named physical
surface groups.

The topology signature is versioned as
`planar-cell-complex-v1:components=N;holes=H`. Components are connected by shared
edges, not by corner contact. Query classification distinguishes `INTERIOR`,
`BOUNDARY`, and `EXTERIOR`; containment guards fail closed on exterior points.

## Consequences

- S5 performs ingestion and validation only. It does not construct, heal, defeature,
  remesh, infer units, run FEM, or touch training workflows.
- Mixed area elements, quads, high-order cells, 3D cells, nonzero z coordinates,
  duplicate nodes or cells, inverted cells, non-manifold edges, missing physical
  tags, ambiguous physical names, and unclassified boundary facets fail with stable
  reason codes.
- A family with one unique canonical mesh identity is fixed-degenerate and cannot be
  asserted as variable until multiple canonical mesh identities provide that
  evidence.
- CI must install the `geometry` optional dependency and enforce geometry coverage
  independently of the physics gate.
- The dependency lock was regenerated with `uv 0.12.5`. The verified geometry extra
  keeps NumPy below 2.3, so the unified lock resolves NumPy 2.2 for all optional
  consumers instead of retaining the prior unconstrained NumPy branches.
