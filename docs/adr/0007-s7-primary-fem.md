# ADR 0007: S7 primary FEM solver boundary

- Status: Accepted
- Date: 2026-08-22
- Step: S7

## Context

S7 introduces the first production FEM boundary after S4's plane-stress physics and
S5/S6's canonical geometry. The solver must preserve every attempted case, including
failures, because downstream sampling and training stages need visible failed FEM
records rather than silently filtered datasets. ADR 0001 also requires peak stress to
come from a recovered field, with P2 or a declared graded mesh, rather than from
element-averaged stress.

## Decision

S7 adds immutable `FemRequest`, `FemResult`, and `FemAttempt` contracts plus a
runtime-checkable `FemBackend` protocol. A request hashes under the S2 `recipe`
domain and closes over the canonical mesh, material, BC/load/QoI specs, solver
configuration, registered body-force provider identities, and declared
`ExecutionProfile`. Successful results hash under the S2 `content` domain and may be
associated with their recipe in the S3 object store. Failed attempts retain the same
recipe hash and one stable `FemFailureCode`.

The first implementation is `ScikitFemBackend`. It builds scikit-fem P1 or P2
triangular vector bases and solves 2D finite-strain, Total-Lagrangian plane stress by
sparse Newton iterations with load continuation and residual-reducing line search.
At every quadrature point, the S4 compressible Neo-Hookean kernel solves `P33 = 0`.
The assembled tangent is the analytic S4 material tangent condensed by the exact
plane-stress Schur complement; it is not a finite-difference tangent.

All S7 geometry is in mm and the 2D model has a fixed out-of-plane reference
thickness of 1 mm. Essential values are displacements in mm. Dead nominal tractions
are first-Piola boundary tractions in MPa on the reference boundary. Registered body
forces return reference-volume force density in N/mm^3; provider content identity is
part of the request recipe. Component-wise Dirichlet, axis-aligned implicit symmetry,
explicit free boundaries, and rigid-body-mode rank checking are bound from semantic
boundary tags.

The result contains displacement at every scalar P1/P2 DOF, quadrature coordinates
and weights, full 3-by-3 deformation gradients and first-Piola stresses, constrained
reaction resultants grouped by BC name, strain and potential energies, dead-load
work, convergence history, and supported QoIs. Stress is also L2-projected to the
scalar P1/P2 basis. Screened `maximum` and `pointwise_peak` reductions use recovered
DOFs on their named boundary tag; a re-entrant corner or mixed essential/natural BC
junction is rejected. P1 peak requests also require an explicit `qoi_graded_mesh`
declaration. Unsupported QoI fields, reductions, or units produce a visible
`UNSUPPORTED_REQUEST` attempt.

## Consequences

- `J <= 0`, plane-stress closure failure, constraint rank loss, Newton
  non-convergence, resource exhaustion, and unsafe or under-resolved peak QoIs are
  distinct visible outcomes. A failing registered body-force provider is also a
  visible unsupported request rather than a dropped batch.
- Mesh identity, solver settings, body-force implementation identity, BLAS thread
  count, and declared library versions participate in recipe identity.
- Reactions and dead-load work use the 1 mm thickness convention. `external_work`
  does not reconstruct work done along a prescribed-displacement path, so it is not
  by itself an energy-balance certificate for displacement-controlled problems.
- Recovered point peaks are a declared numerical estimator, not a mathematical or
  experimental guarantee of the physical maximum stress.
- S7 establishes a real primary-solver implementation and its contract gate. It is
  not independent verification. S8 must still add manufactured solutions, patch and
  convergence tests, error estimation, energy-balance evidence, and verification on
  non-affine fields before downstream accuracy claims can rely on this solver.
- `G7` remains failed until the exact branch head passes the required Linux/Windows
  and Python 3.11/3.12 CI matrix and that evidence is recorded separately.
