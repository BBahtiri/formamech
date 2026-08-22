# ADR 0007: S7 primary FEM solver boundary

- Status: Accepted
- Date: 2026-08-22
- Step: S7

## Context

S7 introduces the first production FEM boundary after S4's plane-stress physics and
S5/S6's canonical geometry. The solver must preserve every attempted case, including
failures, because downstream sampling and training stages need visible failed FEM
records rather than silently filtered datasets.

## Decision

S7 adds immutable `FemRequest`, `FemResult`, and `FemAttempt` contracts. Each request
hashes under the S2 `recipe` domain and closes over the canonical mesh, material,
BC/load/QoI specs, solver configuration, and `ExecutionProfile`. Each successful
result hashes under the S2 `content` domain. Failed attempts record the same recipe
hash and one stable `FemFailureCode`.

The first backend is `ScikitFemBackend`. It uses scikit-fem's triangular vector basis
construction for P1/P2 DOF accounting and uses PI-GINOT's S4 compressible
Neo-Hookean plane-stress closure for constitutive evaluation. The S7 gate exercises
affine finite-strain states, where the constant-strain triangle field is exact. The
backend therefore implements the exact affine path now and keeps broader nonlinear
assembly and non-affine load continuation as S8/S9 hardening work.

The backend binds component-wise Dirichlet and symmetry conditions from semantic
boundary tags, accepts dead nominal traction and body-force load specs, returns
displacement, quadrature-indexed first-Piola stress and deformation gradients,
reactions, energies, convergence history, QoIs, and all attempts.

The QoI regularity screen rejects pointwise peak-stress requests whose target region
touches a re-entrant boundary vertex. Such QoIs must later be replaced by approved
averaged or masked quantities before they can enter training or certification.

## Consequences

- `J <= 0`, closure setup failure, constraint rank loss, declared Newton
  non-convergence, resource exhaustion, and unsafe peak-stress QoIs are distinct
  visible outcomes.
- Mesh identity, solver tolerance, element order, BLAS thread count, and library
  version all participate in recipe identity through the request and execution
  profile.
- Byte-identical retries reuse the recipe hash; any changed input creates a new
  recipe hash.
- S7 is a primary solver boundary and gate harness, not yet the independent
  verification proof. S8 must still add MMS, patch tests, convergence studies, error
  estimators, and stronger evidence for non-affine load cases.
