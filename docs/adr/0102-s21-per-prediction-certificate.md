# ADR 0102: The per-prediction physics certificate (S21)

- Status: Proposed — technical decision; the G21 thresholds are ratified by
  the decision owner with the S21 protocol before any DEVELOPMENT scoring
- Date: 2026-10-11
- Step: S21 (gate `G21`), `STEPS-v2.0.md` §S21; implementation plan Feature 1a

## Context

Served predictions carry a calibrated pointwise stress bound from the S14
equilibrium-residual indicator (`piginot.verification.confidence`), which
covered 87–88% of HOLDOUT vertices. It measures how far the predicted field is
from satisfying equilibrium locally; it does not measure how far the
prediction is from the solution. S21 adds a per-case certificate of that
distance, computed with the reference solver's own discrete problem and no
retraining.

## Decision

1. **Estimator.** For a prediction, the surrogate displacement is evaluated at
   the displacement nodes of the case's S10 FEM space (P1 vertices, or P2
   vertices and mid-edge nodes); Dirichlet DOFs are set to their prescribed
   values and the largest deviation is reported. One Newton step at full load
   solves `K(u) d = -r(u)` with the backend's own assembly and constraint
   handling (`piginot.fem.linearization`). The certificate reports
   `eta = sqrt(d^T K(u) d)`, the energy drop `Pi(u) - Pi(u + d)` and its
   ratio to `eta^2 / 2`, and the relative L2 changes of displacement and first
   Piola stress along `d`, plus any QoI change a campaign registers.
2. **Scope.** `eta` estimates the distance to the converged discrete solution
   on the same mesh. The FEM discretization error is a separate term (the S14
   reference term, see ADR 0101), never folded into the certificate.
3. **Guards.** A certificate is `OK` only if the surrogate state and the Newton
   step are admissible (positive Jacobian, plane-stress closure), the step's
   curvature is positive, the energy decreases, and the free tangent is
   positive definite. Definiteness comes from the inertia of a symmetric-mode
   SuperLU factorization without off-diagonal pivoting (Sylvester); if SuperLU
   pivots anyway the inertia is unknown and the certificate is refused. With
   global (rigid-mode) constraints the tangent test is not applicable and the
   other guards decide. Otherwise the verdict is `INDEFINITE` or
   `INADMISSIBLE` and no estimate is issued.
4. **Reproducibility.** Certificates carry no wall times; their documents are
   byte-identical on reruns of the same profile.

## Evidence so far

On the S7 traction test problem (refined unit square, P1 and P2), with smooth
perturbations of the converged solution over two orders of magnitude, `eta`
equals the true energy-norm error within ±10% and the energy drop equals
`eta^2 / 2` within 10% (`tests/certify`).

## Consequences

- G21 measures, per family on DEVELOPMENT, the rank correlation of the
  certificate with the true per-case error, its effectivity, its cost against
  a cold FEM solve on the reference machine (ADR 0101 decision 3), and
  compares its ranking with the case-aggregated S14 indicator. The protocol
  and thresholds are fixed before scoring.
- `piginot.certify` and `piginot.fem.linearization` are family-free and keep
  the `.[runtime]` extra FEM-free: certification needs the FEM dependencies.
