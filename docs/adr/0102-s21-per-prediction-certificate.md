# ADR 0102: The per-prediction physics certificate (S21)

- Status: Accepted — G21 PASS on 2026-10-11 (`artifacts/gates/S21.json`);
  thresholds ratified by the decision owner in `artifacts/certify/S21-PROTOCOL.json`
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

## Result (G21, reference machine, 64 DEVELOPMENT cases per family)

| Check | DogBone | Notched |
|---|---|---|
| Spearman(eta, true energy error), threshold 0.8 | 1.000 | 1.000 |
| Effectivity in [0.5, 2], threshold 90% of cases | 100% (median 1.000) | 100% (median 1.000) |
| Spearman(stress change, true stress error), threshold 0.7 | 1.000 | 1.000 |
| Same ranking by the S14 indicator | 0.19 | 0.48 |
| Certificate cost / cold FEM solve, threshold 10% | 0.52% | 0.57% |

All 128 certificates are `OK`; the recomputed S10 references reproduce the
S14 coarse terms to 9e-11 and 9e-13; a re-issued certificate is
byte-identical. The surrogate is close enough to the discrete solution that
one Newton step lands on it: the certificate measures each prediction's
discrete error essentially exactly. G21 first failed its determinism check
because the gate re-issued with eight torch threads and the evaluation with
one (summation order, 4.4e-16); the gate now re-issues under the protocol's
conditions.

## Consequences

- G21 measures, per family on DEVELOPMENT, the rank correlation of the
  certificate with the true per-case error, its effectivity, its cost against
  a cold FEM solve on the reference machine (ADR 0101 decision 3), and
  compares its ranking with the case-aggregated S14 indicator. The protocol
  and thresholds are fixed before scoring.
- `piginot.certify` and `piginot.fem.linearization` are family-free and keep
  the `.[runtime]` extra FEM-free: certification needs the FEM dependencies.
