# ADR 0029: Independence vectors behind every accuracy claim

- Status: Accepted
- Date: 2026-10-05
- Step: S17 (`G17.adrs`)

## Context

A verified claim is only as strong as the evidence that is independent of the
thing being verified. A surrogate checked against data it was selected on, or a
solver checked against itself, proves little. STEPS-v1.0 calls each source of
independent evidence an *independence vector* and names the S8 FEM
verification the strongest one v1.0 has. This ADR lists every vector the v1
claims rest on, and the ones that do not exist.

## Decision

The v1 claim ceiling is `HOLDOUT_VERIFIED`, and it rests on these vectors:

1. **Manufactured solutions for the FEM solver (S8).** Symbolic MMS cases
   (pure Dirichlet, mixed, traction with body force, strongly varying `F33`),
   convergence orders for P1 and P2, patch, objectivity, angular momentum,
   energy and reaction checks, and six seeded solver defects. This bounds
   implementation error of the reference solver independently of any
   surrogate.
2. **Discretisation error bounds (S8, S14).** Calibrated Richardson and
   recovery estimators with safety factors 1.97 and 2.61 enter every per-case
   error budget, so the reference's own error is charged against the
   tolerance instead of being assumed zero.
3. **Preregistration and a sealed HOLDOUT (S9, ADR 0009).** Roles, design and a
   blake2b commitment to the HOLDOUT coordinates are committed before any
   label; HOLDOUT is consumed once, after DEVELOPMENT, with a durable record
   written before the reveal is read.
4. **Evidence-role isolation (S10–S13).** Normalisation and selection read
   TRAIN only; the `NORMALIZE_ON_ALL_SPLITS`, `HOLDOUT_ACCESS` and
   `SELECT_ON_HOLDOUT` mutants show the guards are load-bearing.
5. **Mutation testing (S16).** Twenty source-level defects, one per gate, each
   detected by its named primary gate, with signatures compared per release.
6. **A second family through the same code (S17).** The centre-notched plate
   reaches its own HOLDOUT verdict through the framework used for the DogBone,
   with no framework branch on family.

## Not available in v1

- **An independent second solver.** `IndependentSolverAdapter` is shipped as a
  protocol and is unimplemented; every reference comes from one scikit-fem
  code path. The S8 cross-checks (P1 against P2, structured against
  unstructured meshes) bound discretisation error, not implementation error
  shared by both.
- **Physical experiments.** Every claim is a claim about the plane-stress
  Neo-Hookean model; no measurement validates the model itself.

## Consequences

- No bundle states a claim above `HOLDOUT_VERIFIED`, and
  `SEALED_RELEASE_VERIFIED` cannot be emitted (`G18.claim-ceiling`).
- `docs/limitations.md` states both missing vectors.

## Verification

- G8 (`artifacts/gates/S8.json`), G9 and G14 per campaign, the S16 signature
  baseline, and `G17.non-dogbone`.
