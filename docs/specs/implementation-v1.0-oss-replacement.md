# IMPLEMENTATION-v1.0-OSS — project replacement for §1.12 and §1.13

- Status: Adopted by the decision owner on 2026-10-03 ([ADR 0022](../adr/0022-replace-implementation-v1.0-oss.md))
- Replaces: the two sections of `IMPLEMENTATION-v1.0-OSS.md` that `STEPS-v1.0.md`
  cites and that are not available to the project:
  - "the twelve families" of the `G14.reducer` (STEPS §S14);
  - "the twenty mutants from `IMPLEMENTATION-v1.0-OSS.md` §1.13" (STEPS §S16).
- Does not replace: the strategy and rationale of the original document, which
  `STEPS-v1.0.md` already binds where it matters.

This document is written by the project. It is not a reconstruction of the
original text, and nothing in it claims to reproduce the original wording.

## §1.12 The twelve verification families

A verification family is one body of evidence that the claim of a bundle
depends on. Each family is the gate artifact of one upstream step. The
families follow the dependency order of `STEPS-v1.0.md`, so none of them was
chosen from an S14 result.

| ID | Family | Gate | Evidence artifact | What it guarantees for a bundle |
| --- | --- | --- | --- | --- |
| F01 | contracts-identity | G2 | `artifacts/gates/S2.json` | canonical identity, units and immutability of every spec |
| F02 | content-store | G3 | `artifacts/gates/S3.json` | round-trip, corruption detection and retention of stored bytes |
| F03 | physics-kernel | G4 | `artifacts/gates/S4.json` | the constitutive law, its tangent and the plane-stress closure |
| F04 | geometry-tags | G5 | `artifacts/gates/S5.json` | mesh ingestion, topology and semantic tags |
| F05 | morphing-membership | G6 | `artifacts/gates/S6.json` | the family map, its validity and the membership predicate |
| F06 | primary-fem | G7 | `artifacts/gates/S7.json` | the reference solver, its outcomes and its recipes |
| F07 | fem-verification | G8 | `artifacts/gates/S8.json` | convergence, physical invariants and the calibrated estimators |
| F08 | doe-preregistration | G9 | `artifacts/gates/S9.json` | evidence roles, leakage groups and the holdout commitment |
| F09 | dataset-materialisation | G10 | `artifacts/gates/S10.json` | traceability, projection error and normalisation |
| F10 | operator-architecture | G11 | the G11 artifact of the frozen candidate's architecture (`S11-ENERGY.json` for the energy-form operator, ADR 0018) | node order, batching, reachability, safe weights, heads and boundary conditions |
| F11 | objectives-gradients | G12 | `artifacts/gates/S12.json` | residuals, gradients and role isolation of the objectives |
| F12 | training-selection | G13 | `artifacts/gates/S13.json` | determinism, resume, identity, lineage and preregistered selection |

### Reduction rule

1. The checks of a family are the checks recorded in its artifact. Universal
   `G-CODE` checks are owned by the family whose artifact records them, so
   the same check ID in two artifacts counts as two checks.
2. A family's status is the worst status of its checks
   (`ERROR` > `FAIL` > `NOT_RUN` > `SKIPPED` > `PASS`).
3. A missing artifact makes the family `NOT_RUN`. An unreadable artifact, a
   wrong gate ID, or a `PASS` verdict over a non-passing check makes it
   `ERROR`.
4. All twelve families are required. The claim of a bundle is `NOT_VERIFIED`
   unless every family is `PASS` and every required G14 check is `PASS`.
5. `G14.reducer` passes only if the reducer gives `NOT_VERIFIED` for every
   injected case: each family set to `FAIL`, `ERROR` and `NOT_RUN` in turn,
   and each family with one recorded check removed. The all-`PASS` control
   must reach the evidence level.

## §1.13 The twenty mutants

Each mutant is a deliberate defect with one primary gate that must detect it
(`G16.primary`). Mutants 1–15 already exist as gate checks; 16–20 are added by
S14 and S15.

| # | Mutant | Primary gate | Defect |
| ---: | --- | --- | --- |
| 1 | `BODY_FORCE_SIGN` | `G8.seeded-defects` | body force applied with the wrong sign |
| 2 | `BOUNDARY_NORMAL` | `G8.seeded-defects` | traction applied along the inward normal |
| 3 | `TAG_SWAP` | `G8.seeded-defects` | two boundary tags exchanged |
| 4 | `CLOSURE_WRONG_ROOT` | `G8.seeded-defects` | plane-stress closure converges to the wrong root |
| 5 | `INCONSISTENT_TANGENT` | `G8.seeded-defects` | Newton tangent inconsistent with the residual |
| 6 | `ANISOTROPIC_SCALING` | `G8.seeded-defects` | coordinates scaled differently per axis |
| 7 | `NODE_SPLIT` | `G9.no-node-split` | one leakage group split across evidence roles |
| 8 | `DROP_PROJECTION_ERROR` | `G10.projection-error` | projected field shipped without its error envelope |
| 9 | `STRESS_NODE_PAIRING` | `G10.cross-mesh` | stress compared across meshes by node index |
| 10 | `NORMALIZE_ON_ALL_SPLITS` | `G10.normalization` | normalisation statistics computed on non-TRAIN cases |
| 11 | `IGNORE_DIRICHLET_LIFTING` | `G11.lifting-mutant` | essential conditions not imposed by the lifting |
| 12 | `COORDINATE_ANISOTROPY` | `G12.anisotropy` | residual coordinates scaled per axis |
| 13 | `HOLDOUT_ACCESS` | `G12.role-isolation` | an objective reads HOLDOUT evidence |
| 14 | `SELECT_ON_HOLDOUT` | `G13.holdout` | checkpoint selection reads HOLDOUT |
| 15 | `CHOOSE_NONRANKED_CHECKPOINT` | `G13.selection` | a checkpoint outside the preregistered rule is chosen |
| 16 | `ZERO_MISSING_BUDGET_COMPONENT` | `G14.budget` | a missing budget component is counted as zero |
| 17 | `PASS_ON_NOT_RUN` | `G14.reducer` | the reducer treats `NOT_RUN` as `PASS` |
| 18 | `REVEAL_BEFORE_CONSUME` | `G14.holdout-order` | the holdout is read before its consumption record exists |
| 19 | `SERVE_P_MATERIAL` | `G15.substitution` | `P_material` served in place of `P_served` |
| 20 | `UNDETECTED_BUNDLE_TAMPER` | `G15.manifest` | a bundle file changes without the manifest detecting it |

For each mutant S16 records the full sorted set of gates that fire as its
signature baseline (`G16.signatures`).
