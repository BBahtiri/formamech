# ADR 0027: Soft Dirichlet enforcement is the default; hard lifting only where verified

- Status: Accepted — records the STEPS-v1.0 §S11 rule and how v1 applies it
- Date: 2026-10-05
- Step: S17 (`G17.adrs`)

## Context

Component-wise Dirichlet conditions can be imposed on a neural operator in two
ways. **Soft** enforcement adds a residual term to the objective and accepts
the condition to a declared tolerance. **Hard** enforcement builds the
condition into the output through a lifting, so it holds by construction — but
only where the lifting is correct, and an approximate-distance lifting can be
wrong at boundary points or parameter values that were never checked.

STEPS-v1.0 §S11 fixes the policy: soft enforcement is the default with a
declared residual tolerance; hard enforcement via a lifting only where it is
verified across boundary refinement and the full parameter range.

## Decision

1. **Default: soft.** `OperatorConfig.boundary_condition_mode` defaults to
   `soft`, with `soft_dirichlet_residual_tolerance = 1e-6`. The objective
   registry carries the `dirichlet` residual term (`dirichlet_residual`,
   nondimensional by the displacement scale).
2. **Hard lifting needs evidence.** `hard-lifting` is accepted only with a
   `LiftingVerification` record: at least two boundary refinement levels and
   the full parameter range verified, with the maximum boundary residual
   recorded. Without it the configuration is refused.
3. **The v1 operator uses a verified hard lifting.** The energy-form operator
   (ADR 0018) uses the exact affine axis-aligned tension lifting of the quarter
   model: `u_x = d (ξ + ξ(1 − ξ) φ_x)`, `u_y = d η φ_y`, with `d` the prescribed
   loaded-end displacement. Its verification is `G11.lifting`: the constrained
   lines at 320, 480 and 4,800 points per edge, for at least 64 parameter
   vectors (every box corner, the centre and seeded interior draws), relative
   error below 1e-12; and `G11.lifting-mutant`: an identity lifting must miss
   the conditions by more than 1e-3.
4. **Bundles support only verified liftings.** A bundle names its lifting kind;
   the inference closure supports `axis-aligned-tension` only and refuses any
   other kind (`SPEC_INVALID`). A family whose loaded boundary is not an
   axis-aligned line has no hard lifting in v1 and falls back to the soft
   default.

## Consequences

- Both v1 families (DogBone and the centre-notched plate) are quarter models
  with two symmetry lines and an axis-aligned loaded end, so both use the
  verified hard lifting; their free edges are natural conditions of the energy
  minimum.
- A new lifting kind requires a `G11.lifting` measurement over its full
  parameter box before a bundle may declare it.

## Verification

- `G11.lifting` and `G11.lifting-mutant` in `artifacts/gates/S11-ENERGY.json`
  (DogBone) and `artifacts/campaigns/notched-v1/gates/S11-ENERGY.json`
  (notched plate).
- The `IGNORE_DIRICHLET_LIFTING` mutant of the S16 suite fires `G11.lifting`.
- `tests/operators` refuses `hard-lifting` without a complete
  `LiftingVerification`.
