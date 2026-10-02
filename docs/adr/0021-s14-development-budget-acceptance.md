# ADR 0021: The decision owner accepts the S14 DEVELOPMENT budget result

- Status: Accepted — decided by the user (decision owner) on 2026-10-02
- Date: 2026-10-02
- Step: S14 step 3, after DEVELOPMENT and before HOLDOUT
- Amends: the signed `S14-PROTOCOL.json` (digest `99c27839…`) through
  `S14-PROTOCOL-AMENDMENT-1.json`; ADR 0020 decision 4

## Context

Under the signed protocol the frozen S13c candidate was scored on the 64
DEVELOPMENT cases (`S14-DEVELOPMENT.json`):

- `G14.confidence` passes: pooled Spearman 0.641 against 0.6.
- The error budget passes 61 of its 63 gated slice checks. Two peak checks
  exceed the 5% tolerance: `W_grip:2` (p90 of 19 cases, 5.004%) and
  `nominal_axial_strain:2` (maximum of 9 cases, 5.054%).
- The surrogate's own peak error on DEVELOPMENT is at most 2.72% (p90 1.92%).
  The per-case peak budget is about two thirds FEM reference terms
  (discretisation median 1.21%, projection median 0.45%), and on the TRAIN
  refinement check those terms exceeded the extrapolated FEM peak error by a
  factor of 15 or more.

By ADR 0020 decision 4, a failing DEVELOPMENT budget keeps HOLDOUT closed.

## Decision

The decision owner rules that the DEVELOPMENT budget result is a success: the
two measured exceedances (0.004 and 0.054 percentage points) are accepted as
passing `G14.budget`, because they come from the conservative FEM reference
bound and not from the surrogate.

The ruling is recorded, not hidden:

1. `S14-PROTOCOL-AMENDMENT-1.json`, signed by the decision owner, holds a
   waiver that names the exact DEVELOPMENT result (its SHA-256) and exactly
   the two failing rows with their measured statistics. A waiver that does not
   match the measured failures, or that would cover a row that was not
   measured, is ignored.
2. `G14.budget` is reported as passed with the reason "accepted by the
   decision owner (ADR 0021)" and the measured value "FAIL as computed". The
   slice table in `S14-DEVELOPMENT.json` is unchanged.
3. HOLDOUT may now be consumed once. The HOLDOUT budget is judged with the
   unchanged tolerances (5% / 5% / 2%), slices and statistic; this decision
   does not extend to HOLDOUT.
4. The amendment re-freezes the evaluation files that implement the waiver
   (`s14_validate.py`, `s14_gate.py`, `s14_protocol.py`); the scoring code is
   unchanged.

## Consequences

- The DEVELOPMENT budget was accepted after seeing DEVELOPMENT results. The
  claim built on it therefore rests on the decision owner's judgement for those
  two slices; the S14 report says so next to the check.
- HOLDOUT remains an untouched test of the same frozen candidate under the
  unchanged rule. If its budget fails, the reachable claim is
  `DEVELOPMENT_VERIFIED`.
- A tighter FEM reference (for example, references at doubled resolution) is
  the way to reduce the reference share of the budget in a later campaign.

## Verification

- `tests/campaigns/test_dogbone_s14_protocol.py`: amendments must name the
  base protocol and be signed by the decision owner; a waiver covers exactly
  the measured failing rows; HOLDOUT is blocked without it.
- `tests/campaigns/test_dogbone_s14_gate.py`: a waived budget passes with its
  reason and computed result; an unwaived or unmeasured one does not.
