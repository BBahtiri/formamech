# ADR 0030: The decision owner accepts the notched DEVELOPMENT budget result

- Status: Accepted — decided by the user (decision owner) on 2026-10-06
- Date: 2026-10-06
- Step: S17, notched-plate S14 step 3, after DEVELOPMENT and before HOLDOUT
- Amends: the signed notched `S14-PROTOCOL.json` (digest `6d544d82…`) through
  `artifacts/campaigns/notched-v1/S14-PROTOCOL-AMENDMENT-1.json`; the
  HOLDOUT order of ADR 0020 decision 4 as applied to the notched campaign
- Precedent: [ADR 0021](0021-s14-development-budget-acceptance.md)

## Context

Under the signed protocol the frozen notched candidate was scored on the 64
DEVELOPMENT cases (`artifacts/campaigns/notched-v1/S14-DEVELOPMENT.json`):

- `G14.confidence` passes: pooled Spearman 0.625 against 0.6.
- The error budget passes 32 of its 51 gated slice checks. The all-cases p90
  is 5.24% for the stress field and 5.34% for the peak (tolerance 5%), and
  0.58% for the displacement (tolerance 2%). The largest exceedances are on
  the sharpest notches (`notch_radius:0`: stress field 7.78%, peak 8.73%) and
  the widest plates (`width:3`: 7.88%, 8.29%).
- About half of each case's budget is the FEM discretisation term of the
  reference mesh (median 1.22% stress field, 1.09% peak; maximum 4.79% and
  3.93%), next to surrogate medians of 1.72% and 1.01%.

By ADR 0020 decision 4 a failing DEVELOPMENT budget keeps HOLDOUT closed, and
the notched G14 first recorded FAIL with claim level `NOT_VERIFIED`. When the
result was reported, the decision owner answered: "The error is fine i want to
make this as a PASS", and chose to accept the DEVELOPMENT result and open
HOLDOUT, over recording the acceptance at G17 alone or setting a new
tolerance before HOLDOUT.

## Decision

1. **The DEVELOPMENT budget result is accepted as passing `G14.budget`.** The
   signed amendment waives exactly the 19 measured failing rows of the
   DEVELOPMENT result whose LF-normalised SHA-256 it names. Any other result,
   or a row that was not measured, is not covered.
2. **HOLDOUT opens once, with the rule unchanged.** The 64 HOLDOUT cases are
   consumed, solved with the unchanged S10 recipe and judged against the
   unchanged tolerances (5%, 5%, 2%), slices and p90 statistic. The HOLDOUT
   budget alone sets the reachable claim level: `HOLDOUT_VERIFIED` if it
   passes every gated slice, `DEVELOPMENT_VERIFIED` otherwise.
3. **Nothing else changes.** The evaluation code, the FEM reference terms, the
   confidence and latency procedures and the frozen candidate are those of the
   signed protocol; the amendment re-freezes the same file digests.
4. **The record keeps the computed result.** The G14 report states the
   computed FAIL beside the waiver, and every page that reports the notched
   DEVELOPMENT budget shows the measured numbers.

## Consequences

- HOLDOUT of the notched campaign is spent whatever its result.
- If the HOLDOUT budget fails, G14 passes with `DEVELOPMENT_VERIFIED`, the
  bundle serves `DEVELOPMENT_VERIFIED` and `G17.non-dogbone`, which asks for
  `HOLDOUT_VERIFIED`, still fails.
- The exceedances accepted here are larger than those of ADR 0021 (up to 3.7
  percentage points rather than 0.054). The notched accuracy statement rests on
  the HOLDOUT measurement, not on this waiver.
