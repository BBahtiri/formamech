# ADR 0019: S14 verification decisions — tolerance, slices, refusal fixtures, confidence

- Status: Accepted — ratified by the user on 2026-10-02
- Date: 2026-10-02
- Step: S14, before any implementation result or DEVELOPMENT access
- Plan: [`docs/campaigns/s14-verification-plan.md`](../campaigns/s14-verification-plan.md)

## Context

`CP-FREEZE` (2026-10-02) froze the S13c energy-form candidate (run
`0ca34e32…`) and authorised S14 to open DEVELOPMENT and to consume HOLDOUT once.
`STEPS-v1.0.md` §S14 leaves four quantities to the project: the product
tolerance of the error budget, the coverage slices, the composition of the
labelled out-of-distribution fixture set, and the confidence calibration. They
must be fixed before DEVELOPMENT is opened, so that no S14 threshold can be
chosen after seeing DEVELOPMENT or HOLDOUT errors.

## Decision

1. **Product tolerance (S14-D1).** On every coverage slice, the slice statistic
   of `surrogate discrepancy + conservative FEM discretisation bound +
   conservative projection error` must not exceed:

   | Quantity | Tolerance |
   | --- | ---: |
   | in-plane first-Piola stress, relative L2 over the case vertices | 5% |
   | absolute root-peak stress error (maximum Frobenius norm over the root vertices) | 5% |
   | displacement, relative L2 over the case vertices | 2% |

   The discretisation bound is the S8 calibrated estimator (ADR 0002) applied
   to each case; the projection error is the S10 projection artifact of each
   case. A missing component is `NOT_RUN`, never zero.

2. **Coverage slices (S14-D2).** 21 gated slices: all cases, plus the four S9
   strata bins of `L_total`, `W_grip`, `W_gauge`, `R_fillet` and nominal axial
   strain. The slice statistic is the 90th percentile (linear interpolation)
   when a slice holds at least 10 cases and the maximum otherwise. Slices on the
   four derived ratios are reported but not gated.

3. **Refusal fixtures (S14-D3).** 240 labelled out-of-distribution cases, each
   with its expected reason code: 60 primitive-bound violations (five inputs ×
   two sides × three distances × two draws), 48 coupled derived-bound
   violations, 24 material and load contract violations, 36 topology or tag
   violations (including second-family holed plates), 36 query-containment
   violations, and 36 guard-band near-misses just outside every bound.
   `G14.refusal` requires 100% refusal with the expected code on this set and
   at most 1% false refusal on the in-domain set: the 344 public S10 cases plus
   the 500-shape public membership sweep.

4. **Confidence (S14-D4).** Three variants of the equilibrium residual of the
   frozen candidate are preregistered: pointwise, element-patch mean, and patch
   mean times local mesh size. A TRAIN-only pilot on the 64 TRAIN-screen cases
   selects the variant with the largest Spearman correlation with the pointwise
   in-plane stress error, and that variant is frozen in
   `S14-PROTOCOL.json` before DEVELOPMENT is opened. `G14.confidence` is the
   Spearman correlation over all DEVELOPMENT vertices pooled (threshold 0.6).
   The calibration curve is an isotonic fit of the 90% error quantile against
   the indicator on DEVELOPMENT; its coverage is reported on HOLDOUT.

## Consequences

- The constitutive and angular-momentum residuals of the frozen candidate are
  identically zero, so the equilibrium residual is the only confidence signal.
  If it fails `G14.confidence`, the remedy is a new ADR, not a lower threshold.
- `G14.reducer` reduces over the twelve verification families of
  `IMPLEMENTATION-v1.0-OSS.md`. That file is not in the repository; until the
  user provides it, the reducer is built and tested against injected families,
  and the DogBone family list stays incomplete (`NOT_RUN`).
- The tolerance is tighter than the 15% invalidation threshold of ADR 0001.
  A slice that fails is reported; the candidate is not retrained in S14.

## Verification

`S14-PROTOCOL.json` must reproduce these values before DEVELOPMENT is opened,
and `artifacts/gates/S14.json` records the measured value of every check.
