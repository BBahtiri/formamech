# ADR 0104: Calibrated error bounds and envelope gating (S23)

- Status: Accepted — decisions 1–7 and the G23 thresholds accepted by the
  decision owner on 2026-10-13 (chat: "Ok accepted"), before any calibration
  or evaluation; recorded in `artifacts/bounds/S23-PROTOCOL.json`
- Date: 2026-10-12
- Step: S23 (gate `G23`), `STEPS-v2.0.md` §6; implementation plan Feature 3
- Builds on: ADR 0020 (S14 error budget), ADR 0101 (H1/H2, burned HOLDOUTs),
  ADR 0102 (certificate), ADR 0103 (serving modes)

## Context

The implementation plan defines S23 as split-conformal bounds whose score is
the S21 indicator, `|Q error| / eta_Q`. G21 then measured an effectivity of
1.000 on all 128 DEVELOPMENT cases: one Newton step from the prediction lands
on the converged discrete solution, so the certificate already *is* the
prediction's error against that solution. Bounds normalised by it would
restate the certificate.

Two kinds of error remain without a per-request number:

1. **Fast mode** (`.[runtime]`, no FEM installed): its only per-request signal
   is the S14 equilibrium-residual indicator, which ranks the stress error
   weakly (Spearman 0.19 DogBone, 0.48 notched on DEVELOPMENT, ADR 0102).
2. **The reference terms**: certified and exact compare against the FEM
   solution on the recipe mesh. Its discretisation and projection error, the
   S14 reference terms of ADR 0020, is not in the certificate (ADR 0102
   decision 2), yet it is about half of each notched case's budget (ADR 0101).

The calibration data already exist: `S14-DEVELOPMENT.json` of each family
holds, per DEVELOPMENT case and per headline quantity (`p_relative_l2`,
`absolute_peak_error`, `u_relative_l2`), the surrogate term and the
discretisation and projection components. The fresh sealed H1 sets (ADR 0101)
are the unbiased test.

## Decisions

1. **⚑ Scope.** S23 bounds the **S14 budgeted error** — per case and quantity,
   surrogate + discretisation + projection, the quantity the claim levels and
   the 5% / 5% / 2% tolerances apply to:
   - `fast`: an upper bound on the whole budget total;
   - `certified` and `exact`: an upper bound on the reference terms only
     (discretisation + projection), reported next to the certificate and never
     added into it, so the measured and the calibrated parts stay separate.

   The plan's certificate-normalised score is dropped (see Context).
2. **⚑ Method.** Unnormalised split conformal, per family and quantity, with
   `alpha = 0.1`. Calibration set: the 64 DEVELOPMENT cases with their
   recorded S14 terms; no new FEM. The bound is the `k`-th smallest score with
   `k = ceil((n + 1)(1 - alpha))` (`k = 59` for `n = 64`); if `k > n` no bound
   is issued. Normalised scores (the S14 indicator, the distance to TRAIN, a
   Mondrian split by the family's sharpness or slice strata) are reported on
   H1, not used. Exchangeability caveat: DEVELOPMENT was seen before this
   choice and informed the notched acceptance (ADR 0030); H1 is the unbiased
   test.
3. **⚑ Envelope gate.** A request inside the bundle's envelope (otherwise it
   is refused, as today) is `IN_CALIBRATION` if its Mahalanobis distance to
   TRAIN — TRAIN mean and covariance on the primitive axes that vary across
   TRAIN, as in ADR 0101 D7 — is at most the largest distance among the
   calibration cases; otherwise it is `OUT_OF_CALIBRATION`: the prediction is
   served unchanged, with no bound and a visible flag. Conformal guarantees do
   not extend beyond exchangeable queries.
4. **⚑ Coverage test on H1 with multiplicity control.** After calibration is
   frozen, a dedicated reveal command solves every H1 case with its family's
   S14 evaluation (S10 recipe, coarse level, recovery), writes the reveal
   artifact and burns H1; a second reveal is refused. Gated: for each family,
   the budget-total bound (fast) and the reference-terms bound (certified and
   exact) of `p_relative_l2` and `absolute_peak_error` — eight counts. Each
   passes if the number of covered in-calibration H1 cases is at least the
   lower `0.05 / 8` quantile of its exact beta-binomial distribution
   (Bonferroni; for `m = 64`, 48 of 64). The plan's per-check 5% quantile
   (52 of 64) would fail a valid procedure on at least one of eight checks
   with probability up to about 34%. `u_relative_l2` is reported.
5. **Runtime package.** The bounds code lives in a new package
   `piginot.bounds` (NumPy only), not in `piginot.certify` as the plan says,
   because `piginot.certify` imports the FEM stack and fast mode must keep
   working with `.[runtime]` alone.
6. **Calibration sidecar.** Each bundle's calibration is a canonical-JSON
   sidecar `bundles/calibration/<bundle>.json`, bound to the bundle's
   manifest hash like the S22 recipes, holding `alpha`, `n`, `k`, the bounds,
   the hash of the calibration case IDs and the gate statistics. Bundles stay
   byte-identical. Bounds are opt-in (`piginot predict --bounds`,
   `serve(..., calibration=...)`); without them every mode's output is
   unchanged.
7. **⚑ Widths.** Gated: the DogBone budget-total bounds are within the S14
   tolerances (stress field and peak 5%, displacement 2%), as its
   `HOLDOUT_VERIFIED` claim implies. Notched widths are reported only; its
   recorded DEVELOPMENT p90 totals already exceed 5% (5.24%, 5.34%), so its
   bounds will too.

## Consequences

- A fast-mode user gets, per in-calibration request, a 90% upper bound on the
  budgeted error at the tolerance scale; a certified or exact user gets the
  measured certificate plus a 90% bound on what the FEM mesh itself may add.
- An unnormalised bound is one number per family and quantity: it is valid
  for every in-calibration request, not adaptive to the request. Adaptive
  bounds need a FEM-free signal that ranks the error; none of the available
  ones does so well enough today.
- H1 is spent by S23; H2 stays sealed for S25.
- The H1 reveal costs one fine and one coarse S10-recipe solve per case:
  about 256 solves for both families on the reference machine.

## Verification

Gate `G23` (table in `STEPS-v2.0.md` once ratified): protocol first,
isolation from burned and sealed IDs, exact recomputation and determinism of
the calibration, the gate on constructed out-of-calibration requests, the
single H1 reveal, the eight coverage counts, the DogBone widths, unchanged
outputs without `--bounds`, a native Windows suite, and the quality floors.

Kill rule: if a coverage count fails on H1, record FAIL and do not retune on
H1; serve without bounds. Certificates and exact serving are unaffected.
