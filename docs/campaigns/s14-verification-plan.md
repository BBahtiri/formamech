# S14 plan — verification, gates, envelope, confidence

- Status: **Decisions accepted** by the user on 2026-10-02 (S14-D1 … D4, see
  [ADR 0019](../adr/0019-s14-verification-decisions.md)). Steps 1 and 2 are
  done: the TRAIN rehearsal is in `S14-TRAIN-REHEARSAL.json`, and the user
  signed off the frozen `S14-PROTOCOL.json` and the methods of
  [ADR 0020](../adr/0020-s14-protocol-methods.md) on 2026-10-02
  (`S14-PROTOCOL-SIGNOFF.json`). Step 3 opens DEVELOPMENT; HOLDOUT is
  consumed only if DEVELOPMENT passes.
- Date: 2026-10-02
- Base: `codex/s13-closeout` at `6f2034f` (G13 PASS, `CP-FREEZE` written)
- Frozen candidate: S13c energy-form operator, run `0ca34e32…`, manifest
  `a978c3d6…` (`artifacts/campaigns/dogbone-v1/CP-FREEZE.json`)
- Gate: `G14` (`STEPS-v1.0.md` §S14); planned effort 3 weeks

## 1. What S14 must deliver

| Deliverable (STEPS §S14) | Gate check | Threshold |
| --- | --- | --- |
| Error budget per coverage slice: surrogate-to-FEM discrepancy + conservative discretisation error + conservative projection error ≤ product tolerance | `G14.budget` | arithmetic exact on synthetic cases with known components |
| Gate reducer over the twelve verification families; any required `FAIL`, `ERROR` or `NOT_RUN` gives `NOT_VERIFIED` | `G14.reducer` | 100% of injected cases |
| Report with required, passed, failed, errored, skipped and not-run sections, even when empty | `G14.report-sections` | a fully passing run still emits skipped and not-run |
| Envelope checker: family membership, coupled bounds, topology and tag identity, query containment, adversarial guard band, stable reason codes | `G14.refusal`, `G14.reason-codes` | 100% refusal on ≥ 200 labelled out-of-distribution cases; ≤ 1% false refusal in-domain; codes identical across runs |
| Per-query confidence: equilibrium and in-plane constitutive residual at the query points, calibration curve fitted on DEVELOPMENT | `G14.confidence` | Spearman(indicator, observed error) ≥ 0.6 on DEVELOPMENT |
| `G-LATENCY`: median and P99 inference latency, speed-up against the primary FEM on the same query set, one CPU thread | `G14.latency` | measured and reported; target ≥ 100× |
| Holdout reveal: consumption recorded before reveal | `G14.holdout-order` | a crash mid-reveal cannot un-consume |
| No certification wording | `G14.no-certified` | the string `certified` never appears in generated output |
| Coverage of the new verification package | `G14.coverage` | ≥ 85% |

Two facts shape the plan:

- **The confidence indicator has one live term.** The frozen candidate serves
  `P = P_NH(I + grad u)`, so its in-plane constitutive residual and its
  angular-momentum residual are identically zero. Equilibrium (`Div P` in the
  interior, `P N` on the traction-free edges) is the only informative residual.
- **The twelve verification families are not in the repository.**
  `G14.reducer` (and the twenty S16 mutants) are defined in
  `IMPLEMENTATION-v1.0-OSS.md`, which is not in this repository or in the
  connected folders. **Action for the user: provide that file** before step 1.3
  is built.

## 2. Decisions (accepted 2026-10-02, ADR 0019)

| # | Decision | Accepted value | Alternatives considered |
| --- | --- | --- | --- |
| S14-D1 | **Product tolerance** of the error budget | Stress field: in-plane `P` relative L2 ≤ **5%**; peak: absolute root-peak error ≤ **5%**; displacement relative L2 ≤ **2%**. Each is checked on the slice statistic (S14-D2) after adding the per-case conservative FEM discretisation bound (S8 calibrated estimator) and the S10 projection error | 3% / 3% / 1% (tighter; leaves about 1% for the surrogate after a 2% FEM bound) or 10% / 10% / 3% (looser) |
| S14-D2 | **Coverage slices** and slice statistic | 21 slices: all cases, plus four bins each of `L_total`, `W_grip`, `W_gauge`, `R_fillet` and nominal strain (the S9 strata). Statistic: 90th percentile when a slice holds ≥ 10 cases, the maximum otherwise. Derived-ratio slices are reported but not gated because several bins hold fewer than five DEVELOPMENT cases (for example `fillet_to_gauge_ratio` bin 3: 3 cases) | gate derived-ratio slices with merged bins; or use the maximum everywhere |
| S14-D3 | **Out-of-distribution refusal fixtures** (≥ 200, labelled with expected reason codes) | 240 cases: 60 primitive-bound violations (five inputs × two sides × three distances × two draws); 48 coupled derived-bound violations; 24 material and load contract violations; 36 topology or tag violations, including second-family holed plates; 36 query-containment violations; 36 guard-band near-misses just outside every bound. In-domain false-refusal set: the 344 public cases plus the 500-shape public sweep | fewer, larger-margin cases; or add HOLDOUT-like random draws from the envelope |
| S14-D4 | **Confidence indicator and calibration** | A TRAIN-only pilot on the 64 TRAIN-screen cases picks one of three preregistered variants of the equilibrium residual (pointwise; element-patch mean; patch mean × local mesh size) by Spearman correlation with the pointwise stress error. The chosen variant is frozen before DEVELOPMENT opens. `G14.confidence` is the Spearman correlation over all DEVELOPMENT vertices pooled. The calibration curve is an isotonic fit from indicator to the 90% error quantile on DEVELOPMENT; its coverage is then reported on HOLDOUT | per-case (not pooled) Spearman; a single fixed variant with no pilot |

The S1 decision record (ADR 0001) uses a 15% in-envelope stress error as its
invalidation threshold. The proposed 5% budget is deliberately tighter: the
frozen candidate scored 0.91% (field) and 1.69% (p90 peak) on the TRAIN screen,
and S8 bounds the FEM error at 2%.

## 3. Step 1 — Implement (no DEVELOPMENT, no HOLDOUT)

| Item | Location | Content |
| --- | --- | --- |
| 1.1 Error budget | `src/piginot/verification/budget.py` | per-case components, slice statistics, exact arithmetic, fail-closed on missing components |
| 1.2 Report | `src/piginot/verification/report.py` | required, passed, failed, errored, skipped and not-run sections; claim level; no certification wording |
| 1.3 Gate reducer | `src/piginot/verification/reducer.py` | the twelve families of `IMPLEMENTATION-v1.0-OSS.md`; `NOT_VERIFIED` on any required non-PASS |
| 1.4 Envelope checker | `src/piginot/verification/envelope.py` | executable envelope (`executable-envelope.json`), coupled bounds, topology signature and tag identity, query containment, guard band, stable reason codes |
| 1.5 Confidence | `src/piginot/verification/confidence.py` | equilibrium residual by automatic differentiation at query points; isotonic calibration |
| 1.6 Latency | `src/piginot/verification/latency.py` | single-thread timing harness for the operator (u only, and u with P) and for the primary solver |
| 1.7 Holdout reveal | `src/piginot/verification/holdout.py` + campaign runner | durable consumption record written and fsynced before the private reveal is read; refusal on replay |
| 1.8 Campaign glue | `campaigns/dogbone_v1/s14_*.py` | DogBone fixtures, slices, the frozen candidate loader (only from `CP-FREEZE.json`) |

## 4. Step 2 — Test and rehearse on TRAIN

- Unit tests for every gate check: exact budget arithmetic on synthetic
  cases, 100% of injected reducer cases, empty skipped/not-run sections,
  reason-code stability across two runs, the `certified` string scan, and a
  crash-injection test for the consumption record.
- A **TRAIN-only rehearsal** of the whole pipeline on the 64 TRAIN-screen
  cases: budget per slice, the confidence pilot of S14-D4, the refusal
  fixtures, and latency. TRAIN evidence is allowed and opens nothing.
- Then freeze `artifacts/campaigns/dogbone-v1/S14-PROTOCOL.json`: the
  tolerance, slices, statistic, chosen indicator variant, the fixture set
  hash and the evaluation code revision. **User sign-off on the frozen
  protocol before step 3.**

Implemented (`codex/s14-verification`):

| Module | Content |
| --- | --- |
| `campaigns/dogbone_v1/s14_reference.py` | per-case FEM reference error: recovery estimator and one half-resolution Richardson solve (ADR 0020 §1–2); the refinement check |
| `campaigns/dogbone_v1/s14_candidate.py` | `CP-FREEZE`-verified loader, role-gated serving of any `S14Case`, the unchanged S13c metrics, the three indicators with the boundary term |
| `campaigns/dogbone_v1/s14_rehearsal.py` | the TRAIN rehearsal: budget per slice, pilot, refusal run, latency, refinement check |
| `campaigns/dogbone_v1/s14_evaluation.py` | DEVELOPMENT access record; HOLDOUT consumption, verified reveal, S9 strata, S10 recipe |
| `campaigns/dogbone_v1/s14_gate.py` | G14 required and reported checks; claim level |
| `campaigns/dogbone_v1/s14_protocol.py` | protocol builder, evaluation-file digests, sign-off check |
| `campaigns/dogbone_v1/s14_validate.py` | step 3 phases `development`, `holdout`, `gate` |

Sign-off is recorded as `S14-PROTOCOL-SIGNOFF.json` (protocol digest, name,
date). `s14_validate` refuses every phase without it, or if any evaluation
file differs from its frozen digest.

TRAIN rehearsal (`artifacts/campaigns/dogbone-v1/S14-TRAIN-REHEARSAL.json`,
64 TRAIN-screen cases, code `a5ac1b1`):

| Item | Result |
| --- | --- |
| Budget, all cases (p90 of surrogate + FEM terms) | field 2.6%, peak 3.5%, displacement 0.30% |
| Budget, gated slices | 20 of 21 pass; `R_fillet:0` fails (9 cases, maximum statistic: field 6.3%, peak 11.0%), driven by `candidate-001561` |
| FEM reference terms (median) | field 1.04%, peak 1.56%, displacement 0.055% |
| Refinement check (4 cases, doubled resolution) | the terms exceed the extrapolated FEM error in every case (smallest factor 2.6) |
| Confidence pilot (pooled Spearman) | patch-mean 0.644 (selected), patch-mean x h 0.590, pointwise 0.472 |
| Refusal | 240/240 with the expected code; 0/844 false refusals; codes stable |
| Latency (one thread, 969 vertices) | 0.116 s median, 0.153 s p99 (u and P); FEM 138 s; speed-up 1,188x |

HOLDOUT is consumed only if the DEVELOPMENT budget and confidence pass
(ADR 0020, decision 4).

## 4a. Step 3 — DEVELOPMENT result (2026-10-02)

DEVELOPMENT was opened under the signed protocol
(`S14-DEVELOPMENT-ACCESS.json`, then `S14-DEVELOPMENT.json`):

| Check | Result |
| --- | --- |
| Budget, all cases (p90) | field 2.6%, peak 4.1%, displacement 0.21% (limits 5 / 5 / 2%) |
| Budget, gated slices | 61 of 63 slice checks pass. **FAIL**: peak on `W_grip:2` (p90 of 19 cases: 5.004%) and on `nominal_axial_strain:2` (maximum of 9 cases: 5.054%) |
| Surrogate peak error alone | p90 1.9%, maximum 2.7% |
| FEM reference terms in the peak total | median share 67% (discretisation median 1.2%, projection 0.45%) |
| G14.confidence (pooled Spearman, patch-mean, 62,016 vertices) | **0.641**, PASS (threshold 0.6) |
| Latency (8 cases, one thread) | 0.109 s median, 0.174 s p99; FEM 110 s; speed-up 1,009x |

By ADR 0020 decision 4 HOLDOUT stays unspent, because the DEVELOPMENT budget
failed. The failure is set by the conservative FEM reference bound rather
than by the surrogate: on the TRAIN refinement check the bound exceeded the
extrapolated FEM peak error by a factor of 15 or more.

## 5. Step 3 — Validate (opens DEVELOPMENT, then HOLDOUT once)

1. **Open DEVELOPMENT** (64 cases; their FEM results already exist in the S10
   store, and the selective source records the access durably). Score the
   frozen candidate exactly as on the TRAIN screen, fit the calibration curve,
   and evaluate `G14.confidence` and the budget per slice.
2. **Measure latency** on the DEVELOPMENT query sets against the primary
   solver on one thread. For scale, the second-family spike solved one
   1,100-vertex quarter mesh in about 135 s with P2 elements and 64 load steps.
3. **Holdout reveal**: write the consumption record, read the private reveal
   in `pi_ginot_private_reveals`, regenerate the 64 HOLDOUT geometries, solve
   them with the S10 solver policy (about 2.5 hours on one thread), and
   score the frozen candidate once. Report the calibration coverage and the
   budget per slice on HOLDOUT.
4. Write `artifacts/gates/S14.json`, the third ADR 0001 revisit and the
   limitations update.

Out of scope: retraining, changing the candidate or its conditioning,
threshold changes after DEVELOPMENT opens, and any claim above
`HOLDOUT_VERIFIED`.

## 6. Risks

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| DEVELOPMENT or HOLDOUT accuracy worse than the TRAIN screen | budget fails on some slices | the tolerance is fixed before opening; failure is reported, not tuned away |
| Equilibrium residual correlates weakly with the error | `G14.confidence` FAIL | TRAIN pilot before opening; a failure needs a new ADR, not a lowered threshold |
| Stress needs one forward and two reverse passes per query | latency target tighter | measured and reported regardless (STEPS) |
| `IMPLEMENTATION-v1.0-OSS.md` missing | reducer families unknown | user provides the file before item 1.3 |
| Family generality (ADR 0001 Risk 3) | S17 `G17.non-dogbone` | second-family spike (`experiments/holed_plate`, branch `codex/second-family-spike`): the unchanged operator handles a domain with a hole (peak error 2.35%, no folded point) but its stress-field error is 5.05% against a preregistered 5%; a follow-up spike is needed before S17 |
