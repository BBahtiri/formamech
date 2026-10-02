# ADR 0020: S14 protocol methods — FEM reference error, indicator, access and freeze

- Status: Accepted — signed off by the user on 2026-10-02 together with
  `artifacts/campaigns/dogbone-v1/S14-PROTOCOL.json`
  (`S14-PROTOCOL-SIGNOFF.json`, protocol digest `99c27839…`)
- Date: 2026-10-02
- Step: S14 step 2 (TRAIN rehearsal), before any DEVELOPMENT access
- Refines: ADR 0019 (S14-D1 projection component, S14-D4 residual definition)

## Context

ADR 0019 fixed the S14 tolerances, slices, refusal fixtures and the confidence
pilot. Running the pipeline on the 64 TRAIN-screen cases needed five method
choices that ADR 0019 leaves open or states in terms that do not fit the
S14 comparison. All five were made on TRAIN evidence only.

1. **Discretisation bound per case.** ADR 0019 names "the S8 calibrated
   estimator". S8 calibrated two (ADR 0002): the recovered-stress estimator
   (factor 2.61, single mesh, stress only) and Richardson extrapolation
   (factor 1.97, two meshes). Displacement has no single-mesh S8 estimator.
2. **Projection error.** ADR 0019 names "the S10 projection artifact of each
   case". That artifact bounds the cell-mean projection of the raw stress to
   the 15 common normalisation query points of the S10 policy. The S13c and
   S14 comparison does not use that projection: it compares at the mesh
   vertices with the P2 displacement degrees of freedom and the recovered P2
   stress.
3. **Equilibrium residual.** The S14 plan defines it as `Div P` in the
   interior and `P N` on the traction-free edges. The step-1 code evaluated
   only the interior term.
4. **Populations.** ADR 0019 does not say which population gates the budget.
5. **Freeze.** The plan requires the protocol to name the evaluation code.

## Decision

1. **FEM reference error (discretisation component).** Both S8 estimators are
   applied to every case, and the larger bound is used:
   - recovery: `2.61 eta`, with `eta = |P_recovered - P_raw| / |P_raw|` in the
     quadrature L2 norm of the in-plane block, over all cells (`eta_all`) for
     the stress field and over the cells incident to a root vertex
     (`eta_root`) for the peak;
   - Richardson: `1.97 d`, with `d` the relative difference between the S10
     solve and one solve of the same geometry on the half-resolution level
     mesh (18 + 10 columns, 8 rows; every coarse vertex is an S10 vertex),
     taken at the coarse vertices, and the conservative order 1.

   Per quantity: stress field `max(2.61 eta_all, 1.97 d_P)`; root peak
   `max(2.61 eta_root, 1.97 d_peak)`; displacement `1.97 d_u`.

   The order is 1, not the theoretical P2 orders 3 (displacement) and 2
   (stress), because the TRAIN refinement check (one doubled-resolution solve
   for each of four cases) shows the S10 meshes are not in the asymptotic
   range: the apparent orders at the coarse vertices were 2.0 for the
   displacement, 0.45–1.7 for the stress field and 1.4–1.6 or non-monotone
   for the peak. With the theoretical orders the Richardson term alone would
   have under-estimated the peak error of one of the four cases. On TRAIN
   the Richardson term sets the stress-field bound in all 64 cases and the
   recovery term sets the peak bound in 59.

2. **Projection component.** The transfer from the solved field to the
   comparison points: `eta_all` for the stress field and `eta_root` for the
   peak (the recovered field differs from the raw stress by `eta`, so with
   the discretisation bound of the raw stress this is the triangle
   inequality), and zero for the displacement, whose vertex values are P2
   degrees of freedom (checked to sit on the vertices). The S10 transfer
   artifact is not used, because it describes a projection the S14 comparison
   does not make.

3. **Equilibrium residual.** All three preregistered variants add the
   natural-boundary residual `t = |(P N)_free| / E`, averaged over the
   boundary facets incident to a vertex: both traction components on
   `gauge_top` and `right_arc`, the `x` component on `bottom`, the `y`
   component on `left_symmetry` and `right_grip`. The interior term is
   `|Div P| * 1 mm / E`. Variants: pointwise `r + t`; patch mean
   `mean_patch(r) + t`; patch mean times mesh size `mean_patch(r) h / 1 mm + t`.
   The pilot rule of ADR 0019 (largest pooled Spearman correlation) selects
   among these three. The interior-only values are reported alongside.

4. **Populations.** The budget is gated on DEVELOPMENT. HOLDOUT is scored
   once with the same code; its budget sets the claim level when every
   required G14 check passes: `HOLDOUT_VERIFIED` if it passes,
   `DEVELOPMENT_VERIFIED` otherwise. The calibration curve is fitted on
   DEVELOPMENT and its coverage reported on HOLDOUT. HOLDOUT is consumed only
   if the DEVELOPMENT budget passes on every gated slice and the pooled
   DEVELOPMENT Spearman correlation meets 0.6; otherwise the one-time
   HOLDOUT stays unspent for a later candidate.

5. **Access and freeze.** DEVELOPMENT is opened only through
   `s14_validate development`, which first requires
   `S14-PROTOCOL-SIGNOFF.json` naming the exact protocol digest and the
   decision owner, and every evaluation file at its frozen digest; it then
   writes `S14-DEVELOPMENT-ACCESS.json` (exclusive create, fsync) before any
   label is read. HOLDOUT is opened only after the DEVELOPMENT results exist:
   `S14-HOLDOUT-CONSUMED.json` is written and fsynced, then the private S9
   reveal is read and must open the `CP-ENVELOPE` commitment.

## Consequences

- The FEM terms are measured, not assumed. On the 64 TRAIN-screen cases the
  discretisation plus projection terms have medians of 1.04% (stress field),
  1.56% (peak) and 0.055% (displacement). On the four refined cases they
  exceed the extrapolated FEM error by factors of 2.6–5.1 (field), 15–55
  (peak) and 4.0–4.2 (displacement); extrapolated with the observed instead
  of the conservative order, the smallest field factor is still 1.4.
- Each S14 case needs one coarse solve (about 30 s on one thread).
- TRAIN rehearsal (`S14-TRAIN-REHEARSAL.json`): the budget passes on 20 of
  the 21 gated slices (all-case p90: field 2.6%, peak 3.5%, displacement
  0.30%) and fails on `R_fillet:0`, which holds 9 TRAIN cases and therefore
  uses the maximum. One case drives it (`candidate-001561`: fillet 8.5 mm,
  gauge 13.7 mm, strain 0.198; surrogate error 4.9% field and 8.7% peak).
  The same weakness may fail DEVELOPMENT, which by decision 4 keeps HOLDOUT
  unspent.
- The pilot selects `patch-mean` with a pooled Spearman correlation of 0.644
  (0.631 without the boundary term; median per case 0.586).
  `G14.confidence` on DEVELOPMENT may still fail; per ADR 0019 the remedy is
  then a new ADR, not a lower threshold.
- Refusal: 240 of 240 fixtures refused with the
  expected code, 0 of 844 in-domain cases refused, codes stable. Latency on
  one thread: 0.116 s median (0.153 s p99) for displacement and stress at
  969 vertices against 138 s for the FEM solve, a 1,188x speed-up.
- `G14.reducer` stays `NOT_RUN` until `IMPLEMENTATION-v1.0-OSS.md` is
  available, so G14 cannot pass before then. DEVELOPMENT and HOLDOUT
  measurements remain valid; the reducer is a function of their statuses.

## Verification

- `tests/campaigns/test_dogbone_s14_*.py`: component rules, nesting of the
  coarse level, S8 factors, strata reproduction, the S10 recipe for any
  role, consumption before reveal, single DEVELOPMENT opening, protocol
  digests and the sign-off check.
- `S14-TRAIN-REHEARSAL.json`: the refinement check compares the budget
  reference terms with the Richardson-extrapolated error of four
  doubled-resolution solves.
