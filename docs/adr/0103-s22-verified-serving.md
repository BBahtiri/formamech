# ADR 0103: Verified serving — fast, certified and exact (S22)

- Status: Accepted — G22 PASS on 2026-10-12 (`artifacts/gates/S22.json`);
  accepted as implemented by the decision owner on 2026-10-12 (chat: "Accept
  ADR 0103"); the G22 thresholds were accepted on 2026-10-11
  (`artifacts/serving/S22-PROTOCOL.json`)
- Date: 2026-10-12
- Step: S22 (gate `G22`), implementation plan Feature 1b

## Context

S21 showed that one Newton step from the v0.1.0 operators' predictions lands
on the converged discrete solution (effectivity 1.000 on all 128 DEVELOPMENT
cases, at about 0.5% of a cold solve). S22 turns this into serving modes, so
a user can trade speed for certainty and is never silently given a wrong
answer.

## Decisions

1. **Modes.** `piginot.serving.serve` and `piginot predict --mode`:
   - `fast`: the bundle's prediction, the unchanged v0.1.0 code path and
     document;
   - `certified`: the prediction plus the S21 certificate on the bundle's FEM
     problem; `--corrected` adds the one-step corrected displacement at the
     FEM nodes; `--escalate` (with `--tolerance` on the stress change)
     promotes a certificate that is not good enough to `exact`;
   - `exact`: the converged FEM solution, started from the prediction.
2. **Solve ladder.** Warm Newton at full load; then continuation from the
   half-scaled prediction at load factors 0.5, 0.75, 1; then the cold
   load-stepped solve; then an explicit `solve_failed`. Every response names
   its path and lists its attempts.
3. **FEM recipes as sidecars.** The shipped bundles are hash-closed evidence
   and stay byte-identical. Each gets a recipe in `bundles/fem-recipes/`,
   bound to its manifest hash: the canonical S10 request without mesh and
   request ID, the material inputs, and the axial-strain rule (end
   displacement = nominal strain × x-position of the tagged end of the served
   mesh). For all 64 + 64 DEVELOPMENT cases the rebuilt request equals the S10
   request byte for byte (`artifacts/serving/S22-RECIPES.json`).
4. **Warm start in the solver.** `ScikitFemBackend.solve_from(request,
   initial, load_factors)`; `solve_many` is unchanged (a test pins the
   identical result hash when its default load factors are passed
   explicitly).
5. **Displacement-only evaluation.** `EnergyFormServing.displacement`
   evaluates only `u`, chunked exactly as `serve`, bit-identical to the served
   `u` (6402 P2 nodes in 0.12 s instead of 6.4 s).
6. **Packaging.** `fast` needs only the `runtime` extra. `certified` and
   `exact` need the new `verify` extra; without it they raise
   `MissingExtraError` naming it. In `certified` and `exact` the S14 pointwise
   bound is off by default (`confidence=True` restores it); the certificate
   supersedes it.

## Consequences

- A request can now be answered at FEM accuracy for a few percent of a cold
  solve, with a certificate or an explicit failure in every case.
- G22 measures, on the reference machine and the 128 DEVELOPMENT cases,
  never-silent exact serving, equivalence with cold solves (1e-6), the median
  speed-up (≥ 10×), the corrected peak-stress error (≤ 0.1 × the operator's),
  fast-mode identity with the S19 goldens, the light install, old bundles and
  determinism.
