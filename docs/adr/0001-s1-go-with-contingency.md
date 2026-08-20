# ADR 0001: S1 risk spike — GO with contingency

- Status: Accepted
- Date: 2026-08-20
- Decision owner: Betim Bahtiri
- Gate: `G1` (PASS, 2026-08-10) · `artifacts/gates/S1.json`

## Context

`STEPS-v1.0.md` §S1 makes the risk spike a ⚑ decision point: gate `G1` passing is
necessary but not sufficient, and the GO/NO-GO call is reserved to a human. The
question S1 exists to answer is narrow and load-bearing:

> Can a mixed displacement/first-Piola neural operator reach engineering tolerance
> on **stress** for a parameterized 2D family?

If it cannot, the remaining ~34 weeks of framework work are not justified.

`G1` passed on 2026-08-10 with all five required checks green: 60/60 FEM cases
converged, uniaxial extension matched closed form to 4.1e-16, the geometry split
was verified disjoint, `results.json` was complete, and spike isolation held.

Eight architecture iterations followed inside the spike. Iteration 8 froze
`iteration3_joint_adam` and evaluated it against a fresh 50-case campaign,
reporting DogBone 9.80% and centre-notched 13.34% relative L2 on stress, and
declaring the two-family model `NOT_CONFIRMED`.

An independent review found that verdict **confounded**. The centre-notched family
is governed by the net-section factor `k = W / (W − 2d)`. The 20-geometry training
set covered `k ∈ [1.13, 1.56]`; the confirmation draw reached `k = 2.09`. Five of
25 cases fell outside training coverage and carried essentially the whole failure.
Meanwhile **zero** DogBone confirmation cases fell outside its training range, so
the two families were never tested under comparable extrapolation load. The
`NOT_CONFIRMED` verdict was also rendered against gates the implementing agent
authored for itself (a bootstrap CI bound and a P90 limit), not against the
decision rule in `STEPS-v1.0.md`.

A remedial closeout repaired the ground truth and the design of experiments, then
re-measured on fresh data.

## Evidence this decision rests on

Frozen incumbent, unchanged, evaluated on a fresh stratified 50-case campaign
disjoint from training, against P1 ground truth matching its own training targets:

| Family | Slice | n | Pooled relative L2 on P |
|---|---|--:|--:|
| DogBone | in-envelope | 21 | 8.88% |
| DogBone | out-of-envelope | 4 | 9.40% |
| centre-notched | **in-envelope** | 3 | **8.53%** |
| centre-notched | out-of-envelope | 22 | 25.10% |

**Inside the declared envelope the two families are equivalent.** The
centre-notched "generalization problem" does not survive the coverage correction.

A controlled retrain — identical protocol, seeds and solver, differing only in
which 40 geometries were trained on — showed the effect is about envelope width
rather than accuracy:

| | uniform DOE | stratified DOE |
|---|--:|--:|
| centre-notched, whole campaign | 23.24% | **10.62%** |
| centre-notched declared envelope | `k ∈ [1.13, 1.56]` | `k ∈ [1.17, 2.60]` |
| accuracy inside its own envelope | 8.53% | 7.75% |

About 8% over a sliver became about 8% over essentially the whole family
(achievable `k` range 1.07–2.65). DogBone moved 0.03pp, behaving as a control
should.

Ground truth was separately repaired in the closeout: peak-stress remaining error
at the production mesh fell from 1.6–8.9% to 0.44–1.40% against a 2% limit, with
P1 and P2 discretizations agreeing to ≤1.5% (`Gate A`, PASS).

## Decision

**GO with contingency**, per the `STEPS-v1.0.md` §S1 decision table for a measured
stress error in the 5–15% band.

- Proceed to S2 and continue the planned sequence.
- Budget the **3-week architecture iteration between S11 and S12** that the
  contingency branch requires. It is not optional and is not to be reclaimed for
  schedule.

Every measurement above sits in the 5–15% band on the point estimate, which is the
statistic the decision table names. The tighter gates that produced
`NOT_CONFIRMED` were not preregistered by this project and do not govern here.

## Consequences

**Accepted risks.**

- **Peak stress is systematically under-predicted.** Median signed peak error was
  −48% for centre-notched under the original DOE and −27% after the coverage fix.
  This is non-conservative — the model reports less stress than is present — and it
  is the single most important open defect. It is the target of the closeout's
  Phase C and must be carried into S11/S12 as a named requirement, not assumed
  solved.
- **Speedup is unproven at scale.** Measured inference was ~0.018 s against ~0.28 s
  per FEM solve on spike meshes, roughly 15×. The `G14.latency` target is ≥100×
  against the *primary* solver on realistic meshes and remains untested.
- **Family generality is untested.** Both spike families are produced by the same
  mesh generator with the same topology signature, differing only in a profile
  function. `G17.non-dogbone` still has to be earned against a genuinely different
  topology.

**Requirements this decision pushes into later steps.**

- **S9** — the DOE must stratify on derived governing ratios, not the raw parameter
  box. This is the spike's central finding.
- **S10** — supervision and collocation samplers must not down-weight the QoI
  region. Area-weighted sampling starves a notch root, which is where the smallest
  elements sit.
- **S12** — the traction-free condition on `P_served` becomes a first-class named
  objective term. The current S12 list omits boundary tractions.
- **S7** — peak-QoI resolution requires stress recovered at vertices rather than
  element-averaged, and P2 or a graded mesh. Element-averaged recovery was the
  dominant source of ground-truth peak bias, ahead of element order.
- **S14** — the envelope must be declared in derived-ratio space, and out-of-envelope
  cases refused rather than silently aggregated into a headline number.

**Evidence status.** Everything produced by the spike is `role=LEGACY` and
permanently ineligible as release evidence. This decision authorises continuing;
it establishes no accuracy claim about any shipped artifact.

## Verification

This decision stops being valid if any of the following is measured:

- In-envelope stress error exceeds 15% on a preregistered fresh campaign at any
  later step, on the point estimate.
- The peak-stress bias is not materially reduced by S12, or persists into S14's
  error budget as a non-conservative term.
- `G17.non-dogbone` fails against a genuinely distinct topology, indicating the
  framework is not family-general.
- The S11–S12 contingency iteration is consumed by schedule pressure without being
  run, in which case the basis for GO with contingency no longer holds.

Revisit at `CP-ENVELOPE` (S9) and again at `CP-FREEZE` (S13).
