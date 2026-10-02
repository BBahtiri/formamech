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

---

## Revisit 1 — `CP-ENVELOPE` (S9)

- Date: 2026-08-27
- Trigger: this ADR's own Verification section — *"Revisit at `CP-ENVELOPE` (S9)"*
- Decision status: **unchanged.** GO with contingency stands. Nothing below
  re-validates it; this records what has and has not moved since 2026-08-20.

The original decision text above is not edited. Two of its three accepted risks
were measured after it was written, in the closeout committed as `56dac44`, and
the register it presents is now stale in those two rows. The row that is still
accurate is the one bearing on the S9 family decision.

### Method

Every figure below is recomputed from the committed closeout artifacts, not
quoted from a commit message. The pooled statistic is

```
pooled_relative_l2_P = sqrt( Σ err ) / sqrt( Σ ref )
```

over the cases in a slice, where each row's `err` and `ref` are **squared**
norms. That formula reproduces all six `pooled_relative_l2_P` values recorded in
`experiments/spike/_closeout_phaseB/phaseB_attribution.json` to machine
precision, and is then applied unchanged to `phaseC_result.json`.

Ground truth differs between arms and the numbers are **not** comparable across
that boundary. Phase B attribution used P1 ground truth, matching the frozen
incumbent's own training targets. Phase C and its baseline used P2. The `−48%`
and `−27%` figures in the decision text above are both correct and are measured
against different solvers.

### Risk 1 — peak stress systematically under-predicted

**Status: materially reduced in the spike, not closed.** The decision text's
`−27%` is superseded.

Phase C preregistered three switches in `431c39b` before training, then enabled
them at weight 1.0. Same DOE, same fresh 50-case campaign, same P2 ground truth,
same seeds; only the switches differ.

| slice | n | switches OFF | switches ON |
|---|---|---|---|
| dogbone, in-envelope | 21 | −12.90% | **−4.78%** |
| dogbone, all | 25 | −12.90% | **−4.38%** |
| centre-notched, in-envelope | 22 | −27.44% | **−5.26%** |
| centre-notched, all | 25 | −28.07% | **−4.33%** |

Cases under-predicting, in-envelope: dogbone 21/21 → 20/21; centre-notched
22/22 → 14/22.

The median remains negative in every slice. The bias is smaller, not absent, and
the evidence is `role=LEGACY`. The requirement this ADR pushes into S11/S12 is
**not** discharged.

### Risk 2 — speedup unproven at scale

**Status: figure superseded, risk open.** The decision text's `~15×` came from
spike meshes against the pre-Phase-A solver. Phase C measures, against the P2
solver on the same meshes:

| aggregation | speedup |
|---|---|
| total FEM time / total inference time | 116.7× |
| mean of per-case ratios | 118.9× |
| median of per-case ratios | 122.9× |
| ratio of medians | 123.1× |

Every aggregation clears the `G14.latency` threshold of ≥100×. It does not
discharge the risk: `G14.latency` is against the **primary** solver on
**realistic** meshes, and neither condition is met by spike evidence. The "at
scale" half of this risk is untouched.

### Risk 3 — family generality untested

**Status: unchanged, and now the dominant carried risk.** Both spike families
are produced by the same generator with topology signature
`quarter-single-component-no-holes`, differing only in a profile function.
Nothing in the closeout addressed this. `G17.non-dogbone` still has to be earned
against a genuinely different topology, and no step between here and S17
currently schedules that work.

### What did not change, and should not be misread

The closeout moved peak bias and envelope width. It did not move headline field
accuracy:

| slice | n | switches OFF | switches ON |
|---|---|---|---|
| dogbone, in-envelope | 21 | 9.32% | 8.69% |
| centre-notched, in-envelope | 22 | 8.74% | 8.76% |

In-envelope pooled stress error sits at ~8.7% across every closeout arm, against
this ADR's 15% invalidation threshold. That margin is real but not comfortable,
and it has not improved.

Phase A's ground-truth correction stands: production remaining error at P2 on
the four worst-case geometries is 0.44%, 0.54%, 0.64% and 1.40% against a 2%
limit. One of the four (`dogbone-3778c574631c19f0`) records
`richardson_study_p2.status = NOT_RUN` with a direct settled bound of 0.439%
substituted. `A.order` permits that explicitly — "observed order **or** a settled
bound" — and the substitution is visible rather than silent, but it is a
weaker form of evidence than the other three.

### Correction to the S1 review

The review that produced this ADR reported that in-envelope the two families were
equivalent — dogbone 8.88%, centre-notched 8.53%. Both figures are correct, but
the centre-notched slice contains **n = 3** cases: 22 of the 25 fresh draws fell
outside the incumbent's training envelope. That is the coverage confound, stated
correctly, but it will not support a claim that either family is more accurate
than the other. The defensible claim is narrower: the iteration-8 gap was a
coverage artifact, not a family-difficulty result. After the covering DOE, the
in-envelope centre-notched slice holds 22 of 25 cases.

### Requirements pushed into later steps — status at S9

| Step | Requirement | Status at `CP-ENVELOPE` |
|---|---|---|
| S7 | vertex-recovered stress, P2 or graded mesh | Implemented in the primary FEM kernel; `G7` closed |
| S9 | DOE stratified on derived governing ratios | Enforced — `checkpoint.py` refuses a `CP-ENVELOPE` that declares no derived stratification axes |
| S10 | samplers must not down-weight the QoI region | Open; S10 not started |
| S12 | traction-free `P_served` as a named objective term | Open |
| S14 | envelope declared in derived-ratio space, OOE cases refused | Open |

### Invalidation conditions

| Condition | Verdict at S9 |
|---|---|
| In-envelope stress error > 15% on a preregistered fresh campaign | **Not met.** ~8.7% on closeout evidence, which is `LEGACY` and cannot discharge the condition either way |
| Peak bias not materially reduced by S12, or persists into S14 | **Not yet measurable.** S12 has not run. Spike evidence is indicative only |
| `G17.non-dogbone` fails against a genuinely distinct topology | **Not yet measurable.** This is Risk 3 and remains live |
| S11–S12 contingency iteration consumed by schedule pressure | **Not yet.** S10 has not started |

Nothing invalidates the decision. Three of the four conditions cannot be
evaluated until S12, S14 and S17, so this revisit narrows the register without
strengthening the GO.

### Bearing on the S9 family choice

Risk 3 is the only accepted risk that is neither reduced nor superseded, and it
is the one the production-family decision acts on. Choosing the family that most
resembles the prior single-family work postpones the first real test of
`G17.non-dogbone` to the release gate. That is a legitimate risk-management
choice if made deliberately; it is not supported by any accuracy evidence in this
repository, and the sign-off should record which of the two it is.

Next revisit: `CP-FREEZE` (S13), unchanged.

---

## Revisit 2 — `CP-FREEZE` (S13)

- Date: 2026-10-02
- Trigger: Revisit 1 — *"Next revisit: `CP-FREEZE` (S13), unchanged."*
- Decision status: **unchanged.** GO with contingency stands. The evidence
  below is TRAIN-role only; it narrows the register, it does not close it.

### What moved since Revisit 1

The S11 two-head operator went through the S11–S12 contingency iteration
(closed 2026-09-09) and then failed S13: six registered hybrid fine-tunes and
the S13b peak pilot missed the frozen TRAIN-screen guardrails (ADRs 0014,
0015), and the one-case recovery diagnostics traced the failure to inaccurate
displacement gradients (ADR 0016). S13 replaced the operator with the
energy-form PI-GINOT (ADR 0018, accepted 2026-10-02). `CP-FREEZE` freezes its
single registered trial (run `0ca34e32…`). On the 64 preregistered
TRAIN-screen cases, which were not used for fitting:

| Statistic | S11 incumbent | Frozen candidate |
| --- | ---: | ---: |
| Median P-field relative L2 | 19.20% | 0.91% |
| Median signed root-peak error | −7.76% | −0.29% |
| Median absolute root-peak error | 8.09% | 0.83% |
| Cases under-predicting the peak | 55 / 64 | 36 / 64 |

### Risk 1 — peak stress systematically under-predicted

**Status: materially reduced on TRAIN evidence, not closed.** The median
signed error is −0.29% and the under-predicting share is 36 of 64, so the bias
is no longer systematic on this population. The evidence is TRAIN-role; S14
must confirm it on DEVELOPMENT and HOLDOUT and keep any residual bias out of
the error budget as a non-conservative term.

### Risk 2 — speedup unproven at scale

**Status: unchanged, open.** Nothing was measured. The frozen candidate serves
stress from the gradient of its displacement, so each query needs one forward
and two reverse passes; `G14.latency` must measure exactly that path against
the primary solver.

### Risk 3 — family generality untested

**Status: unchanged and sharpened.** The frozen candidate is conditioned on the
DogBone family parameters and its exact lifting assumes two symmetry lines and
one loaded edge. `G17.non-dogbone` therefore tests the conditioning interface
as well as the topology. An early second-family spike is recommended before
S14 completes.

### Invalidation conditions

| Condition | Verdict at `CP-FREEZE` |
| --- | --- |
| In-envelope stress error > 15% on a preregistered fresh campaign | **Not met.** 0.91% on the preregistered TRAIN screen; the fresh-campaign test is S14 |
| Peak bias not materially reduced by S12, or persists into S14 | **Reduced on TRAIN evidence** (−7.76% → −0.29%); S14 must confirm |
| `G17.non-dogbone` fails against a genuinely distinct topology | **Not yet measurable** |
| S11–S12 contingency iteration consumed by schedule pressure without being run | **Not met.** It ran and closed on 2026-09-09 |

Nothing invalidates the decision. Next revisit: after the S14 holdout reveal.

## Revisit 3 — S14 holdout reveal

- Date: 2026-10-02
- Trigger: Revisit 2 — *"Next revisit: after the S14 holdout reveal."*
- Decision status: **unchanged.** GO with contingency stands. The frozen
  candidate has now been measured on DEVELOPMENT and once on HOLDOUT under the
  signed S14 protocol (ADRs 0019–0021).

### What moved since Revisit 2

S14 scored the `CP-FREEZE` candidate on the 64 DEVELOPMENT cases and, after
the durable consumption record, on the 64 HOLDOUT cases revealed from the S9
commitment. Each total adds the surrogate error to conservative FEM reference
terms (ADR 0020):

| Statistic | DEVELOPMENT | HOLDOUT |
| --- | ---: | ---: |
| Surrogate P-field relative L2, median | 0.98% | 0.97% |
| Surrogate absolute root-peak error, median / max | 0.70% / 2.72% | 1.03% / 2.48% |
| Budget p90, field / peak / displacement | 2.6% / 4.1% / 0.21% | 2.8% / 4.0% / 0.29% |
| Gated budget slices passing | 61 / 63 (accepted, ADR 0021) | 63 / 63 |
| Pooled Spearman of the confidence indicator | 0.641 | 0.620 |
| Single-thread speed-up against the P2 FEM | 1,009x | — |

### Risk 1 — peak stress systematically under-predicted

**Status: closed for the DogBone envelope on HOLDOUT evidence.** The largest
surrogate peak error on HOLDOUT is 2.48%, and with the FEM reference terms
every HOLDOUT case stays inside the 5% peak tolerance.

### Risk 2 — speedup unproven at scale

**Status: closed for this query set.** Displacement and stress at all 969
vertices take 0.109 s median (0.174 s p99) on one CPU thread against 110 s for
the S10 FEM solve: 1,009x, above the 100x target.

### Risk 3 — family generality untested

**Status: unchanged, open.** S14 measures only the DogBone family. The
second-family spike (`codex/second-family-spike`) remains the only evidence:
the unchanged operator handles a domain with a hole, but its field error
(5.05%) misses the preregistered 5%. `G17.non-dogbone` is still ahead.

### Invalidation conditions

| Condition | Verdict after the holdout reveal |
| --- | --- |
| In-envelope stress error > 15% on a preregistered fresh campaign | **Not met.** HOLDOUT p90 field 2.8% including FEM terms; largest surrogate field error 2.3% |
| Peak bias not materially reduced by S12, or persists into S14 | **Not met.** HOLDOUT surrogate peak error at most 2.48% |
| `G17.non-dogbone` fails against a genuinely distinct topology | **Not yet measurable** |
| S11–S12 contingency iteration consumed by schedule pressure without being run | **Not met** |

Nothing invalidates the decision. G14 is not yet passed: `G14.reducer` is
`NOT_RUN` until the twelve verification families of
`IMPLEMENTATION-v1.0-OSS.md` are available. Next revisit: at `G17`.
