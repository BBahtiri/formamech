Work in: "C:\Users\betim\Documents\Data_Science\pi_ginot_large"

You are implementing PI-GINOT v1.0, an open-source framework that produces verified,
problem-specific physics-informed neural operators for 2D finite-strain elasticity.

Authorized scope for this task: the **S1 closeout iteration** only. This is iteration 9
of the risk spike. It is the last one. Nothing from S2 onward.

Canonical documents, in precedence order:
  1. STEPS-v1.0.md                 — the executable sequence and quality gates (binding)
  2. S1-iteration8-review.md       — the independent review that motivates this task
  3. AGENT-PROMPT-S0-S1.md         — the original S0/S1 brief (context for scope rules)

────────────────────────────────────────────────────────────────────────
WHY THIS TASK EXISTS
────────────────────────────────────────────────────────────────────────
Iteration 8 reported DogBone 9.80% and centre-notched 13.34% relative L2 on stress, and
declared the two-family model NOT_CONFIRMED on the centre-notched bootstrap upper bound
and P90.

An independent review of the artifacts found that verdict is confounded. The controlling
physical variable for the centre-notched family is the net-section factor

    k = width / (width - 2 * notch_depth)

The 20-geometry training set covers k up to 1.56. The 25-case confirmation set reaches
k = 2.09, with five cases above the training maximum. Those five cases carry essentially
the entire failure. Restricted to the region training actually covers, centre-notched
scores 8.76% (CI 5.96-11.80%), P90 12.57%, 95% of cases under 15% — passing every
preregistered gate. Meanwhile **zero** DogBone confirmation cases fall outside the
training range of its governing ratios. The two families were never tested under
comparable extrapolation load.

The review further identified that the peak-stress weakness has three mechanical causes,
two of which are code paths already present and set to zero:

  (a) `_sample_supervised_indices` (run.py:1415) draws supervised query points by lumped
      nodal area. On a waisted specimen the smallest elements are at the notch root, so
      the highest-stress region is the lowest-probability region. Measured: the QoI band
      is 2.48% of nodes but 0.74-0.92% of the area measure; the single peak node enters
      the loss roughly once every 35 epochs.
  (b) `peak_supervised_p_weight = 0.0` — the peak-QoI supervision term is implemented and
      disabled.
  (c) `traction_free_on_P_served = 0.0` — the traction-free constraint on the served
      stress is implemented and disabled, at the free surface where the QoI lives.

And the ground truth is not converged. Richardson extrapolation on the six recorded
16/32/64 refinement sequences puts the production mesh (nx=32) 1.6-8.9% **below** the
extrapolated peak, always low, at observed order 0.7-2.1. `ElementTriP1` gives
element-wise constant stress; `_nodal_fields` (fem.py:355) then applies an unweighted
count average that pulls the root peak down further. The existing convergence check
(`all_last_changes_within_limit`) tests the last increment, which shrinks automatically
for any convergent sequence and cannot detect slow convergence.

Your job is to remove those confounds and measure the mixed (u, P) formulation cleanly,
once, so the human GO/NO-GO decision rests on a number that means what it says.

────────────────────────────────────────────────────────────────────────
BEFORE EDITING ANYTHING
────────────────────────────────────────────────────────────────────────
1. Read STEPS-v1.0.md §0 (gate policy) and §S1 completely.
2. Read S1-iteration8-review.md completely.
3. Run `git status --short --branch`. Note that all of experiments/ is currently
   untracked. **The no-commit prohibition from the original brief is lifted for this
   task** — see PROHIBITIONS below for what replaces it.
4. State back to me, before your first file write:
   - the four phases below as you understand them, and what each one isolates
   - the exact list of paths you intend to create or modify
   - which gate checks you expect to be NOT_RUN in this environment, and why
   - any place where this brief contradicts STEPS-v1.0.md

────────────────────────────────────────────────────────────────────────
METHOD: strict TDD, every step
────────────────────────────────────────────────────────────────────────
RED     — write the failing test first and run it to demonstrate the failure
GREEN   — implement the smallest change that passes
IMPROVE — remove duplication without adding scope

Never write implementation before its test. Never widen scope during IMPROVE.

**Phases run in order and each is measured separately.** The point of the ordering is
attribution: if you change the DOE and the loss together, you will not know which bought
you what, and you will have repeated iteration 8's mistake in the opposite direction.

────────────────────────────────────────────────────────────────────────
PHASE 0 — Commit the existing spike
────────────────────────────────────────────────────────────────────────
Before any new work: commit the current state of experiments/ and artifacts/ as-is, on a
branch. Do not clean it up, do not reformat it, do not rewrite history. Eight iterations
and 3,738 seconds of training are currently untracked and one mistake from gone.

Commit message must record that these artifacts are role=LEGACY and ineligible as
release evidence.

Then refresh `artifacts/gates/S1.json` so it reflects iteration 8 rather than the Aug 10
original, and commit that separately.

────────────────────────────────────────────────────────────────────────
PHASE A — Repair the ground truth
────────────────────────────────────────────────────────────────────────
Everything downstream is measured against the FEM peak. Fix it first.

A1. Raise the element order to `ElementTriP2` in `_solve_displacement_control`
    (fem.py:259), or grade the mesh toward the notch root in `geometry.py`, or both.
    State which you chose and why. If you grade the mesh, the grading must be a declared
    deterministic function of the geometry parameters, not a hand-tuned constant.

A2. Replace nodal stress recovery in `_nodal_fields` (fem.py:355). The current unweighted
    count average is biased low at the root. Use an area-weighted patch average at
    minimum; a recovery-based estimator is better.

A3. Replace the convergence check. Delete `all_last_changes_within_limit`. Implement
    Richardson extrapolation over at least three levels, reporting per geometry:
    observed order, extrapolated limit, and **estimated remaining error at the
    production mesh**. This is the S8 `G8.estimator-*` machinery arriving early; write it
    so it can be lifted into S8 conceptually, but do not build S8.

A4. Select refinement geometries by **worst case, not fixed stride**. Iteration 8 sampled
    family_index 0, 12, 24. Sample instead the tightest ligament, the smallest notch
    radius, and the smallest fillet radius in each family — the cases that are hardest to
    resolve and that drive the verdict.

A5. Re-run `G1.sanity`. Record it. Note in your report that uniaxial extension of a
    square is an affine deformation which P1 and P2 both reproduce exactly, so this check
    validates the constitutive law, the closure and the BC plumbing, and says nothing
    about resolution at a concentration. Do not present it as evidence of the latter.

**Gate A** — all must pass before Phase B begins:

| Check | Threshold |
|---|---|
| `A.sanity` | uniaxial extension of a square matches closed form to 1e-6 |
| `A.order` | observed convergence order recorded for every refinement geometry |
| `A.remaining` | estimated remaining error on peak stress at the production mesh **≤ 2%** for every refinement geometry, both families |
| `A.worst-case` | refinement geometries are the extremal cases, not a fixed stride, and this is asserted by test |
| `A.solves` | ≥50 of 60 FEM cases converge on the repaired solver |

The 2% figure is deliberate: you cannot gate a model peak error at 20% against a ground
truth carrying 8% of its own. If 2% is unreachable at acceptable cost, **stop and report
the achievable number with its cost** — do not loosen it silently. A loosened threshold
requires an ADR under STEPS §0 rule 2.

────────────────────────────────────────────────────────────────────────
PHASE B — Fix the DOE, and measure it alone
────────────────────────────────────────────────────────────────────────
Change the sampling, change nothing else. The training protocol stays exactly
`iteration3_joint_adam` with all three switches still at zero. This isolates the DOE
effect.

B1. Define the derived governing ratios explicitly in `geometry.py`:

      centre-notched:  k = width / (width - 2*notch_depth)     (net-section factor)
                       d/r = notch_depth / notch_radius        (notch sharpness)
      dogbone:         W_grip / W_gauge
                       R_fillet / W_gauge

B2. Replace uniform box sampling in `sample_geometry_cases` with a design that is
    space-filling **in the derived ratio space**, stratified, with declared bounds. The
    training set must cover the ratio range the evaluation set will draw from. Record the
    achieved coverage: min, max, and stratum counts per ratio, per family, per role.

B3. Declare the ratio ranges as the family envelope and write them into the artifact. Any
    evaluation case outside the declared envelope must be **labelled as such in the
    per-case record**, not silently mixed into the aggregate. This is the S9/S14 envelope
    concept arriving early; implement the labelling only, not the refusal machinery.

B4. Retrain the unchanged iteration-3 protocol on the new training set. Same optimizer,
    same learning rate, same epochs, same seeds, same loss weights.

B5. Run a **diagnostic** campaign. Label it `attribution_only` in the artifact. Its
    numbers exist to attribute the DOE effect and are explicitly **not** quotable as a
    result. Report the delta against iteration 8's in-coverage figures.

No gate. Phase B produces an attribution number and a covered training set.

────────────────────────────────────────────────────────────────────────
PHASE C — Flip the three switches, then confirm once
────────────────────────────────────────────────────────────────────────
C1. Fix the sampler. `_sample_supervised_indices` (run.py:1415) must stop down-weighting
    the QoI region. Stratified sampling across the domain, or importance sampling on
    stress magnitude, or simple oversampling of the QoI band — your choice, but state it
    and justify it. Assert by test that the expected number of QoI-band draws per epoch
    is at least proportional to the band's node count.

C2. Set `peak_supervised_p_weight > 0`. Choose the value on a **training-only** screen, in
    the same style as iteration 7's `training-only guardrails`. Do not touch any
    evaluation data to select it.

C3. Set `traction_free_on_P_served > 0`. This is the physically load-bearing one — the
    notch root is a free surface and `P·N = 0` is what fixes the peak magnitude.
    `_traction_loss` already exists. Same training-only selection rule.

    `partial_traction` and `section_resultant` stay at zero unless you can show on the
    training-only screen that they help; if you enable them, say so and show the screen.

C4. Retrain. Freeze the checkpoint. Record full provenance as iteration 8 did — that part
    of the existing code is good and should be kept.

C5. **Preregister the confirmation gates in a git commit that provably precedes the
    training artifact.** Not a JSON file written by the same process 34 seconds before the
    result. A commit. This is the STEPS §S9 `commitment-order` discipline adopted one step
    early; it costs nothing and it is what makes the word "preregistered" true.

C6. Run **one** fresh confirmation campaign. Fresh geometry seeds, disjoint from training
    and from every prior campaign including the consumed 50-case iteration-8 set. Prove
    disjointness the way iteration 8 did — that freshness proof is good work, keep it.

**Gate C — the confirmation gate:**

| Check | Threshold |
|---|---|
| `C.solves` | all attempted FEM cases converge, or failures recorded with reasons |
| `C.freshness` | geometry IDs, mesh IDs, mesh-qualified node IDs and RNG seeds all disjoint from training and from all prior campaigns |
| `C.prereg-order` | preregistration commit SHA provably precedes the checkpoint mtime and the campaign artifact |
| `C.envelope` | every evaluation case labelled in-envelope or out-of-envelope against the Phase B declaration |
| `C.stress-in-envelope` | relative L2 on P, in-envelope cases, per family — **report, do not gate** |
| `C.stress-oob` | relative L2 on P, out-of-envelope cases, per family — **report separately** |
| `C.peak-signed` | signed relative peak error reported, with the under/over-prediction split |
| `C.no-tuning` | zero optimizer steps and zero backward calls during the campaign, asserted |

Note what is deliberately **not** in that table: pass/fail thresholds on the stress
numbers. Iteration 8's `NOT_CONFIRMED` was a verdict against gates the agent authored,
not against STEPS. STEPS §S1 gives exactly one decision rule — ≤5% GO, 5-15% GO with
contingency, >15% STOP, on the point estimate — and reserves the call to the human.
Report the numbers. Do not render a verdict.

────────────────────────────────────────────────────────────────────────
PHASE D — Fix the reporting
────────────────────────────────────────────────────────────────────────
D1. `relative_peak_stress_error` (iteration8.py:471) takes `abs(pred - fem)/fem` and
    discards the sign. In iteration 8, 25 of 25 DogBone peaks were under-predicted, ratio
    0.685-0.962, and centre-notched reached 0.44 — a systematic non-conservative bias
    that is invisible in the report. Report signed error and the under/over split.

D2. `_stress_magnitude` (run.py:2028) is the Frobenius norm of the first Piola tensor. It
    is not von Mises and it is not a principal stress. Rename the metric so it says what
    it is, or change the QoI. Do not label it "peak notch stress" in any report.

D3. The P90 gate at n=25 interpolates between the 22nd and 23rd order statistics — two or
    three observations. Either report it with a bootstrap CI, or drop it and report the
    full ECDF instead.

D4. `_bootstrap_interval` (confirm.py:370) clamps the interval to contain the point
    estimate. It is conservative, so it can only make gates harder, but it is not a
    percentile interval and must not be described as one. Either implement BCa or
    document the clamping in the artifact.

D5. Record the wall-clock numbers STEPS §S1 asks for: mean seconds per FEM solve, mean
    seconds per operator inference. This is the first `G-LATENCY` datapoint.

D6. Answer, on the record, the question STEPS §S1 asks and iteration 8 left implicit:
    **does the peak notch/fillet stress QoI survive a regularity check, or is it
    singular?** You now have the Phase A convergence orders to answer it properly.

────────────────────────────────────────────────────────────────────────
ANTI-SCOPE — what must not exist when you are done
────────────────────────────────────────────────────────────────────────
- No S2+ code. No contracts module, no content store, no real geometry ingestion, no
  morphing, no production FEM backend, no MMS harness, no DOE framework, no dataset
  manifests, no trainer, no verification gates, no bundle, no CLI subcommands beyond
  `--version`.
- **No new architecture variants.** The three switches and the sampler only. Do not add
  stress-head variants, do not change the vendored upstream PI-GINOT, do not widen the
  head, do not add parameter-vector conditioning. Architecture is a separate decision the
  human has not yet made.
- **No third geometry family.** The central-hole topology question is deferred until
  these numbers exist.
- **No reuse of the consumed 50-case iteration-8 campaign** for anything except the
  attribution comparison in Phase B, which must be labelled as such.
- No tuning against any confirmation campaign.
- Nothing from the spike copied into `src/`. `tests/test_spike_isolation.py` must still
  pass.

────────────────────────────────────────────────────────────────────────
PROHIBITIONS
────────────────────────────────────────────────────────────────────────
- Commit freely on a branch. **Do not push, do not tag, do not merge to main, do not
  publish.** The no-commit rule from the original brief is replaced by this one.
- Do not install tools globally. Use the project venv.
- Do not modify credentials or use paid compute.
- Do not hand-author uv.lock.
- Do not fabricate a SHA, a gate result, a benchmark number, or a coverage figure.
- Do not weaken a threshold to make a gate green. If a threshold is wrong, say so and
  stop. `A.remaining ≤ 2%` in particular is a real constraint, not a target to negotiate.
- Do not report NOT_RUN as PASS.
- Do not render the GO/NO-GO verdict. Report numbers.
- Do not edit STEPS-v1.0.md or S1-iteration8-review.md.

────────────────────────────────────────────────────────────────────────
FINAL REPORT FORMAT
────────────────────────────────────────────────────────────────────────
1. Files created and changed, with line counts, and the branch name.
2. RED evidence: the failing test output that preceded each implementation.
3. Gate A table with measured values against thresholds.
4. Phase B attribution: the DOE effect, stated as a delta, explicitly labelled
   attribution-only and not a result.
5. Phase C: the three headline numbers per family, split in-envelope and out-of-envelope,
   with signed peak error and the under/over-prediction split. **No interpretation, no
   recommendation, no verdict.**
6. Achieved DOE coverage in derived-ratio space: min, max, stratum counts, per family,
   per role.
7. The preregistration commit SHA and proof it precedes the training artifact.
8. Every NOT_RUN check with its reason.
9. Anything you found that contradicts STEPS-v1.0.md or S1-iteration8-review.md, or that
   you had to guess at. Be specific. A wrong guess recorded is recoverable; a wrong guess
   hidden is not.
10. Confirmation that no S2+ code, no architecture change, no third family, no push, no
    global install and no paid compute occurred.

*** Then STOP. The GO/NO-GO decision is mine. Do not proceed to S2. ***
