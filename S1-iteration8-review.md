# Review — S1 iteration 8 confirmation campaign

Independent read of the repo at `pi_ginot_large`, against `STEPS-v1.0.md` §S1 and
`AGENT-PROMPT-S0-S1.md`. Everything below is derived from the committed artifacts and
source, not from the coder's summary.

---

## 1. Bottom line

The engineering is good and the reported numbers are real. But the headline conclusion
is **wrong in attribution**, and the recommended next experiment follows from the wrong
diagnosis.

The coder concludes: *"a model-generalization issue"* on centre-notched geometry, and
proposes targeting centre-notched generalization next.

What the data actually shows: **the centre-notched result is confounded by a design-of-
experiments coverage gap.** Five of the 25 confirmation geometries have a ligament
tighter than anything in the 20-geometry training set, and those five carry essentially
the whole failure. Restricted to the region the training set covers, centre-notched
passes every preregistered gate.

| centre-notched slice | n | Stress L2 | 95% CI | P90 | frac ≤ 15% | verdict |
|---|--:|--:|--:|--:|--:|---|
| all cases (as reported) | 25 | 13.34% | 9.05–17.31% | 26.55% | 0.80 | fails CI + P90 |
| within training ligament coverage | 20 | **8.76%** | **5.96–11.80%** | **12.57%** | **0.95** | **passes all** |
| beyond training ligament coverage | 5 | 26.4% | — | — | 0.20 | — |

The controlling variable is the net-section factor `k = width / (width − 2·notch_depth)`
— the ligament tightening ratio, which sets the stress amplification.

- Training set (20 centre-notched geometries): `k` spans 1.13 → **1.56**.
- Confirmation set: `k` spans 1.12 → **2.09**, with five cases at 1.65, 1.65, 1.66,
  1.96, 2.09 — all outside training range.
- Those five cases are, in rank order, the worst results in the campaign.
- `corr(notch_depth, L2 stress error) = +0.60`. No other parameter comes close.

The dogbone comparison makes the confound explicit: **zero** of the 25 dogbone
confirmation cases fall outside the training range of its governing ratios
(`W_grip/W_gauge`, `R_fillet/W_gauge`). So the two families were not tested under
comparable extrapolation load. DogBone "passing" and centre-notched "failing" is, in
substantial part, a difference in how far each confirmation draw wandered outside its
training envelope — not a difference in family difficulty.

Both families draw parameters uniformly from an axis-aligned box, but the physics is
governed by a derived ratio. A uniform box draw gives narrow coverage of that ratio's
tail in 20 samples and wider coverage in 25 — pure luck of the draw, in opposite
directions for the two families.

> **Important caveat, and please hold the coder to it too:** the 8.76% figure is a
> *post-hoc subgroup analysis on a consumed campaign*. It is a hypothesis, not a
> passing result, and it must not be reported as one. Its only legitimate use is to
> design the next experiment.

---

## 2. What is genuinely well done

Worth saying plainly, because the list below is long and mostly critical.

- **Fresh-campaign discipline.** Iteration 3's own artifact records
  `held_out_evaluation_status: confirmatory_not_pristine` — the coder noticed that
  iterations 1–2 had already looked at the original held-out set, and built a
  genuinely fresh 50-case campaign rather than quietly reusing it. That is the right
  instinct and it is rare.
- **Freshness proof is real.** Geometry IDs, mesh IDs, mesh-qualified node IDs and RNG
  seeds are all verified disjoint from training. `G1.split` is honestly satisfied.
- **The bootstrap is correct.** Whole-geometry resampling of a ratio-of-sums estimator,
  10,000 resamples, fixed seed. Resampling cases rather than nodes is the right unit
  and is often gotten wrong.
- **Evaluation metric is correct.** Area-weighted (lumped P1) discrete L2 over all mesh
  nodes, not an unweighted node average.
- **Checkpoint integrity holds.** The two `.pt` files have different SHA-256s, which
  looked alarming. I unpacked both: all 196 tensor storages are byte-identical
  (`8ea3668b…`). The difference is archive naming and provenance metadata only. The
  evaluated model *is* the frozen model.
- **The plane-stress closure is right.** Damped Newton in `log(F33)`, monotone residual,
  correct IFT derivative. `G1.sanity` returns 4.1e-16.
- **The notch geometry is what it claims.** `h(x) = W/2 − d·(1−(x/L)²)³` with
  `L = √(6dr)` gives `h″(0) = 1/r` exactly, so `notch_radius` really is the root radius
  of curvature. Docstring verified against the algebra.
- **Honest self-reporting throughout.** `historical_checkpoint_missing: true`,
  `reproduction_kind: semantic_protocol_reproduction`, and `commit_scope: "base commit
  plus uncommitted local S1 worktree; S1 commit prohibited"` are all volunteered rather
  than glossed. The "campaign is now consumed" warning at the end is correct and should
  be enforced.

---

## 3. Three protocol defects that explain the peak-stress failure

The peak-stress numbers are much worse than the field numbers (centre-notched median
peak error 22.9%, worst 55.9%). The coder frames this as a generalization limit. It
isn't — it's three switches, two of which are already implemented in the codebase and
set to zero.

### 3.1 Area-weighted sampling starves the notch root

`_sample_supervised_indices` (`run.py:1415`) draws the 96 supervised query points by
`torch.multinomial` on **lumped nodal area**. On this mesh, `eta = linspace(0,1,ny+1)`
is normalised through the thickness, so element height scales with `h(x)` — which is
*smallest at the notch root*. The highest-stress region is therefore the
lowest-probability region in the sampler.

Measured on the worst confirmation geometry (`k = 2.09`):

| | uniform | area-weighted |
|---|--:|--:|
| share of the 9 QoI (notch-root) nodes | 2.48% | **0.92%** |
| expected QoI draws per epoch, out of 96 | 2.4 | **0.89** |
| draws/epoch for the single peak node | 0.26 | **0.029** |

The node where the peak actually occurs enters the supervised loss roughly **once every
35 epochs**. Over 1,500 epochs that is ~43 gradient contributions, each worth 1/96 of
one geometry's stress MSE. The model was effectively never told what the peak is.

### 3.2 Peak-QoI supervision is implemented and disabled

`peak_supervised_p_weight = 0.0` in the frozen run (`iteration8_reproduction.py:178`).
The `peak_supervised_p` term exists, samples from `qoi_weights`, and is switched off.

### 3.3 The traction-free condition on `P_served` is implemented and disabled

`loss_weights` in the checkpoint provenance:

```
traction_free_on_P_served      : 0.0
partial_traction_on_P_served   : 0.0
section_resultant_on_P_served  : 0.0
```

`_traction_loss`, `_partial_traction_loss` and `_section_resultant_loss` all exist in
`run.py` and all carry weight zero. The notch root **is** a free surface; `P·N = 0` is
exactly the constraint that makes the stress state uniaxial there and fixes the peak
magnitude. The served stress currently gets equilibrium at 24 interior collocation
points, an in-plane constitutive consistency term, and nothing at all on the boundary
where the QoI lives.

Consequence, which the report does not mention:

- **25 of 25 dogbone peaks are under-predicted** (pred/FEM ratio 0.685–0.962).
- Centre-notched: 14/25 under, median ratio 0.82, worst **0.44** — the model predicts
  less than half the true peak on the tightest ligament.

This is a systematic non-conservative bias. For a stress surrogate that is a more
serious finding than the L2 number, and it is invisible in the report because
`relative_peak_stress_error` takes `abs(pred − fem)/fem` and throws the sign away
(`iteration8.py:471`).

### 3.4 Architectural note (secondary, but relevant)

The stress head is **4,484 of 614,939 parameters** — a single 64-wide GELU layer
consuming the decoder features that the *displacement* head was optimised to produce,
plus one scalar (`STRESS_HEAD_CONDITIONING = ("y_max/x_max",)`), with output scaled by
`E·u_delta/x_max`.

For centre-notched, `x_max` is fixed at 25 mm, so that output scale is a **constant
30.4 MPa for every case in the family**. The head has no access to `notch_depth` or
`notch_radius` except through the 320-point boundary cloud, and must synthesise the
entire amplification factor (`K_t,gross` observed range 1.67–2.72) from a 64-d feature
vector through one hidden layer. The variant registry already contains
`coordinate_conditioned_mlp` and `fully_conditioned_mlp`; neither is in use.

Worth flagging against the target design: `STEPS` S11 specifies "FiLM conditioning on
the parameter vector" and `G11.reachability` requires every spec parameter to have a
non-zero gradient path to the output. The spike model has no parameter-vector
conditioning at all, despite the head being named `conditioned_mlp`.

---

## 4. Ground truth is not as solid as the report implies

The report says "mesh refinement passed." That claim is doing more work than it can bear.

**The refinement study has not converged.** Richardson extrapolation on the six recorded
16/32/64 sequences:

| geometry | h32 | h64 | observed order | extrapolated | error at nx=32 |
|---|--:|--:|--:|--:|--:|
| dogbone-5d72ef5d | 40.42 | 41.77 | 0.75 | 43.76 | **−7.6%** |
| dogbone-bdfeae7a | 46.13 | 47.86 | 0.70 | 50.65 | **−8.9%** |
| dogbone-3397de98 | 29.36 | 30.20 | 1.05 | 30.98 | −5.2% |
| centre_notched-fb223277 | 62.46 | 63.67 | 1.83 | 64.14 | −2.6% |
| centre_notched-b9c497bf | 63.75 | 64.57 | 2.14 | 64.81 | −1.6% |
| centre_notched-e94f38dd | 52.23 | 53.98 | 1.34 | 55.12 | −5.2% |

Every case is monotone increasing and observed order is ~0.7–2.1. The production mesh
(nx=32) sits **1.6–8.9% below** the extrapolated limit, always low. So the peak-stress
ground truth carries a systematic negative bias of the same order as several of the
gates being applied to it.

Three contributing causes:

1. **`ElementTriP1` (`fem.py:259`).** Stress is element-wise constant, then averaged to
   nodes by an *unweighted* count average (`_nodal_fields`, `fem.py:355`). At the notch
   root that averaging pulls in elements away from the root and biases the peak low.
2. **No mesh grading.** `x` and `eta` are both `linspace`. There is no refinement toward
   the root. On the tightest geometry there are roughly four elements across the root
   radius.
3. **The convergence gate is mis-specified.** `all_last_changes_within_limit` tests
   `|h64 − h32| / h64 < 0.1`. The last increment shrinks automatically for *any*
   convergent sequence regardless of remaining error — it cannot detect slow
   convergence, which is exactly the failure mode here. The correct test is Richardson
   extrapolation with an estimated remaining error, which is what `STEPS` S8
   (`G8.estimator-brackets`, `G8.estimator-conservative`) will require anyway.

**Also: `G1.sanity` cannot support the weight placed on it.** Uniaxial extension of a
square is an affine deformation. P1 elements reproduce it *exactly* — hence 4e-16. It
validates the constitutive law, the closure and the BC plumbing. It says nothing
whatsoever about the mesh's ability to resolve a stress concentration. The prompt's own
warning ("a wrong solver produces confidently wrong ground truth") is not yet retired.

**Sampling of the refinement study.** The three geometries per family were chosen at
`family_index` 0, 12, 24 — a fixed stride, not the worst cases. The tightest-ligament
geometries, which are both the hardest to resolve and the ones driving the verdict, were
never refinement-checked.

---

## 5. Metric and statistics issues

- **Signed error is discarded.** As above: 100% of dogbone peaks under-predicted, and
  that fact appears nowhere. Report signed relative error, or at minimum the
  under/over-prediction split.
- **The QoI is the Frobenius norm of `P`** (`run.py:2028`), not von Mises or a maximum
  principal stress. Defensible for a spike, but it is not "peak notch stress" in the
  engineering sense, and it is the number the headline gate is applied to. Name it
  honestly or change it.
- **P90 at n=25 is not a measurement.** `np.quantile(per_case_P, 0.9)` on 25 points
  interpolates between the 22nd and 23rd order statistics — it is determined by two or
  three observations, and here those observations are precisely the out-of-coverage
  cases. A hard gate on it, with no CI, was underpowered by construction.
- **The 0.80 fraction gate passed by exact equality** (20/25 vs threshold 0.80). One
  more bad draw flips it. Worth noting as fragile rather than passing.
- **`_bootstrap_interval` clamps the interval to contain the point estimate**
  (`confirm.py:370`). Conservative, so it can only make gates harder — but it means the
  reported interval is not a pure percentile interval and should not be described as
  one. For a ratio-of-sums estimator with this right tail, BCa would be the defensible
  choice.
- **The gates are stricter than your own decision rule, and that should be surfaced.**
  `STEPS` §S1 gives one decision table: ≤5% GO, 5–15% GO-with-contingency, >15% STOP. It
  is on the *point estimate*. Both families are in the 5–15% band. The CI-upper ≤ 15%,
  P90 ≤ 20%, fraction ≥ 0.80, u ≤ 7.5% and worst-peak ≤ 60% gates were authored by the
  coder, not by you.

  Tightening is permitted (§0 rule 2, the ratchet), and tighter gates here are arguably
  good practice. But `NOT_CONFIRMED` is a verdict against a self-authored bar, and the
  report presents it as though it were the project's decision. **By `STEPS-v1.0.md` as
  written, this run is GO-with-contingency — and §S1 reserves that call explicitly to
  you.** Ask where the 60% worst-peak threshold came from; it is the one number that
  looks reverse-engineered (measured 55.9%).

---

## 6. Scope observation: the two families are the same shape class

`geometry.py` generates both families through the same `_mapped_mesh(x, height, ny)`
call, on the same structured grid, with the same `TOPOLOGY_SIGNATURE =
"quarter-single-component-no-holes"`. Dogbone and centre-notched differ only in the
profile function `h(x)` — both are a quarter-symmetric strip whose half-height is
minimum at `x = 0` and rises monotonically.

The stated purpose of family (b) in the prompt is: *"Family (b) is the important one. It
is the proof that nothing here is dogbone-specific."* As built, it does not deliver
that proof. It varies a profile function within one shape class. A true centre-notched
plate — an internal slit or hole, different topology, `hole_count = 1` — would test
family-generality; a smooth C² waisted strip does not. There is also a genuine
ambiguity here about whether you meant a centre notch (internal) or a double-edge notch
(what was built); `geometry.py`'s own docstring flags that the protocol does not
prescribe the construction. That's yours to adjudicate, and it materially changes how
hard the family is.

---

## 7. Process

- **Nothing from S1 is in git.** `git status` shows `?? experiments/` and
  `?? artifacts/gates/S1.json`; only S0 is committed. The coder was correct to obey the
  "do not commit" prohibition and correct to record `commit_scope` honestly — this is
  your action, not a coder failure. But right now eight iterations and 3,738 seconds of
  training sit in an untracked working directory.
- **The preregistration is self-witnessed.** `iteration8_preregistration.json` was
  written at `21:22:31`, the checkpoint at `21:17:32`, the result at `21:23:05` — all by
  the same process in the same run, with `created_before_fem_solves` as a self-asserted
  boolean and the FEM results served from a content-addressed cache. There is no
  external ordering witness. `STEPS` S9 demands a git-committed commitment hash for
  exactly this reason. S1 does not formally require it, but the report leans on the word
  "preregistered" and the artifact does not yet earn it.
- **`artifacts/gates/S1.json` is stale.** Dated Aug 10, before iterations 4–8. It still
  records the original S1 numbers. If G1 is meant to be the standing gate record it
  needs to reflect iteration 8.
- **The "92.86% coverage / 126 tests" figure is over throwaway spike code**, which
  `pyproject.toml` excludes from the package. Fine as a hygiene signal; it is not a
  `STEPS` coverage floor and shouldn't be presented next to gate results as though it
  were.

---

## 8. What I'd do next

**Do not** run "an experiment targeting centre-notched generalization." That accepts a
diagnosis the data doesn't support and risks spending an iteration tuning against a
confound.

Ordered by information gained per unit of compute:

1. **Fix the DOE first — it is nearly free and it is the actual finding.**
   Reparameterise or stratify on the derived governing ratios: net-section factor
   `k = W/(W−2d)` and `d/r` for centre-notched; `W_grip/W_gauge` and `R_fillet/W_gauge`
   for dogbone. Sample to fill those, not the raw box. Declare the ratio range as the
   envelope and record it. This is the S9 "coupled-domain DOE with declared
   interactions and difficult regions" arriving early, and it is where the S1 spike is
   actually delivering its lesson.

2. **Flip the three switches that are already in the code**, then re-screen on training
   data only:
   - `peak_supervised_p_weight > 0`
   - `traction_free_on_P_served > 0` (this is the physically load-bearing one)
   - replace area-weighted supervised sampling with stratified or
     `stress-magnitude`-weighted sampling, or simply oversample the QoI band

   These are cheap and directly target the mechanism in §3. My expectation is that
   traction-free plus peak supervision moves the peak error more than any architecture
   change would.

3. **Then, and only then, consider architecture.** Give the stress head its own trunk
   rather than a 4.5k-parameter tap off the displacement features, and add explicit
   parameter-vector conditioning (`fully_conditioned_mlp` is already in the registry).
   `STEPS` already budgets a 3-week architecture iteration between S11 and S12 for the
   5–15% band — this is that budget, and it should be spent after (1) and (2), not
   before.

4. **Repair the ground truth in parallel.** P2 elements or a graded mesh at the root, and
   replace the last-change convergence test with Richardson + estimated remaining error.
   Until the FEM peak is converged to better than the tolerance you are gating on, the
   peak-stress gate is measuring the solver as much as the model.

5. **Report signed peak error** and split under/over-prediction. Non-conservative bias
   is the finding an engineer reading this needs first.

6. **Commit everything, then preregister the next campaign properly** — thresholds and
   holdout commitment hash in a git commit that provably precedes the first training
   artifact. That is the S9 machinery; adopting it one step early costs nothing and
   makes the next verdict mean something.

**One process rule to hold:** the 50-case campaign is consumed. The coder is right about
that, and §1's caveat applies to my own re-slicing too. Everything above is diagnosis;
the next campaign has to be fresh, and its gates fixed before it runs.

---

## 9. Two questions worth putting to the coder

1. Where did `family_worst_peak_error_max = 0.6` come from? Measured value is 0.559. If
   there is a calibration behind it, it should be in the preregistration; if not, it
   should be tightened or dropped.
2. Was the coverage asymmetry between the two families' confirmation draws noticed? If
   it was and was judged immaterial, the reasoning belongs in the report. If it was not,
   that is the gap in the analysis, and it is the one that changes the conclusion.
