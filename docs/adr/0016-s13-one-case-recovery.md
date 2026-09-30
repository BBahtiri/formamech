# ADR 0016: Diagnose S13 stress and peak fitting on one TRAIN case

- Status: Active non-gate diagnostic preregistration
- Date: 2026-09-30
- Step: S13 recovery; G13 remains failed

## Evidence before the next run

The exact S11 baseline checkpoint and S10 FEM result for TRAIN-fit case
`candidate-002485` are bound by the selective source verifier. The operator
queries 969 mesh vertices; 3,729 is the P2 scalar-DOF count, not the operator
query count. The v3-v5 diagnostics train normalized `DATA_U + DATA_P` using
deterministic 57-vertex minibatches and measures the full vertex field and S13
interior-collocation determinants after training.

At 800 steps and learning rate `5e-6`, v3 reduced combined data MSE by 55.70%
but ended at 10.40% P-field and 10.10% absolute root-peak error. At the same
800 steps and `2e-5`, v4 reduced MSE by 62.71%, P error to 8.48%, and peak
error to 9.78%. V5 kept `2e-5` and extended the same schedule to 4,000 steps.
Its first 800 optimizer losses were byte-identical to v4. It reduced MSE by
87.38%, U error to 1.41%, and P error to 5.71%; the peak error remained 9.14%.
Its 969 vertex and interior-collocation determinant checks were valid. V5
therefore failed only the frozen P and peak thresholds of 5% each.

The canonical reports are `S13-OVERFIT-DIAGNOSTIC.json` (v3),
`S13-OVERFIT-LR-V4.json`, and `S13-OVERFIT-BUDGET-V5.json` in the DogBone
artifact directory. These results support a peak-target mismatch after
substantial field fitting, but one case cannot establish multi-case behavior.

## Predeclared v6 test

V6 changes one training term relative to v5. It adds `0.1` times the squared
difference between predicted and FEM maximum root-QoI Frobenius stress, using
all 20 registered `right_arc` vertices for this case and excluding
`right_grip`. The term uses normalized stress units and the same exact maximum
definition as S13 screening. The case, S11 state, S10 result, normalization,
learning rate (`2e-5`), 4,000-step budget, vertex sampler, and success
thresholds remain v5's. The initial squared peak residual is about `0.001053`;
the coefficient makes its initial weighted contribution about `0.000105`
against combined field MSE `0.000477`. This tenfold reduction from the failed
S13b coefficient-1 peak pilot is deliberate.

The frozen v6 plan hash is
`b0ef164be120e8552940c559ec10d72d6dc0072afc3910ba38e2ed2b8b1ac1d3`.
Its sampler hash is
`f894eeab7ea354a2d2ee0085ba8fef17b521df59329071e380e02749495df861`,
identical to v5's. The primary outcome is the existing conjunctive one-case
diagnostic: at least 80% combined data-MSE reduction, U relative L2 at most
2%, P relative L2 at most 5%, absolute root-peak error at most 5%, and zero
invalid S13 interior-collocation points. A secondary useful signal requires
at least 25% lower absolute peak error than v5, no more than 5% relative
regression in P-field error, U error at most 2%, and zero invalid interior
points. Both outcomes will be reported without relaxing a threshold.

The diagnostic uses only TRAIN labels and is not a G13 trial. It cannot create
`CP-FREEZE`, open DEVELOPMENT or HOLDOUT, or unblock S14. Any production
recovery still requires a separately ratified full-population protocol and
complete gate evidence.

## V6 outcome

The completed canonical report is `S13-OVERFIT-PEAK-V6.json`, SHA-256
`b8da0b8d7aa5319f0f1c17ad30c98b4ab6ef6e88b8e283f40c20a3d0a25fe596`.
It used all 4,000 frozen steps and no DEVELOPMENT access record appeared. V6
reduced combined data MSE by 86.28%, kept U relative L2 at 1.42%, and had
zero invalid vertex or interior-collocation determinants. The absolute peak
error fell to 0.36%, a 96.04% reduction relative to v5. P relative L2 rose
from 5.71% to 6.19%, an 8.46% relative regression. The primary one-case
classification fails its P threshold, and the secondary signal fails its 5%
P non-regression condition. The coefficient is not promoted.

## Predeclared v7 P-field weighting test

V7 is a separate one-factor contrast against v5, not an adjustment to v6.
Its training loss is `DATA_U + 2 * DATA_P`, with no peak term. V5 uses
`DATA_U + DATA_P`; both use normalized targets. The initial v5 unweighted
field components are `0.000225069` for U and `0.000251590` for P, so doubling
P deliberately prioritizes a comparable-sized component. The reported
`normalized_data_mse` remains **unweighted** `DATA_U + DATA_P`; v7 cannot
improve that metric by changing its definition. The case, S11 state, S10
result, normalization, learning rate (`2e-5`), 4,000-step budget, vertex
sampler, and success thresholds are unchanged from v5. The only trained
objective difference is the P-field coefficient. A versioned canonical plan
includes that coefficient directly, without changing historical v3-v6 plan
hashes. An uncommitted first draft bound only the diagnostic ID; this was
corrected before any v7 training or result was produced.

The frozen v7 plan hash is
`7325533cb7bed56e881a6cfd8028963204212ddfcefa3bf368312a18107acbca`.
Its sampler hash is
`f894eeab7ea354a2d2ee0085ba8fef17b521df59329071e380e02749495df861`,
identical to v5's. The primary result uses the existing conjunctive one-case
criteria, including root-peak error at most 5%. Because v7 has no peak loss,
we additionally predeclare a P-field signal: final P relative L2 at most 5%,
U relative L2 at most 2%, at least 80% reduction in unweighted combined data
MSE, and zero invalid interior-collocation points. The peak error is always
reported, even if this secondary signal holds; it cannot be ignored for the
primary classification. We will not select a coefficient or threshold after
seeing v7. This sequential TRAIN-only diagnostic is mechanistic evidence,
not an independent statistical success or a G13 trial. DEVELOPMENT and
HOLDOUT remain closed; CP-FREEZE and S14 remain blocked.

## V7 outcome

The completed canonical report is `S13-OVERFIT-P-WEIGHT-V7.json`, SHA-256
`02f701ef87b02b13c9c5321e35eaf9433161cb2f0fbe1b9aef22bc5159dbb4d4`.
It contains the frozen v7 plan hash, the v5 sampler hash, all 4,000 optimizer
losses, and the same S10/S11/initial-state bindings as v5. No DEVELOPMENT
access record appeared. The unweighted combined data MSE fell 88.08%; U
relative L2 was 1.45%, P relative L2 was 5.10%, and absolute root-peak error
was 7.92%. Both vertex and interior-collocation invalid counts were zero.
Relative to v5, P error improved 10.64%, but it remains above the 5% limit.
The preregistered secondary P-field signal fails narrowly, and the primary
one-case classification fails both P-field and peak limits. The coefficient
is not promoted; G13, CP-FREEZE, and S14 remain unchanged.

## Predeclared frozen-state spatial/physics audit

No further loss coefficient is selected from this one case. The next
TRAIN-only, non-gate diagnostic replays the **unchanged** v5, v6, and v7 plans
from the same pinned S11 checkpoint and selective S10 FEM result. It must
verify each canonical report's SHA-256, plan and sampler hashes, initial and
final model-state hashes, and exact optimizer loss trace before using a replay
for interpretation. It must not write a model checkpoint or access
DEVELOPMENT/HOLDOUT. A replay mismatch stops the audit rather than yielding a
partial comparison.

For each frozen final model, measure P relative L2 separately on the 20
registered right-arc root-QoI vertices and the other 949 vertices; the root
share of total squared P error; the predicted and FEM peak vertex IDs,
coordinates, magnitudes, and 2×2 P components; and root/non-root RMS of the
S13 constitutive residual evaluated at mesh vertices, with minimum
deformation determinants. These are descriptive diagnostics with no new
success threshold or gate claim. Vertex constitutive residual measures
internal consistency of predicted P and the gradient of predicted u; the FEM
stress target is a recovered vertex field, so this residual is **not** itself
an FEM error measure. We will use the spatial pattern to choose a separately
predeclared architecture or physics intervention, not to relabel any failed
v5-v7 arm as passing.

## Frozen-state spatial/physics audit outcome

All three 4,000-step replays reproduced their complete optimizer loss traces
and final state hashes exactly. The canonical report is
`S13-OVERFIT-SPATIAL-PHYSICS-AUDIT.json`, SHA-256
`6ea2b9d6bcb748c6da232675b2030d42177a978794071e56b1f5c169272524e5`.
The DEVELOPMENT access record remained absent.

| Arm | Root P relative L2 | Non-root P relative L2 | Root squared-error share | Non-root constitutive RMS | Minimum vertex det F |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5 | 17.08% | 5.37% | 12.60% | 0.3891 | 0.09267 |
| v6 | 15.10% | 5.97% | 8.37% | 0.4379 | 0.07393 |
| v7 | 15.37% | 4.80% | 12.77% | 0.8283 | 0.03428 |

The 20 root vertices are only 2.06% of the 969 queries, but carry
8.37-12.77% of squared P error. Most total error remains outside the root;
the root is not the entire failure. V7 gets the non-root field below 5%,
while its root tensor error keeps the full field above that threshold.

V6's near-correct scalar peak is at vertex 730, whereas the FEM peak is at
645. Its peak tensor has normalized P12/P21/P22 of
0.04760/0.03965/0.02638, versus FEM 0.009273/0.009321/0.0005686.
Matching a Frobenius maximum therefore did not establish accurate peak
location or components. V5 and v7 peak at 628. These observations explain
why the scalar peak loss is not promoted, without asserting that they
identify the complete cause of the original multi-case S13 failure.

The constitutive residual is an internal-consistency diagnostic, not a
vertex FEM-error estimate. These arms trained on data only. V7's improved
P values coexist with worse constitutive consistency and a lower positive
determinant; good displacement values alone do not establish good
displacement derivatives. Further budget may also help: the last 170-step
mean loss falls about 22% from step 3,000 to 4,000 in all three arms. The
next capacity test therefore needs a matched extra-budget control, not just
a comparison with the original v5 endpoint. No gate state changes.

## FEM recovery reference

An exploratory read-only probe was performed after the spatial audit; its
result is now reproduced by `s13_fem_recovery_reference.py`, not presented
as a preregistered success test. The canonical report
`S13-FEM-RECOVERY-REFERENCE.json` has SHA-256
`a3aa61288694358dcbb6298022d502348aeaa5074db1ef864e2ec984fa7dda7c`.
It reconstructs the existing S10 P2 displacement and recovered stress at
the stored physical quadrature, without solving a new FEM problem. The
maximum difference between reconstructed displacement gradient and stored
F-I is `2.31e-14`.

Recovered versus raw in-plane P has area-weighted relative L2 differences
of 0.2081% on all 1,792 cells, 0.6058% on the 41 cells incident to any of
the 20 registered root vertices, and 0.1888% on the 1,751-cell complement.
The report binds request/result hashes, physical quadrature weights,
population definitions, and the 760 MPa component-RMS normalization.
These are quadrature populations, not the unweighted operator-vertex
populations in the spatial audit; their percentages are not directly
interchangeable. The small recovered/raw discrepancy does not explain the
large model root error by itself, nor does it verify an exact continuum
stress. TRAIN-only access was confirmed; DEVELOPMENT remained closed.

## Predeclared matched stress-capacity contrast

The next bounded diagnostic uses the existing S11c local-stress options,
not a replacement PI-GINOT implementation. ADR 0012's candidate C improved
stress and peak fitting but failed its displacement guardrail. Here shared
parameters and the displacement head remain frozen in both arms, so the
test isolates stress-path capacity while retaining the exact v5 deformation.
This does not overturn the S11 selection or establish physics consistency.

The runner first replays v5 from the pinned S11 checkpoint and checks every
optimizer loss and the final state against the pinned v5 report. Both arms
then start with identical v5 P and u values:

- Control: train only the original linear `p_head`.
- Enriched: use the already implemented local boundary scales `(0.08, 0.25)`
  in the operator's normalized coordinate frame and two residual stress
  layers; train only `p_head`, local projections, and stress residual blocks.

Migration copies every baseline state tensor exactly and rejects unexpected
keys or shape/dtype mismatches. New parameters use seed 13002. The local
projection weights/biases and each residual block's final linear layer are
zeroed, making the added path neutral before training. The report includes
both complete configs, module digests, copied/new state-name allowlists,
trainable names/counts, initial/final states, and the shared frozen-state
hash. Full vertex and interior-collocation displacement, physical Jacobian,
and determinant arrays must remain exactly equal to v5, not just their
minimum values or invalid counts.

Both arms receive fresh CPU float64 Adam with learning rate `2e-5`, betas
`(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, and 4,000 additional steps.
The deterministic 57-vertex v5 schedule restarts identically for both arms;
this is matched fine-tuning, not continued Adam state. The only objective is
the unchanged `DATA_U + DATA_P`, with no peak term, reweighting, or physics
term. Every finite loss is retained. Displacement data loss is constant
because its parameters are frozen. Six implementation-file SHA-256 hashes
are captured before execution and must be unchanged afterward.

The canonical contrast plan hash is
`8ad9644d53e8363d95030f6f1a67fcc7cea1fcc0616f6f0678244b3533846ae2`.
The 4,000-step fine-tuning sampler hash remains
`f894eeab7ea354a2d2ee0085ba8fef17b521df59329071e380e02749495df861`.
The original one-case criteria stay fixed: cumulative combined data-MSE
reduction at least 80% relative to the **original pinned S11 initial state**,
U relative L2 at most 2%, P relative L2 at most 5%, absolute root-peak error
at most 5%, and zero invalid interior points. Additional MSE reduction from
v5 is reported separately; it is not substituted as the 80% denominator.

The enriched contrast succeeds only if it meets those original criteria,
reduces root tensor relative L2 by at least 25% versus the matched control,
and has no more than 5% relative non-root P regression versus that control.
All raw metrics and both classifications are reported, even on failure.
No coefficient or threshold will be selected after this result. This is one
TRAIN case, non-gate mechanistic evidence, with no saved model checkpoint,
new FEM solve, DEVELOPMENT/HOLDOUT access, or production promotion. G13
remains failed, CP-FREEZE withheld, and S14 blocked; a production recovery
still requires a separately ratified full-population protocol.

## Matched stress-capacity outcome

The completed report `S13-STRESS-CAPACITY-CONTRAST.json` has SHA-256
`a87d52ce180718d7709462855e90994e62080291d45fbcde68f5e854d607f33e`.
A complete second execution reproduced the entire canonical report
byte-for-byte, including both 4,000-loss traces and final state hashes. Both
executions first reproduced v5 exactly. All six code fingerprints remained
unchanged; DEVELOPMENT access records remained absent.

| Arm | Global P relative L2 | Root P relative L2 | Non-root P relative L2 | Absolute peak error |
| --- | ---: | ---: | ---: | ---: |
| v5 warmstart | 5.7065% | 17.0820% | 5.3729% | 9.1445% |
| Matched linear-head control | 5.6526% | 17.1463% | 5.3119% | 8.9428% |
| Existing stress enrichment | 3.9391% | 11.8003% | 3.7085% | 5.4355% |

The control trains 260 parameters; the enriched arm trains 41,988. Both
preserve every frozen state tensor and the full displacement/Jacobian/
determinant arrays. U relative L2 remains 1.4089%, with zero invalid vertex
or interior-collocation points. Combined MSE falls 87.48% from original S11
in the control and 90.37% in the enriched arm. Additional reduction from
v5 is 0.85% versus 23.75%, respectively.

The enriched arm reduces root tensor error by 31.18% and non-root error by
30.19% versus the matched control, meeting both slice-comparison conditions.
It passes the original one-case global P limit, but misses the peak limit:
5.4355% is greater than 5%. Its conjunctive classification and the capacity
contrast both remain **false**. Both arms still peak at vertex 628 rather
than FEM vertex 645. The capacity hypothesis gains mechanistic support, but
the arm is not promoted and no threshold is relaxed.

Non-root constitutive RMS is 0.38915 in the control and 0.38938 with
enrichment; root RMS is 0.29972 versus 0.30045. The supervised P improvement
has not repaired displacement-derived material consistency. Frozen U makes
this expected scope limitation, not evidence of a material-kernel defect.

The next descriptive check should use the stored physical FEM quadrature,
not a vertex-gradient average: compare model grad u with stored F-I, served
P with raw FEM P, and material stress recomputed from FEM gradients with
raw FEM P. The latter is a pipeline-consistency check, not independent
physics verification. Report every inadmissible quadrature point and its
area weight; conditional material metrics must state their admissible
coverage rather than drop those points silently. This can distinguish
kinematic fitting from a material/layout/scaling mismatch before selecting
a separately preregistered training intervention. No new loss coefficient
is selected here. G13 remains failed, CP-FREEZE withheld, and S14 blocked.
