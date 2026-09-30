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
