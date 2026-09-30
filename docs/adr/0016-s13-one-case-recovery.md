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
