# ADR 0015: Diagnose and pilot S13 peak-stress preservation

- Status: Accepted — bounded TRAIN-only pilot ratified 2026-09-26
- Date: 2026-09-25
- Step: S13 recovery, after the failed six-trial TRAIN screen
- Gate: `G13` remains open; `CP-FREEZE` is withheld

## Observed outcome

All six registered S13 trials completed 1,500 optimizer steps and were screened
on the frozen 64-case TRAIN-screen population. The selector admitted none. The
exact trial results, frozen guardrails, checkpoint lineage, and source-log digest
are retained in `artifacts/campaigns/dogbone-v1/S13-TRAIN-SCREEN-FAILED.json`.
No DEVELOPMENT or HOLDOUT label was used for this decision.

Trial 0 (`learning_rate=1e-5`, `physics_weight_scale=1e-4`) is the best of the
six on median absolute root peak error, yet its error is 0.12255 versus the
S11 incumbent's 0.08086. Its median signed peak error is -0.12255 versus
-0.07762 for S11. The trial's median P-field relative L2 improves from
0.191996 to 0.142356; its sampled invalid-collocation count is zero. Thus
field improvement is real, but it does not satisfy the peak-stress requirement.

A read-only, per-case replay compared the pinned S11 model with trial 0's
verified final checkpoint on the same 64 TRAIN-screen FEM cases. It used
`iter_train_screen()`, `prepare_operator_case()`, the pinned S11 normalization,
and the existing vertex predictor. The baseline median signed peak was
-0.07762409959123628, agreeing with the frozen S11 metric up to floating-point
roundoff; trial 0 was -0.12254976158510011, agreeing with its batch screen.
The S11 model underpredicted the peak in 55/64 cases, versus 63/64 for trial 0.
Trial 0 worsened absolute peak error in 57/64 cases while improving P-field
relative L2 in all 64. This is broad peak bias, not a few outliers. The replay
did not run FEM, train, open DEVELOPMENT/HOLDOUT, or select a checkpoint.

## Diagnosis and limit

S11's supervised objective includes a differentiable root-QoI peak-P mean
square term at weight 1.0. S13 retains normalized DATA_U and DATA_P plus five
strong-form physics residuals, but no peak term; its frozen selector still
requires improvement in that peak. This objective/selection mismatch is a
plausible mechanism for the measured field/peak trade-off, not proof of
causality. The physics residuals, optimizer schedule, and full-vertex training
queries also differ from S11.

S11's training peak used a sampled QoI query suffix. S13's selection peak is
the maximum Frobenius norm of P on `qoi_indices` from the full mesh: `right_arc`
excluding `right_grip`. Any S13 recovery term should target those registered
indices, not assume a 64-query suffix.

## Ratified bounded experiment; not authorized as G13 evidence

Create a separate, preregistered S13b TRAIN-only pilot. Keep the S11 initial
weights, 192 TRAIN-fit cases, 64 TRAIN-screen cases, normalization, sampler
prefix, finite-strain constitutive residual, orientation recovery, and frozen
selection guardrails. Do not alter the six S13 results or their protocol hash.

Use the trial-0 configuration (learning rate 1e-5, physics scale 1e-4) for a
matched 800-step comparison. The existing trial-0 step-0800 checkpoint can be
the control after its lineage is verified; train one new peak arm from the
pinned S11 model for 800 steps. Add an explicit supervised root-QoI peak-P
mean-square loss with coefficient 1.0 in normalized stress units, bound to a
new S13b protocol and run hash. The implementation must not silently change
S12's objective-registry or the legacy S13 hashes. Test the exact QoI indices,
gradient path, role isolation, deterministic replay, and checkpoint binding
before the pilot run.

The prepared implementation makes `DATA_PEAK` an opt-in objective term. Its
residual is the signed difference between predicted and FEM root-QoI peak
stress; the registry squares and averages it at weight 1.0. The default
strong-HYBRID registry remains byte-identical to the legacy recipe hash
`b6bb39d2b12b08c81ebc5fb81fa78782c6ea6f4abe3ced8325483b9ead08d887`;
the opt-in peak registry has distinct hash
`c32436ec8646dc5eea647f003453672a4605b4229498d5d5e2eaf0b3b363c36b`.
The ratified one-arm pilot plan hash is
`9c58e3ea932ad03dfe298a2759e06a0dd017d09a8face21513e2c2e66dfae89e`.
Its control is the original S13 trial-0 step-0800 checkpoint, whose manifest
hash is `dedb463a0a8eb490c4b413a911c0d02b9ac84625a506acd2054cdca9d493fcbd`;
it was loaded read-only with the original binding, not relabelled as S13b.

Screen both arms on the same 64 TRAIN-screen cases at step 800 and count invalid
collocation points across all 256 TRAIN cases. A useful signal requires the new
arm to reduce median absolute and p90 absolute peak errors and improve median
underprediction by at least 10% relative to the matched control, with zero
invalid points and no more than 2% regression in either median field L2. These
are pilot comparison criteria, not replacements for the frozen S11-derived
guardrails. Even a successful pilot cannot create `CP-FREEZE` or close `G13`;
those require a newly ratified full-budget protocol and its own complete gate
evidence.

## Ratification and boundary

The user approved continuing with this exact coefficient, 800-step
matched-control design, and separate S13b identity on 2026-09-26. That
authorizes this TRAIN-only pilot, not a full six-trial search or a gate claim.
If the pilot fails, inspect the optimizer/physics trade-off instead of
widening the frozen selection thresholds. S14 remains blocked throughout.

## Completed TRAIN-only pilot outcome

The matched step-800 comparison is retained in
`artifacts/campaigns/dogbone-v1/S13B-TRAIN-PILOT.json` (commit `abb9a00`,
SHA-256 `6523895342ad68edf624991b65a4c1a6ad30bc295d5e944afb3c69d31077037b`).
Both checkpoints passed their pinned lineage checks. The same 64 registered
TRAIN-screen FEM cases were evaluated for each arm, and the invalid-point count
used all 256 TRAIN cases. DEVELOPMENT and HOLDOUT remained unopened.

The peak arm reduced median absolute root-peak error from 0.114342 to 0.043127
(62.3%) and p90 absolute peak error from 0.160802 to 0.076469 (52.4%). Its
median P-field relative L2, however, increased from 0.152807 to 0.166150
(8.73%), beyond the preregistered 2% non-regression allowance. U-field relative
L2 changed from 0.037101 to 0.036733. The peak arm also had one invalid
collocation point versus zero for the control. A separate matched per-case
TRAIN replay found P-field error worse in 63/64 cases and absolute peak error
better in 62/64; the field regression is not an isolated outlier.

The read-only determinant locator reproduced the single invalid count and the
same peak-checkpoint manifest. Its non-gate record is
`artifacts/campaigns/dogbone-v1/S13B-TRAIN-INVALID-POINT.json`: TRAIN-FIT case
`candidate-002485`, collocation index 25, coordinates
`(24.720325328268974, 2.341752661726852)`, with
`det(F) = -0.001147183246821691`. This is a genuine sampled orientation
reversal, not roundoff near the `1e-12` admissibility margin. The result does
not certify or refute global validity away from the sampled points.

The preregistered pilot signal is **false**. The improved peak does not justify
promoting this arm, relaxing the frozen thresholds, or claiming `G13` evidence.
`G13` remains not passed, `CP-FREEZE` is withheld, and S14 remains blocked.
The scalar maximum-peak loss plausibly overweights the QoI relative to the
distributed P-field objective, but the replay does not establish that as the
unique cause of either failure. A reduced peak coefficient is a bounded next
hypothesis; its coefficient, comparison rule, and budget need a separate
preregistration before any further training.
