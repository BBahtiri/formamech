# ADR 0015: Diagnose and pilot S13 peak-stress preservation

- Status: Proposed
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

## Proposed bounded experiment; not yet authorized as G13 evidence

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

Screen both arms on the same 64 TRAIN-screen cases at step 800 and count invalid
collocation points across all 256 TRAIN cases. A useful signal requires the new
arm to reduce median absolute and p90 absolute peak errors and improve median
underprediction by at least 10% relative to the matched control, with zero
invalid points and no more than 2% regression in either median field L2. These
are pilot comparison criteria, not replacements for the frozen S11-derived
guardrails. Even a successful pilot cannot create `CP-FREEZE` or close `G13`;
those require a newly ratified full-budget protocol and its own complete gate
evidence.

## Decision needed

Before implementing the new objective or running this pilot, ratify its
coefficient, 800-step matched-control design, and separate S13b identity. If
the pilot fails, inspect the optimizer/physics trade-off instead of widening
the frozen selection thresholds. S14 remains blocked throughout.
