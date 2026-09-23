# ADR 0014: S13 orientation recovery during hybrid training

- Status: Proposed
- Date: 2026-09-23
- Step: S13
- Gate: `G13` remains open

## Context

The selected S11 checkpoint is a supervised neural operator, not an admissible
deformation parameterization. On a diagnostic replay of the registered 192
TRAIN-fit cases, 19 cases had 29 interior collocation points with predicted
`det(I + grad_X u) <= 1e-12`; the minimum was about `-0.630`. The scan did not
open DEVELOPMENT or HOLDOUT and is not G13 evidence. Without a local recovery
path, S4's plane-stress closure correctly refuses these points, so the S13
fine-tune stops before the full TRAIN-fit population can be traversed.

## Proposed decision

Keep the S4/S12 constitutive residual exactly unchanged wherever
`J = det(I + grad_X u) > 1e-12`. For each invalid interior point, replace its
four in-plane constitutive residual components with the same dimensionless
recovery residual:

`r_orientation = 100 * (0.1 + 1e-12 - J)`.

No invalid point is dropped or passed through an undefined material-law
extension. The determinant is expanded explicitly as a 2-by-2 polynomial so
the recovery gradient remains nonzero at an exactly singular deformation;
the generic determinant backward path produced a zero gradient there in the
observed PyTorch runtime. The `0.1` buffer keeps the residual at least `10`
even when `J` is zero or only just below the admissibility threshold. The
gain, buffer and policy ID are bound into the S13 protocol before any
TRAIN-screen selection.

The proposed eligibility check counts invalid predicted interior points over
the complete registered TRAIN population, both fit and screen, at the
persisted S10 collocation coordinates. A candidate with any invalid point is
ineligible for CP-FREEZE, irrespective of its peak-stress error. Accuracy and
hybrid-loss ranking remain TRAIN-screen-only. The latter is the arithmetic mean
of per-case strong-HYBRID totals under the trial's fixed physics scale, with
no path-dependent adaptive training weights. The 64-screen/all-TRAIN metric
policy has a separate frozen protocol identifier. DEVELOPMENT and HOLDOUT are
not training or selection inputs.

## Consequences and limits

This is an optimization recovery term, not a constitutive law or a proof of
global orientation preservation. It is discontinuous at the admissibility
threshold, where S4's original constitutive residual takes over; the
bounded pilot and full-trial diagnostics must reveal whether training can
cross that boundary stably. Zero invalid points at finite S10 collocation
locations does not prove positivity everywhere in the continuous domain.
The discrete FEM mesh and S6 morph validity are separate guarantees.

If every registered trial remains inadmissible, fails the S11-derived
accuracy guardrails, or fails deterministic/resume/lineage checks, no
checkpoint is selected: CP-FREEZE and G13 remain withheld. This proposal
does not relax the preregistered six-trial search or the S11 incumbent
guardrails.

## Verification still required

Unit tests must show bitwise equality with S12 constitutive residuals for
valid points, a positive penalty and nonzero corrective gradient at singular
and threshold points, and exclusion of invalid candidates. A full TRAIN-fit
pilot must traverse the initially folded cases without nonfinite losses.
Production evidence additionally requires the complete registered trials,
TRAIN-screen metrics, deterministic rerun and resume comparison, checkpoint
lineage, safe export, and the remaining G13 checks.
