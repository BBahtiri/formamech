# ADR 0017: Proposed joint U/P strong-HYBRID contrast

Date: 2026-10-01

Status: **ACCEPTED — exact proposal ratified in this chat on 2026-10-01;
PARKED, NOT EXECUTED — superseded by ADR 0018 (S13c) by user decision on
2026-10-02**

Parking note (2026-10-02): the user adopted the S13c energy-form operator
(ADR 0018) as the S13 model and decided not to run this contrast. The runner,
tests, proposal and approval record are committed unchanged for the record.
No result report exists for this ADR. Any future execution needs a fresh
decision that names the exact proposal and plan hashes below.

The user responded "i approve and continue" to the sign-off request naming the
exact artifact below. `S13-JOINT-HYBRID-APPROVAL.json` records that ratification;
it is an operator evidence record, not cryptographic attestation. The original
draft artifact is preserved byte-for-byte, including its draft-only flags.

## Question and boundary

Does adding the existing strong-HYBRID physics objective improve displacement
gradients in a matched joint U/P fine-tune from the exact v5 data-only state?
This tests a TRAIN-only, one-case intervention, not the full S13 population and
not the isolated effect of constitutive loss. No new PI-GINOT architecture or
physics implementation is proposed. The production operator still serves its
learned P head; material stress reconstructed from grad U is diagnostic only.

The prior physical-quadrature comparison found v5 U relative L2 of 1.6135%,
gradient relative H1 seminorm error of 296.1507%, and material-from-model-gradient
stress error against raw FEM P of 268.2270%, globally. The last quantity is
**not** the internal constitutive mismatch served P minus M(model grad U).
These are findings from a DATA_U + DATA_P diagnostic, not proof that the
original full-population HYBRID run has the same cause.

The public `S13-JOINT-HYBRID-PROPOSAL.json` and immutable `JointHybridPlan`
specify the proposed policy. The writer reads only the two committed public
reports named in the plan, verifies their canonical bytes, SHA-256 and lineage,
and emits an unauthorized draft. It contains no training entry point. Its
existence, passing tests, and the user's instruction to continue drafting do not
authorize optimization, including historical warmstart reconstruction.

The exact draft artifact SHA-256 is
`b613b4dc884e0b34d5a253fd81154abd11406e3b5d67ef1d89ab1e88f6cf7e0a`;
the plan content hash is
`c7e698556c23b4163d5ff6a22a1a30cd600904312e7f7e2a445bafdf58b718e2`.
These identify a proposal, not accepted training evidence.

## Exact proposed contrast

Use only TRAIN-fit `candidate-002485`, its pinned request/result, original S11
initialization, and exact v5 baseline architecture/state. DEVELOPMENT and HOLDOUT
remain closed. No new FEM solve, checkpoint, enrichment or architecture promotion.
The pinned v5 and quadrature reports also bind source identity and historical
implementation fingerprints; these must be checked before and after execution.

After approval, reconstruct v5 once with its **historical** 4,000-update algorithm,
including no gradient clipping, and require equality of every optimizer loss,
sampler hash and final model-state hash. A failure aborts; no tolerance relaxation.
Clone that terminal state into both arms and jointly train every shared, U-head
and P-head parameter. Their initial states and predictions must agree exactly;
their final U, grad U and determinants are **not** required to remain equal.
Verify every parameter name/count/trainability plus config and module digest,
not just tensor-state hashes. Do not reuse `build_stress_arm(enriched=False)`,
which freezes non-stress parameters. Both arms restart the same sampler at step
zero and receive independent fresh optimizers, never v5 optimizer moments.

| Arm | Fixed normalized objective |
| --- | --- |
| Matched data-only control | DATA_U + DATA_P |
| Candidate | DATA_U + DATA_P + 0.001 × (strong equilibrium + constitutive + angular momentum + Dirichlet + traction-free) |

Each term uses the existing residual implementation and unweighted mean-square
reduction. DATA_P supervises recovered FEM stress at the same 57 selected mesh
vertices per update in both arms; it does not supervise raw quadrature stress.
No peak term, stress reweighting, direct FEM-gradient/H1 labels, adaptive weighting,
learning-rate scheduler, coefficient search, or extra orientation loss in control.

The proposed physics scale `1e-3` is the ordinal middle of the original S13 grid
`{1e-4, 1e-3, 1e-2}`. This is a **new, approval-required choice**, not a historically
selected scale or an outcome-calibrated recommendation. Static weights remove
adaptive-controller and independently responsive scheduler changes from the
contrast; this diagnostic is not a replay of the original adaptive HYBRID package.

Each arm gets fresh CPU/float64 Adam, learning rate `2e-5`, betas `(0.9, 0.999)`,
epsilon `1e-8`, zero weight decay, foreach/fused false, and exactly 4,000 additional
updates. Seed 13001 and the identical pinned v5 57-of-969 vertex schedule are
used in both. Runtime is PyTorch `2.13.0+cpu`, eight intra-op/eight inter-op threads.
Global gradient clipping at norm 1 is shared by both arms. This reuses production
S13's rule but is **new relative to v5**; control is a matched fine-tune with fresh
Adam and clipping, not an unclipped continuation. Per-step preclip gradient norm
and clip coefficient/exposure must be recorded. Unequal exposure is an outcome
of the different losses and limits interpretation as an unclipped physics effect.

The finite budget is one common 4,000-update reconstruction plus two 4,000-update
arms: 12,000 total optimizer updates, 8,000 new diagnostic updates. No rerun,
extension, third arm or retuning without a separately approved protocol.

## Sampling, validity and evaluation

Candidate physics uses the same recorded 256 collocations: 128 interior, 64
background boundary and 64 right-arc boundary-quota points. Strong equilibrium,
constitutive and angular terms apply to the 128 interior points; boundary terms
use their existing incidence/component masks. These are fixed sampled points,
not physical-area-weighted FEM quadrature. The existing builder supervises all
vertices, so the future runner needs a tested adapter enforcing the shared
57-vertex data schedule; it must not call the full-vertex builder unmodified for
candidate and silently make this an unequal-data contrast. The generic S13
engine refuses DATA_ONLY; a zero physics scale is not a control implementation.

Candidate retains the existing v2 orientation treatment: det F admissible only
above `1e-12`; on invalid interior rows replace all four constitutive residual
components by `100 * (0.1 + 1e-12 - det F)`. The `0.1` is an invalid-row offset,
not a minimum determinant for valid F. The material law is never extended to
invalid points and points are never removed. Both arms abort on nonfinite loss,
gradients or parameters; any material-kernel failure on admissible F also fails
closed. Finite invalid points are recorded rather than secretly adding a physics
loss to the data-only arm. Sampled positivity is not continuous injectivity.

Evaluate at additional-update counts 0, 1,000, 2,000, 3,000 and 4,000. Only the
fixed endpoint 4,000 is compared; no best-step selection or early stopping.
Evaluation must preserve model parameters, optimizer state, RNG and mode.
Record complete per-step objective/term-loss traces and clipping diagnostics,
with sparse physical metrics at the declared evaluation steps.

Retain the existing physical-quadrature evaluator: all 1,792 cells × seven
points = 12,544, reference-domain area weights in mm squared, cell-major chunks
of 512, and the existing root-incident/non-root populations. Report:

- Unconditional U relative L2, grad U relative H1 seminorm (not full F), and
  served P relative L2 against raw in-plane FEM P, including invalid points.
- Material-from-gradient P relative L2 against raw FEM P, internal constitutive
  A=served P-M(model grad U), B and C decomposition, component RMS/biases, and
  signed cross energies on explicitly admissible support.
- Invalid point IDs/determinants/area, admissible counts/area coverage, det ranges,
  vertex/interior validity, and FEM-gradient/material consistency checks.
- Candidate/control and candidate/pinned-v5 endpoint ratios and signed changes
  for all/root/non-root populations. Conditional ratios are NA unless their
  admissible point supports match exactly; zero denominators are NA, not invented
  finite ratios. Unconditional metrics always retain the full population.

There is **no new numeric repair threshold or pass label**. Descriptive paired
results may motivate a subsequent separately approved change; they cannot
establish accuracy recovery. Report the unchanged historical vertex/recovered-P
classification separately: at least 80% DATA MSE reduction from the **original
S11** state, U error ≤2%, P error ≤5%, absolute root-peak error ≤5%, and zero
invalid interior collocations. Its invalid-vertex count is diagnostic, not a
retroactively added criterion. Additional MSE reduction from v5 is separate and
never substituted for the original denominator.

## Before execution and afterward

Human approval must name the exact proposal/plan hashes. Then implement and
review a dedicated runner with synthetic tests for shared sampling, fresh
optimizers, joint parameter scope, residual equivalence, clipping/finite aborts,
snapshot non-mutation, exact replay, and role/private-access boundaries. Pin the
reviewed runner and dependencies before opening TRAIN payloads or optimizing.
The approval is not permission to edit fingerprinted historical modules or to
weaken replay checks. Use one exclusive canonical public result report, no saved
model checkpoint; preserve failure reasons and access/runtime/code checks.

An authority, runtime-pin, clean-revision or safe-output-path refusal before
the execution boundary does not start a training attempt and does not publish
an approved TRAIN result. It raises a diagnostic without loading checkpoints
or source labels. After that boundary passes, any initialization, replay,
training or postflight failure produces the single canonical FAILED report,
including the failure stage and all completed joint-arm evidence; no retry.

No runtime or storage estimate is measured for this proposed contrast, and no
remote/paid job or push is authorized. Execution is local and bounded by the
stated update budget. A full 192-fit/64-screen recovery policy requires separate
ratification after interpreting this diagnostic. G13 remains **NOT_PASSED**,
CP-FREEZE **WITHHELD**, S14 **BLOCKED**, regardless of a favorable one-case result.
