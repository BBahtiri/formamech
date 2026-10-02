# ADR 0018: Replace the S13 operator with the energy-form PI-GINOT (S13c)

- Status: Accepted — ratified by the user on 2026-10-02 (decision D1). The
  same decision parks the joint-hybrid contrast of ADR 0017 unexecuted (D2).
  Drafted as ADR 0017 on `codex/s13c-enhanced-model` and renumbered when the
  S13 lines were merged, because ADR 0017 was already the ratified contrast
- Date: 2026-10-01; accepted 2026-10-02
- Step: S13 recovery (S13c), branch `codex/s13c-enhanced-model`
- Supersedes for S13c: the S11 baseline-A operator as the S13 model; amends
  `G11.heads` for the new architecture only (see Decision 2)

## Context

S13 has no admissible checkpoint. The six registered hybrid fine-tunes of
the S11 baseline improved the P field (median relative L2 0.192 to 0.142) but
made the root peak worse (median absolute error 0.081 to 0.123, all signed
negative); the selector admitted none (`S13-TRAIN-SCREEN-FAILED.json`,
ADR 0015). The S13b peak arm fixed the peak (0.114 to 0.043) only by
regressing the P field by 8.7% and producing one folded collocation point
(`S13B-TRAIN-PILOT.json`, `S13B-TRAIN-INVALID-POINT.json`).

The one-case recovery diagnostics on `codex/s13-recovery` (ADR 0016, read
only) located the mechanism: the S11 operator reproduces displacement values
to about 1.6% but its displacement gradients are wrong (relative H1
seminorm error 296%), so the stress recomputed from its own kinematics is
268% off. The served stress comes from an independent `P` head; field, peak
and physics residuals therefore pull against each other, and the predicted
peak sits at the wrong vertex (628 instead of 645).

The energy-form PI-GINOT in `pi-ginot-enhanced` (published commit `12c9c1c`)
addresses exactly this. It trains on the total potential energy with exact
Dirichlet lifting, derives stress from the displacement gradient, and on the
same dog-bone family (same parameter box, material and quarter model, but a
1 mm load) reported 1.0% in-range K_t error and 0.5-1.3% von Mises relative
L2 against an independent FEM reference on held-out geometries.

## Decision

1. **Architecture** `pi-ginot-energy-parametric-film-v1`
   (`src/piginot/operators/energy_operator.py`), a port of the enhanced
   operator with the oracle (parameter) conditioning used for its reported
   results: a parameter MLP to 32 latent tokens of width 64; NeRF query
   features (6 octaves); a pooled-token bias; six pre-norm residual
   cross-attention blocks, each followed by identity-initialised FiLM; a
   linear correction head initialised at N(0, 1e-5). 573,474 parameters for
   five conditioning inputs. Attention is written explicitly so the
   double-backward path does not depend on the SDPA backend.
2. **Served stress is constitutive.** `P_served = P_NH(I + grad u_pred)` through
   the S4 plane-stress kernel. This deliberately amends `G11.heads` ("`P` is not
   derived constitutively at inference") for this architecture: there is no
   independent `P` head, so field, gradient, stress and peak are one
   consistent object and `L_constitutive` is identically zero by construction.
   Decided with the user on 2026-10-01.
3. **Exact Dirichlet lifting.** For the quarter model
   `u_x = d (xi + xi (1 - xi) phi_x)`, `u_y = d eta phi_y` with `xi = x / (L/2)`,
   `eta = y / (W_grip/2)` and `d = epsilon L / 2`. The left symmetry, bottom
   symmetry and loaded-end conditions hold exactly; the free edges are natural
   conditions of the energy minimum. The correction is measured in units of
   `d`, so one network serves the whole 10-20% strain range.
4. **Objective** `potential-energy-stratified-triangles-v1`
   (`src/piginot/objectives/energy.py`): `Pi / (E A)` on the S10 morph mesh of
   each case with three uniform points per triangle redrawn every step from the
   torch CPU generator, plus `1e3 * mean(relu(0.05 - J)^2)`. The S4 energy is
   evaluated only where `det F > 1e-6`; the material law is never extended to
   folded points. Training reads no FEM labels. The S12 registry and every
   legacy S12/S13 hash are unchanged.
5. **Trainer** (`src/piginot/training/energy_engine.py`): the S13 identity and
   persistence surface is reused unchanged (`TrainingRunBinding`, sampler plan,
   checkpoint/resume including Adam moments and generator state), with the
   enhanced schedule: Adam 3e-4 for 1,600 steps, cosine to 3e-6 at 2,400,
   batch 4, gradient clip 1.0, float32 on two threads.
6. **Conditioning** on `L_total`, `W_grip`, `W_gauge`, `R_fillet` and the
   nominal strain, z-scored with the frozen S11 TRAIN statistics. E and nu are
   constant in the envelope.
7. **Selection is unchanged.** One registered trial; the frozen S11-derived
   S13 guardrails and ranking apply. The ranking tie-breaker field
   `hybrid_loss` carries the mean TRAIN-screen nondimensional potential energy.
8. **Reproducibility fixes** found while porting (also benefit S13):
   checkpoint safetensors headers are key-sorted (the writer randomised
   `__metadata__` order per process, so identical runs had different
   `weights_hash`); S10 requests rebuilt off Windows recover the libm-dependent
   bits of 2 node sets and 34 quality-angle pairs, accepted only when every
   request-plan hash then matches exactly.

## Consequences

- `G11` must be re-run for this architecture with the amended heads rule.
  Boundary-cloud checks (`permutation`, `resampling`, `mask`) do not apply to a
  parameter-conditioned operator; query-order and query-batch invariance,
  reachability and safetensors are covered by tests.
- The energy objective is a new term family; its exact fixtures (identity,
  uniaxial plane stress), energy-gradient-equals-stress and finite-difference
  checks are in `tests/objectives/test_energy.py`.
- The operator is specific to the parameterised family; a boundary-cloud
  encoder can be restored later without changing the decoder or objective.
- No DEVELOPMENT or HOLDOUT label is read; `CP-FREEZE`, `artifacts/gates/S13.json`
  and S14 remain a separate decision.

## Verification

Preregistration: `artifacts/campaigns/dogbone-v1/S13C-PROTOCOL.json`. Unit and
integration tests: `tests/operators/test_energy_operator.py`,
`tests/objectives/test_energy.py`, `tests/training/test_energy_engine.py`,
`tests/campaigns/test_dogbone_s13c_*.py`,
`tests/campaigns/test_dogbone_s10_portability.py`. TRAIN-screen evidence:
`artifacts/campaigns/dogbone-v1/S13C-TRAIN-SCREEN.json` (outcome below).

## Outcome (TRAIN-only, 2026-10-01)

The single registered trial trained for all 2,400 steps from revision
`8736c1a` (run `0ca34e32…`, final manifest `a978c3d6…`) in 2 h 54 min of
wall time on two CPU cores. The determinant barrier never activated and no
quadrature point was inadmissible; the smallest sampled `det F` during
training was 0.947.

TRAIN-screen evidence, `S13C-TRAIN-SCREEN.json` (64 FEM cases, unchanged S13
metric code, evaluator revision `fdfda00`):

| Metric | S11 incumbent | S13 trial 0 | S13b peak arm | S13c | Frozen limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Median absolute root-peak error | 8.086% | 12.255% | 4.313% | **0.828%** | ≤ 8.086% |
| p90 absolute root-peak error | 13.635% | 17.334% | 7.647% | **1.693%** | ≤ 13.635% |
| Median signed root-peak error | -7.762% | -12.255% | -4.111% | **-0.288%** | ≥ -6.986% |
| Median P-field relative L2 | 19.200% | 14.236% | 16.615% | **0.909%** | ≤ 19.584% |
| Median u-field relative L2 | 3.731% | 3.502% | 3.673% | **0.081%** | ≤ 3.805% |
| Invalid interior points (256 TRAIN cases) | 0 | 0 | 1 | **0** | 0 |

`select_checkpoint` with the frozen policy admits the trial (no failed
guardrail). Descriptive diagnostics on the same cases: the model strain
energy exceeds the stored FEM strain energy by a median 0.0087% (maximum
0.074%); root-vertex P relative L2 is 2.97% and the remaining field 0.82%;
the predicted root peak lies a median 0.54 mm from the FEM peak vertex (the
same vertex in 19 of 64 cases); 36 of 64 cases under-predict the peak, versus
55 of 64 for the S11 incumbent. The minimum `det F` is 0.990 at the replayed
interior points and 0.963 at the screen-case vertices.

`S13C-G13-REPLAY-PROBE.json`: an independent process replayed the registered
trial from a clean checkout of `8736c1a`, interrupted at step 200 and resumed
to 400; model export, full training state and manifest are byte-identical to
the registered step-400 checkpoint (`G13.determinism` and `G13.resume` PASS
on real data).

Issue status: I1-I5 are resolved on TRAIN evidence. I6 is not: closing `G13`,
writing `CP-FREEZE` and starting S14 still require (a) ratifying this ADR,
including the `G11.heads` amendment and a G11 re-run for the
parameter-conditioned operator, where the boundary-cloud checks are not
applicable, and (b) the remaining `G13` gate artifact. DEVELOPMENT and HOLDOUT
remain unopened.

## Acceptance and the amended G11 check list (2026-10-02)

The user ratified this ADR on 2026-10-02 together with three consequences:
the joint-hybrid contrast of ADR 0017 is parked unexecuted, the S13 lines are
merged into `codex/s13-closeout`, and G11 is re-run for this architecture
before `G13` and `CP-FREEZE` are written.

`artifacts/gates/S11.json` stays the record for the S11 two-head operator.
The re-run for `pi-ginot-energy-parametric-film-v1` is a separate artifact,
`artifacts/gates/S11-ENERGY.json`, measured on the registered S13c checkpoint
in float64. Its required checks are fixed here, before measurement:

| Check | Rule | Threshold |
| --- | --- | --- |
| `G11.node-order` | reverse the query order; compare `u` and `P` | relative change < 1e-6 |
| `G11.query-batch` | split the queries into three batches | relative change < 1e-6 |
| `G11.reachability` | every conditioning parameter has a non-zero gradient path to `u` and to `P` | 5 of 5 |
| `G11.safetensors` | a pickle payload is refused before deserialisation | true |
| `G11.heads` (amended) | `u` is `(B, Q, 2)`, `P` is `(B, Q, 2, 2)`; the operator's only output layer has two features (no independent stress head); served `P` equals the gradient of the S4 plane-stress energy density at the predicted `F` | all true; relative difference < 1e-10 |
| `G11.lifting` (replaces `G11.permutation` and `G11.resampling`) | Dirichlet values on `x = 0` (`u_x = 0`), `y = 0` (`u_y = 0`) and `x = L/2` (`u_x = d`) for 64 parameter vectors (the 32 corners of the parameter box, its centre and 31 seeded interior draws) at 320, 480 and 4,800 points per edge | maximum error / `d` < 1e-12 |
| `G11.lifting-mutant` (replaces `G11.mask`) | the `IGNORE_DIRICHLET_LIFTING` mutant replaces the lifting by the identity (offset 0, scale 1) | fires: control < 1e-12 and mutant > 1e-3 |
| `G11.coverage` | `tests/operators` coverage of `piginot.operators` | >= 80% |

The boundary-cloud checks do not apply because this operator receives no
boundary point cloud; the lifting checks are the full-range,
boundary-refinement verification that S11 requires before hard Dirichlet
enforcement may be used. Two S13 deliverables are read as follows for this
architecture: the declared search space is the single registered trial, and
EMA adaptive weighting is not used because the objective is one energy term
with a fixed barrier penalty (the S13 adaptive controller remains available
to the hybrid objectives). `G13` is written to `artifacts/gates/S13.json` and
the frozen candidate to `artifacts/campaigns/dogbone-v1/CP-FREEZE.json`.
