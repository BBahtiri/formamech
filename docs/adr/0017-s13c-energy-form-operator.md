# ADR 0017: Replace the S13 operator with the energy-form PI-GINOT (S13c)

- Status: Proposed — implementation and TRAIN-only evidence complete; awaiting
  user ratification before any `CP-FREEZE`, `G13` or S14 decision
- Date: 2026-10-01
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
