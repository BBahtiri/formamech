# S13c plan — replace the S13 model with the energy-form PI-GINOT

- Status: Steps 1-3 complete on branch `codex/s13c-enhanced-model`; the
  registered checkpoint passes every frozen S13 guardrail on TRAIN evidence
  (see "Outcome" at the end); ADR 0018 (drafted as ADR 0017) accepted by the
  user on 2026-10-02; `G13`/`CP-FREEZE` are closed out on `codex/s13-closeout`
- Date: 2026-10-01
- Base: `codex/s13b-evaluation` at `9f90ca3` (copied from `pi_ginot_large_s13b_eval`)
- New implementation source: `pi-ginot-enhanced`, published commit `12c9c1c`
  ("energy-form neural operator for stress concentrations on five specimen families")
- Gate: `G13` remains open until the evidence below exists and the user ratifies it

**Decisions taken with the user (2026-10-01):**

1. The served stress is the **constitutive stress of the predicted displacement**
   (P = Neo-Hookean(I + grad u)), as in the enhanced implementation. ADR 0018
   records the amendment of the S11 two-head rule (`G11.heads`).
2. The registered validation trial uses **2,400 optimizer steps**
   (1,600 at 3e-4, then cosine to 3e-6), run in the cloud workspace.

## 1. The S13 issues this plan targets

| # | Issue | Evidence |
|---|---|---|
| I1 | No S13 trial is admissible. Peak stress gets worse while the field improves: the best trial has median absolute root-peak error 12.25% (S11 incumbent 8.09%) and signed error -12.25% | `S13-TRAIN-SCREEN-FAILED.json`, ADR 0015 |
| I2 | Adding a peak loss trades the field for the peak: S13b peak arm 4.31% peak error, but P-field L2 +8.7% (limit 2%) and one invalid point (`det F = -0.0011`) | `S13B-TRAIN-PILOT.json`, `S13B-TRAIN-INVALID-POINT.json` |
| I3 | The S11 operator predicts accurate displacement values but wrong displacement gradients (grad u relative H1 error 296%; stress recomputed from its grad u is 268% off), so the separately served `P` head and the physics residuals fight each other | ADR 0016 |
| I4 | Orientation (det F) is only protected by a soft recovery term (ADR 0014) | ADR 0014, I2 |
| I5 | Peak location is wrong (vertex 628 instead of the FEM peak at 645) | ADR 0016 |
| I6 | `G13`, `CP-FREEZE` and S14 are blocked | ADRs 0014-0016 |
| I7 | Repository hygiene: shared git config is damaged (`core.bare`, `core.worktree`, "S9 test" author) | project review 16 Sep |

I7 is already fixed **in the copy**: it is a standalone repository with its own
`.git`, author `Betim Bahtiri`, and no link to the shared `pi_ginot_large/.git`.

## 2. Why the enhanced implementation addresses I1-I5

The DogBone problem is the same in both repositories: quarter model,
`L_total` 40-70, `W_grip` 16-26, `W_gauge` 6-14, `R_fillet` 8-20 mm, E = 760 MPa,
nu = 0.23, symmetry on x = 0 and y = 0, prescribed `u_x` on the loaded end.
Only the load differs (1 mm in the enhanced study; nominal strain 10-20% here).

The enhanced operator:

- is trained on the **total potential energy** (Deep Ritz), with stratified
  quadrature resampled every step and a J-barrier (1e3 at J = 0.05);
- imposes the Dirichlet conditions **exactly** (`u = u_d xi + xi (1 - xi) phi_u`,
  `v = eta phi_v`); free-edge tractions are natural conditions of the minimum;
- derives stress from the displacement gradient through the material law,
  so field, gradient, stress and peak are one consistent object (I3, I5);
- conditions a FiLM cross-attention decoder on the shape parameters;
- reported 1.0% in-range K_t error and 0.5-1.3% von Mises relative L2 on held-out
  dog-bones (`pi-ginot-enhanced/docs/phase9_6_results.md`).

## 3. Step 1 — Implement (only in the copy)

| Item | Location | Content |
|---|---|---|
| 1.1 Operator | `src/piginot/operators/energy_operator.py` | Port of the enhanced operator: parameter-token encoder, NeRF query encoding, 6 cross-attention + FiLM blocks, exact affine Dirichlet lifting, constitutive served stress via the S4 kernel. Fixed `architecture_id` and module digest; safetensors export through the existing checkpoint code. No family names in `src/`. |
| 1.2 Energy objective | `src/piginot/objectives/energy.py` | Stratified triangle quadrature (torch RNG, so resume is exact), plane-stress Neo-Hookean potential energy from the S4 kernel, J-barrier, safe handling of inadmissible points. |
| 1.3 Energy trainer | `src/piginot/training/energy_engine.py` | Deterministic trainer reusing `TrainingRunBinding`, `DeterministicSamplerPlan`, checkpoint/resume (model, Adam, RNG, sampler, schedule); paper schedule (constant, then cosine), gradient clipping 1.0. |
| 1.4 DogBone campaign | `campaigns/dogbone_v1/s13c_*.py` | Preregistered S13c protocol; geometry-only TRAIN materialization (training needs no FEM labels); quarter-model lifting; trial runner (fresh/resume); TRAIN-screen evaluator reusing the S11/S13 metric code; frozen S13 selection. |
| 1.5 Decision record | `docs/adr/0018-s13c-energy-form-operator.md` | Rationale, deviation from the S11 two-head contract (if constitutive stress is chosen), evidence and outcome. |

## 4. Step 2 — Test

Acceptance: everything green in the copy.

- **Operator:** shapes, exact Dirichlet values on all three constrained edges,
  determinism, query-batch and query-order invariance, parameter reachability,
  safetensors round trip, module-digest stability.
- **Energy:** zero energy and stress at the identity, energy gradient equals the
  S4 stress, homogeneous plane-stress stretch matches the analytic solution,
  stratified quadrature is exact for constants and linear fields, barrier
  activates only below J_min, finite-difference gradient check.
- **Trainer (G13 checks):** two runs give identical weight hashes (determinism);
  interrupt at 50% and resume gives identical weights (resume); `run_hash`
  changes with data, config, code or profile (identity); out-of-space trial
  refuses (search-space); non-TRAIN roles refused (holdout).
- **Campaign:** lifting satisfies all BCs on morphed meshes across the
  parameter box; protocol hashes stable; evaluator reductions and both
  selection mutants (`SELECT_ON_HOLDOUT`, `CHOOSE_NONRANKED_CHECKPOINT`) fire.
- **Quality gates:** full `pytest` (all pre-existing tests still pass),
  `ruff check`, `ruff format --check`, `mypy --strict src/`, coverage of the new
  modules >= 90% and of `piginot.training` >= 80%.

## 5. Step 3 — Validate against the S13 issues

1. **Pilot** (50 steps): finite energy, minimum J above the barrier, seconds per step.
2. **Registered trial:** 192 TRAIN-fit geometries, batch 4, seed 0, float32,
   2,400 steps: Adam 3e-4 for 1,600 steps, then cosine to 3e-6 (fixed in the
   protocol before training).
3. **TRAIN-screen evaluation** on the 64 registered FEM cases with the unchanged
   S13 metric code, plus the invalid-J count over all 256 TRAIN cases.
   DEVELOPMENT and HOLDOUT stay closed.
4. **Frozen selection** (`select_checkpoint` with the S11-derived guardrails,
   unchanged):

   | Guardrail | Limit |
   |---|---|
   | median absolute peak error | <= 8.086% |
   | p90 absolute peak error | <= 13.635% |
   | median signed peak error | >= -6.986% |
   | u relative L2 | <= 3.805% |
   | P relative L2 | <= 19.584% |
   | invalid collocation points | 0 |

5. **Report:** `artifacts/campaigns/dogbone-v1/S13C-TRAIN-SCREEN.json`, a
   comparison against the S11 incumbent, S13 trial 0 and the S13b peak arm, an
   issue-by-issue table (I1-I7), and the ADR outcome.

**Out of scope here:** writing `CP-FREEZE`, `artifacts/gates/S13.json`, or
opening HOLDOUT. Those stay a user decision after reviewing the evidence.

## 6. Outcome

| Step | Result |
|---|---|
| 1. Implement | Energy-form operator, potential-energy objective and trainer in `src/piginot`; S13c protocol, data, trial, evaluator and evidence writer in `campaigns/dogbone_v1`; ADR 0018. Reproducibility fixes: canonical safetensors headers, exact cross-platform S10 request verification. |
| 2. Test | New tests for operator, energy, trainer, protocol, data/lifting, trial, evaluator, evidence and portability; pre-existing S11 execute-harness failures fixed. Full suite at `d9b89ee`: 2,355 passed, 0 failed, 0 skipped; branch coverage 90.56% overall, `energy_operator` 95%, `objectives/energy` 99%, `energy_engine` 98%. Package floors, now also in CI: operators 89.41% (>= 80), objectives 93.76% (>= 85), training 83.33% (>= 80). `ruff check`, `ruff format --check` and `mypy --strict src/` clean. |
| 3. Validate | Registered 2,400-step trial: median abs. root-peak error 0.83% (limit 8.09%), p90 1.69% (13.63%), median signed -0.29% (>= -6.99%), P L2 0.91% (19.58%), u L2 0.08% (3.81%), 0 invalid points over 256 TRAIN cases. Replay probe: byte-identical after interrupt/resume. Re-evaluating the checkpoint at `d9b89ee` reproduces `S13C-TRAIN-SCREEN.json` exactly; only the recorded evaluator revision differs. |

| Issue | Status |
|---|---|
| I1 no admissible S13 trial | Resolved on TRAIN evidence (`S13C-TRAIN-SCREEN.json`) |
| I2 field/peak trade-off, folded point | Resolved: peak and field improve together; min det F 0.990 |
| I3 grad u inconsistent with served P | Resolved by construction; strain energy within 0.009% of FEM |
| I4 orientation only softly protected | Improved: exact lifting + barrier never activated; still a sampled check |
| I5 wrong peak location | Improved: median 0.54 mm from the FEM peak vertex |
| I6 G13 / CP-FREEZE / S14 blocked | ADR 0018 accepted on 2026-10-02; G11 re-run, G13 gate artifact and CP-FREEZE follow on `codex/s13-closeout` |
| I7 damaged shared git config | Resolved in the copy (standalone repository, correct author) |
