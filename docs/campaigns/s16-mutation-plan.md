# S16 plan — mutation suite

- Status: **Complete — G16 PASS** (`artifacts/gates/S16.json`). Decisions
  S16-D1 … D5 are the proposed defaults of
  [ADR 0024](../adr/0024-s16-mutation-suite.md); they await the decision
  owner's confirmation.
- Date: 2026-10-05
- Base: `codex/s15-bundle` (G15 PASS)
- Gate: `G16` (`STEPS-v1.0.md` §S16); mutants from
  [the replacement spec](../specs/implementation-v1.0-oss-replacement.md) §1.13

## 1. What S16 must deliver

| Check | Threshold |
| --- | --- |
| `G16.primary` | 20 of 20 mutants fire their named primary gate |
| `G16.signatures` | complete fired-gate signature recorded for each and committed as baseline |
| `G16.regression` | a CI job compares signatures per release; losing a secondary detection is reported as a coverage regression even when the primary still fires |
| `G16.runtime` | the full suite completes in under 45 minutes on CI |

The project adds `G16.coverage` (≥ 85% of `piginot.mutation`), as for every
earlier gate.

## 2. Design

| Part | Location |
| --- | --- |
| Engine: source edits, runner, signatures, comparison | `src/piginot/mutation/` |
| The twenty mutants | `campaigns/dogbone_v1/s16_mutants.py` |
| The twenty-two detectors | `campaigns/dogbone_v1/s16_detectors.py` |
| Command line and gate | `campaigns/dogbone_v1/s16_mutation.py`, `s16_gate.py` |
| Release job | `.github/workflows/mutation.yml` |
| Baseline | `artifacts/mutation/S16-SIGNATURES.json` |
| Hardened G8 controls | `src/piginot/fem/seeded_defects.py` |

Each mutant is a source edit of the production function that holds the defect:

| # | Mutant | Edited function | Edit |
| ---: | --- | --- | --- |
| 1 | `BODY_FORCE_SIGN` | `fem.binding._registered_body_force` | negate the body-force form |
| 2 | `BOUNDARY_NORMAL` | `fem.binding._traction_field` | pass `-normals` to the provider |
| 3 | `TAG_SWAP` | `fem.binding.facets_for_tag` | `left` and `right` resolve to each other |
| 4 | `CLOSURE_WRONG_ROOT` | `physics.plane_stress._nearest_representable_root` | return the negative root |
| 5 | `INCONSISTENT_TANGENT` | `physics.neo_hookean.consistent_tangent` | scale by 0.9 |
| 6 | `ANISOTROPIC_SCALING` | `fem.backend._dof_coordinates` | stretch x by 1.2 |
| 7 | `NODE_SPLIT` | `doe.roles.CaseRoleRegistry.validate_node_observations` | accept split roles |
| 8 | `DROP_PROJECTION_ERROR` | `datasets.projection._sample_envelope` | zero envelope |
| 9 | `STRESS_NODE_PAIRING` | `datasets.projection._physical_integral` | drop the quadrature weights |
| 10 | `NORMALIZE_ON_ALL_SPLITS` | `campaigns…s10_closeout._normalization_evidence` | fit on every role |
| 11 | `IGNORE_DIRICHLET_LIFTING` | `operators…AxisAlignedTensionLifting.__call__` | drop the end displacement |
| 12 | `COORDINATE_ANISOTROPY` | `objectives.equilibrium.strong_equilibrium_residual` | x scale for every derivative |
| 13 | `HOLDOUT_ACCESS` | `objectives.registry.evaluate_objective` | admit HOLDOUT |
| 14 | `SELECT_ON_HOLDOUT` | `training.selection._validate_result_matrix` | admit non-TRAIN results |
| 15 | `CHOOSE_NONRANKED_CHECKPOINT` | `training.selection.SelectionDecision.selected_result` | serve the last-ranked trial |
| 16 | `ZERO_MISSING_BUDGET_COMPONENT` | `verification.budget.CaseBudgetEntry.total` | count a missing component as 0 |
| 17 | `PASS_ON_NOT_RUN` | `verification.reducer.reduce_families` | count NOT_RUN as PASS |
| 18 | `REVEAL_BEFORE_CONSUME` | `verification.holdout.reveal_after_consumption` | read before the record check |
| 19 | `SERVE_P_MATERIAL` | `bundle.inference._fields` | serve `P_material` |
| 20 | `UNDETECTED_BUNDLE_TAMPER` | `bundle.manifest.verify_bundle` | compare hashes only when sizes differ |

## 3. Steps

1. Implement the engine, the registry and the detectors; harden the G8
   controls; test the engine (`tests/mutation`) and the registry against the
   spec table (`tests/campaigns/test_dogbone_s16_mutation.py`).
2. Run the suite, commit the baseline.
3. Validate: rerun at the committed revision and compare with the baseline;
   rerun without one detector to show a coverage regression; write
   `artifacts/gates/S16.json`.

## 4. Results

Measured at the committed revision on one CPU thread
(`artifacts/mutation/S16-RUN.json`).

| Check | Result |
| --- | --- |
| `G16.primary` | PASS. 20 of 20 mutants fire their primary gate; the unmutated control passes all 22 detectors before and after. |
| `G16.signatures` | PASS. `artifacts/mutation/S16-SIGNATURES.json` records the sorted fired-gate set of every mutant: 20 primaries and 13 secondary detections. A rerun reproduces it exactly, also under another hash seed. |
| `G16.regression` | PASS. `.github/workflows/mutation.yml` compares the signatures on every release. Removing one detector (`G7.loads-fields-store`) gives `COVERAGE_REGRESSION`, exit code 3, for `BODY_FORCE_SIGN`, `BOUNDARY_NORMAL` and `CLOSURE_WRONG_ROOT`, while all twenty primaries still fire. The hosted release run happens with the first published release. |
| `G16.runtime` | PASS. The full mutation suite takes 49 s; the release job's timeout is 45 minutes. |
| `G16.coverage` | PASS. 99.15% branch coverage of `piginot.mutation`. |

Signatures with secondary detections:

| Mutant | Fired gates |
| --- | --- |
| `BODY_FORCE_SIGN` | G7.loads-fields-store, G8.seeded-defects |
| `BOUNDARY_NORMAL` | G7.loads-fields-store, G8.seeded-defects |
| `TAG_SWAP` | G8.patch, G8.seeded-defects |
| `CLOSURE_WRONG_ROOT` | G15.fields, G15.refuse, G15.substitution, G4.closure-residual, G7.loads-fields-store, G8.patch, G8.seeded-defects |
| `INCONSISTENT_TANGENT` | G4.tangent, G8.seeded-defects |
| `ANISOTROPIC_SCALING` | G8.patch, G8.seeded-defects |
| `IGNORE_DIRICHLET_LIFTING` | G11.lifting, G11.lifting-mutant, G15.fields |

The other thirteen mutants fire only their primary gate.
