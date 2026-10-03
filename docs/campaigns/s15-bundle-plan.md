# S15 plan — bundle and CLI

- Status: **Implemented; validation pending.** Decisions S15-D1 … D5 are the
  proposed defaults of [ADR 0023](../adr/0023-s15-bundle-and-cli.md); they await
  the decision owner's confirmation.
- Date: 2026-10-03
- Base: `codex/s14-verification` (G14 PASS, claim `HOLDOUT_VERIFIED`)
- Gate: `G15` (`STEPS-v1.0.md` §S15)

## 1. What S15 must deliver

| Deliverable (STEPS §S15) | Gate check | Threshold |
| --- | --- | --- |
| Self-contained bundle: spec, checkpoint identity, morph and membership, weights (`safetensors`), preprocessing, schemas, inference closure, verification report, machine-readable limitations, claim level, all failed / skipped / not-run work with reasons | `G15.manifest` | the manifest hash closes over every contained file; changing any byte is detected |
| `load_bundle`, `predict`, `inspect` | `G15.fields` | `predict` returns `u`, `P_served`, `F_2D`, `F33`, `J_3D`, `sigma`, QoIs, units, identities, in-domain verdict, reason codes, claim level, confidence |
| Served-stress authority | `G15.substitution` | `P_material` is never silently served in place of `P_served` |
| Envelope refusal at inference | `G15.refuse` | out-of-envelope requests refuse with a reason code |
| Limitations | `G15.limitations` | the required limitation paragraph is in every report |
| Fresh install | `G15.fresh-install` | the bundle loads and predicts in a clean virtual environment with runtime dependencies only |
| CLI | — | `family compile`, `pilot`, `campaign`, `simulate`, `train`, `verify`, `holdout-reveal`, `bundle`, `predict`, `inspect`, `store prune` |
| Coverage | `G15.coverage` | ≥ 85% |

## 2. Decisions

See [ADR 0023](../adr/0023-s15-bundle-and-cli.md). In short:

| # | Decision | Default |
| --- | --- | --- |
| S15-D1 | Bundle format | a directory: `bundle.json` plus `spec/reference-mesh.json`, `spec/morph-map.json`, `spec/envelope.json`, `spec/model.json`, `weights/model.safetensors`, `report/verification.json`, `report/limitations.json`; every byte hashed, every part decoded exactly and cross-checked |
| S15-D2 | Runtime dependencies | `numpy`, `torch`, `safetensors`, `rfc8785` (`piginot[runtime]`) |
| S15-D3 | Confidence at a query point | the largest S14 `patch-mean` indicator over the vertices of the containing cells, with the DEVELOPMENT 90% error quantile |
| S15-D4 | Claim level | the G14 claim for in-envelope predictions; `NOT_VERIFIED` otherwise |
| S15-D5 | CLI scope | framework: `bundle verify`, `predict`, `inspect`, `store prune`; campaign plugin via `--campaign`: `family compile`, `pilot`, `campaign`, `simulate`, `train`, `verify`, `holdout-reveal`, `bundle build` |

ADR 0023 also records the served-stress binding (decision 6), the refusal of
inadmissible predictions (7), the refusal scope (8) and the advanced S5
anti-scope test (9).

## 3. Step 1 — Implement

| Item | Location |
| --- | --- |
| Manifest and integrity | `src/piginot/bundle/manifest.py` |
| Exact spec codecs | `src/piginot/bundle/specs.py` |
| Architectures and served-stress authority | `src/piginot/bundle/serving.py` |
| `load_bundle`, `predict`, `inspect` | `src/piginot/bundle/inference.py` |
| Atomic build | `src/piginot/bundle/build.py` |
| Limitation paragraph | `src/piginot/bundle/limitations.py` |
| CLI | `src/piginot/cli.py` |
| DogBone bundle and plugin | `campaigns/dogbone_v1/s15_bundle.py`, `campaigns/dogbone_v1/cli.py` |
| Validation | `campaigns/dogbone_v1/s15_validate.py` |
| User guide | `docs/bundle.md` |

## 4. Step 2 — Test

`tests/bundle/` covers the manifest (every single-byte flip, truncation,
deletion, extra file and symbolic link; the `UNDETECTED_BUNDLE_TAMPER` mutant),
the codecs, loading refusals, every predicted field, refusals through the S14
fixtures, the served-stress authority (the `SERVE_P_MATERIAL` mutant through a
two-head stand-in architecture) and the CLI. `tests/test_s15_anti_scope.py`
runs the inference closure with FEM-only imports blocked.

## 5. Step 3 — Validate

1. Build the DogBone bundle from the `CP-FREEZE` weights.
2. `s15_validate measure`: tamper the real bundle, re-sign it with another
   stress authority, check every report for the limitation paragraph, serve
   every field, reproduce the S14 vertex fields bit for bit on TRAIN cases, run
   the S14 refusal fixtures and the 844 in-domain cases through `predict`.
3. `s15_validate fresh-install`: build the wheel, install `piginot[runtime]`
   into a new virtual environment, and verify, inspect and predict from a
   directory outside the repository, also with FEM-only imports blocked.
4. `s15_validate gate`: write `S15-VALIDATION.json`, `S15-GATE.json` and
   `artifacts/gates/S15.json`.
