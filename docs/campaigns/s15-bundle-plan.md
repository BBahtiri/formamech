# S15 plan — bundle and CLI

- Status: **Complete — G15 PASS** (`artifacts/gates/S15.json`). Decisions
  S15-D1 … D5 are the proposed defaults of
  [ADR 0023](../adr/0023-s15-bundle-and-cli.md); they await the decision
  owner's confirmation.
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

## 6. Results

Bundle `dogbone-v1-s13c-cp-freeze`, manifest `009db9d3…`, built from the
`CP-FREEZE` weights (`57b8f4c2…`), claim `HOLDOUT_VERIFIED`. Measurements are
in `artifacts/campaigns/dogbone-v1/S15-VALIDATION.json`.

| Check | Result |
| --- | --- |
| `G15.fresh-install` | PASS. Wheel installed with `piginot[runtime]` into a new Python 3.11 environment (torch 2.14.1); `bundle verify`, `inspect` and `predict` run from a directory outside the repository, also with `scipy`, `skfem`, `meshio` and `sympy` imports blocked. On one thread its fields equal the repository environment's (torch 2.13.0) exactly. The runtime requires no FEM-only package; torch itself brings `sympy`. |
| `G15.fields` | PASS. Every field at vertices and queries, finite, with units, identities, verdict, claim and confidence. On 8 TRAIN cases `u` and `P_served` equal the S14 serving path bit for bit, and so does the confidence indicator, computed in the same process, on the one case checked (it differs from the two-thread S14 rehearsal cache by at most 1.4e-16). |
| `G15.substitution` | PASS. Declared and provided authority `constitutive-plane-stress-neo-hookean`; a re-signed bundle declaring another authority is refused with `BUNDLE_STRESS_AUTHORITY_MISMATCH`; `P_material` appears only as the constitutive gap (0.0); the `SERVE_P_MATERIAL` mutant test passes. |
| `G15.refuse` | PASS. 204 of 204 request-form S14 fixtures refused with the expected code (60 primitive, 48 derived, 24 material and load, 36 query, 36 guard band); 0 of 844 in-domain cases refused. The 36 mutated-mesh fixtures are not applicable. |
| `G15.manifest` | PASS. 50 of 50 tampered copies of the real bundle refused (byte flips in every file and the manifest, truncation, deletion, extra file, symbolic link). |
| `G15.limitations` | PASS. The paragraph is in the bundle, the in-domain and refused predictions, `inspect`, and the report of a refused bundle. |
| `G15.coverage` | PASS. 98.06% branch coverage of `piginot.bundle` and `piginot.cli` (106 tests). |

Serving on one CPU thread: 0.44 s for the fields, 4.9 s with the confidence
bound at every vertex. Rebuilding the bundle at the committed revision with
the same timestamp reproduces it byte for byte. Peak root first-Piola stress at the reference geometry
and 15% strain: 129.4 MPa, with a 90% stress-error bound of 0.6–2.4 MPa at the
five reference queries.
