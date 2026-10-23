# ADR 0107: Material registry, strain-energy formulas and the admission gate (S25)

- Status: Accepted — decisions 1–9 and the G25 thresholds accepted by the
  decision owner on 2026-10-13 (chat: "Accept as written"; viability material
  "Mooney–Rivlin"; kill rule "Registry, Neo-Hookean training only"), before
  any S25 run; recorded in `artifacts/materials/S25-PROTOCOL.json`
- Date: 2026-10-13
- Step: S25 (gate `G25`), `STEPS-v2.0.md` §7
- Builds on: ADR 0004 (S4 physics domain and plane-stress closure), ADR 0002
  (S8 FEM thresholds), ADR 0105 (release 0.2.0; decisions 3–5), ADR 0106
  (ProblemSpec)

## Context

Every kernel in the project — the S4 physics, the plane-stress closure, the
S7 FEM solver, the energy objective and serving — is the compressible
Neo-Hookean model `W = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2`, written by
hand with its stress, tangent and a closed-form Newton closure. ADR 0105
makes 0.2.0 trainable for any isotropic compressible hyperelastic law, from a
built-in library or a user's energy formula, parsed as data and admitted by
an automatic gate before training. The energy method needs nothing else from
a material: it minimises `∫ W` with the closure `P33 = 0`.

## Decisions

1. **One registry, Neo-Hookean untouched.** The five functions of
   `piginot.physics` (energy, first Piola stress, tangent, plane-stress
   closure, admissibility) dispatch on `MaterialSpec.model`: the closed-form
   model `compressible_neo_hookean` runs its existing kernels, bit for bit; a
   formula model runs `piginot.materials`. `MaterialSpec` is unchanged. The
   FEM solver, the energy objective, the operator's served stress and the
   admission gate all go through the dispatch, so a new model needs no other
   code. Shipped bundles and their sidecars are not touched.
2. **⚑ Built-in library.** Three formula models, in the decoupled form of
   common FE codes with `kappa/2 (J - 1)^2` as volumetric energy:
   - `mooney_rivlin`: `c10 (I1bar - 3) + c01 (I2bar - 3)`; `mu0 = 2(c10 + c01)`;
   - `yeoh`: `c10 (I1bar - 3) + c20 (I1bar - 3)^2 + c30 (I1bar - 3)^3`;
     `mu0 = 2 c10`;
   - `gent`: `-mu Jm/2 ln(1 - (I1bar - 3)/Jm)`; `mu0 = mu`;
   with `K0 = kappa` (parameter `bulk_modulus`). The small-strain moduli were
   derived symbolically. Signs that published fits leave free (Yeoh `c20`,
   Mooney–Rivlin `c01`) are left to the admission gate.
3. **⚑ Formula language** (`piginot.contracts.energy_formula`, schema
   `piginot.energy-formula-v1`). A whitelist parser walks Python's syntax tree
   and accepts only numbers, parameter names, the invariants `I1`, `I2`, `J`,
   `I1bar = J^(-2/3) I1`, `I2bar = J^(-4/3) I2`, `+ - * /`, `**` (or `^`)
   with a numeric exponent, and `exp`, `log`, `sqrt` of one argument;
   nothing is evaluated as code. Parameters are in `MPa` or `1`; the energy
   must be a stress and every function argument dimensionless. A formula
   declares its small-strain shear and bulk moduli as expressions of its
   parameters and may list parameters that must be positive. Its identity is
   the SHA-256 of the canonical form (expressions as trees); a user formula
   is registered as model `energy-<first 16 hex>`, so a name never changes
   meaning. Exponents follow Python's precedence: `J^(-2/3)` needs the
   parentheses. The contract stores the tree; scalar values (the declared moduli)
   are computed by `piginot.materials.values` (S2: contracts do not
   evaluate).
4. **Stress, tangent and closure by automatic differentiation.** `P = dW/dF`
   by reverse mode (differentiable, for training and the served stress); the
   tangent by one reverse pass per stress component; the closure by a
   safeguarded Newton solve of `F33 P33 = 0` in `log F33`, with a bracket
   search that stays inside the energy's domain, a stopping step of 16
   machine epsilons, and the implicit-function derivative
   `dF33/dF2 = -(dP33/dF2)/(dP33/dF33)`. Measured: a Neo-Hookean formula
   equals the closed form within 1e-12 (energy, stress, tangent, closure and
   its gradient); one energy training step of a DogBone case costs about 30%
   more with Mooney–Rivlin; an FEM solve about 3.5 times more.
5. **Stress scale.** Objectives and the confidence indicator scale stresses
   by Young's modulus. A formula model uses
   `E0 = 9 K0 mu0 / (3 K0 + mu0)` of its declared moduli
   (`piginot.physics.reference_modulus`); Neo-Hookean keeps `E`.
   *Implementation note (2026-10-14):* the confidence indicator
   (`piginot.verification.confidence`) is S14 evaluation code frozen by
   `S14-PROTOCOL.json`, so it is left unchanged and still reads
   `elastic_modulus`; no formula model is served before S28, which gives
   the indicator its scale outside the frozen file. The objectives use
   `reference_modulus` as decided.
6. **⚑ Admission gate** (`piginot.materials.admission`,
   `piginot.problem.admit_material`). A model is trained only if it passes,
   over a declared range of principal stretches (default `[0.6, 1.6]`, set
   per spec) and at every corner and the centre of its parameters' box:

   | Check | Pass condition |
   |---|---|
   | reference | `|W(I)|`, `|P(I)|` ≤ 1e-10 `mu0` |
   | objectivity | `W(QF)`, `P(QF) = Q P(F)` within 1e-10 (relative), 1,000 random `F`, `Q` |
   | symmetry | isotropy: `W(FQ)`, `P(FQ) = P(F) Q` within 1e-10 |
   | consistency | `P` vs central differences of `W`, tangent vs those of `P` (step 1e-6): ≤ 1e-6 |
   | moduli | tangent at `I` equals the isotropic tensor of the declared `mu0`, `K0` within 1e-8 |
   | closure | on a 12 × 12 grid of in-plane stretches with random rotations: `F33 P33` strictly increasing with exactly one root over `F33 ∈ [0.05, 20]`; solved `|P33|` ≤ 1e-10 `mu0` |
   | ellipticity | smallest eigenvalue of the plane-stress acoustic tensor over 180 directions ≥ 1e-6 `mu0` at those points |
   | fem | the reference solver on the S8 `strongly_varying_f33` field, body force by automatic differentiation of the model's own stress, P2 levels 3–4: displacement L2 order within 0.15 of 3, first-Piola L2 order within 0.2 of 2 (ADR 0002) |

7. **ProblemSpec material block.** `material` takes `model` (a built-in) or
   `energy` (a formula document; the model is then its content name), the
   property-to-parameter map and an optional `stretch_range`. The family
   specs and their hashes are unchanged.
8. **⚑ Viability run.** One registered DogBone run trained on Mooney–Rivlin
   with the same small-strain moduli as the shipped material
   (`c10 = 0.4 mu`, `c01 = 0.1 mu`, `bulk_modulus = K`, with `mu = 308.94`
   MPa and `K = 469.14` MPa from `E = 760` MPa, `nu = 0.23`), the S13c recipe
   otherwise unchanged (label-free, same seed, initial weights, sampler plan,
   2,400 steps), bound to a Mooney–Rivlin DogBone spec. References: the 64
   TRAIN-screen geometries re-solved with the reference FEM solver on their
   S10 meshes and settings, with the Mooney–Rivlin material (owner machine).
   It is preregistered in `artifacts/materials/S25-PROTOCOL.json` and judged
   by the frozen S13 guardrails (median |peak| ≤ 8.09%, p90 ≤ 13.63%, median
   signed ≥ −6.99%, P L2 ≤ 19.58%, u L2 ≤ 3.81%, no invalid point).
   DEVELOPMENT, HOLDOUT, H1 and H2 stay closed. Kill rule: if it fails, 0.2.0
   ships the registry and the admission gate but the S28 compiler trains only
   the closed-form Neo-Hookean model, and this ADR records why.
9. **Anti-scope.** No anisotropy (S26), no path-dependent law, no new bundle
   format (serving a formula model comes with the first bundle trained with
   one, S28), no change to the energy objective's form, to the shipped
   bundles or to the S8 records.

## Gate G25

| Check | Pass condition |
|---|---|
| `G25.protocol-first` | `S25-PROTOCOL.json` committed and this ADR accepted before the first FEM re-solve and the viability run |
| `G25.formula-safety` | The parser refuses every construct outside the grammar (failing-formula tests); no `eval`, `exec`, `compile`, `sympify` or `parse_expr` in the formula path (syntax-tree scan) |
| `G25.neo-hookean-identity` | Dispatch bit-identical to the closed-form kernels (energy, stress, tangent, closure and its gradient); S19 fast-serving goldens unchanged; the Neo-Hookean formula within 1e-12 of the closed form |
| `G25.admission` | Every built-in model and the closed-form model admitted (decision 6, all eight checks) at its test parameters over `[0.6, 1.6]` |
| `G25.admission-mutants` | Each seeded defect fails its check: not stress-free (reference), wrong moduli (moduli), negative shear stiffness (ellipticity, closure), energy undefined in the range, anisotropic energy (symmetry), non-objective energy (objectivity), scaled stress and scaled tangent (consistency) |
| `G25.fem` | Autograd manufactured fields equal the symbolic S8 fields within 1e-10 for Neo-Hookean; the FEM solver converges at the optimal orders for a formula model |
| `G25.viability` | Decision 8 guardrails pass |
| `G25.binding-compat` | Existing run bindings, bundles and G13–G24 records still verify |
| `G25.anti-scope` | Decision 9 tests; `.[runtime]` imports `piginot.materials` without FEM packages; branch coverage of `piginot.materials` and the formula contract ≥ 90% |
| `G25.portability` | Suite records on Linux and Windows |
| `G25.quality` | Full suite, ruff, mypy `--strict`, global coverage ≥ 90.5% |

## Result of the viability run (Linux/AVX2, 2026-10-14)

The 64 TRAIN-screen S10 requests were re-solved with Mooney–Rivlin at
revision `80dff0c` (index `artifacts/materials/S25-MR-REFERENCES.json`); the
registered run `9da65236…` was trained at revision `2db0142` for 2,400
steps, with the training batches' minimum det F between 0.974 and 0.996 at
the six checkpoints, and scored against the re-solves by the frozen S13 guardrails
(`artifacts/materials/S25-MR-TRAIN-SCREEN.json`):

| Metric | Mooney–Rivlin (S25) | Neo-Hookean (S13c) | Frozen limit |
|---|---|---|---|
| Median absolute root-peak error | 0.70% | 0.83% | ≤ 8.09% |
| p90 absolute root-peak error | 1.73% | 1.69% | ≤ 13.63% |
| Median signed root-peak error | −0.32% | −0.29% | ≥ −6.99% |
| P-field relative L2 | 0.93% | 0.91% | ≤ 19.58% |
| u-field relative L2 | 0.084% | 0.08% | ≤ 3.81% |
| Invalid interior points (minimum det F) | 0 (0.990) | 0 (0.990) | 0 |

The model's strain energy is within 9.2e-5 (median) of the FEM energy.
Every guardrail passes, so the kill rule does not apply: the S28 compiler may
train any admitted formula model. The Mooney–Rivlin task is not a relabelled
Neo-Hookean one: at the hardest corner of the box (20% strain, gauge 6 mm) the
reference solution's peak |P| is 9.4% and its strain energy 5.7% below
Neo-Hookean's, and its solve took 423 s against 122 s (cloud, one core). The
S24 same-CPU replays, at revisions that contain this step's dispatch, are
byte-identical to the base runs (ADR 0106, result). DEVELOPMENT, HOLDOUT,
H1 and H2 stayed closed.

## Consequences

- A user can name a built-in model or write a formula in the problem spec;
  admission tells them before any training whether it can be used, and why
  not.
- The plane-stress closure of a formula model costs about ten times the
  closed form's per point; training stays dominated by the operator.
- S26 adds fibre families (anisotropic invariants) to the same language and
  gate; S28 adds the material stage and bundle support.

## Verification

- `tests/contracts/test_energy_formula.py`, `tests/materials/`,
  `tests/problem/test_material.py`, `tests/test_s25_anti_scope.py`.
- Viability run: `campaigns/dogbone_v1/s25_mooney_rivlin.py` (`protocol`,
  `references`, `train`, `evaluate`), records
  `artifacts/materials/S25-MR-REFERENCES.json` and
  `artifacts/materials/S25-MR-TRAIN-SCREEN.json`; its glue is tested in
  `tests/campaigns/test_dogbone_s25_mooney_rivlin.py` without training or
  re-solving.
- Gate: `campaigns/g25_gate.py`, `tests/campaigns/test_g25_gate.py`.
