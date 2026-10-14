# ADR 0108: Anisotropic hyperelastic models with in-plane fibre families (S26)

- Status: Accepted — decisions 1–10 and the G26 thresholds accepted by the
  decision owner on 2026-10-14 (chat: "Accept as written"; angle unit
  "Degrees, new unit deg"; viability material "±30°, k1=50 MPa, k2=2"; kill
  rule "Registry, isotropic training only"), before any S26 run; recorded in
  `artifacts/materials/S26-PROTOCOL.json`
- Date: 2026-10-14
- Step: S26 (gate `G26`), `STEPS-v2.0.md` §7
- Builds on: ADR 0107 (material registry, formula language, admission gate),
  ADR 0106 (ProblemSpec, quarter-model symmetry conditions), ADR 0105
  (release 0.2.0: "isotropic and anisotropic hyperelastic models … HGO in
  S26"), ADR 0100 (platform findings)

## Context

S25 made any isotropic compressible hyperelastic law trainable: a formula
over `I1`, `I2`, `J`, `I1bar`, `I2bar`, stress, tangent and the plane-stress
closure by automatic differentiation, and an admission gate before training.
ADR 0105 adds fibre-reinforced materials in S26, with the
Holzapfel–Gasser–Ogden (HGO) model as the built-in.

Four facts shape the design:

1. **Full, not isochoric, fibre invariants.** Writing the fibre term of a
   compressible HGO model in the isochoric invariant `I4bar = J^(-2/3) I4`
   makes it insensitive to volume change: under hydrostatic tension an
   anisotropic sphere stays a sphere. The full invariant `I4 = a·C a` gives
   the correct anisotropic response (Nolan, Gower, Destrade, Ogden and
   McGarry, "A robust anisotropic hyperelastic formulation for the modelling
   of soft tissue", 2014).
2. **The tension-only switch on the true fibre stretch.** Fibres are commonly
   switched off in compression. The switch must test `I4 > 1`; codes that
   test the deviatoric stretch never activate the fibres under hydrostatic
   tension (Vergori, Destrade, McGarry and Ogden, "On anisotropic elasticity
   and questions concerning its finite element implementation").
3. **In-plane fibres leave the closure alone.** With `a = (cos θ, sin θ, 0)`,
   `I4` and `I5` depend only on the in-plane block of `C`, not on `F33`. A
   fibre term in `I4` therefore adds nothing to `P33`, and the plane-stress
   closure root is the matrix's.
4. **The quarter model needs a symmetric material.** Both shipped families
   are quarter models with symmetry conditions on `x = 0` and `y = 0`. They
   are valid only if the material is symmetric under the reflections across
   those planes, which maps a fibre angle `θ` to `−θ`. Two families at `±β`
   (or one at 0° or 90°) keep that symmetry; one family at 30° does not.

Measured for the proposed viability material (uniaxial plane stress,
`μ = 308.94` MPa, `K = 469.14` MPa, HGO fibres `k1 = 50` MPa, `k2 = 2`,
`β = ±30°`): the axial first Piola stress rises from 69.1 to 92.0 MPa at a
stretch of 1.1 (+33%) and from 126.9 to 188.1 MPa at 1.2 (+48%); the
transverse stretch at 1.2 falls from 0.957 to 0.938.

## Decisions

1. **⚑ Fibre families in the formula language** (schema
   `piginot.energy-formula-v1`, additive). A formula may declare
   `"fibres": {"<label>": "<angle parameter>", …}`, at most two families,
   each in the plane at the angle of its parameter from the x axis. Each
   family adds the invariants `I4<label> = a·C a` and `I5<label> = a·C² a`
   (full, not isochoric; decision 1 of the context). A new function
   `ramp(x) = max(x, 0)` (one dimensionless argument, derivative 0 at 0)
   writes the tension-only switch. A formula without `fibres` is unchanged:
   same canonical form, same identity, same registry name.
2. **⚑ Built-in library.** Matrix `μ/2 (I1bar − 3) + κ/2 (J − 1)^2` in all
   three; small-strain moduli `μ0 = μ`, `K0 = κ` (fibres carry no load at the
   reference):
   - `hgo`: two families at `±β` with shared `k1` (MPa) and `k2` (1):
     `Σ k1/(2 k2) (exp(k2 ramp(I4 − 1)^2) − 1)`; parameters
     `shear_modulus`, `bulk_modulus`, `k1`, `k2`, `fibre_angle`;
   - `hgo_single`: one family at `fibre_angle`;
   - `goh`: `hgo` with dispersion `kappa_d ∈ [0, 1/3]` (Gasser, Ogden and
     Holzapfel, 2006), fibre strain `E = kappa_d (I1 − 3) + (1 − 3 kappa_d)
     (I4 − 1)` in place of `I4 − 1` (its `I1` term couples to `F33`, so the
     closure is solved as in S25).
3. **⚑ Fibre angles.** Angle parameters are in degrees: the contracts gain the
   unit `deg` of a new dimension `angle` (additive). Kernels convert to
   radians.
4. **Kernels.** `piginot.materials` builds `a` per family from the material's
   angle values and evaluates the formula in torch as in S25: stress by
   reverse mode, tangent by nine reverse passes, closure by the safeguarded
   Newton solve. The first Piola stress is continuous; its derivative jumps
   where a fibre crosses `I4 = 1`. The FEM solver, the energy objective and
   the dispatch need no other change.
5. **⚑ Admission gate for fibre models.** The eight S25 checks over the
   stretch range and the corners and centre of the parameter box (angles
   included), with two changes:
   - **symmetry** tests the declared symmetry group instead of isotropy: the
     proper rotations by π about `e3`, `e1`, `e2`, each fibre axis and its
     in-plane normal that map the set of families (with their parameters)
     to itself. An energy that depends on an undeclared direction fails.
     Formulas without fibres keep the isotropy check;
   - **fem** uses the S8 manufactured field on top of a homogeneous in-plane
     stretch that keeps every fibre in tension (`I4 ≥ 1.05` everywhere), so
     the solution is smooth and the optimal orders apply; the compression
     branch is the matrix, already admitted.
   The **moduli** check is unchanged: the tangent at `F = I` must be the
   isotropic tensor of the declared matrix moduli, so in 0.2.0 a fibre term
   must carry no load at the reference (written with `ramp`, or vanishing to
   second order). A fibre term without a switch is refused there.
6. **⚑ ProblemSpec rule `material-symmetry`.** A symmetry condition on a
   region whose normal is `e1` or `e2` requires the material to be symmetric
   under the reflection across that plane: the set of fibre families must be
   closed under `θ → −θ` at every point of the parameter box. Two families
   at `±β`, or one at 0° or 90°, pass; one family at 30° is refused with this
   rule's name.
7. **Stress scale.** As in S25: objectives scale stresses by `E0` of the
   declared matrix moduli (`piginot.physics.reference_modulus`). Fibre
   stiffening appears only under strain.
8. **⚑ Viability run.** One registered DogBone run on `hgo` with the shipped
   material's matrix moduli (`μ = 308.94` MPa, `κ = 469.14` MPa), `k1 = 50`
   MPa, `k2 = 2`, `β = 30°` (two families at ±30°, orthotropic about the
   loading axes, so the quarter model stays valid). Otherwise the S25
   pipeline unchanged: the 64 TRAIN-screen geometries re-solved with the
   reference FEM solver and this material, the S13c recipe (label-free, same
   seed, initial weights, sampler plan, 2,400 steps), a run binding bound to
   an HGO DogBone spec, judged by the frozen S13 guardrails. DEVELOPMENT,
   HOLDOUT, H1 and H2 stay closed.
9. **⚑ Kill rule.** If the viability run fails a guardrail, 0.2.0 keeps fibre
   formulas in the registry and the admission gate, but the S28 compiler
   trains isotropic models only; this ADR records why.
10. **Anti-scope.** No out-of-plane fibres, no fibre field varying in space,
    no more than two families, no fibre term loaded at the reference, no
    viscoelasticity, damage or recruitment model, no new bundle format
    (S28), no change to the isotropic formulas, their hashes or the S25
    records.

## Gate G26

| Check | Pass condition |
|---|---|
| `G26.protocol-first` | `S26-PROTOCOL.json` committed once, with this ADR accepted, before the first re-solve and the viability run |
| `G26.formula-safety` | Every malformed fibre construct is refused (undeclared family, a third family, an angle that is not an angle parameter, `ramp` of a stress, `I4` of an unknown label); no `eval`, `exec`, `compile` in the formula path |
| `G26.isotropic-identity` | Identities of the S25 built-ins and of every formula without fibres unchanged; S25 admission reports unchanged; Neo-Hookean dispatch bit-identical; S19 fast goldens equal at their recorded thread count (`campaigns/fast_goldens.py`) |
| `G26.fibre-kernels` | `hgo` stress equals its closed form within 1e-12; zero fibre stress wherever `I4 ≤ 1`; the `hgo` closure root equals the matrix's within 1e-12 |
| `G26.admission` | `hgo`, `hgo_single` and `goh` admitted at their test parameters over `[0.6, 1.6]`, and the viability material, all eight checks |
| `G26.admission-mutants` | An energy whose fibre direction differs from the declared angle fails symmetry; a fibre term without switch fails moduli; `k1 < 0` fails ellipticity; the S25 mutants still fail |
| `G26.fem` | Fibres in tension: displacement L2 order within 0.15 of 3, first-Piola L2 order within 0.2 of 2 (ADR 0002) |
| `G26.material-symmetry` | A quarter-model spec with one family at 30° is refused (`material-symmetry`); `±β` and 0° pass |
| `G26.viability` | Decision 8 guardrails pass |
| `G26.binding-compat` | Existing run bindings, bundles, family spec hashes and G13–G25 records still verify |
| `G26.anti-scope` | Decision 10 tests; branch coverage of `piginot.materials` and the formula contract ≥ 90% |
| `G26.portability` | Suite records on Linux and Windows |
| `G26.quality` | Full suite, ruff, mypy `--strict`, global coverage ≥ 90.5% |

## Consequences

- A user can name `hgo`, `hgo_single` or `goh`, or write a fibre formula,
  in a ProblemSpec; admission and the `material-symmetry` rule say before
  any training whether it can be used on that geometry.
- Fibre models whose fibres resist compression, or that load the reference
  state, wait for an anisotropic declaration of the small-strain tangent.
- A single off-axis family needs a geometry without symmetry conditions
  (S27, S28).

## Implementation notes (2026-10-14)

- A family's angle is a `deg` parameter or its negative
  (`"b": "-fibre_angle"`), which writes the `±β` pair of `hgo` and `goh`.
  The formula language also gained `bounds` (`goh`: `kappa_d ∈ [0, 1/3]`).
  Both keys are absent from formulas without them, so every S25 identity is
  unchanged.
- "Families with their parameters" (decision 5) is read as: two families may
  be swapped by a symmetry rotation when they share their angle parameter
  (`β` and `−β`). Families with separate angle parameters are each mapped to
  themselves, which any energy in `I4`, `I5` satisfies.
- The FEM check's field is the S8 field at half amplitude on an equibiaxial
  stretch of 1.2 (smallest `I4` 1.10, for fibres at 90°). It is solved by
  Newton at full load from the field's interpolant (the S22 warm start):
  load stepping scales the data through states where fibres cross the switch.
  Measured orders, displacement and first Piola: `hgo` 3.016 / 1.980,
  `hgo_single` 3.016 / 1.974, `goh` 3.007 / 1.994.
- The `material-symmetry` rule is checked on the spec's parameter ranges, so
  it needs no mesh; both in-plane reflections map an angle `θ` to `−θ`.

## Verification

- `tests/contracts/test_energy_formula_fibres.py`, `tests/materials/test_fibres.py`,
  `tests/problem/test_material_symmetry.py`, `tests/test_s26_anti_scope.py`.
- Viability run: `campaigns/dogbone_v1/s26_hgo.py` (the S25 recipe as
  `campaigns/dogbone_v1/material_run.py`), records
  `artifacts/materials/S26-HGO-REFERENCES.json` and
  `artifacts/materials/S26-HGO-TRAIN-SCREEN.json`; tested in
  `tests/campaigns/test_dogbone_s26_hgo.py`.
- Gate: `campaigns/g26_gate.py`, `tests/campaigns/test_g26_gate.py`; fast
  goldens at their recorded thread count (`campaigns/fast_goldens.py`).
