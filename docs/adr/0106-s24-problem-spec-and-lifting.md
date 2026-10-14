# ADR 0106: ProblemSpec, exact lifting from the spec, FEM requests from the spec (S24)

- Status: Accepted — decisions 1–8 and the G24 thresholds accepted by the
  decision owner on 2026-10-13 (chat: "Accept as written"; harmonic lifting
  "Defer beyond 0.2.0"; viability run "Full 2,400 steps"; kill rule "Keep
  untrimmed lines"), before any S24 training run; recorded in
  `artifacts/problem/S24-PROTOCOL.json`; amendment 1 (replay reference)
  accepted on 2026-10-14, before any replay comparison, recorded in
  `artifacts/problem/S24-REPLAY-AMENDMENT.json`
- Date: 2026-10-13
- Step: S24 (gate `G24`), `STEPS-v2.0.md` §7; implementation plan Feature 4
- Builds on: ADR 0018 (energy-form operator and its quarter lifting),
  ADR 0027 (soft default; a hard lifting only where verified), ADR 0103 (FEM
  recipes), ADR 0105 (release 0.2.0, sequence S24–S29)

## Context

Every problem the project solves today is described in campaign code: the
DogBone and notched-plate families each have their own geometry, boundary
conditions, lifting and FEM request builders under `campaigns/`. A user who
wants a new problem, and S28's one-command compiler, need the problem as
data: one file that says what the geometry, the parameter ranges, the
material, the boundary conditions and the lifting are, validated before
anything runs and identified by a hash that training runs bind.

The only exact lifting is the quarter-model lifting of ADR 0018,
`u_x = d (ξ + ξ(1 − ξ) φ_x)`, `u_y = d η φ_y`. It needs the constrained
boundaries to be the lines `x = 0`, `x = a` and `y = 0` of the frame. Any other
Dirichlet boundary has no exact lifting.

The S2 contract `ProblemFamilySpec` (schema 1.0) exists but no campaign builds
one; it lacks the reference geometry, the lifting, the DOE budget and the FEM
settings, and its boundary conditions cannot express a value that depends on
the request.

## Decisions

1. **⚑ ProblemSpec contract.** A new contract `piginot.contracts.ProblemSpec`
   (schema `piginot.problem-spec-v1`), read from a JSON file (the plan's YAML
   is dropped: the strict JSON reader exists and YAML would add a dependency
   and a second parser). `ProblemFamilySpec` stays unchanged and unused. A spec
   holds:
   - `physics`: `2d-plane-stress-finite-strain`, the only value in 0.2.0;
   - `parameters`: one range per request input — role (`geometry`, `material`,
     `load`), `lo`, `hi`, canonical unit (`mm`, `MPa`, `1`), distribution
     (`uniform`, or `fixed` when `lo = hi`);
   - `constants`: named values with a unit that expressions may use and that
     are not request inputs (the notched plate's half length, 25 mm);
   - `constraints`: the derived feasibility bounds of the envelope, as
     expressions of the parameters;
   - `geometry`: the reference mesh by content id, the morph map by SHA-256,
     the boundary regions (semantic tags);
   - `material`: the model and which parameters set its properties
     (`compressible_neo_hookean` only, until S25);
   - `boundary_conditions`: name, region, kind (`dirichlet`, `symmetry`,
     `traction_free`, `neumann`), constrained components, value; a reserved
     `load_path` that must be null;
   - `lifting`: the kind and its data (decision 3);
   - `qois`, `verification` (quantile and the per-quantity tolerances), `split`
     (the DOE budget per role), `fem` (solver settings and material name).

   Values are expressions over the parameters and constants, parsed by the S2
   safe parser (names, numbers, `+ − * /`) and evaluated left to right in
   float64, never by `eval`; a bare number is a constant in the expected unit.
   A Dirichlet value is either an expression or a nominal strain
   (`u_c = ε × the region's coordinate along c`, the rule of ADR 0103's
   recipes). Identity: BLAKE2b-256 in the `problem` domain over the RFC 8785
   canonical form, in which each expression is its syntax tree, so key order,
   whitespace, number spelling, the spelling of an expression and the file's
   Unicode encoding do not change it.
2. **⚑ Validation.** A spec is refused, with a named reason, for: unknown or
   missing keys; a missing or wrong unit; `lo > hi`; `fixed` with `lo ≠ hi`; an
   expression naming an unknown parameter or of the wrong dimension; a
   component that is both Dirichlet and Neumann on one region; a non-null
   `load_path`; a lifting that does not cover exactly the Dirichlet components.
   Against the reference mesh it is also refused for: a region absent from the
   mesh; rigid-body modes left free (the constraint rows `[1, 0, −y]` for each
   node with `u_x` fixed and `[0, 1, x]` for each node with `u_y` fixed must
   have rank 3); a nominal-strain region that is not perpendicular to its
   component. `neumann` conditions validate but the builders refuse them:
   traction work is not in the energy objective (anti-scope).
3. **⚑ Lifting package and kinds.** `piginot.lifting` holds the one
   interface (`AffineDirichletLifting`: `u = offset + scale ⊙ φ`, with
   `offset = g` extending the Dirichlet data and `scale` vanishing on each
   component's Dirichlet set) and two kinds:
   - `quarter`: the ADR 0018 lifting, moved without numerical change (the
     bundle name stays `axis-aligned-tension`; `piginot.operators` re-exports
     it);
   - `adf`: R-function approximate distance functions (Sukumar and Srivastava
     2022) over line segments and minor circular arcs, traversed with the
     domain on their left, whose end points are expressions of the
     parameters. Per component `c`: `φ_c` is the R-equivalence of order 1 of
     the trimmed piece distances, computed in coordinates scaled by the
     frame's larger extent `l`; `g_c` is their transfinite interpolation;
     `scale_c = d l φ_c / extent_c` with `d` the declared displacement scale.
     Values and gradients are finite everywhere, at piece end points and
     joints included; on a piece the gradient of `φ_c` is the inward unit
     normal, also for a boundary node a rounding error (≤ 1e-9 `l`) outside
     it. A line may be declared untrimmed (`"trimmed": false`, its whole
     supporting line) when that line meets the domain only along the piece;
     with untrimmed lines `adf` reduces to `quarter` (agreement to 1e-12), so
     the two kinds differ only by the trimming at the ends of each piece.

   A harmonic lifting (Laplace and Poisson solves on the mesh) is **deferred
   beyond 0.2.0**: it puts a mesh and a sparse solve into fast serving, so
   `.[runtime]` would no longer be FEM-free, and every 0.2.0 family has
   straight or circular Dirichlet boundaries that `adf` covers. Default per
   geometry class: `quarter` for the two shipped families (bit-identity);
   `adf` for a new 2D family whose Dirichlet regions are lines and arcs; any
   other family is refused by the S28 compiler in 0.2.0.
4. **Lifting verification (ADR 0027).** A spec's lifting is verified over
   the parameter box — every corner and the centre that meet the
   constraints, and 64 seeded feasible interior points — on the morphed
   mesh (`piginot.problem.verify_lifting`): at the nodes of every Dirichlet
   region the lifted displacement, for a random correction, meets the FEM
   data within 1e-12 of the displacement scale. For `adf` also: every region
   node lies within 1e-9 `l` of its condition's pieces; no mesh node lies
   more than 1e-9 `l` on the wrong side of a piece; the data are met within
   1e-12 at 320 and 4,800 points per piece. Mutants (each piece dropped, and
   the identity lifting) must miss by more than 1e-3. The result is a
   `LiftingVerification` record. A bundle may declare `adf` only with that
   record; bundle support for it comes with the first bundle trained with it
   (S28).
5. **⚑ ADF viability run.** One registered DogBone run, the S13c recipe
   unchanged except the lifting: `adf` with the trimmed segments of the
   DogBone spec, the same seed, initial weights (the lifting has no
   parameters), sampler plan, 2,400 steps and execution profile; owner
   machine (about 3 h). It is preregistered in `artifacts/problem/S24-PROTOCOL.json`
   before its first step and judged on the 64 TRAIN-screen cases by the
   frozen S13 guardrails: median |peak error| ≤ 8.09%, p90 ≤ 13.63%, median
   signed peak error ≥ −6.99%, P-field relative L2 ≤ 19.58%, u-field
   relative L2 ≤ 3.81%, no invalid collocation point. Its P-field L2 is
   reported against S13c's 0.91%. DEVELOPMENT, HOLDOUT, H1 and H2 are not
   opened. Kill rule: if it fails a guardrail, trimmed pieces are not offered
   by the S28 compiler in 0.2.0 (untrimmed lines, which reduce to `quarter`
   on the shipped geometries, stay) and this ADR records why.
6. **Regression of the shipped families.** Both family specs live under
   `campaigns/` (`campaigns/dogbone_v1/problem_spec.json`,
   `campaigns/notched_v1/problem_spec.json`); family names stay out of
   `src/`. From each spec: the `quarter` lifting equals the bundle's lifting
   bit for bit on the 64 DEVELOPMENT meshes; the parameter ranges and
   constraints equal the bundle's envelope; the FEM requests built from the
   spec equal the S10 requests of the 64 DEVELOPMENT cases byte for byte
   (`piginot.problem.fem_request`); shipped bundles and sidecars are not
   touched. On the owner machine, a clean 400-step replay of each family's
   registered run at the S24 revision matches the registered step-400 tensors
   byte for byte; the checkpoint manifests and safetensors headers may differ
   only in `run_hash` and the two file hashes that cover it
   (`campaigns/s24_replay.py`).
7. **Run binding.** `TrainingRunBinding` gains an optional
   `problem_spec_hash`, omitted from the canonical form when absent (as the
   S20 platform fingerprint of the execution profile is), so every existing
   binding keeps its run hash. Runs from a spec (the viability run, S28)
   bind it.
8. **Anti-scope.** No 3D, no new material (S25), no UI, no harmonic lifting,
   no change to the energy objective, the operator, the shipped bundles or
   their sidecars; no traction loads.

## Gate G24

| Check | Pass condition |
|---|---|
| `G24.protocol-first` | `S24-PROTOCOL.json` committed and this ADR accepted before the viability run's first step |
| `G24.lifting-exact` | Property tests, 1,000 random feasible parameter vectors and network outputs per kind and family: the Dirichlet data are met at points along every piece, end points included, within 1e-12 of the displacement scale; values and gradients finite at end points and joints |
| `G24.lifting-reduces` | `adf` with untrimmed lines equals `quarter` within 1e-12 (offset and scale) on 1,000 random points and parameter vectors |
| `G24.lifting-verified` | Decision 4 for both family specs and their `adf` variants: node and piece residuals ≤ 1e-12; conformity and orientation ≤ 1e-9; every mutant misses by > 1e-3 |
| `G24.regression-dogbone` | Decision 6: lifting bit-identical on 64 DEVELOPMENT meshes; envelope equal; 64 of 64 FEM requests byte-identical; step-400 replay tensors byte-identical to the same-CPU v0.2.0-base run (amendment 1) |
| `G24.regression-notched` | The same for the notched plate |
| `G24.spec-hash-stable` | Hash unchanged under key order, whitespace, number spelling and UTF-8/UTF-16 encoding; changed by every semantic edit of a leaf |
| `G24.spec-validation` | Each rejection rule of decision 2 has a failing-spec test |
| `G24.adf-viability` | Decision 5 guardrails pass; P-field L2 reported |
| `G24.binding-compat` | A binding without `problem_spec_hash` has exactly the previous canonical fields; the binding, checkpoint and bundle tests pass; G13–G23 records still verify |
| `G24.anti-scope` | No family token in `src/`; `.[runtime]` imports no FEM module; branch coverage of `piginot.lifting`, `piginot.problem` and the ProblemSpec contract ≥ 90% |
| `G24.portability` | Suite records on Linux and Windows |
| `G24.quality` | Full suite, ruff, mypy `--strict`, global coverage ≥ 90.5% |

The plan's `harmonic.smoke` check is dropped with the harmonic lifting
(decision 3).

## Result before the training runs (cloud, 2026-10-13)

`python -m campaigns.s24_regression --write` →
`artifacts/problem/S24-REGRESSION.json`, both families:

| Check | DogBone | Notched plate |
|---|---|---|
| Ranges, constraints, mesh, morph map, material, QoIs, Dirichlet set equal the bundle | yes | yes |
| `quarter` lifting bit-identical on DEVELOPMENT meshes | 64 / 64 | 64 / 64 |
| Frame bit-identical | 64 / 64 | 64 / 64 |
| FEM request from the spec byte-identical to S10 | 64 / 64 | 64 / 64 |
| Lifting verification, `quarter` and `adf` | passed | passed |

The `adf` variants' residuals are about 2e-16 of the displacement scale.

## Amendment 1 — replay reference on the same CPU (2026-10-14)

Accepted by the decision owner on 2026-10-14 (chat: "Same-CPU reference"),
before any replay comparison was made.

Decision 6 required a 400-step replay on the owner machine to match the
registered step-400 tensors. The registered runs were trained on
Linux/AVX512, and trained weights depend on the CPU instruction set: on the
owner machine (Linux/AVX2) the S13c recipe gives model state `9492c444…` at
step 400, against `e45edbdc…` registered (ADR 0100, decision 5 and
findings). The registered rule therefore cannot pass on that machine,
whatever the code.

The replay reference becomes a 400-step run of the same recipe at the v0.2.0
base revision (`399a146`, tag `v0.2.0-base`) on the same machine:

- DogBone: the `G19.resume` straight run (`artifacts/gates/S19.json`, model
  state `9492c444…`, run `eb3c5e15…`);
- notched: a new 400-step run in the `v0.2.0-base` checkout.

The rule is otherwise unchanged: every tensor of `model.safetensors` and
`state.safetensors` equal, `optimizer.json` and `adaptive.json`
byte-identical, manifest fields equal except `run_hash`,
`model_export_hash` and `tensor_payload_hash`, and safetensors metadata equal
except `run_hash`. In addition, the two run bindings must be equal except
`implementation_revision`, and the reference's revision must be the base
revision. The registered AVX512 hashes are reported beside each record and
are required only on that profile. The comparison is stronger than the
registered one in one respect: it covers every change from S19 to the
replay revision, not only S24's. `S24-PROTOCOL.json` stays as committed;
`G24.protocol-first` also requires the amendment to be committed once,
before the replay records (`campaigns/s24_replay.py`,
`campaigns/g24_gate.py`).

## Consequences

- S27 and S28 take a ProblemSpec as their input; S25 replaces the material
  block's single model with the registry.
- The quarter lifting stays the lifting of both shipped bundles; nothing they
  serve changes.
- A spec's `split`, `verification` and `fem` blocks are read by S28; in S24
  they are validated, hashed and compared with the shipped families' records.

## Verification

- `tests/contracts/test_problem_spec.py`, `tests/lifting/`,
  `tests/problem/`, `tests/campaigns/test_problem_specs.py`,
  `tests/campaigns/test_dogbone_s24_adf.py`, `tests/campaigns/test_g24_gate.py`.
- `campaigns/s24_regression.py` writes `artifacts/problem/S24-REGRESSION.json`.
- `campaigns/dogbone_v1/s24_adf.py` writes the protocol, trains the viability
  run and writes `artifacts/problem/S24-ADF-TRAIN-SCREEN.json`;
  `campaigns/s24_replay.py` writes `artifacts/problem/S24-REPLAY-<family>.json`;
  `campaigns/g24_gate.py` writes `artifacts/gates/S24.json`.
