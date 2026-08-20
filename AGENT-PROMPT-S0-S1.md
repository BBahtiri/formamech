
Work in: <"C:\Users\betim\Documents\Data_Science\pi_ginot_large"

You are implementing PI-GINOT v1.0, an open-source framework that produces verified,
problem-specific physics-informed neural operators for 2D finite-strain elasticity.

Authorized scope for this task: STEP S0 and STEP S1 only. Nothing else.

Canonical documents, in precedence order:
  1. STEPS-v1.0.md            — the executable sequence and quality gates (binding)
  2. IMPLEMENTATION-v1.0-OSS.md — strategy and rationale (context)
  3. IMPLEMENTATION-v3.0.md   — long-term target architecture (REFERENCE ONLY)

IMPLEMENTATION-v3.0.md is superseded as an executable plan. Do not issue work from
it. In particular, do NOT build: work-package manifests, a validated WP/DG graph,
compare-and-set acceptance heads, project-scoped decision isolation, sealed audits,
custodial evidence, Ed25519 signing, revocation state machines, or a second FEM
stack. Those were deliberately cut or deferred. If you find yourself building
governance infrastructure, you have misread the scope.

────────────────────────────────────────────────────────────────────────
PRODUCT CONTEXT (so you understand what you are NOT building yet)
────────────────────────────────────────────────────────────────────────
The finished framework will accept a user-defined, fixed-topology parameterized 2D
geometry family with boundary conditions, loads and material, generate FEM-labelled
and physics-collocation data, train a hybrid neural operator with a mixed
displacement/first-Piola formulation, verify it against an explicit error budget, and
package a signed-manifest inference bundle that refuses out-of-envelope inputs.

S0 and S1 implement none of that. S0 is repository scaffolding. S1 is a throwaway
measurement that decides whether the framework is worth building at all.

────────────────────────────────────────────────────────────────────────
BEFORE EDITING ANYTHING
────────────────────────────────────────────────────────────────────────
1. Read STEPS-v1.0.md completely, especially section 0 (gate policy), S0 and S1.
2. Read any AGENTS.md or CLAUDE.md in the tree.
3. Run `git status --short --branch`. Inventory every existing file, tracked and
   untracked. Preserve all pre-existing user files and changes.
4. Check availability of: git, python (>=3.11), uv, pip, a C compiler.
5. Verify the working directory is not inside a sync folder. If it is, STOP and
   report this before writing anything.
6. State back to me, before your first file write:
   - the S0 and S1 success criteria as you understand them
   - the exact list of paths you intend to create
   - which gate checks you expect to be NOT_RUN in this environment, and why

────────────────────────────────────────────────────────────────────────
METHOD: strict TDD, every step
────────────────────────────────────────────────────────────────────────
RED     — write the failing test first and run it to demonstrate the failure
GREEN   — implement the smallest change that passes
IMPROVE — remove duplication without adding scope

Never write implementation before its test. Never widen scope during IMPROVE.

────────────────────────────────────────────────────────────────────────
STEP S0 — Repository foundation and CI
────────────────────────────────────────────────────────────────────────
Create exactly these paths and no others:

  pyproject.toml
  uv.lock                       (ONLY if uv is actually available; never hand-author)
  .gitignore
  .pre-commit-config.yaml
  LICENSE                       (MIT)
  README.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  CITATION.cff
  .github/workflows/ci.yml
  src/piginot/__init__.py       (exposes __version__ only)
  src/piginot/cli.py            (a --version command and nothing else)
  tests/test_version.py
  tests/test_no_fixture_names.py
  tests/test_spike_isolation.py
  docs/mkdocs.yml
  docs/index.md
  docs/limitations.md           (stub, but honest — see below)
  docs/adr/0000-template.md

Requirements:
- Python >=3.11,<3.14. Optional extras: [fem], [ml], [dev].
- CI matrix: {ubuntu-latest, windows-latest} x {3.11, 3.12}.
- Every external GitHub Action pinned to a full 40-character commit SHA. No tags.
- CI runs lint, format check, mypy --strict, and pytest with coverage.
- Pre-commit config wires ruff and ruff-format.

The two guard tests are load-bearing and must be written NOW, not retrofitted:
- tests/test_no_fixture_names.py: walks src/ and asserts no file contains the
  case-insensitive substrings "dogbone", "gauge", "fillet", or "grip". The framework
  is geometry-agnostic; this test is how that stays true.
- tests/test_spike_isolation.py: asserts no module under src/ has any import path,
  direct or transitive, into experiments/.

docs/limitations.md must already say, in plain language, that this framework performs
numerical verification against FEM, not experimental validation; that it establishes
nothing about geometries outside a bundle's declared envelope; and that finite
verification sampling is evidence, not a proof of a uniform error bound. Write it
honestly even though there is no code behind it yet.

GATE G0 — all of these must pass before you touch S1:
  G0.install    pip install -e ".[dev]" succeeds in a clean venv
  G0.lint       ruff check . and ruff format --check . clean
  G0.types      mypy --strict src/ clean
  G0.tests      pytest green, at least 3 tests
  G0.ci         CI green on all four matrix cells (or NOT_RUN with reason if you
                cannot trigger CI; say so explicitly)
  G0.guards     both guard tests present and passing
  G0.location   working directory confirmed not inside a sync folder
  G0.actions    every external action pinned to a 40-char SHA

Write artifacts/gates/S0.json using the schema in STEPS-v1.0.md section 3.

*** STOP HERE. Report G0 results and wait for my confirmation before starting S1. ***

────────────────────────────────────────────────────────────────────────
STEP S1 — Risk spike (THROWAWAY CODE, quarantined)
────────────────────────────────────────────────────────────────────────
Purpose: measure whether a mixed (u, P) neural operator can reach engineering
tolerance on stress for a parameterized 2D family. This single number decides whether
the remaining ~34 weeks of work are justified. Everything here is disposable.

Hard quarantine rules:
- All spike code lives under experiments/spike/.
- experiments/ is excluded from the package in pyproject.toml.
- Nothing in src/ imports anything in experiments/. The guard test enforces this.
- Every artifact produced is tagged role="LEGACY" and is permanently ineligible as
  release evidence.
- When the real modules are built in S2 onward, spike code is REWRITTEN under test,
  never copy-pasted.

Build:

experiments/spike/geometry.py
  Two families, analytic point and mesh generation:
  (a) parametric dogbone, 4 parameters: L_total 40-70, W_grip 16-26, W_gauge 6-14,
      R_fillet 8-20 (mm), quarter model with double symmetry;
  (b) centre-notched plate, 3 parameters: width, notch radius, notch depth.
  Family (b) is the important one. It is the proof that nothing here is dogbone-specific.

experiments/spike/fem.py
  Crude 2D finite-strain plane-stress solver via scikit-fem (preferred, pip-installable)
  or FEniCSx. Compressible Neo-Hookean:
      psi(F) = (mu/2)(tr(F^T F) - 3) - mu*ln(J) + (lambda/2)(ln J)^2
      P(F)   = mu*(F - F^-T) + lambda*ln(J)*F^-T
      mu = E/(2(1+nu)),  lambda = E*nu/((1+nu)(1-2nu))
  Plane-stress closure: solve for F33 > 0 from
      g(F33) = mu*(F33^2 - 1) + lambda*ln(J_2D) + lambda*ln(F33) = 0
  g is strictly increasing for F33 > 0, so the positive root is unique. Damped Newton.
  Material: E = 760 MPa, nu = 0.23, units mm/N/MPa.
  Fixed fine mesh. No error estimator, no tag abstraction, no provenance. Correctness
  only. Newton with load continuation.

experiments/spike/model.py
  Take the existing physics_decoder architecture (branch geometry encoder + trunk
  query decoder with cross-attention and FiLM conditioning) and ADD A SECOND HEAD
  that predicts first-Piola stress P directly.
  This is the first test of the mixed formulation and it is the point of the spike.
  P_served = P_predicted is authoritative. The constitutive law P_material(F) is a
  consistency oracle only — never substituted for P_served.

experiments/spike/run.py
  1. Generate 30 geometries per family, solve all 60. If fewer than 50 converge, stop
     and report — that is itself the finding.
  2. Split BY GEOMETRY, never by node: 20 train / 10 evaluate per family.
  3. Train hybrid: FEM supervision on u and P, plus equilibrium residual
     ||Div_X(P_served) + B|| on interior collocation points.
     Note: L_constitutive compares P_served to P_material on IN-PLANE components only.
     The 33 component is identically zero by construction of the closure and carries
     no information.
  4. Learning-rate sweep only. No further tuning.
  5. Emit experiments/spike/results.json.

results.json must contain, per family, evaluated on the 10 held-out geometries:
  - relative L2 error on u
  - relative L2 error on P
  - worst-case relative error on peak notch/fillet stress
  - number of FEM cases attempted, converged, failed (with failure reasons)
  - wall-clock: mean seconds per FEM solve, mean seconds per operator inference
  - whether the peak-stress QoI is well-posed or singular at the notch root

GATE G1:
  G1.solves     >= 50 of 60 FEM cases converged
  G1.sanity     spike FEM reproduces uniaxial extension of a unit square against the
                closed-form solution to 1e-6
  G1.split      evaluation geometries share no nodes, meshes or RNG seeds with training
  G1.report     results.json complete as specified above
  G1.isolation  tests/test_spike_isolation.py passes

Write artifacts/gates/S1.json.

*** Then STOP and report. Do not proceed to S2. ***

The GO/NO-GO decision is MINE, not yours. Report the numbers; do not interpret them
as permission to continue. For reference, the rule is:
    P error <= 5%   -> proceed
    P error 5-15%   -> proceed with an extra architecture iteration budgeted
    P error > 15%   -> do not build the framework; iterate on architecture first

────────────────────────────────────────────────────────────────────────
PROHIBITIONS
────────────────────────────────────────────────────────────────────────
- Do not commit, push, publish, or tag.
- Do not install tools globally. Use a project venv.
- Do not modify credentials or use paid compute.
- Do not hand-author uv.lock. If uv is absent, omit it and report NOT_RUN.
- Do not fabricate a SHA, a gate result, a benchmark number, or a coverage figure.
- Do not weaken a threshold to make a gate green. If a threshold is wrong, say so and
  stop.
- Do not report NOT_RUN as PASS.
- Do not create convenience files outside the authorized path lists.
- Do not edit STEPS-v1.0.md, IMPLEMENTATION-v1.0-OSS.md, or IMPLEMENTATION-v3.0.md.
- Do not implement anything from S2 onward: no contracts module, no content store, no
  real geometry ingestion, no morphing, no production FEM backend, no MMS harness, no
  DOE, no dataset manifests, no trainer, no verification gates, no bundle, no CLI
  subcommands beyond --version.

────────────────────────────────────────────────────────────────────────
FINAL REPORT FORMAT
────────────────────────────────────────────────────────────────────────
1. Files created and changed, with line counts.
2. RED evidence: the failing test output that preceded each implementation.
3. GREEN results: exact command output for every gate check.
4. Gate tables for G0 and G1 with measured values against thresholds.
5. Every NOT_RUN check with its reason.
6. For S1: the three headline numbers per family, stated plainly, with no
   interpretation and no recommendation.
7. Anything you found that contradicts STEPS-v1.0.md, or that you had to guess at.
   Be specific. A wrong guess recorded is recoverable; a wrong guess hidden is not.
8. Confirmation that no S2+ code, no commit, no push, no global install and no paid
   compute occurred.
```

---

## Notes for you, not for the agent

**Why the stop between S0 and S1.** S0 is mechanical and S1 is a research task with a
real chance of surprising you. Reviewing the scaffolding first costs ten minutes and
means you are not debugging repo layout and FEM convergence in the same session.

**Expect S1 to take multiple sessions.** Three weeks of human effort does not compress
into one agent run. The prompt is written so the agent can be re-invoked: the gate
artifacts and `results.json` are the resumable state. If the session ends mid-spike,
tell the next one which sub-deliverables already exist.

**The most likely failure is the spike FEM, not the model.** Finite-strain plane
stress with the `F33` closure is where crude solvers go wrong, and a wrong solver
produces confidently wrong ground truth. `G1.sanity` is deliberately the second check
in the gate — if uniaxial extension of a square does not match the closed form to 1e-6,
stop and fix that before reading any accuracy number.

**Prompt for S2 onward** should follow the same shape: one or two steps, the gate
table verbatim from `STEPS-v1.0.md`, an explicit anti-scope list naming the steps that
come later, and a hard stop at the gate.
