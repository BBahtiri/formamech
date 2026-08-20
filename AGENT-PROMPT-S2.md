Work in: "C:\Users\betim\Documents\Data_Science\pi_ginot_large"

You are implementing PI-GINOT v1.0, an open-source framework that produces verified,
problem-specific physics-informed neural operators for 2D finite-strain elasticity.

Authorized scope for this task: STEP S2 only. Nothing else.

Canonical documents, in precedence order:
  1. STEPS-v1.0.md                          — the executable sequence and gates (binding)
  2. docs/adr/0001-s1-go-with-contingency.md — the S1 decision and the requirements it
                                               pushes into later steps
  3. S1-iteration8-review.md                 — why those requirements exist

S1 closed on 2026-08-20 with a human GO-with-contingency decision. You are the first
step of the framework proper. Everything under experiments/ is role=LEGACY and is
never a source of code, only of lessons.

────────────────────────────────────────────────────────────────────────
PRODUCT CONTEXT (so you understand what you are NOT building yet)
────────────────────────────────────────────────────────────────────────
The finished framework accepts a user-defined, fixed-topology parameterized 2D
geometry family with boundary conditions, loads and material; generates FEM-labelled
and physics-collocation data; trains a hybrid neural operator with a mixed
displacement/first-Piola formulation; verifies it against an explicit error budget;
and packages an inference bundle that refuses out-of-envelope inputs.

S2 builds none of that. S2 builds the vocabulary everything else is written in:
immutable, hashable, unit-safe specifications. Nothing in S2 computes physics,
touches a mesh, stores anything, or trains anything. If a spec cannot be described,
hashed and compared exactly, no later step can claim reproducibility — which is the
whole basis of the project's verification story.

────────────────────────────────────────────────────────────────────────
BEFORE EDITING ANYTHING
────────────────────────────────────────────────────────────────────────
1. Read STEPS-v1.0.md completely: section 0 (gate policy), S2, and skim S9, S11,
   S13, S14 so you know what will consume these contracts.
2. Read docs/adr/0001-s1-go-with-contingency.md, especially "Requirements this
   decision pushes into later steps".
3. Run `git status --short --branch`. Work on a branch off the current HEAD.
4. State back to me, before your first file write:
   - the S2 deliverables and the G2 checks as you understand them
   - the exact list of paths you intend to create
   - which gate checks you expect to be NOT_RUN in this environment, and why
   - your reading of the DERIVED-QUANTITY question in the S1 CARRY-FORWARD section
     below, which is the one place this brief goes beyond STEPS-v1.0.md

────────────────────────────────────────────────────────────────────────
METHOD: strict TDD, every step
────────────────────────────────────────────────────────────────────────
RED     — write the failing test first and run it to demonstrate the failure
GREEN   — implement the smallest change that passes
IMPROVE — remove duplication without adding scope

Never write implementation before its test. Never widen scope during IMPROVE.

Nothing from experiments/spike/ is copied. If a spike module looks useful, it is
REWRITTEN under test here. `tests/test_spike_isolation.py` enforces this and must
keep passing at every commit.

────────────────────────────────────────────────────────────────────────
STEP S2 — Contracts, units, canonical identity
────────────────────────────────────────────────────────────────────────
Objective: immutable, hashable, unit-safe specifications.

Build under `src/piginot/contracts/`:

**1. Frozen specification dataclasses**

  ProblemFamilySpec, GeometryFamilySpec, ParameterSpec, MaterialSpec,
  BoundaryConditionSpec, LoadSpec, QoISpec, ToleranceSpec, CampaignPlan,
  CanonicalMesh

  All frozen. Mutating any field raises. `CanonicalMesh` is a CONTRACT ONLY in this
  step — the shape of the payload, its required fields and its validation rules.
  The implementation lands in S5; do not write mesh algorithms here.

**2. Canonical internal unit system**

  mm, N, MPa, N·mm. Conversion happens explicitly at the DTO boundary and nowhere
  else. **No bare dimensional value appears anywhere in the API** — every dimensional
  quantity carries its unit until it crosses into canonical form. A float where a
  quantity is required is a typed error, not a silent assumption.

**3. `hash_spec(obj, domain) -> str`**

  BLAKE2b-256 over RFC 8785 (JCS) canonical JSON, with domain separation across
  exactly four domains: `problem`, `recipe`, `content`, `bundle`.

  Note two traps. RFC 8785 fixes key ordering, number serialisation and string
  escaping — do not hand-roll it from `json.dumps(sort_keys=True)`, which agrees on
  none of the hard cases. And Python's `json` module silently keeps the LAST value
  for a duplicate key; `G2.duplicate` requires rejection, so parsing needs
  `object_pairs_hook` or an equivalent.

**4. `ExecutionProfile`**

  Python version, library versions, BLAS thread count, device, dtype, RNG algorithm
  and seed, container digest if present. This is what makes a result attributable to
  the machine that produced it. The S1 spike learned this the hard way: its
  iteration-8 checkpoint could not be reproduced byte-identically because the
  profile was not pinned.

**5. Cross-runtime test vectors**

  Committed under `tests/fixtures/canonical/`. Generated once, then verified — never
  regenerated to make a failing test pass. A changed vector is either a bug or a
  deliberate, ADR-recorded format change.

────────────────────────────────────────────────────────────────────────
S1 CARRY-FORWARD — read this before designing ParameterSpec
────────────────────────────────────────────────────────────────────────
This section goes beyond STEPS-v1.0.md as written. Flag it in your state-back if you
disagree; do not silently skip it.

The S1 spike's central finding was that a geometry family's behaviour is governed by
DERIVED ratios, not by its raw parameters. The centre-notched family is controlled by
the net-section factor `k = width / (width - 2*notch_depth)`. Sampling the raw
parameter box uniformly covered `k` only to 1.56 out of an achievable 2.65, and that
coverage gap — not model capability — produced the apparent failure that nearly
stopped the project.

ADR 0001 pushes the consequence into S9 (the DOE must stratify on derived ratios) and
S14 (the envelope must be DECLARED in derived-ratio space). Neither can happen if the
S2 contract can only express per-parameter boxes.

So `ParameterSpec` and the envelope contract must be able to express **a named derived
quantity as an expression over parameters, with its own bounds**, alongside plain
per-parameter bounds.

Scope boundary, and hold it firmly: S2 defines the CONTRACT and its VALIDATION — a
name, an expression, bounds, units, and the rules that make it well-formed and
hashable. S2 does NOT evaluate expressions against geometry, does not sample, does
not check membership. That is S5, S6 and S9. If you find yourself writing an
evaluator over meshes, you have left S2.

Keep the expression language deliberately small and total: arithmetic over named
parameters and numeric literals, no calls, no control flow, no attribute access. It
must be safely parseable, hashable, and round-trippable through canonical JSON. `eval`
is not acceptable.

────────────────────────────────────────────────────────────────────────
GATE G2 — verbatim from STEPS-v1.0.md
────────────────────────────────────────────────────────────────────────
  G2.vectors    canonical JSON test vectors byte-identical on Linux and Windows,
                Python 3.11 and 3.12
  G2.strict     unknown field, non-finite number, and bare dimensional value each
                raise a typed error
  G2.units      physically equivalent inputs in different supported units hash EQUAL
                after conversion and UNEQUAL before it
  G2.domains    one payload under four domains yields four distinct hashes
  G2.immutable  mutating any frozen spec raises
  G2.duplicate  duplicate JSON keys rejected on parse
  G2.coverage   >=90% on piginot.contracts

Plus the universal checks at every gate:

  ruff check .
  ruff format --check .
  mypy --strict src/
  pytest --cov=src/piginot --cov-fail-under=90
  pip-audit
  pytest tests/test_no_fixture_names.py
  pytest tests/test_spike_isolation.py

Write artifacts/gates/S2.json using the schema in STEPS-v1.0.md section 3. Record the
MEASURED value on every check, including passes — that is what makes the ratchet
enforceable. Any required check that is FAIL, ERROR or NOT_RUN makes the gate verdict
FAIL. NOT_RUN is never PASS.

G2.vectors spans two operating systems and two Python versions, so it is a CI check,
not a local one. If you cannot trigger CI, record it NOT_RUN with that reason and say
so plainly — do not assert it from a single-platform run.

────────────────────────────────────────────────────────────────────────
ANTI-SCOPE — what must not exist when you are done
────────────────────────────────────────────────────────────────────────
- No lifecycle state machines, no approval ledgers, no supersession semantics.
- No storage, no content-addressed store, no filesystem layout (that is S3).
- No geometry algorithms, no mesh construction, no morphing (S5, S6).
- No physics, no constitutive kernel, no closure (S4).
- No DOE, no sampling, no role assignment (S9).
- No expression EVALUATION against geometry or parameter values beyond what the
  contract's own validation requires.
- No CLI subcommands beyond `--version`.
- Nothing copied from experiments/spike/.

────────────────────────────────────────────────────────────────────────
PROHIBITIONS
────────────────────────────────────────────────────────────────────────
- Commit freely on a branch. Do not push, tag, or merge to main.
- Do not install tools globally. Use the project venv.
- Do not hand-author uv.lock.
- Do not fabricate a SHA, a gate result, a benchmark number, or a coverage figure.
- Do not weaken a threshold to make a gate green. If a threshold is wrong, say so and
  stop. Loosening one requires an ADR naming the measurement that justified it.
- Do not report NOT_RUN as PASS.
- Do not regenerate a committed test vector to make a test pass.
- Do not edit STEPS-v1.0.md, docs/adr/0001-*, or S1-iteration8-review.md.
- Do not implement anything from S3 onward.

Note on the repository's git state: some git operations may leave stale lock files
(.git/index.lock, .git/HEAD.lock) and stray .git/objects/*/tmp_obj_* files. If a git
command fails with "Unable to create ... File exists", remove the stale lock and
retry. A `git gc` clears the accumulated temp objects.

────────────────────────────────────────────────────────────────────────
FINAL REPORT FORMAT
────────────────────────────────────────────────────────────────────────
1. Files created and changed, with line counts, and the branch name.
2. RED evidence: the failing test output that preceded each implementation.
3. GREEN results: exact command output for every G2 check and every universal check.
4. Gate table for G2 with measured values against thresholds.
5. Every NOT_RUN check with its reason.
6. The derived-quantity contract: its shape, its expression grammar, and the exact
   line where S2 stops and S9 would begin.
7. Anything you found that contradicts STEPS-v1.0.md or ADR 0001, or that you had to
   guess at. Be specific. A wrong guess recorded is recoverable; a wrong guess hidden
   is not.
8. Confirmation that no S3+ code, no push, no global install, no paid compute, and no
   copy from experiments/spike/ occurred.

*** Then STOP and report. Do not proceed to S3. ***
