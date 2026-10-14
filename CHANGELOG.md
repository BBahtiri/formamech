# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 0.2.0.dev0

Development line S19–S29 (`STEPS-v2.0.md`, ADR 0105), forked from `v0.1.0`.

### Added

- **Material registry and admission gate** (S25, ADR 0107 accepted):
  strain-energy formulas as data (`piginot.contracts.energy_formula`, a
  whitelist parser with unit checks); built-in Mooney–Rivlin, Yeoh and Gent
  models; `piginot.materials`, which derives stress, tangent and the
  plane-stress closure of any formula by automatic differentiation; the
  `piginot.physics` functions, the FEM solver and the energy objective accept
  every registered model while the Neo-Hookean path stays bit-identical; the
  admission gate (eight checks, including optimal FEM convergence on a
  manufactured solution); `material.energy` and `material.stretch_range` in
  the ProblemSpec; the preregistered Mooney–Rivlin viability run
  (`artifacts/materials/S25-PROTOCOL.json`,
  `campaigns/dogbone_v1/s25_mooney_rivlin.py`), which passes every frozen
  guardrail at the accuracy of the Neo-Hookean run (P-field L2 0.93%
  against 0.91%), and gate G25 (`campaigns/g25_gate.py`).
- **ProblemSpec and exact liftings from the spec** (S24, ADR 0106 accepted):
  the contract `piginot.contracts.ProblemSpec` (`piginot.problem-spec-v1`, a
  validated JSON problem description with a BLAKE2b identity); the package
  `piginot.lifting` with the quarter lifting (moved without numerical change)
  and an ADF lifting over line segments and circular arcs; `piginot.problem`,
  which builds FEM requests, query frames and liftings from a spec and
  verifies a lifting over the parameter box; an optional `problem_spec_hash`
  in run bindings; the two shipped families as specs under `campaigns/`, which
  reproduce every DEVELOPMENT lifting bit for bit and every S10 FEM request
  byte for byte (`artifacts/problem/S24-REGRESSION.json`); the preregistered
  ADF viability run, which passes every frozen guardrail (P-field L2 2.04%),
  and the 400-step replays, byte-identical to same-CPU runs of the v0.2.0
  base for both families (`artifacts/problem/S24-PROTOCOL.json`, amendment
  `S24-REPLAY-AMENDMENT.json`, `campaigns/dogbone_v1/s24_adf.py`,
  `campaigns/s24_replay.py`) and gate G24 (`campaigns/g24_gate.py`).
- **Calibrated bounds** (S23, ADR 0104 accepted): the runtime package
  `piginot.bounds` (split-conformal bounds, Mahalanobis calibration gate,
  calibration sidecars), `piginot predict --bounds` and
  `serve(..., calibration=...)`; the preregistered protocol and the
  DEVELOPMENT calibration of both bundles (`bundles/calibration/`); the
  single H1 reveal command `campaigns/s23_reveal.py`; gate G23
  (`campaigns/g23_gate.py`). G23 PASS: every gated H1 coverage count passed;
  the reported DogBone displacement bound did not (46 of 63), so `--bounds`
  withholds it (H1 results in `bundles/h1-results/`).
- **Verified serving** (S22): `piginot predict --mode fast|certified|exact`,
  FEM recipe sidecars in `bundles/fem-recipes/`, warm-start FEM solves, the
  `verify` extra (ADR 0103).
- **Per-prediction physics certificate** (S21): `piginot.certify` (ADR 0102).
- **Evaluation harness, fresh sealed holdouts, notched post-mortem** (S20,
  ADR 0101).
- **Fork and baseline lock** (S19): a standalone fork of `v0.1.0` (ADR 0100),
  the fork manifest tool `scripts/verify_fork.py`, gate `G19`
  (`campaigns/g19_gate.py`) and the baseline records in `artifacts/fork/`.

### Changed

- **S22 follow-up**: ADR 0103 accepted; the G21 and G22 tables are in
  `STEPS-v2.0.md`; the user guide explains what the certified and exact modes
  measure and what they do not; `docs/limitations.md` states the limits of
  S20–S22, so the 0.1.0 limitations sign-off must be renewed before the next
  release; `piginot predict` documents `--tolerance` and the other serving
  options, and refuses `--escalate` or `--tolerance` where they would be
  ignored. Native Windows runs of S21–S22 are still to be recorded.

### Fixed

- The notched S13 protocol could not be rebuilt on the 0.2.0 line: its
  request plan bound the live package version (0.2.0.dev0 since the S19
  fork). It now binds the S10 profile the dataset was planned under, so the
  notched trainer, `s13_close` and `s14` start again.
- Gates G24 and G25 ignore each other's gate and suite records, so one
  suite run serves both.
- Full-suite findings of S24 and S25: contracts no longer evaluate
  expressions (the S2 rule; values moved to `piginot.problem.expressions`
  and `piginot.materials.values`); `piginot.problem.fem` renamed
  `piginot.problem.requests` (the S5 module rule); the confidence indicator,
  S14 evaluation code frozen by its protocol, restored unchanged; test
  repositories set their own Git identity.

## [0.1.0] - 2026-10-06

The first public release. It completes the STEPS-v1.0 plan (S0–S18). Read
[the limitations](docs/limitations.md) before using any number below.

### Added

- **Contracts and storage** (S2, S3): typed contracts with units,
  RFC 8785 canonical identity, and a content-addressed store with retention
  tiers and crash visibility.
- **Physics and FEM** (S4, S7, S8): a compressible Neo-Hookean plane-stress
  kernel; a primary P2 finite-element solver; manufactured-solution
  verification and recovery-based error estimators.
- **Geometry and families** (S5, S6): geometry ingestion with semantic
  boundary tags; morph maps (`CircularTransitionMap` for the DogBone, `BlendedNotchMap` for the notched plate) and family
  membership.
- **Evidence roles and datasets** (S9, S10): a preregistered design of
  experiments with a committed HOLDOUT, projection and collocation datasets,
  and normalisation fitted on TRAIN only.
- **Operator and objectives** (S11, S12): the S11 boundary cross-attention
  operator and a nondimensional objective registry with gradient checks. The
  served operator is the energy-form `pi-ginot-energy-parametric-film-v1`,
  with exact Dirichlet lifting and constitutive stress (S13c).
- **Training and selection** (S13): deterministic, resumable training under a
  preregistered protocol, with TRAIN-only selection and frozen candidates.
- **Verification** (S14): per-case error budgets that include the FEM
  reference terms, slice statistics, refusal envelopes, a confidence
  indicator, latency, and a twelve-family reducer. Claim levels stop at
  `HOLDOUT_VERIFIED`.
- **Bundles and command line** (S15): hash-closed bundles, `piginot predict`
  and `piginot inspect`, refusals with reason codes, and limitations on every
  response.
- **Mutation suite** (S16): twenty source-level mutants, each detected by its
  primary gate, with committed fired-gate signatures.
- **Examples, benchmarks and docs** (S17): a fixed-mesh example that runs end
  to end in CI, benchmark tables that reproduce from pinned seeds, a refusal
  suite over two families, and a strict mkdocs site.
- **Release** (S18): the upstream-sources record, claim-ceiling tests, a
  validated `CITATION.cff`, a TestPyPI publish-and-install workflow for Linux,
  macOS and Windows, and the G18 gate.
- **Bundles and user guide**: both trained bundles under `bundles/`, with
  example requests, and a [user guide](docs/user-guide.md) covering
  prediction, the end-to-end example and building a model for a new
  geometry.

### Verification results

| Family | Claim level | HOLDOUT p90 (stress field / peak / displacement) | Decisions |
| --- | --- | --- | --- |
| DogBone (`dogbone-v1`) | `HOLDOUT_VERIFIED` | 2.75% / 4.02% / 0.29% | DEVELOPMENT budget accepted by the decision owner (ADR 0021) |
| Centre-notched plate (`notched-v1`) | `DEVELOPMENT_VERIFIED` | 6.01% / 6.40% / 0.53% (fails the 5% tolerance) | DEVELOPMENT budget accepted (ADR 0030); `G17.non-dogbone` accepted (ADR 0031) |

Gates G0–G17 pass. G14's DogBone budget check, G14's notched budget check and
`G17.non-dogbone` pass through the decision owner's signed acceptances, which
record the computed results alongside.

### Changed

- The S1 spike no longer redistributes the third-party GINOT modules. They
  are pinned in `experiments/spike/upstream_pi_ginot/THIRD_PARTY.json` and
  fetched with `python -m experiments.spike.fetch_upstream`
  ([upstream sources](docs/adr/upstream-sources.md)).
- The Git history is rewritten to remove the same modules from every commit
  after `6f5f0cb`. `artifacts/release/HISTORY-REWRITE.json` maps the original
  commit ids recorded in earlier evidence to the rewritten ones, and
  `campaigns/history.py` resolves them.
- The repository moved to `github.com/BBahtiri/verified-neural-operators`
  with the rewritten history. The project, the package and the command keep
  the name PI-GINOT (`piginot`). The S0–S8 gate records link to CI runs of
  the former repository, `BBahtiri/PI-GINOT-Agentic-Framework`; recorded
  evidence is not edited.
- `uv.lock` is regenerated; it had not been updated for the S15 `runtime`
  extra.
- The sdist holds the package, licence, README, citation, changelog,
  `pyproject.toml` and `uv.lock` only.

### Removed

- The agent prompts (`AGENT-PROMPT-*.md`) and the S1 coding review
  (`S1-iteration8-review.md`) are no longer tracked.

[0.1.0]: https://github.com/BBahtiri/verified-neural-operators/releases/tag/v0.1.0
