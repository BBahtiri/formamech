# PI-GINOT v2 — Steps S19–S26 with Quality Gates

**Extends:** `STEPS-v1.0.md`. Its gate policy (§0) applies unchanged: `NOT_RUN`
is never `PASS`, thresholds only tighten, every gate writes
`artifacts/gates/S<NN>.json`, anti-scope is enforced by tests, coverage floors
are per module.

**Plan:** "PI-GINOT S19–S26 implementation plan: top 5 features" (Claude Doc,
10 Oct 2026). Decision records of this line start at ADR 0100 (ADR 0100).

---

## 0. Additional rules for S19+

1. One branch per step, `codex/sNN-<slug>`; one preregistration per run;
   the protocol commit is an ancestor of every run's bound revision.
2. HOLDOUT is never read by selection, tuning, calibration, acquisition or
   compiler code; a revealed holdout is burned; every holdout is revealed once
   per frozen model, by a dedicated reveal command.
3. Family names and numbers stay in `campaigns/`; `src/piginot` stays
   family-free (`tests/test_no_fixture_names.py`).
4. Contracts are additive: v0.1.0 bundles, run bindings and gate artifacts keep
   loading and verifying. New schemas carry a version field.
5. `.[runtime]` stays FEM-free; FEM-backed serving goes into a new `.[verify]`
   extra.
6. New modules ≥ 90% branch coverage; existing package floors unchanged; the
   global floor of `src/piginot` is the S18 figure (91%).
7. All commits authored by Betim Bahtiri; this line never pushes and never
   closes `G18.install`.

## 1. Sequence

| Step | Name | Feature | Effort | Depends | Gate |
|---|---|---|---|---|---|
| **S19** | Fork and baseline lock | — | 0.5–1 d | S18 | `G19` |
| **S20** | Evaluation harness, notched-HOLDOUT root cause, fresh sealed holdouts | 2 | 1 wk | S19 | `G20` ⚑ |
| **S21** | Per-prediction physics certificate | 1 | 1.5 wk | S20 | `G21` |
| **S22** | Verified serving with Newton warm start | 1 | 1.5 wk | S21 | `G22` |
| **S23** | Calibrated conformal bounds and envelope gating | 3 | 1 wk | S20, S22 | `G23` ⚑ |
| **S24** | ProblemSpec and generalized exact BC lifting | 4 | 2–3 wk | S20 | `G24` ⚑ |
| **S25** | Cross-family geometry encoder | 2 | 4–6 wk | S20, S23, S24 | `G25` ⚑ |
| **S26** | One-command campaign compiler and evidence report | 5 | 2–3 wk | S23, S24 | `G26` ⚑ |

⚑ = the step needs a signed owner decision (ADR 0101, 0104–0107).
S24 may run in parallel with S21–S23, and S26 with S25, in separate clones.

## 2. Gate G19 — fork and baseline lock

Evaluated by `python -m campaigns.g19_gate --write`.

| Check | Pass condition |
|---|---|
| `G19.isolation` | no remote; no object alternates; `.git` a directory; one worktree |
| `G19.ancestry` | `HEAD` descends from `399a146`; `v0.1.0` and `v0.2.0-base` resolve to it |
| `G19.identity` | every commit after the base, and the configured user, are Betim Bahtiri |
| `G19.quality` | `ruff check`, `ruff format --check`, `mypy --strict src/` clean |
| `G19.suite` | full suite at the base: ≥ 3,075 passed, 0 failed, 0 errors |
| `G19.coverage` | branch coverage of `src/piginot` ≥ 90.5% (S18 recorded 91% to the integer) |
| `G19.manifests` | `bundles/` and `artifacts/` equal the base, in Git and byte for byte (BLAKE2b) |
| `G19.bundles` | both bundles load with recorded manifests and claim levels; examples served byte-identically twice and against the golden of this execution profile |
| `G19.s13c-rescore` | the registered S13c checkpoint re-scores `S13C-TRAIN-SCREEN.json` exactly, apart from the evaluator revision |
| `G19.resume` | clean-checkout S13c training stopped at 200 and resumed to 400 is byte-identical (every checkpoint file) to a straight 0 → 400 run on the same execution profile; the registered probe hash (Linux/AVX512) is reported, and required only on that profile (ADR 0100 decision 5) |

Anti-scope: no source change besides the version, docs, the fork script and the
gate; no push; no new remote.

## 3. Gates G20–G26

The checks, thresholds and kill criteria of S20–S26 are specified in the
implementation plan, one table per step. Each step copies its table into this
document, with the owner's ratified thresholds, before its first run.
