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
| `G19.s13c-rescore` | the `TrialResult` printed by re-scoring the registered S13c checkpoint equals the `trial_result` recorded in `S13C-TRAIN-SCREEN.json`: exactly on the registered profile; elsewhere every non-numeric leaf exactly and numeric leaves within a relative 1e-10 (ADR 0100 decision 6); the selection is `selected` / `ELIGIBLE` |
| `G19.resume` | clean-checkout S13c training stopped at 200 and resumed to 400 is byte-identical (every checkpoint file) to a straight 0 → 400 run on the same execution profile; the registered probe hash (Linux/AVX512) is reported, and required only on that profile (ADR 0100 decision 5) |

Anti-scope: no source change besides the version, docs, the fork script and the
gate; no push; no new remote.

## 3. Gate G20 — evaluation harness, notched post-mortem, fresh holdouts

Decision record: ADR 0101. Post-mortem protocol:
`artifacts/campaigns/notched-v1/POSTMORTEM-PROTOCOL.json`.

| Check | Pass condition |
|---|---|
| `G20.protocol-first` | the protocol commit is an ancestor of every post-mortem artifact |
| `G20.harness` | the evaluation harness recomputes the recorded S14 DEVELOPMENT and HOLDOUT budget statistics of both families from their per-case records, tolerance 0; off-by-one case and swapped-split mutants are detected |
| `G20.burned` | selection and calibration paths raise on burned HOLDOUT IDs (tests, with a mutant) |
| `G20.holdouts` | per family, H1 and H2 commitments of ≥ 64 cases each, disjoint from each other and from every used case ID and leakage group; protocol commit precedes the draw |
| `G20.rca` | every hypothesis of the protocol is SUPPORTED, NOT_SUPPORTED or NOT_RUN with its evidence; ADR 0101 records the ranking |
| `G20.portability` | native Windows suite: 0 failed, every platform skip carries a reason; Linux suite unchanged |
| `G20.binding` | run binding v2 records operating system and CPU capability; every v1 binding and gate artifact still verifies |
| `G20.quality` | ruff, ruff format, mypy --strict clean; new modules ≥ 90% branch coverage; global ≥ 90.5% |

## 4. Gate G21 — per-prediction physics certificate

Evaluated by `python -m campaigns.g21_gate --write`. Decision record: ADR 0102.
Protocol and ratified thresholds: `artifacts/certify/S21-PROTOCOL.json`
(64 DEVELOPMENT cases per family, reference machine of ADR 0101 decision 3).

| Check | Pass condition |
|---|---|
| `G21.protocol-first` | the protocol is committed once, before both results files |
| `G21.reference-machine` | both results ran on Linux x86_64 with AVX2 |
| `G21.recompute` | the records cover the protocol's cases in order; the summaries recompute exactly |
| `G21.integrity` | every certificate `OK`; recomputed references reproduce the S14 coarse terms within 1e-6 relative; mesh identity; vertex gap and Dirichlet deviation ≤ 1e-9 |
| `G21.rank` | Spearman(η, true energy-norm error) ≥ 0.8 per family |
| `G21.effectivity` | η / true error in [0.5, 2.0] for ≥ 90% of cases per family |
| `G21.stress` | Spearman(stress change along the step, true stress error) ≥ 0.7 per family |
| `G21.versus-s14` | the certificate ranks the stress error at least as well as the S14 indicator |
| `G21.cost` | median certificate time ≤ 10% of a cold S10-recipe FEM solve |
| `G21.determinism` | a re-issued certificate equals the recorded one byte for byte |
| `G21.quality` | ruff, ruff format, mypy --strict clean; recorded Linux suite of this code tree with 0 failed; global branch coverage ≥ 90.5%; `piginot.certify` ≥ 90% |

Kill rule: Spearman < 0.6 on both families would have made η an acquisition
signal only. Not triggered (1.000 on both).

Changes from the implementation plan: the plan's `consistency` (energy drop
against η²/2) and `guards` (constructed bad states) are covered by
`tests/certify` inside `G21.quality` rather than gated separately; the
reaction-force and P-field QoI ranks are reported, not gated; there is no
native Windows check (see the S22 follow-up in the changelog).

## 5. Gate G22 — verified serving

Evaluated by `python -m campaigns.g22_gate --write`. Decision record: ADR 0103.
Protocol and ratified thresholds: `artifacts/serving/S22-PROTOCOL.json`
(64 DEVELOPMENT cases per family, reference machine).

| Check | Pass condition |
|---|---|
| `G22.protocol-first` | the protocol is committed once, before both results files |
| `G22.reference-machine` | both results ran on Linux x86_64 with AVX2 |
| `G22.recompute` | the records cover the protocol's cases in order; the summaries recompute exactly |
| `G22.integrity` | every cold solve succeeds; the recipes rebuild every S10 request byte for byte |
| `G22.never-silent` | every exact request converges on some rung of the solve ladder or returns an explicit `solve_failed` |
| `G22.equivalence` | exact results equal cold solves within 1e-6 relative |
| `G22.speedup` | median speed-up of exact over cold ≥ 10× per family (plan: 3×) |
| `G22.correction` | median corrected peak-stress error ≤ 0.1 × the operator's, per family |
| `G22.fast-unchanged` | fast-mode documents of the example requests equal the S19 goldens |
| `G22.light-install` | without the FEM packages, fast serves and the other modes raise `MissingExtraError` naming `piginot[verify]` |
| `G22.old-bundles` | the gated v0.1.0 bundles load unchanged and their recipes bind their manifests |
| `G22.determinism` | an exact result hash and a certified document re-served under the protocol's conditions equal the recorded ones |
| `G22.quality` | as `G21.quality`, with `piginot.serving` ≥ 90% |

Changes from the implementation plan: bundle schema v2 is replaced by FEM
recipe sidecars (`bundles/fem-recipes/`, bound to each manifest hash), so the
shipped bundles stay byte-identical; `G22.light-install` blocks the FEM imports
in a subprocess instead of installing a fresh `.[runtime]` environment (a real
install test belongs to the release step); timings ran with three worker
processes, one run per case.

## 6. Gates G23–G26

The checks, thresholds and kill criteria of S23–S26 are specified in the
implementation plan, one table per step. Each step copies its table into this
document, with the owner's ratified thresholds, before its first run.
