# ADR 0100: A standalone fork for the S19+ line

- Status: Proposed — decision 1–4 need the decision owner's sign-off
- Date: 2026-10-10
- Step: S19 (gate `G19`)
- Plan: `STEPS-v2.0.md` §S19, and the implementation plan
  "PI-GINOT S19–S26 implementation plan: top 5 features"

## Context

Release 0.1.0 (ADR 0032) is tagged at `399a146` (`v0.1.0`, branch
`codex/s18-release`) and published to the private repository
`github.com/BBahtiri/verified-neural-operators`. One check of G18,
`G18.install` (the TestPyPI run), is still open and belongs to the release line.

The S19+ work adds five features: per-prediction certificates, verified
serving, calibrated bounds, a declarative problem specification and a campaign
compiler, plus a cross-family geometry encoder. It runs in parallel with
another assistant that keeps working on the release line. Two lines in one
repository have already collided once: test subprocesses that inherited
`GIT_DIR` damaged the shared Git configuration (71 commits are still authored
`S9 test`), and ADR numbers have clashed (two ADR 0017 drafts).

## Decisions

1. **A standalone clone, history preserved.** The S19+ line is a full clone of
   `codex/s18-release` at `399a146` with no remote, no object alternates and
   no worktree links. History is not rewritten; the 71 `S9 test` commits stay
   as they are, because rewriting them would change every recorded revision
   again. Every new commit is authored by Betim Bahtiri, and G19 checks it.
2. **ADR numbers from 0100.** This line numbers its decision records from
   0100, so it cannot collide with the release line, which is at 0032.
3. **Tags.** `v0.2.0-base` (annotated) marks the fork base; it and `v0.1.0`
   resolve to the same commit. Development versions are `0.2.0.devN`.
4. **Merge-back.** Work returns to the shared repository once, as a Git
   bundle, through a single decision record signed by the decision owner. This
   line never pushes, never adds a remote and never closes `G18.install`.

## G19 checks that need data outside Git

Three baseline checks cannot run in the one-core cloud workspace. Coverage
(`G19.coverage`) needs a multi-core machine. Two more need the ignored S10 run root and the registered S13c run
directory, which are not in Git:

- `G19.s13c-rescore`: the registered checkpoint re-scores
  `S13C-TRAIN-SCREEN.json` exactly, apart from the evaluator revision;
- `G19.resume`: a clean-checkout S13c training run stopped after step 200 and
  resumed to 400 reproduces the registered step-400 model-state hash.

Where that data is absent they are `NOT_RUN`, and `NOT_RUN` is never `PASS`,
so G19 stays open until they run on a machine that has the data. The commands
are in [S19-FORK.md](../fork/S19-FORK.md).

## Consequences

- G19 is evaluated by `python -m campaigns.g19_gate --write`.
- The coverage floor of the fork is the S18 figure as recorded: 91% to the
  integer, so the gate requires at least 90.5%. `G19.suite` (pass/fail) and
  `G19.coverage` are separate checks, because a full branch-coverage run is
  impractical on the one-core cloud workspace; coverage is measured on the
  decision owner's machine.
- Bundle outputs are compared byte for byte only within one execution profile
  (Python, operating system, machine, torch version and CPU capability),
  because float results may differ across CPU instruction sets.
  `artifacts/fork/S19-BUNDLE-OUTPUTS.json` holds one golden per profile.
