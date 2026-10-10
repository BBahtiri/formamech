# ADR 0100: A standalone fork for the S19+ line

- Status: Proposed — decisions 1–4 need the decision owner's sign-off;
  decision 5 accepted by the decision owner on 2026-10-10
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
5. **`G19.resume` is judged on the profile that runs it** (accepted by the
   decision owner on 2026-10-10). The check passes when a clean-checkout S13c
   run stopped at step 200 and resumed to 400 is byte-identical, file for
   file, to a straight 0 → 400 run on the same execution profile. The
   registered replay probe hash `e45edbdc…` (Linux, AVX512) is reported beside
   it and is required only on that profile. The first wording in
   `STEPS-v2.0.md` ("reproduces the registered hash") demanded cross-CPU bit
   identity, which the hardware does not provide (see findings below); resume
   correctness is what the check is for, and it is not weakened.

## G19 checks that need data outside Git

Three baseline checks cannot run in the one-core cloud workspace. Coverage
(`G19.coverage`) needs a multi-core machine. Two more need the ignored S10 run
root and the registered S13c run directory, which are not in Git:

- `G19.s13c-rescore`: re-scoring the registered checkpoint prints a
  `TrialResult` equal, leaf for leaf, to the `trial_result` recorded in
  `S13C-TRAIN-SCREEN.json`, and the selection stays `ELIGIBLE`. (The first
  gate version compared that output with the whole report document, which
  can never match; corrected on 2026-10-10 before G19 closed.)
- `G19.resume`: a clean-checkout S13c training run stopped after step 200 and
  resumed to 400 equals a straight run to 400 on the same profile (decision 5).

Where that data is absent they are `NOT_RUN`, and `NOT_RUN` is never `PASS`,
so G19 stays open until they run on a machine that has the data. The commands
are in [S19-FORK.md](../fork/S19-FORK.md).

## Findings from the local G19 runs (2026-10-10)

The owner ran the three local checks on a Lunar Lake laptop (8 threads),
first on native Windows and then under WSL2 Ubuntu.

1. **Linux (WSL2), Python 3.11.15, torch 2.13.0+cu130, AVX2:** full suite at
   the base tree 3,075 passed, 1 skipped, 0 failed; branch coverage 91.44%.
   The pinned S13c initial weights reproduce. Resume is bit-exact: stopped at
   200 and resumed to 400 equals straight 0 → 400 (model-state hash
   `9492c444…`, manifest `d9568fc8…`). The trained weights differ from the
   registered AVX512 probe (`e45edbdc…`): AVX2 and AVX512 kernels round
   differently during training. The S13c note that a new run needs "AVX2 or
   AVX512" is true for the initial weights only.
2. **Native Windows, Python 3.12, torch 2.13.0+cpu, AVX2:** 11 failed, 3,063
   passed, 2 skipped. None is a defect of `src/piginot`:
   - six S13c tests: the initial weights differ from the pin on Windows'
     `torch+cpu` build, so S13c training cannot run there;
   - two tests expect POSIX file mode `0o600` for private files; Windows has
     no POSIX modes, so the owner-only guarantee is not enforced there;
   - one test starts Python without `SYSTEMROOT`, so Windows sockets (and
     torch) cannot initialise: a test bug;
   - two runtime tree-hash tests: `os.stat` and `os.fstat` report different
     file identities on Windows: a portability bug in campaign code;
   - symlink tests need Developer Mode or an administrator shell.
3. **Identity gap.** `ExecutionProfile` binds versions, threads, device, dtype
   and seed, but not the operating system or CPU instruction set, so runs on
   AVX2 and AVX512 at one revision share a `run_hash` and differ in weights.

Findings 2 and 3 are S20 work items: Windows portability (explicit skips with
reasons, the two real fixes, a `.python-version` pin to 3.11) and run binding
version 2 with `os` and `cpu_capability` (additive; v1 hashes stay
verifiable). G19 is evaluated on WSL2 Linux.

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
