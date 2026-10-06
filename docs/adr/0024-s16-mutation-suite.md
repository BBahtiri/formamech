# ADR 0024: S16 mutation suite — source-level mutants, gate detectors and signatures

- Status: Accepted — S16-D1 to S16-D5 accepted as implemented by the decision
  owner on 2026-10-06, at the release sign-off (ADR 0032)
- Date: 2026-10-05
- Step: S16 (gate `G16`)
- Builds on: ADR 0022 (the twenty mutants of §1.13), ADR 0023 (bundle)

## Context

`G16` asks that each of the twenty mutants fire its named primary gate, that
the full sorted set of gates each mutant fires be committed as a baseline, that
a CI job compare those signatures per release and report a lost secondary
detection as a coverage regression, and that the suite finish in under 45
minutes on CI.

The existing evidence for mutants 1–15 does not run the production code it
guards: the G8 seeded defects compare analytic values, several gates call a
defective copy of a function next to production, and the G9 and G10 artifacts
record hand-entered booleans. A real defect in the production function would
therefore not reach most of those checks.

## Decisions

1. **S16-D1 Mutants are source edits of production code.** A mutant is one or
   two exact snippet replacements in the function that holds the defect
   (`piginot.mutation.SourceEdit`). The engine reads that function's source,
   applies the edit, compiles it in the function's own module namespace and
   swaps the code object into the original function object, so every
   reference runs the defect: imports by name, default arguments, bound
   methods, properties and decorator wrappers. Leaving the context restores
   the code. A snippet must occur exactly once, so a mutant that no longer
   matches its source fails loudly instead of silently testing nothing. The
   engine (`src/piginot/mutation`) is imported by no production module.
2. **S16-D2 Detectors are gate procedures on fixed fixtures.** Each detector
   runs the procedure of one gate check against the production code currently
   loaded, on a CI-sized fixture with an answer fixed independently of that
   code (an analytic value, a hand-built expectation or a committed artifact).
   Twenty-two detectors cover G4, G7–G15. A detector *fires* when its status
   is not `PASS`, so an exception under a mutant counts as a detection. The
   unmutated control must pass every detector before and after the mutants
   run, and every run starts with the registered caches cleared.
3. **S16-D3 Gate checks that missed real defects are hardened.**
   `G8.seeded-defects` adds production controls: a body-force solve and a
   normal-aware traction solve whose reactions are known exactly, and the
   anisotropic-scaling defect checks the coordinates the backend reports.
   `G13.selection` is checked against an independently ranked fixture, and
   `G14.holdout-order` checks that the reader is never called before the
   consumption record exists, not only that the record exists afterwards. The
   committed S8, S13 and S14 artifacts are unchanged; G16 records the
   hardened procedures passing on the unmutated code.
4. **S16-D4 Signatures and regressions.** The baseline
   (`artifacts/mutation/S16-SIGNATURES.json`) holds, per mutant, its primary
   gate, its sorted fired-gate set and a hash of its edits. A comparison is
   `PASS`, `MUTANT_CHANGED` (edits changed; re-record), `COVERAGE_REGRESSION`
   (a secondary detection lost while the primary still fires) or
   `PRIMARY_LOST`, with exit codes 0, 2, 3 and 4. New detections are reported
   and do not fail. The release workflow (`.github/workflows/mutation.yml`)
   runs the suite on every published release, `v*` tag or manual dispatch and
   fails on any status other than `PASS`.
5. **S16-D5 Runtime.** `G16.runtime` measures the full mutation suite (control,
   twenty mutants, control). The release job has a 45-minute timeout. The full
   pytest suite is reported separately.

## Consequences

- All twenty primaries fire; seven mutants also fire secondary gates (for
  example `CLOSURE_WRONG_ROOT` fires seven), which the baseline protects.
- A refactor that changes a mutated line makes the mutant fail to apply. The
  mutant must then be re-pointed and the baseline re-recorded deliberately.
- Detectors use small fixtures; they show that each gate procedure catches the
  defect, not that the original campaign evidence would have changed.
- The engine compiles edited source of the project's own modules at test time;
  it never runs in production code paths.
