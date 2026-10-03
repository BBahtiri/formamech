# ADR 0022: Replace the missing sections of IMPLEMENTATION-v1.0-OSS.md

- Status: Accepted — the decision owner asked on 2026-10-03 to replace
  `IMPLEMENTATION-v1.0-OSS.md`
- Date: 2026-10-03
- Steps: S14 (`G14.reducer`), S16 (mutation suite)
- Specification: [`docs/specs/implementation-v1.0-oss-replacement.md`](../specs/implementation-v1.0-oss-replacement.md)

## Context

`STEPS-v1.0.md` names `IMPLEMENTATION-v1.0-OSS.md` as its companion for
strategy and rationale, and cites it for two concrete lists:

- S14: "Gate reducer over the twelve families";
- S16: "the twenty mutants from `IMPLEMENTATION-v1.0-OSS.md` §1.13".

The file is not in the repository or in any connected folder. Without it,
`G14.reducer` stays `NOT_RUN`, so G14 cannot pass and S16 cannot start.

## Decision

1. The project writes its own replacement of the two cited sections. It is
   adopted as binding for S14 and S16 in place of the missing text. It does
   not claim to reproduce the original.
2. **Families (§1.12).** The twelve families are the gate artifacts of the
   twelve upstream steps S2–S13, in dependency order. They were not chosen
   from any S14 result. For the operator family the artifact is the one that
   gates the frozen candidate's architecture (`S11-ENERGY.json`, ADR 0018).
   Every family is required. A family's status is the worst status of its
   recorded checks; a missing artifact is `NOT_RUN`, and an inconsistent one
   is `ERROR`.
3. **Mutants (§1.13).** Twenty mutants, each with one primary gate check.
   Fifteen are the mutants and seeded defects already detected by G8–G13.
   Five are new and guard the S14 and S15 claims: a zeroed budget component,
   a reducer that passes `NOT_RUN`, a reveal before consumption, a
   substituted `P_material`, and an undetected bundle change.
4. Any later recovery of the original file does not change this record. A
   difference would need a new ADR.

## Consequences

- `G14.reducer` becomes measurable. The reducer is tested on injected cases
  and then applied to the twelve recorded families.
- S16 has a fixed list of twenty mutants. Five of them need new harnesses,
  in S14 and S15 code.
- The families add no new evidence of their own. They make the G14 claim
  depend explicitly on every upstream gate still being `PASS`.

## Verification

- `tests/verification` and `tests/campaigns/test_dogbone_s14_*.py`: every
  injected `FAIL`, `ERROR`, `NOT_RUN` and missing check in each family gives
  `NOT_VERIFIED`; the all-`PASS` control reaches the evidence level.
- `artifacts/gates/S14.json` records the twelve family statuses.
