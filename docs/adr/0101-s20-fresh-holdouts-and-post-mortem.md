# ADR 0101: Fresh sealed holdouts and the notched post-mortem (S20)

- Status: Accepted — decisions 1–4 decided by the decision owner on 2026-10-11
  ("1–3 ok"); the failure classification (section "Post-mortem result") is
  added when the post-mortem is complete
- Date: 2026-10-11
- Step: S20 (gate `G20`), `STEPS-v2.0.md` §S20
- Builds on: ADR 0020 (HOLDOUT order), ADR 0030 and ADR 0031 (notched
  acceptance), ADR 0100 (fork)

## Context

Both v0.1.0 HOLDOUT sets are spent:

- the DogBone HOLDOUT was revealed in S14–S16 and the bundle is
  `HOLDOUT_VERIFIED`;
- the notched HOLDOUT was consumed in S14 (`S14-HOLDOUT-CONSUMED.json`) and
  failed its budget: all-cases p90 6.01% stress field and 6.40% peak against
  5% (`S14-HOLDOUT.json`), so the bundle is `DEVELOPMENT_VERIFIED` (ADR 0031).

The S21–S25 work needs unseen test cases twice: once to test certification of
the v0.1.0 models (S22–S23), once to test the new cross-family model (S25).
One shared set would leak: whatever S22–S23 reveal about where v0.1.0 fails
would inform the S25 design that the same set then grades.

## Decisions

1. **Fresh holdouts from the existing envelope.** New cases are drawn from
   each family's existing `CP-ENVELOPE` with the existing S9 DOE machinery
   (`piginot.doe`), under a newly committed seed, disjoint from every case
   ID and leakage group used so far. Each family gets two disjoint sealed
   sets:
   - `H1`: evaluation of certification for the v0.1.0 models (S22, S23);
   - `H2`: reserved for the S25 model; never opened before S25 CP-FREEZE.
2. **Size.** At least 64 cases per set, so 4 × 64 = 256 cases. Their
   coordinates are committed now (salted hash, secret material on the
   decision owner's machine, as in S9); the FEM reference for a set runs only
   at its reveal, with the unchanged S10 recipe of its family.
3. **Reference machine for timing gates.** Timing checks in S21, S22 and S26
   are measured on the decision owner's laptop under WSL2 (Linux x86_64,
   Python 3.11.15, torch 2.13.0+cu130, AVX2, 8 logical CPUs).
4. **Spent holdouts are burned.** The case IDs of the DogBone HOLDOUT and the
   notched HOLDOUT carry a burned mark; selection, tuning, calibration,
   acquisition and compiler code refuse them. Their results may be used
   descriptively in the notched post-mortem, never to select or tune.

## Post-mortem procedure

The notched post-mortem follows
`artifacts/campaigns/notched-v1/POSTMORTEM-PROTOCOL.json`, committed before
any per-case analysis. It ranks five non-exclusive hypotheses by preregistered
criteria; the result is recorded below and decides how S25 starts.

Viewed before the protocol was written: the aggregate budget and component
summaries of `S14-DEVELOPMENT.json` and `S14-HOLDOUT.json`, public since S14,
and ADR 0030. They show that the DEVELOPMENT budget also failed (5.24%,
5.34%), and that the reference discretisation term is about half of each
case's budget, with HOLDOUT maxima of 5.42% (stress field) and 4.75% (peak).

## Post-mortem result

Not yet recorded.

## Consequences

- S22 and S23 evaluate on `H1`; S25 on `H2`. Each set is revealed once per
  frozen model by a dedicated reveal command.
- 256 FEM reference solves are budgeted, about 100 s each for the notched
  family on the reference machine.
- Timing thresholds are only comparable on the reference machine; timing
  evidence from another machine is reported, not gated.
