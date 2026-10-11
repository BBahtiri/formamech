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

5. **Run binding v2 (technical).** `ExecutionProfile` gains an optional
   `platform_fingerprint` (`os`, `machine`, `cpu_capability`), filled by
   `piginot.training.platform_fingerprint.platform_fingerprint()`. Fields
   marked `omit_if_none` are left out of the canonical form when unset, so
   every v1 profile, run binding, `run_hash` and gate artifact keeps its bytes
   and hash. Every training or evaluation run started from S20 on binds the
   fingerprint; the frozen S0–S18 campaign code is not changed, so its
   recorded runs stay reproducible as registered.
6. **Windows portability (technical).** Platform-dependent tests skip with an
   explicit reason from `tests/platform_support.py` (symlinks, POSIX modes,
   the S13c initial-weights pin); the minimal child environment of the S15
   anti-scope test keeps `SYSTEMROOT`; the S13 runtime tree hash compares
   path and handle stat records on size and modification time only where
   `os.stat` and `os.fstat` use different APIs; `.python-version` pins 3.11.

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

### Public part (2026-10-11)

`python -m campaigns.notched_v1.postmortem --write` →
`artifacts/campaigns/notched-v1/POSTMORTEM-PUBLIC.json` (diagnostics D1–D6).
All-slice p90 values (tolerance 5% for both stress quantities):

| | DEVELOPMENT field | DEVELOPMENT peak | HOLDOUT field | HOLDOUT peak |
|---|---|---|---|---|
| Recorded budget total | 5.24% | 5.34% | 6.01% | 6.40% |
| Without the reference discretisation term | 2.60% | 3.76% | 2.61% | 4.67% |
| Surrogate term alone | 2.45% | 3.01% | 2.50% | 3.97% |
| Reference discretisation term alone | 2.85% | 2.21% | 3.56% | 2.80% |

Removing the reference term leaves 0–5 failing slices instead of 9–12. The
median case spends 43–44% of its budget on the reference term. The surrogate
error is 1.33× (field) and 1.65× (peak) larger on the sharpest notches than on
the bluntest. TRAIN-screen medians equal the DEVELOPMENT surrogate medians
(0.97× field, 1.20× peak): there is no train-to-test gap. From DEVELOPMENT to
HOLDOUT the surrogate p90 is unchanged for the field (+2%) and higher for the
peak (+32%).

| Hypothesis | Verdict | Primary margin |
|---|---|---|
| (e) reference discretisation dominates | SUPPORTED | 25% below tolerance without the reference term (DEVELOPMENT, worst stress quantity) |
| (c) optimization not converged | SUPPORTED by its criterion | TRAIN/DEVELOPMENT median ≥ 0.97 against 0.8 |
| (b) resolution at the notch root | SUPPORTED | sharpest/bluntest 1.65 against 1.5 |
| (a) envelope extrapolation | NOT_RUN | peak p90 +32% meets its first criterion; needs D7 |
| (d) metric fragility | NOT_SUPPORTED | without the 3 largest cases the field p90 still exceeds 5% |

Reading: the frozen notched network reaches about 2.5% (field) and 3–4%
(peak) p90 against the reference, on training and unseen cases alike. The
budget fails mainly because the reference recipe's conservative
discretisation bound at the notch root adds a comparable amount. Criterion
(c) is met because errors do not shrink from DEVELOPMENT to TRAIN; that is
equally consistent with a resolution floor (b), so (c) and (b) are read
together. By the protocol's decision rule, S25 starts with (e): refine the
reference recipe at the notch root and re-score DEVELOPMENT before any model
change.

### Owner-machine part

D7 (envelope distance), D8 (reference refinement) and D9 (quadrature probe):
pending.

## Consequences

- The draw is `campaigns/fresh_holdouts.py`, run once per family on the
  decision owner's machine. It verifies the family's spent S9 HOLDOUT against
  `CP-ENVELOPE.json` before using it, rejects every fresh case within the S9
  near-duplicate distance (0.015, normalized weighted metric) of any existing
  case, splits 128 cases 64/64 by alternating within each primitive stratum in
  a secret-keyed order, and commits each set with the exact-64 salted S9
  commitment. The private reveal (Git-ignored primary plus external backup)
  is written before the public `FRESH-HOLDOUTS.json`, which holds case IDs
  and commitments only.
- S22 and S23 evaluate on `H1`; S25 on `H2`. Each set is revealed once per
  frozen model by a dedicated reveal command.
- 256 FEM reference solves are budgeted, about 100 s each for the notched
  family on the reference machine.
- Timing thresholds are only comparable on the reference machine; timing
  evidence from another machine is reported, not gated.
