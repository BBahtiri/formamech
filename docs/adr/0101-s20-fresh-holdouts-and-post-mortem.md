# ADR 0101: Fresh sealed holdouts and the notched post-mortem (S20)

- Status: Accepted — decisions 1–4 decided by the decision owner on 2026-10-11
  ("1–3 ok"); decisions 5–6 technical; post-mortem result recorded
  2026-10-11
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

### Owner-machine part (2026-10-11, WSL2 reference machine)

D7 (`POSTMORTEM-D7.json`): pooled Spearman correlation between the
Mahalanobis distance to TRAIN and the surrogate term is 0.46 for the stress
field, 0.16 for the peak and 0.64 for the displacement (DEVELOPMENT and
HOLDOUT agree; median distance 1.98 and 2.11). The peak changed from
DEVELOPMENT to HOLDOUT (+32%) without depending on distance; the field
depends on distance without changing. No stress quantity meets both parts of
criterion (a).

D8 (`POSTMORTEM-D8.json`): the S10 notched run root is not on the owner's
machine, so the 8 DEVELOPMENT cases with the largest stress-field reference
term were re-solved with the S10 recipe; the recomputed references reproduce
the recorded S14 coarse-level terms to 6.5e-12 (relative). Against a
reference at twice the resolution, the observed reference error is
0.39–0.74% (field), 0.28–0.49% (peak) and about 0.01% (displacement). The
recorded discretisation bound is larger by a median factor of 6.8 (field),
7.7 (peak) and 12.2 (displacement); per case 6.3–7.3, 6.3–11.1 and
9.1–16.2. A two-level comparison underestimates the true error by at most a
factor 2 for first-order convergence, so the bound is at least 3.4×
(field) and 3.8× (peak) larger than the reference error it stands for.

| Hypothesis | Verdict | Primary margin |
|---|---|---|
| (e) the reference discretisation term dominates the budget | SUPPORTED | without it, 25% below tolerance (DEVELOPMENT, worst stress quantity); D8: the term is a conservative bound, at least 3.4× the reference error it estimates |
| (c) optimization not converged | SUPPORTED by its criterion | TRAIN/DEVELOPMENT medians 0.97 and 1.20 against 0.8 |
| (b) resolution at the notch root (model) | SUPPORTED | sharpest/bluntest peak surrogate 1.65 against 1.5 |
| (d) metric fragility | NOT_SUPPORTED | field p90 still above 5% without the 3 largest cases |
| (a) envelope extrapolation | NOT_SUPPORTED | D4 and D7 never coincide on one quantity |

**Classification.** The notched HOLDOUT failed mainly because of how the
reference error is charged, not because of the network: the budget adds, per
case, a reference-error bound built from one coarse level with a safety
factor, and that bound is several times larger than the reference error a
finer mesh shows. The network's own error is about 2.5% (field) and 3–4%
(peak) at p90, with no train-to-test gap. A smaller, real model limitation
remains at the sharpest notches, where single peak errors reach 8.5%
(candidate-001677). Criterion (c) holds because errors do not shrink from
DEVELOPMENT to TRAIN; together with (b) it reads as a resolution floor at the
notch root rather than an unfinished optimization.

**Consequence for S25 (decision rule of the protocol).** S25 starts with (e):
a notched v2 reference recipe (finer reference mesh at the notch root and a
three-level error estimate replacing the single coarse level), re-scoring
DEVELOPMENT with the unchanged 5% tolerances and the unchanged network,
before any model change. Changing how the reference term is computed is a
change of the evaluation method and needs its own owner DECISION ADR; the
tolerances are not loosened. Only then does (b) local resolution at the notch
root follow. The field error's dependence on distance (D7) is handled by the
envelope gating of S23. D9 was not run (see protocol rule above).

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
