# ADR 0009: S9 preregistration over a sealed audit service

- Status: Accepted
- Date: 2026-08-24
- Step: S9

## Context

S9 must prevent evidence-role leakage before S10 creates datasets and before S13
creates training artifacts. The open-source project has no independent audit service
that can keep an evaluation campaign sealed on behalf of the implementer. It can,
however, make the campaign immutable and commit to the withheld holdout before any
production artifact exists.

ADR 0001 adds a second load-bearing constraint: the DOE must cover governing derived
ratios, not merely look uniform in primitive parameters. The S1 failure showed that a
raw-parameter design can appear space filling while leaving the high-concentration
part of the executable family almost empty.

## Decision

S9 uses preregistration with a cryptographic holdout commitment rather than claiming
an unavailable sealed audit.

The DOE is generated from domain-separated BLAKE2b Latin-hypercube streams. Candidate
parameters are evaluated only through the public S6 family-membership predicate.
Every variable primitive parameter and every compiled derived quantity is a declared,
fixed-width normalized axis. Selection satisfies preregistered marginal strata,
boundaries, interactions and difficult-region quotas rarest-first, then fills the
remaining budget by weighted maximin distance. Candidate/selection, independent
validation, and guard-band streams have distinct identities; reserves are a
preregistered maximin ranking over the unused candidate stream. Rejected proposals
and negative guard-band cases remain visible with their reasons. The version-one
guard policy targets primitive-bound negatives. Derived-bound and family-membership
rejections remain in the proposal ledger, but are not claimed as targeted guard
strata.

Coverage is a vector, not a score. It records empirical fill distance against an
independently generated admissible validation stream, minimum stratum count,
marginal-stratum coverage, boundary coverage, interaction coverage,
difficult-region coverage, and sorted uncovered or unresolved requirement IDs. No
single geometric percentage is accepted as a substitute. `G9.coverage >= 90%` is the
S9 source-package branch-coverage floor established by the global gate policy.

The five evidence roles are `PILOT`, `TRAIN`, `DEVELOPMENT`, `HOLDOUT` and `LEGACY`.
A case can be assigned exactly once, even if the second write repeats the same role.
Cases joined by geometry lineage, morph seed or a preregistered near-duplicate
tolerance form one transitive `leakage_group_id` and are assigned atomically.
Production-label authorization fails until the exact case and leakage group already
exist in the role registry. Materialized node observations are checked against that
case-level registry, which makes `RANDOM_NODE_SPLIT` observable.

Replacement uses an immutable, role-specific ordered reserve policy. A caller reports
a failure and reason but cannot choose the replacement. Every failure produces an
append-only event naming the next unused reserve or explicit exhaustion. A failed
reserve cannot later re-enter the campaign. The rule is fixed before outcomes exist;
later failures do not change role assignments or hide the failed case.

The production holdout contains exactly 64 unique coordinate mappings with one fixed
schema. Its public record is:

```json
{
  "scheme": "blake2b-256",
  "campaign_hash": "...",
  "n_holdout_points": 64,
  "commitment": "H(salt || canonical_json(sorted_holdout_coordinates))",
  "created_utc": "...Z"
}
```

The salt is at least 32 random bytes. Salt and coordinates are retained only in an
ignored private reveal file and never enter Git, CP-ENVELOPE, gate artifacts, tests or
training inputs. Reveal comparison is constant-time. The commitment file must be
clean, tracked and added in exactly one Git commit. Its live filesystem mtime and its
first-add commit must both strictly precede each non-legacy training artifact. Legacy
exclusion is based on the explicit role, never a path convention.

`CP-ENVELOPE` binds content hashes for the compiled family, pilot evidence, final
executable envelope, DOE plan, role registry and replacement policy, together with
the public commitment, derived axes and fixed role/reserve budget. It discloses no
holdout coordinate.

## Consequences

- S10 can consume immutable case roles and refuse HOLDOUT cases before constructing
  collocation or normalization data.
- S13 can prove both live-mtime and Git-ancestry ordering when it creates its first
  production artifact.
- A fresh checkout cannot reconstruct historical mtimes because Git does not store
  them. The live check is therefore performed at artifact creation and recorded;
  Git first-add ancestry remains independently reproducible from repository history.
- The commitment proves that a later reveal matches the preregistered coordinates. It
  does not prove that the implementer never viewed the private file, and the project
  does not claim an independent sealed audit.
- Finite empirical fill distance and sampled family membership are evidence over
  their declared streams, not proofs of continuous-domain coverage or injectivity.
- Version-one guard points exercise primitive bounds. The independent validation
  stream and rejection ledger expose derived-bound failures, but do not prove dense
  coverage immediately outside every derived boundary.

## Verification

G9 binds this decision through 1,000-trial role and leakage properties, the named
node-split mutant, holdout-visibility checks, injected replacement failures, exact
commitment/reveal vectors, temporary-repository mtime and Git-order tests,
non-aggregated DOE metrics, strict typing/lint/security checks, and at least 90%
branch coverage over `src/piginot/doe`.
