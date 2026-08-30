# S9 Production Campaign Sign-Off

- Status: Production family ratified; remaining campaign choices open
- Step: S9
- Blocks: G9 and S10
- Branch: `codex/s9-doe-preregistration`
- Contract baseline: `CP-ENVELOPE` schema 1.1 and ADR 0009

This page records the scientific campaign choices that must be approved before a
production `CP-ENVELOPE` can be created. It is deliberately not a commitment,
not a private reveal, and not a gate artifact. No DOE seed, allocation seed,
holdout salt, holdout coordinate, or production role label belongs in this file.

## Current Verdict

G9 remains open. The user requested alignment with the pinned DogBone repository,
and ADR 0010 now records owner ratification of one exact executable family to
de-risk the first ceremony. The repository still lacks approval of its complete
material/load envelope, DOE budgets and quotas, leakage tolerance, reserve
distribution, public identifiers, and private reveal-handling decision. S10 must not
begin until those choices are approved and committed, then used to create the first
production `CP-ENVELOPE`.

## Evidence Basis

- `artifacts/gates/S1.json` records `GO_WITH_CONTINGENCY` and the stress-error
  basis for continuing after the spike.
- `docs/adr/0001-s1-go-with-contingency.md` requires S9 to stratify DOE coverage
  on derived governing ratios, not just raw primitive parameters.
- `artifacts/gates/S6.json` records the executable morphing and membership
  boundary: fixed connectivity, S5 canonical mesh validation, semantic tag
  transport, no silent healing, no remeshing, and sampled Jacobian positivity
  through `triangle-degree-5-dunavant-v1`.
- `docs/adr/0006-s6-morphing-membership.md` states that S6 establishes exact
  discrete mesh validity and sampled continuous injectivity, not a global
  inverse-map proof.
- `artifacts/gates/S8.json` records accepted FEM verification evidence at
  `072f500` with signed-off thresholds in
  `docs/adr/0002-fem-thresholds.md`.
- `docs/adr/0009-s9-preregistration-over-sealed-audit.md` fixes the S9 closure
  ceremony, exact-64 holdout, CSPRNG-origin campaign secrets, leakage boundary,
  replacement rule, and `CP-ENVELOPE` schema 1.1.
- `docs/adr/0010-s9-first-production-family.md` accepts the exact pinned DogBone
  family. `artifacts/campaigns/dogbone-v1/family-validation.json` records a disjoint
  public sweep with 500 accepted and zero morph-validation failures.
- `artifacts/campaigns/dogbone-v1/fem-solvability-pilot.json` records the ordered
  results of 12 preselected reference, boundary, and difficult-geometry P2 solves at
  the pinned source material and displacement. Its content hash is
  `e2aa1e0e2cd9d20b31043d9c6cb066c138c9b3cb00c7432111ddd38e6f156928`.

## Approval Items

Each row needs an explicit owner decision before the production ceremony.

| Item | Required Decision | Current Position |
| --- | --- | --- |
| Production family | Exact compiled family, reference mesh identity, morph map recipe, validity policy hash, and S6 membership predicate. | Approved 2026-08-29. Recipe `dogbone-quarter-no-hole-v1`; compiled hash `a8143a75…973a4`; mesh `3bfc1043…d82c`; analytic circular-transition map; policy `b3c2df59…931d`. |
| Pilot evidence | Non-empty ordered set of content-addressed `EvidenceReference` records with role `PILOT` or `LEGACY`. | Available for approval: FEM solvability pilot `e2aa1e0e…6928` and DogBone family validation `4dfac913…eb92d` as `PILOT`, plus S1 gate `cd0b01e8…9470` as `LEGACY`. The finite FEM cost/solvability pilot is complete; the exact checkpoint evidence set is not yet approved. |
| Executable envelope | Primitive bounds, derived bounds, material bounds, load bounds, and refusal policy for out-of-envelope inputs. | Geometry proposal ready: original four geometry ranges plus `arc_feasibility_ratio >= 1.35`. Source-aligned fixed-point proposal exercised by the FEM pilot: `E = 760 MPa`, `nu = 0.23`, prescribed displacement `1 mm`. Those material/load bounds are not approved, so no DogBone executable-envelope preimage or hash exists. |
| Derived axes | Non-empty ordered list of derived stratification axes to bind into `CP-ENVELOPE`. | Proposed order: `arc_feasibility_ratio`, `fillet_to_gauge_ratio`, `grip_to_gauge_ratio`, `length_to_gauge_ratio`, all compiled through S2 and evaluated by S6. Not approved. |
| DOE axis policy | Axis weights, stratum counts, minimum per stratum, boundary widths, boundary quotas, interactions, difficult-region quotas, and coverage thresholds. | Exact proposal is recorded below. A public non-production dry run resolved every declared requirement with fill distance `0.0865461`, minimum stratum count 10, and all four quota fractions equal to 1.0. Not approved. |
| Role budget | Positive `PILOT`, `TRAIN`, `DEVELOPMENT`, exact `HOLDOUT=64`, and explicit non-negative `LEGACY` and `RESERVE` counts. | Proposed first campaign: `PILOT=24`, `TRAIN=256`, `DEVELOPMENT=64`, `HOLDOUT=64`, `LEGACY=0`, `RESERVE=32`; DOE `sample_count=408`, `reserve_count=32`. The S1 gate remains a `LEGACY` evidence reference, but no pre-S9 case enters the production case ledger. Not approved. |
| Candidate streams | `candidate_count`, `validation_count`, `validation_candidate_count`, `guard_band_count`, and `guard_band_fraction`. | Proposed first campaign: `candidate_count=4096`, `validation_count=512`, `validation_candidate_count=4096`, `guard_band_count=32`, `guard_band_fraction=0.05`. Not approved. |
| Leakage policy | Near-duplicate tolerance in normalized primitive-plus-derived axis space. | Proposed first campaign: `near_duplicate_tolerance=0.015` under `geometry-lineage-morph-seed-chebyshev-v1`. The DOE records descriptors, and closure replays them before accepting the role allocation. The private DOE-plan hash binds both policy and descriptors. Not approved. |
| Reserve distribution | Ordered reserve policy per replaceable role and exhaustion behavior. | Proposed first campaign: `PILOT=4`, `TRAIN=20`, `DEVELOPMENT=8`, `HOLDOUT=0`, `LEGACY=0` reserves, ranked by the committed reserve order. Version one permits reserves only for `PILOT`, `TRAIN`, and `DEVELOPMENT`; `HOLDOUT` is never replaced. Not approved. |
| Public naming | Stable public `manifest_id` and replacement `rule_id`. | Proposed: manifest `dogbone-v1-public-cases`; replacement rule `role-local-design-order-v1`. Guard and DOE stream algorithm IDs remain fixed by code. Not approved. |
| Private reveal handling | Local ignored path and access controls for salt plus exact holdout coordinates. | Proposed: `.piginot-private/dogbone-v1/reveal-v1.json`, created before public commitment, readable only by the ceremony operator, with a separately access-controlled backup. The reveal must not enter Git, gate artifacts, tests, or training inputs. Not approved. |

The production family is the first decision. It fixes the reference mesh, primitive
and derived vocabulary, executable envelope, meaningful difficult regions, and the
cost basis for every population count. Numeric DOE or budget values approved before
that choice would be unsupported assumptions and must not be used.

The requested first-campaign source direction has now been resolved to the exact
accepted decision in ADR 0010. The decision retains the pinned reference
implementation's four primitive ranges and makes its fixed-connectivity exclusion
explicit rather than hiding failed geometries. Its hash may enter a later production
commitment only after the remaining approval items on this page are ratified.

## FEM Solvability and Cost Pilot

The public pilot population was fixed before any FEM outcome was observed. It
contains the reference geometry; the worst accepted angle, Jacobian, and boundary
feature cases from the public S6 validation stream; coupled parameter-box boundary
sentinels; one near-arc-margin case; and two high-concentration cases. Each request
uses the ratified family, the pinned source values `E = 760 MPa`, `nu = 0.23`, and
prescribed displacement `1 mm`, P2 displacement, eight load steps, solver tolerance
`1e-10`, and the verified S7 scikit-fem backend.

On the artifact's recorded Windows/Python 3.12 CPU environment, all 12 of 12 cases
converged. Total solve time was 261.094054 s; median time was 21.019403 s; p95 and
maximum time were 25.600437 s. No load step required more than eight Newton
iterations, and the largest final residual was
`7.128736607124267e-11`. The uncompressed canonical FEM result payloads totalled
52,117,983 bytes, with a maximum single payload of 4,403,090 bytes. Every P2 solve
had 7,458 scalar displacement DOFs (3,729 vector-valued basis sites).

This is finite sentinel evidence and a local cost measurement. It is not a uniform
solvability proof, a portable performance benchmark, or approval of the fixed
material/load point. Population projections must identify whether they use average
or conservative per-case timing and whether result payloads are stored as canonical
JSON, compressed, or converted to a later verified dataset representation.

Sequential single-host projections from this finite pilot are:

| Production solves | Mean-time projection | p95-time projection | Raw JSON size projection |
| --- | ---: | ---: | ---: |
| 408 active + 32 reserve | 2.66 h | 3.13 h | 1.69 GiB |
| 512 | 3.09 h | 3.64 h | 2.07 GiB |
| 1024 | 6.19 h | 7.28 h | 4.14 GiB |
| 2048 | 12.38 h | 14.56 h | 8.28 GiB |

The proposed first role budget keeps the original DogBone training scale of 256
training cases, adds an exact 64-case sealed holdout, reserves 64 cases for visible
development decisions, and keeps 24 production `PILOT` cases for pipeline shakeout
after commitment but before using the training labels. Thirty-two reserves give
approximately 7.8% replacement capacity over active production cases without ever
replacing `HOLDOUT`.

## Proposed DOE Policy and Public Dry Run

The proposed axis table is finite and exact. Fixed material/load coordinates retain
one visible stratum but weight `0.01`, so zero-width axes do not materially dilute the
weighted distance. `arc_feasibility_ratio` uses two strata because the ratified upper
bound of 20 is reachable only near a measure-small primitive corner; an equal
four-stratum policy left its highest cell empty even in a 2,400-candidate public
stream. Its load-bearing lower edge is covered by explicit difficult-region quotas
instead of a misleading symmetric boundary quota.

| Axis | Source | Weight | Strata | Minimum/stratum | Boundary width | Minimum/boundary |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `L_total` | primitive | 1.00 | 4 | 8 | 0.10 | 8 |
| `W_grip` | primitive | 1.00 | 4 | 8 | 0.10 | 8 |
| `W_gauge` | primitive | 1.00 | 4 | 8 | 0.10 | 8 |
| `R_fillet` | primitive | 1.00 | 4 | 8 | 0.10 | 8 |
| `elastic_modulus` | primitive, fixed | 0.01 | 1 | 1 | 0.10 | 0 |
| `poisson_ratio` | primitive, fixed | 0.01 | 1 | 1 | 0.10 | 0 |
| `prescribed_displacement` | primitive, fixed | 0.01 | 1 | 1 | 0.10 | 0 |
| `arc_feasibility_ratio` | derived | 1.50 | 2 | 8 | 0.10 | 0 |
| `fillet_to_gauge_ratio` | derived | 1.50 | 4 | 8 | 0.15 | 8 |
| `grip_to_gauge_ratio` | derived | 2.00 | 4 | 8 | 0.15 | 8 |
| `length_to_gauge_ratio` | derived | 1.25 | 4 | 8 | 0.15 | 8 |

Proposed interactions require two samples per cell for
`W_grip x W_gauge`, `W_gauge x R_fillet`, and
`L_total x grip_to_gauge_ratio`. Proposed difficult regions are:

- `high-concentration`: normalized `grip_to_gauge_ratio` in `[0.75, 1]` and
  `arc_feasibility_ratio` in `[0, 0.10]`, minimum 32;
- `near-arc-margin`: normalized `arc_feasibility_ratio` in `[0, 0.025]`, minimum 16;
- `short-high-concentration`: normalized `L_total` in `[0, 0.25]`,
  `grip_to_gauge_ratio` in `[0.75, 1]`, and `arc_feasibility_ratio` in
  `[0, 0.10]`, minimum 8.

The proposed acceptance thresholds are maximum empirical fill distance `0.10`,
minimum stratum count 8, all marginal/boundary/interaction/difficult-region coverage
fractions equal to 1.0, and no unresolved requirement identifier.

A public, non-production dry run used seed
`public-dogbone-s9-proposal-dry-run-v2`; it did not create campaign secrets, roles, a
holdout, or a commitment. Of 4,096 candidate proposals, 3,842 passed membership; of
4,096 independent validation proposals, 3,841 passed. The selector produced 408
active cases, 32 reserves, 512 validation points, and 32 negative guards with no
unresolved quota. Measured fill distance was `0.08654600038744202`, minimum stratum
count was 10, difficult-region counts were 34, 75, and 9 respectively, and all 440
active-plus-reserve points had distinct leakage groups at tolerance `0.015`. This
dry run shows that the proposal is executable; it is not the production DOE and its
public seed must never be reused for closure.

This proposal does not close the family-generality risk from ADR 0001.
`tests/test_no_fixture_names.py` checks `G17.fixture-names` hygiene, but
`G17.non-dogbone` remains `NOT_RUN` until a genuinely different notched-plate topology
reaches a verified bundle through the same API. The circular-transition provider is
currently exercised by only this campaign.

## Derived-Quantity Requirement

The derived axes are not optional process decoration. They are the core S1 lesson.
The centre-notched spike failure was a DOE coverage failure in the governing ratio
`k = W / (W - 2d)`, not a raw-parameter failure. A production campaign that is
space filling in raw parameters but sparse in the governing derived ratios would
repeat the same defect with a cleaner pipeline.

For the first production campaign, approve derived axes only after the family is
chosen. At minimum:

- A notched or net-section-driven family must declare a normalized concentration
  ratio axis and give its high-ratio region an explicit difficult-region quota.
- A tapered or dogbone-like family must declare whichever reduced-section,
  taper, or gauge-region ratio controls stress concentration for that family.
- Every declared derived axis must be produced by the S2 derived-expression
  machinery and evaluated by the S6 membership predicate; no textual `eval` or
  post-hoc notebook calculation is acceptable.

## Ceremony Preconditions

The production ceremony may start only after the approval items above are
committed:

1. Generate one campaign-secret bundle with `generate_campaign_secrets()`.
2. Build `CoupledDesignSpec` using `campaign_secrets.doe_seed`.
3. Generate the coupled DOE through the approved typed executable envelope, which
   wraps the exact S6 family-membership predicate and the approved material/load
   bounds.
4. Allocate roles with `campaign_secrets.allocation_seed`.
5. Build the replacement policy from preregistered reserves.
6. Persist the private reveal outside Git before publication.
7. Call `build_preregistration()` once with the approved campaign inputs and the
   same `CampaignSecrets` bundle.
8. Commit only the public S9 artifacts: `CP-ENVELOPE`, public non-holdout case
   manifest, replacement policy, and any public provenance needed by ADR 0009.
   Keep private DOE-plan preimages, production role-allocation preimages, holdout
   coordinates, salt, and reveal material outside Git.
9. Create downstream artifacts only after the public commitment is tracked and
   clean; capture live mtime evidence at artifact creation and verify Git first-add
   ancestry before claiming `G9.commitment-order`.

## Non-Approvals

The following are intentionally not approved by this draft:

- no owner-ratified complete material/load executable envelope;
- no owner-ratified exact pilot-evidence set;
- no role budget;
- no DOE quota table;
- no leakage tolerance;
- no reserve distribution;
- no production commitment;
- no `CP-ENVELOPE`;
- no S10 start.

## Sign-Off Record

Partial decision record only; this is not a full S9 campaign sign-off.

- Decision owner: Betim Bahtiri
- Recorded UTC timestamp: 2026-08-29T13:00:53Z
- Approved production family: `dogbone-quarter-no-hole-v1`
- Approved compiled-family hash:
  `a8143a75e62c614d40f52049cfc88c1cda74d38c9bd488cc3bed4c5678d973a4`
- Approved reference mesh:
  `3bfc1043f5aa9e989469adacb0ee1d0b483825ffabcd5dfac4bf9caca580d82c`
- Approved validity-policy hash:
  `b3c2df59eed939f1aa14a69d05a6fd2dc57be571ffba33c6355982713cc6931d`

Full campaign approval still requires the exact scientific and population values
above, plus the resulting hashes of the pilot evidence, executable envelope, private
DOE plan including leakage policy and descriptors, role registry, reserve policy,
and public case manifest.
