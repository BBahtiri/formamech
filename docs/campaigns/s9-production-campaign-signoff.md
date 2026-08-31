# S9 Production Campaign Sign-Off

- Status: Complete scientific campaign policy ratified; production ceremony pending
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
de-risk the first ceremony. On 2026-08-30 the owner ratified the evidence basis,
material point, role budget, candidate streams, non-load DOE policy and thresholds,
leakage tolerance, reserve distribution, public identifiers, and private
reveal-handling policy. On 2026-08-31 the owner then ratified the exact 10%–20%
nominal-strain envelope, its 64-step solver recipe, load-axis policy, and qualified
evidence hash. The scientific campaign choices are complete. G9 remains open, and
S10 must not begin, until the one-shot production ceremony creates and commits the
first `CP-ENVELOPE` and all G9 evidence passes.

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
- `artifacts/campaigns/dogbone-v1/fem-strain-envelope-pilot.json` records the
  ordered results of 24 P2 solves using the replacement load primitive
  `nominal_axial_strain`, with right-grip displacement computed as
  `nominal_axial_strain * L_total / 2`. Its content hash is
  `4ff6dba3f311ff949d5cf662b2d68dd358e5b6f2592157677797f6173d0f14d2`.
- `artifacts/campaigns/dogbone-v1/fem-strain-envelope-qualification.json` records
  the same population with 32 continuation steps. Its content hash is
  `fa5f031b761101987d8af642b86e2b893babf4d0e6e9289b27846bd12479f325`.
- `artifacts/campaigns/dogbone-v1/fem-strain-envelope-qualified.json` records the
  final uniform 64-step qualification. Its content hash is
  `2f6cfcb9ed56e072a25ed3b09ade2de9666ade5c7be15fcd33489eb67e4502a0`.
- `artifacts/campaigns/dogbone-v1/s9-strain-load-proposal-dry-run.json` records the
  executable, secret-free DOE proposal. Its content hash is
  `42cb8f7ee8a16f89663289ecf39cc959d656f3fbba152c9826c469f0a920be34`.

## Approval Items

Each row needs an explicit owner decision before the production ceremony.

| Item | Required Decision | Current Position |
| --- | --- | --- |
| Production family | Exact compiled family, reference mesh identity, morph map recipe, validity policy hash, and S6 membership predicate. | Approved 2026-08-29. Recipe `dogbone-quarter-no-hole-v1`; compiled hash `a8143a75…973a4`; mesh `3bfc1043…d82c`; analytic circular-transition map; policy `b3c2df59…931d`. |
| Pilot evidence | Non-empty ordered set of content-addressed `EvidenceReference` records with role `PILOT` or `LEGACY`. | Approved campaign input: FEM solvability pilot `e2aa1e0e…6928`, DogBone family validation `4dfac913…eb92d`, and uniform 64-step strain qualification `2f6cfcb9…02a0` as `PILOT`, plus S1 gate `cd0b01e8…9470` as `LEGACY`. The failed 16/32-step screens remain visible diagnostics and are not checkpoint inputs. |
| Executable envelope | Primitive bounds, derived bounds, material bounds, load bounds, and refusal policy for out-of-envelope inputs. | Approved: original four geometry ranges plus `arc_feasibility_ratio >= 1.35`; `E = 760 MPa`; `nu = 0.23`; `nominal_axial_strain in [0.10, 0.20]`; policy `nominal-axial-strain-times-half-length-v1`, applying `u = strain * L_total / 2`; and 64 uniform load steps. Approved envelope hash: `be25dabbc501dfc52dec6335254c07ae0d0355d533364de9dcc6c21907522690`. |
| Derived axes | Non-empty ordered list of derived stratification axes to bind into `CP-ENVELOPE`. | Approved 2026-08-30 in order: `arc_feasibility_ratio`, `fillet_to_gauge_ratio`, `grip_to_gauge_ratio`, `length_to_gauge_ratio`, all compiled through S2 and evaluated by S6. |
| DOE axis policy | Axis weights, stratum counts, minimum per stratum, boundary widths, boundary quotas, interactions, difficult-region quotas, and coverage thresholds. | Approved in full on 2026-08-31. The load axis uses weight `0.05`, four strata, both boundary quotas, a strain-by-concentration interaction, and a high-strain/high-concentration difficult region. Its paired public dry run resolved every requirement with fill distance `0.0923852`, minimum stratum count 8, and all four quota fractions equal to 1.0; closure refuses any non-empty `unresolved_quota_ids`. |
| Role budget | Positive `PILOT`, `TRAIN`, `DEVELOPMENT`, exact `HOLDOUT=64`, and explicit non-negative `LEGACY` and `RESERVE` counts. | Approved 2026-08-30: `PILOT=24`, `TRAIN=256`, `DEVELOPMENT=64`, `HOLDOUT=64`, `LEGACY=0`, `RESERVE=32`; DOE `sample_count=408`, `reserve_count=32`. The S1 gate remains a `LEGACY` evidence reference, but no pre-S9 case enters the production case ledger. |
| Candidate streams | `candidate_count`, `validation_count`, `validation_candidate_count`, `guard_band_count`, and `guard_band_fraction`. | Approved 2026-08-30: `candidate_count=4096`, `validation_count=512`, `validation_candidate_count=4096`, `guard_band_count=32`, `guard_band_fraction=0.05`. |
| Leakage policy | Near-duplicate tolerance in normalized primitive-plus-derived axis space. | Approved 2026-08-30: `near_duplicate_tolerance=0.015` under `geometry-lineage-morph-seed-chebyshev-v1`. The DOE records descriptors, and closure replays them before accepting the role allocation. The private DOE-plan hash binds both policy and descriptors. |
| Reserve distribution | Ordered reserve policy per replaceable role and exhaustion behavior. | Approved 2026-08-30: `PILOT=4`, `TRAIN=20`, `DEVELOPMENT=8`, `HOLDOUT=0`, `LEGACY=0` reserves, ranked by the committed reserve order. Version one permits reserves only for `PILOT`, `TRAIN`, and `DEVELOPMENT`; `HOLDOUT` is never replaced. |
| Public naming | Stable public `manifest_id` and replacement `rule_id`. | Approved 2026-08-30: manifest `dogbone-v1-public-cases`; replacement rule `role-local-design-order-v1`. Guard and DOE stream algorithm IDs remain fixed by code. |
| Private reveal handling | Local ignored path and access controls for salt plus exact holdout coordinates. | Approved 2026-08-30: `.piginot-private/dogbone-v1/reveal-v1.json`, created before public commitment, readable only by the ceremony operator, with a separately access-controlled backup. The reveal must not enter Git, gate artifacts, tests, or training inputs. |

The recorded decisions now fix the reference mesh, geometry primitive and derived
vocabulary, complete executable envelope, load-coupled difficult regions, solver
recipe, and population cost basis. Numeric DOE or budget values chosen before the
family decision would have been unsupported assumptions and must not be substituted
for these ratified values.

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
solvability proof, a portable performance benchmark, or approval of the fixed load
point. Population projections must identify whether they use average
or conservative per-case timing and whether result payloads are stored as canonical
JSON, compressed, or converted to a later verified dataset representation.

For the ratified 408-active plus 32-reserve population, the fixed-load pilot projects
2.57 h from the median case, 2.66 h from the mean, and 3.13 h from the observed p95
case on that host. Mean payload size projects to 1.911 GB decimal (1.780 GiB) of
uncompressed canonical JSON. The 512 validation points are membership probes, not
FEM solves, and add no solve or result-storage cost.

The fixed `1 mm` load row is rejected for production. Across the approved geometry
range `L_total in [40, 70] mm`, it induces half-model nominal strain from 5.00% down
to 2.86%, monotonically entangling `length_to_gauge_ratio` with load magnitude.

## Strain-Controlled Load Pilot

The replacement load primitive under review is `nominal_axial_strain`, with the
right-grip displacement computed per case as `u = nominal_axial_strain * L_total / 2`.
This preserves a constant nominal axial strain across the family and removes the
fixed-displacement confound with `L_total`.

The public strain qualification used the same 12 sentinel geometries, approved
material point, P2 displacement basis, solver tolerance `1e-10`, and S7 backend. It
evaluated two endpoints, 10% and 20% nominal axial strain. The initial 16-step screen
recorded 21 successes in 24 attempts:

| Case | Nominal strain | Displacement | Failure |
| --- | ---: | ---: | --- |
| `wide_gauge_long` | 10% | 3.5 mm | `CLOSURE_FAILURE` |
| `wide_gauge_long` | 20% | 7.0 mm | `NEGATIVE_JACOBIAN` |
| `arc_margin` | 20% | 7.0 mm | `CLOSURE_FAILURE` |

The transparent 32-step follow-up recorded 23/24 successes; only
`wide_gauge_long` at 20% remained a closure failure. A targeted 64-step diagnostic
then succeeded for that case, so the final qualification reran the complete fixed
population uniformly at 64 steps rather than mixing continuation recipes. All 24/24
final attempts succeeded. Maximum observed Newton iterations were 7, maximum final
residual was `8.874514776412887e-11`, and the minimum recorded deformation Jacobian
was `0.9909092600273233`.

All 12 geometry pairs showed right-reaction scaling 5.46% to 5.70% below the linear
2:1 reference. That establishes measurable finite-strain response on these finite
sentinels; it is not a domain-wide proof. On the recorded host, the 64-step recipe
projects 13.32 h at the median, 16.31 h at p95, and 19.17 h at the observed maximum
for 440 solves. Mean payload size projects to 1.915 GB decimal (1.784 GiB).

This evidence supports proposing the exact 10%–20% envelope with 64 uniform steps.
It does not ratify the proposal, and the earlier failures remain part of the evidence
rather than being overwritten.

The proposed first role budget keeps the original DogBone training scale of 256
training cases, adds an exact 64-case sealed holdout, reserves 64 cases for visible
development decisions, and keeps 24 production `PILOT` cases for pipeline shakeout
after commitment but before using the training labels. Thirty-two reserves give
approximately 7.8% replacement capacity over active production cases without ever
replacing `HOLDOUT`.

## Proposed DOE Policy and Public Dry Run

The proposed axis table is finite and exact. Fixed material coordinates retain one
visible stratum but weight `0.01`, so zero-width axes do not materially dilute the
weighted distance. The variable strain coordinate has distance weight `0.05`; this is
not a claim that load is unimportant. Its marginal and joint coverage are instead
enforced directly by four strata, both endpoint bands, a strain-by-concentration
interaction, and a high-strain difficult-region quota, while the already-ratified
`0.10` fill threshold remains focused on morphology coverage.

`arc_feasibility_ratio` uses two strata because the ratified upper bound of 20 is
reachable only near a measure-small primitive corner; an equal four-stratum policy
left its highest cell empty even in a 2,400-candidate public stream. Its load-bearing
lower edge is covered by explicit difficult-region quotas instead of a misleading
symmetric boundary quota.

| Axis | Source | Weight | Strata | Minimum/stratum | Boundary width | Minimum/boundary |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `L_total` | primitive | 1.00 | 4 | 8 | 0.10 | 8 |
| `W_grip` | primitive | 1.00 | 4 | 8 | 0.10 | 8 |
| `W_gauge` | primitive | 1.00 | 4 | 8 | 0.10 | 8 |
| `R_fillet` | primitive | 1.00 | 4 | 8 | 0.10 | 8 |
| `elastic_modulus` | primitive, fixed | 0.01 | 1 | 1 | 0.10 | 0 |
| `poisson_ratio` | primitive, fixed | 0.01 | 1 | 1 | 0.10 | 0 |
| `nominal_axial_strain` | primitive, variable | 0.05 | 4 | 8 | 0.10 | 8 |
| `arc_feasibility_ratio` | derived | 1.50 | 2 | 8 | 0.10 | 0 |
| `fillet_to_gauge_ratio` | derived | 1.50 | 4 | 8 | 0.15 | 8 |
| `grip_to_gauge_ratio` | derived | 2.00 | 4 | 8 | 0.15 | 8 |
| `length_to_gauge_ratio` | derived | 1.25 | 4 | 8 | 0.15 | 8 |

Proposed interactions require two samples per cell for
`W_grip x W_gauge`, `W_gauge x R_fillet`,
`L_total x grip_to_gauge_ratio`, and
`nominal_axial_strain x grip_to_gauge_ratio`. Proposed difficult regions are:

- `high-concentration`: normalized `grip_to_gauge_ratio` in `[0.75, 1]` and
  `arc_feasibility_ratio` in `[0, 0.10]`, minimum 32;
- `near-arc-margin`: normalized `arc_feasibility_ratio` in `[0, 0.025]`, minimum 16;
- `short-high-concentration`: normalized `L_total` in `[0, 0.25]`,
  `grip_to_gauge_ratio` in `[0.75, 1]`, and `arc_feasibility_ratio` in
  `[0, 0.10]`, minimum 8;
- `high-strain-high-concentration`: normalized `nominal_axial_strain` and
  `grip_to_gauge_ratio` in `[0.75, 1]`, with `arc_feasibility_ratio` in
  `[0, 0.10]`, minimum 8.

The proposed acceptance thresholds are maximum empirical fill distance `0.10`,
minimum stratum count 8, all marginal/boundary/interaction/difficult-region coverage
fractions equal to 1.0, and no unresolved requirement identifier.

Public dry run `public-dogbone-s9-strain-proposal-dry-run-v1` deliberately reused the
prior public seed `public-dogbone-s9-proposal-dry-run-v2` for a paired policy
comparison. It did not create campaign secrets, roles, a holdout, or a commitment.
Of 4,096 candidate proposals, 3,829 passed membership; of 4,096 independent
validation proposals, 3,821 passed. The selector produced 408 active cases, 32
reserves, 512 validation points, and 32 negative guards with no unresolved quota.
Measured fill distance was `0.09238521285946635`, minimum stratum count was 8, all
four coverage fractions were 1.0, and all 440 active-plus-reserve points had distinct
leakage groups at tolerance `0.015`. The proposal envelope, design spec, and design
hashes are respectively `be25dabb…690`, `0e879a43…179`, and `ac6c3f82…2f27`.
This dry run shows that the load-aware proposal is executable; it is not the
production DOE, and its public seed must never be reused for closure.

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

## Remaining Ceremony Preconditions

Scientific approval is complete. The following artifacts and gates do not yet exist:

- no production commitment;
- no `CP-ENVELOPE`;
- no passing G9 artifact;
- no S10 start.

## Sign-Off Record

Complete scientific decision record; this is not a claim that the production
ceremony or G9 has passed.

- Decision owner: Betim Bahtiri
- Recorded UTC timestamp: 2026-08-29T13:00:53Z
- Approved production family: `dogbone-quarter-no-hole-v1`
- Approved compiled-family hash:
  `a8143a75e62c614d40f52049cfc88c1cda74d38c9bd488cc3bed4c5678d973a4`
- Approved reference mesh:
  `3bfc1043f5aa9e989469adacb0ee1d0b483825ffabcd5dfac4bf9caca580d82c`
- Approved validity-policy hash:
  `b3c2df59eed939f1aa14a69d05a6fd2dc57be571ffba33c6355982713cc6931d`
- Additional owner ratification recorded 2026-08-30: evidence basis, material
  point `E = 760 MPa`, `nu = 0.23`, role budget, candidate stream sizes, DOE
  quota policy except the load-axis row, leakage tolerance, reserve distribution,
  public identifiers, and private reveal handling.
- Explicitly rejected 2026-08-30: fixed `1 mm` prescribed displacement.
- Owner ratification recorded 2026-08-31: `nominal_axial_strain in [0.10, 0.20]`, load policy
  `nominal-axial-strain-times-half-length-v1`, the 64-step solver recipe, load-axis
  weight `0.05`, its interaction and difficult-region quotas, and the qualified-load
  evidence hash `2f6cfcb9…02a0`.

The production ceremony must now materialize the resulting hashes of the approved
pilot evidence, executable envelope, private DOE plan including leakage policy and
descriptors, role registry, reserve policy, and public case manifest without changing
any scientific or population value above.
