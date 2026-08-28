# S9 Production Campaign Sign-Off

- Status: Family direction approved; exact family and remaining campaign choices proposed
- Step: S9
- Blocks: G9 and S10
- Branch: `codex/s9-doe-preregistration`
- Contract baseline: `CP-ENVELOPE` schema 1.1 and ADR 0009

This page records the scientific campaign choices that must be approved before a
production `CP-ENVELOPE` can be created. It is deliberately not a commitment,
not a private reveal, and not a gate artifact. No DOE seed, allocation seed,
holdout salt, holdout coordinate, or production role label belongs in this file.

## Current Verdict

G9 remains open. The user approved DogBone as the first production-family direction,
and ADR 0010 now proposes its exact executable family. The repository still lacks
human approval of that exact preimage, its complete material/load envelope, DOE
budgets and quotas, leakage tolerance, reserve distribution, public identifiers, and
private reveal-handling decision. S10 must not begin until those choices are approved
and committed, then used to create the first production `CP-ENVELOPE`.

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
- `docs/adr/0010-s9-first-production-family.md` proposes the exact pinned DogBone
  family. `artifacts/campaigns/dogbone-v1/family-validation.json` records a disjoint
  public sweep with 500 accepted and zero morph-validation failures.

## Approval Items

Each row needs an explicit owner decision before the production ceremony.

| Item | Required Decision | Current Draft Position |
| --- | --- | --- |
| Production family | Exact compiled family, reference mesh identity, morph map recipe, validity policy hash, and S6 membership predicate. | Direction approved and exact proposal ready. Recipe `dogbone-quarter-no-hole-v1`; compiled hash `f5bac941…b35d`; mesh `3bfc1043…d82c`; analytic circular-transition map; policy `b3c2df59…931d`. Exact preimage still needs owner ratification. |
| Pilot evidence | Non-empty ordered set of content-addressed `EvidenceReference` records with role `PILOT` or `LEGACY`. | Proposed: DogBone family validation `c83e09c3…d75a` as `PILOT`, plus S1 gate `cd0b01e8…9470` as `LEGACY`. A FEM cost/solvability pilot is still required before approving population counts. |
| Executable envelope | Primitive bounds, derived bounds, material bounds, load bounds, and refusal policy for out-of-envelope inputs. | Geometry proposal ready: original four geometry ranges plus `arc_feasibility_ratio >= 1.35`. Material/load bounds are not approved, so no DogBone executable-envelope preimage or hash exists. |
| Derived axes | Non-empty ordered list of derived stratification axes to bind into `CP-ENVELOPE`. | Proposed: `grip_to_gauge_ratio`, `fillet_to_gauge_ratio`, `length_to_gauge_ratio`, and `arc_feasibility_ratio`, all compiled through S2 and evaluated by S6. |
| DOE axis policy | Axis weights, stratum counts, minimum per stratum, boundary widths, boundary quotas, interactions, and difficult-region quotas. | Not approved. The high-concentration region must have an explicit quota rather than relying on maximin fill alone. |
| Role budget | Positive `PILOT`, `TRAIN`, `DEVELOPMENT`, exact `HOLDOUT=64`, and explicit non-negative `LEGACY` and `RESERVE` counts. | Not approved. The DOE `sample_count` must equal `pilot + train + development + holdout`; `reserve_count` must equal `reserve`. |
| Candidate streams | `candidate_count`, `validation_count`, `validation_candidate_count`, `guard_band_count`, and `guard_band_fraction`. | Not approved. Candidate counts must be large enough to expose S6 family-membership and derived-bound rejections. |
| Leakage policy | Near-duplicate tolerance in normalized primitive-plus-derived axis space. | Not approved. The algorithm is fixed as `geometry-lineage-morph-seed-chebyshev-v1`; the DOE records descriptors, and closure replays them before accepting the role allocation. The private DOE-plan hash binds both policy and descriptors. |
| Reserve distribution | Ordered reserve policy per replaceable role and exhaustion behavior. | Not approved. Version one permits reserves only for `PILOT`, `TRAIN`, and `DEVELOPMENT`; `HOLDOUT` is never replaced. |
| Public naming | Stable public `manifest_id` and replacement `rule_id`. | Not approved. `s9-public-cases` is only the code default; guard and DOE stream algorithm IDs are fixed by code. |
| Private reveal handling | Local ignored path and access controls for salt plus exact holdout coordinates. | Not approved. The reveal file must not enter Git, gate artifacts, tests, or training inputs. |

The production family is the first decision. It fixes the reference mesh, primitive
and derived vocabulary, executable envelope, meaningful difficult regions, and the
cost basis for every population count. Numeric DOE or budget values approved before
that choice would be unsupported assumptions and must not be used.

The approved first-campaign direction has now been resolved to the exact proposal in
ADR 0010. The proposal retains the pinned reference implementation's four primitive
ranges and makes its fixed-connectivity exclusion explicit rather than hiding failed
geometries. Ratification of ADR 0010 is still required before its hash enters a
production commitment.

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

- no owner-ratified exact compiled production-family record;
- no owner-ratified complete material/load executable envelope;
- no role budget;
- no DOE quota table;
- no leakage tolerance;
- no reserve distribution;
- no production commitment;
- no `CP-ENVELOPE`;
- no S10 start.

## Sign-Off Record

Approval requires replacing this section with the actual decision record, including
the decision owner, UTC timestamp, approved public identifiers, exact scientific and
population values above, and the resulting hashes of the family, pilot evidence,
envelope, private DOE plan (including leakage policy and descriptors), role registry,
reserve policy, and public case manifest.
