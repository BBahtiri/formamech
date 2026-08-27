# S9 Production Campaign Sign-Off

- Status: Draft for human approval
- Step: S9
- Blocks: G9 and S10
- Drafted against branch: `codex/s9-doe-preregistration`
- Drafted against commit: `cea6fff`

This page records the scientific campaign choices that must be approved before a
production `CP-ENVELOPE` can be created. It is deliberately not a commitment,
not a private reveal, and not a gate artifact. No DOE seed, allocation seed,
holdout salt, holdout coordinate, or production role label belongs in this file.

## Current Verdict

G9 remains open. The S9 contracts, role controls, replacement policy, secret
factory, and checkpoint schema are implemented, but the repository still lacks a
human-approved production campaign. S10 must not begin until the choices below
are approved and committed, then used to create the first production
`CP-ENVELOPE`.

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

## Approval Items

Each row needs an explicit owner decision before the production ceremony.

| Item | Required Decision | Current Draft Position |
| --- | --- | --- |
| Campaign identifier | Stable human-readable ID for the first production campaign. | `s9-production-v1` is only a placeholder. |
| Production family | Exact compiled family, reference mesh identity, morph map recipe, validity policy hash, and S6 membership predicate. | Not approved. The S1 spike families are `role=LEGACY` and cannot be reused as release evidence. |
| Executable envelope | Primitive bounds, derived bounds, material bounds, load bounds, and refusal policy for out-of-envelope inputs. | Not approved. The envelope must be declared in derived-ratio space. |
| Derived axes | Non-empty ordered list of derived stratification axes to bind into `CP-ENVELOPE`. | Must include the governing concentration or net-section ratio for any notched/tapered family. Other axes depend on the approved family. |
| DOE axis policy | Axis weights, stratum counts, minimum per stratum, boundary widths, boundary quotas, interactions, and difficult-region quotas. | Not approved. The high-concentration region must have an explicit quota rather than relying on maximin fill alone. |
| Role budget | Positive `PILOT`, `TRAIN`, `DEVELOPMENT`, exact `HOLDOUT=64`, `LEGACY`, and `RESERVE` counts. | Not approved. The DOE `sample_count` must equal `pilot + train + development + holdout`; `reserve_count` must equal `reserve`. |
| Candidate streams | `candidate_count`, `validation_count`, `validation_candidate_count`, `guard_band_count`, and `guard_band_fraction`. | Not approved. Candidate counts must be large enough to expose S6 family-membership and derived-bound rejections. |
| Leakage tolerance | Exact rule for transitive `leakage_group_id` creation: geometry lineage, morph seed, near-duplicate tolerance, and any family-specific equivalence. | Not approved. This must be fixed before role allocation. |
| Reserve distribution | Ordered reserve policy per replaceable role and exhaustion behavior. | Not approved. Version one permits reserves only for `PILOT`, `TRAIN`, and `DEVELOPMENT`; `HOLDOUT` is never replaced. |
| Private reveal handling | Local ignored path and access controls for salt plus exact holdout coordinates. | Not approved. The reveal file must not enter Git, gate artifacts, tests, or training inputs. |

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
3. Generate the coupled DOE through the approved S6 family-membership predicate.
4. Allocate roles with `campaign_secrets.allocation_seed`.
5. Build the replacement policy from preregistered reserves.
6. Persist the private reveal outside Git before publication.
7. Call `build_preregistration()` once with the approved campaign inputs and the
   same `CampaignSecrets` bundle.
8. Commit the public `CP-ENVELOPE`, public manifest, role ledger, replacement
   policy, and gate artifact in the order required by ADR 0009.

## Non-Approvals

The following are intentionally not approved by this draft:

- no production family;
- no executable envelope;
- no role budget;
- no DOE quota table;
- no leakage tolerance;
- no reserve distribution;
- no production commitment;
- no `CP-ENVELOPE`;
- no S10 start.

## Sign-Off Record

Approval requires replacing this section with the actual decision record, including
the decision owner, UTC timestamp, approved campaign identifier, and the committed
hashes of the family, envelope, DOE plan, role budget, leakage policy, and reserve
policy.
