# ADR 0011: S10 dataset trust boundaries

- Status: Accepted
- Date: 2026-09-04
- Step: S10

## Context

S10 is the first step that turns the preregistered public campaign into dataset
artifacts. It inherits the S9 commitment, public manifest, role ledger and approved
DogBone production family, but it must not inherit private material or fabricate
training evidence before FEM results exist.

Two earlier decisions are load-bearing here. ADR 0001 requires samplers not to
down-weight QoI regions simply because those regions have small physical measure.
ADR 0006 establishes exact discrete mesh validity for accepted morphs, but only a
finite sampled continuous-injectivity check for generic RBF and FFD maps. S10 must
therefore avoid any generic cross-mesh claim that depends on an unproved global
inverse map.

## Decision

S10 dataset construction is fail-closed around explicit source identities.

Public FEM requests are materialized only for active non-HOLDOUT public S9 cases.
The DogBone request materializer pins the ratified campaign hash, manifest hash,
compiled-family hash, reference-mesh identity, validity-policy hash, material
point, 10%--20% nominal-strain load policy, P2 element order, 64 load steps and
solver tolerances. A `PublicFemRequestPlan` then binds each case to the exact
request recipe hash, complete request content hash, mesh identity, mesh content
hash and one execution-profile hash. Dataset snapshots must match the CP-ENVELOPE,
public manifest and request plan before accepting any FEM attempt.

Successful FEM attempts are accepted only when the canonical `FemResult` bytes are
resolvable from the content-addressed `ObjectStore` and match the recorded result
and store hashes. Failed attempts carry only a sanitized failure code and no result
payload. Extraction requires the snapshot, request plan, FEM attempt and store
together, and returns field artifacts with units plus the snapshot, plan, store and
result hashes.

Collocation uses deterministic public replay seeds derived from the committed
snapshot, case, role, mesh content, semantic boundary/load specifications, stream
index and quota policy. Callers cannot supply raw seed bytes. Persisted collocation
batches must replay byte-identically against the exact FEM request already bound
into the snapshot; generic replay alone is not a persistence trust decision.
QoI quotas are semantic and measure-independent so small notches, roots or other
declared QoI regions cannot be starved by area- or length-weighted background
sampling.

Projection artifacts carry sample-scoped transfer-error envelopes and must be
recomputed from their quadrature source field before crossing a trust boundary.
Cross-mesh stress comparison is admitted only as physical-domain integration over
stored quadrature samples. S10 does not approve raw node pairing and does not claim
a verified generic pullback through S6 RBF/FFD maps.

Normalization statistics may be fitted from active TRAIN records only, and every
record must resolve to the successful FEM result in one exact TRAIN role snapshot.
The statistics retain that snapshot's content hash. HOLDOUT is outside the public
S10 normalization contract, and non-active reserves are not normalization sources.

## Consequences

The S10 implementation can construct and verify the dataset trust boundary before
the expensive production FEM batch is run. That prevents later code from accepting
self-consistent but unplanned requests, path-only result claims, forged collocation
points, forged projections, raw-node cross-mesh comparisons, or normalization
statistics fitted on DEVELOPMENT, PILOT, HOLDOUT or reserve material.

The cost is stricter ceremony: a dataset artifact is not trusted merely because its
local dataclass is shape-valid. The source-aware constructors and verifiers are the
trusted persistence boundary.

G10 remains FAIL until the production public FEM result population, snapshots,
derived artifacts, replay evidence and normalization statistics are actually
materialized. The S10 code tests are implementation evidence, not production
dataset evidence, and S11 must not begin from fixture-only checks.

## Verification

S10 implementation evidence is local and fixture-scoped:

- request-plan tests reject HOLDOUT cases, reserve cases, altered request content,
  altered meshes and execution-profile divergence;
- snapshot tests require CP/manifest/request-plan ancestry and materialized
  ObjectStore result bytes, while sanitizing failure messages;
- extraction tests require snapshot, request plan, attempt and store together;
- collocation tests verify public seed derivation, byte-identical replay, mesh
  content binding, request-content binding, semantic QoI quotas, maximum population
  checks and exact-request-bound persistence;
- projection tests recompute projected fields and physical-domain integral
  comparisons from their sources;
- normalization tests bind statistics to an exact TRAIN snapshot and reject records
  whose FEM source hash, role, status, case, or leakage group does not resolve there;
- the scoped S10 package coverage threshold is 85%.

Implementation commit `59f0defacea8311a8a1c9710ff3f17a1171d9a79` passed 170
scoped tests at 85.42% branch coverage and the full local suite at 1,536 passed with
2 skipped. This is local implementation evidence; no cross-platform CI or production
FEM population is claimed.

The production G10 gate must separately record non-fixture evidence for every
active public production case before the verdict can change from FAIL to PASS.
