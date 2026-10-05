# Limitations

PI-GINOT performs numerical verification against finite element method (FEM)
solutions. That is not experimental validation, and it does not show that predictions
agree with measurements from physical specimens.

Verification for a bundle establishes nothing about geometries outside that bundle's
declared input envelope. Inputs outside the envelope must not be treated as covered by
the bundle's results. Finite verification sampling is evidence about the evaluated
cases, not a mathematical proof of a uniform error bound over every admissible input.

The verified S4 material envelope is `1 <= E <= 10000 MPa` and
`0 <= nu <= 0.49`; values outside it are unsupported. The S7 primary solver currently
supports only that compressible Neo-Hookean law in 2D finite-strain plane stress on
linear triangular reference meshes. It offers P1 and P2 displacement, sparse Newton
solves with load continuation, component-wise essential conditions, axis-aligned
implicit symmetry, dead nominal traction, and registered reference-volume body
forces. The plane-stress model uses a fixed reference thickness of 1 mm. It does not
yet support dynamics, contact, follower loads, mixed or nearly-incompressible
formulations, multiple materials, remeshing, or alternative constitutive laws.

S7 returns full 3D deformation-gradient and first-Piola tensors at quadrature points
and an L2-recovered first-Piola field. Boundary maximum and pointwise peak-stress QoIs
are refused at re-entrant or mixed-BC junctions and require P2 or a declared graded
mesh. A recovered peak remains mesh- and recovery-dependent; it is not a certified
physical maximum.
Reported external work covers assembled dead loads at the final state and excludes
work reconstructed along prescribed-displacement paths.

S8 adds manufactured-solution, patch, convergence, error-estimation, physical
balance, and seeded-defect checks for the primary FEM implementation. Those checks
verify the stated S8 population only. They are not experimental validation, not a
certificate for arbitrary geometries or load histories, and not evidence for a second
concrete FEM implementation. The accepted thresholds and passing G8 evidence apply
only to the exact recorded branch head and test population.

S5 accepts explicitly unitized, planar, linear triangle meshes with named line
boundary groups and positive named region groups. It rejects healing, defeaturing,
remeshing, inferred units, quads, high-order elements, mixed area elements, 3D
meshes, unclassified facets, duplicate nodes or cells, inverted elements,
non-manifold edges, and query points outside the mesh when containment is required.
S6 deforms an existing S5 reference mesh with interpolatory Gaussian RBF maps,
tensor-product Bernstein FFD maps, or the exact circular-transition profile map used
by the proposed first production family. It preserves connectivity and accepts a
morph only when the declared topology and tags remain unchanged, feature-size and
mesh-quality bounds pass, and `det(grad(Phi))` is strictly above the configured margin
at every point of a fixed seven-point, degree-5 triangle rule. This finite check is an
executable acceptance predicate; it is not a mathematical proof of global injectivity
or of the Jacobian sign between quadrature points. A later cross-mesh comparison must
not call an RBF/FFD pullback verified solely because an S6 report passed. The circular
transition map separately exposes an analytic inverse and positive piecewise
determinant; that family-specific property does not strengthen the generic RBF/FFD
claim. S6 does not heal or remesh failed instances.

Family membership enforces exact named canonical quantities, inclusive primitive and
derived bounds, the reference mesh identity, and a hashed validity policy. S9 samples
that predicate only through a typed executable envelope with explicit non-empty
material and load bounds. Every primitive coordinate, including fixed coordinates,
is bound into the DOE; fixed axes use one stratum. Campaign closure verifies the
compiled-family and membership identities, then replays accepted cases and recorded
guard rejections against the same envelope. It also regenerates the complete design
from the private DOE seed and requires canonical byte identity before returning a
checkpoint. It reports a vector of finite coverage diagnostics. Those diagnostics
are evidence about the sampled population, not a proof that every admissible point
is covered. Version-one targeted guard points cross primitive bounds; derived-bound
failures remain visible in the rejection ledger but are not claimed as dense guard
strata.

S9 provides contracts for immutable case roles and transitive leakage groups,
deterministic non-holdout reserve replacement, and an exact-64 BLAKE2b-256 holdout
commitment. The production ceremony begins with three 32-byte OS-CSPRNG draws for the
DOE seed, allocation seed, and holdout salt; closure accepts only that generated
bundle and CP-ENVELOPE schema 1.1 binds its generator identity. This code path guards
against accidental caller-chosen secrets, but cannot attest that hostile code in the
same Python process was unmodified. The required near-duplicate tolerance and the
pre-role lineage, morph-seed, and normalized-coordinate descriptors are replayed at
design construction and campaign closure, then committed through the private DOE-plan
hash; they are finite campaign metadata, not proof that no unrecorded correlation
exists. The production `CP-ENVELOPE` and exact-64 commitment are now public, while
the reveal remains ignored and operator-only with a separately access-controlled
backup. A DogBone source direction is recorded and ADR 0010 ratifies its exact
production-family preimage.
The first campaign's family, material point, population budgets, DOE quotas, leakage
tolerance, reserve distribution, public identifiers, private reveal policy, and exact
10%–20% strain-controlled load envelope now have owner ratification. Fixed `1 mm`
displacement remains rejected because it makes nominal strain vary monotonically with
specimen length. The ratified strain-controlled load primitive removes that confound.
Its 16-step screen
recorded 21/24 successes, its 32-step follow-up recorded 23/24, and its final uniform
64-step qualification recorded 24/24. Those finite sentinels support the ratified
10%–20% envelope with the 64-step recipe; they are not a domain-wide solvability
proof. G9 is PASS, and its live order record deliberately witnesses zero downstream
artifacts at the pre-training boundary. S10 implements fail-closed public request
plans, role snapshots, store-verified FEM extraction, deterministic replayable
collocation, sample-scoped projection error artifacts, physical-domain cross-mesh
stress-integral comparison, and TRAIN-only normalization contracts. The first
production dataset is now materialized: 344 successful FEM cases, 88,064
collocation points, 344 projected fields, 343 physical-integral comparisons and
256 TRAIN-normalized samples. Normalization statistics retain the exact TRAIN
snapshot identity and require every input record to resolve to its successful FEM
result. G10 is PASS and S11 may begin from the exact committed report, but no S11
training or accuracy evidence exists yet. Every later artifact batch must prove
CP-ENVELOPE ancestry and live file ordering. The absence of family-specific words
and branches in `src/`
addresses only `G17.fixture-names`; `G17.non-dogbone` remains `NOT_RUN` because the
circular-transition provider currently has one campaign consumer and no
distinct-topology example has reached a verified bundle through the same API.

The public DogBone FEM solvability pilot covers 12 preselected geometry sentinels at
one fixed material/load point. Its 12 successful solves support local campaign-cost
sizing only; they do not establish a domain-wide failure rate, portable runtime, or
approval of the fixed load value. The recorded payload sizes are for uncompressed
canonical FEM results, not a final dataset storage format. The public strain-load
qualification covers the same 12 sentinels at 10% and 20% nominal axial strain. The
final 64-step population recorded 24 successful solves, minimum deformation Jacobian
`0.9909092600273233`, and reaction-scaling departure from the linear reference of
5.46%–5.70%. Its 440-case host-local cost projection is 13.32 h at the median,
16.31 h at p95, and 1.915 GB decimal (1.784 GiB) of canonical result bytes. None of
those measurements proves every envelope point will converge or transfers as a
portable runtime guarantee.

S10 projection errors are conservative only over stored quadrature samples that
participate in the projection. They are not continuum error bounds. S10 cross-mesh
comparison deliberately uses physical-domain integrals and does not treat a generic
RBF or FFD S6 morph as a verified inverse pullback.

S11 fixes a contract-hardened PI-GINOT architecture. Its G11 evidence uses the
ratified DogBone reference mesh, one normalized parameter vector, 100 random
boundary permutations and one 320-versus-480 resampling comparison. Those finite
checks establish the named structural invariants, not full-envelope robustness.
The later four-candidate contingency is also limited: it used 192 TRAIN-fit and 64
TRAIN-screen cases, not HOLDOUT, and its registered fallback kept DEVELOPMENT
unopened. Candidate C materially improved all stress metrics but exceeded the 2%
displacement-regression guardrail, so the baseline remains authoritative for S12.
The independent `P_predicted` head is the sole served-stress path; its constitutive,
angular and traction-free terms are now registered by S12, subject to the finite
verification scope below. Hard distance lifting is unavailable without separate
full-range refinement evidence. The safetensors loader rejects pickle and validates
metadata, schema and exact byte layout before materialization, but it does not
authenticate learned tensor values.

[S12](adr/0013-s12-objective-registry.md) fixes the nondimensional objective
registry, and G12 is PASS on finite fixtures. Its perturbation checks use
channel-isolated analytic residual fixtures rather than one coupled physical state,
and its parameter reachability evidence is a single deterministic probe, not a
guarantee of useful sensitivity after training. Constant spatial fields in
derivative-based checks must remain graph-connected as `0 * X + c`; detached
constants are outside the verified contract. The weak/strong check establishes
same-field integration-by-parts consistency on one recovered P2
manufactured-solution (MMS) field with one interior bubble test function. It proves
neither equilibrium of the recovered stress nor `1e-12` FEM discretization error.
S12 opened neither DEVELOPMENT nor HOLDOUT, ran no production training, and created
no checkpoint.

[S13](adr/0018-s13c-energy-form-operator.md) freezes a different operator from the
S11 baseline described above. The frozen energy-form candidate serves the
constitutive stress `P_NH(I + grad u)` of its predicted displacement and has no
independent stress head; the `P_predicted` statement above applies to the S11
operator only. Its Dirichlet conditions are imposed by an exact affine lifting,
verified over the full primitive parameter box at three boundary refinements
(`artifacts/gates/S11-ENERGY.json`). It is conditioned on the five DogBone
parameters (`L_total`, `W_grip`, `W_gauge`, `R_fillet`, nominal axial strain) with
fixed `E` and `nu`, and does not encode an arbitrary boundary point cloud. Every
selection number is TRAIN-role evidence on 64 TRAIN-screen cases that were not
used for fitting; DEVELOPMENT and HOLDOUT were unopened at `CP-FREEZE`. Bitwise
determinism and resume are shown on one recorded Linux CPU profile, and a new
training run reproduces the pinned initial weights only on a CPU whose PyTorch
kernels dispatch AVX2 or AVX512.

S14 measured that candidate on 64 DEVELOPMENT and, once, on 64 HOLDOUT DogBone
cases under a signed protocol (ADRs 0019–0021). Its numbers carry these limits:

- The error budget is a slice statistic (p90, or the maximum for slices with
  fewer than ten cases), not a per-case guarantee. Two DEVELOPMENT cases exceed
  the 5% peak tolerance, and the two DEVELOPMENT slices they fail were
  accepted as passing by the decision owner after the result was seen
  (ADR 0021). HOLDOUT passes every slice under the unchanged rule.
- The FEM reference terms are conservative estimates from the S8 recovery and
  Richardson estimators on one coarser solve, applied to vertex values. The
  peak term is not a rigorous pointwise bound. The reference itself is the
  recovered P2 stress of the S10 mesh, not an exact solution.
- The confidence indicator ranks errors only moderately (pooled Spearman 0.641
  on DEVELOPMENT, 0.620 on HOLDOUT). Its 90% calibration curve, fitted on
  DEVELOPMENT, covers 86.8% of HOLDOUT vertices.
- Latency is measured on one Linux CPU thread at the 969 mesh vertices of a
  case, against the S10 P2 solve with 64 load steps; other query sets,
  hardware or solver settings change the ratio.
- G14 passes with the claim level `HOLDOUT_VERIFIED`. Its reducer runs over
  twelve verification families that the project defined itself (ADR 0022),
  because `IMPLEMENTATION-v1.0-OSS.md` is not available; they are the upstream
  gate artifacts, not an independent audit.

S15 serves that candidate as a bundle ([ADR 0023](adr/0023-s15-bundle-and-cli.md),
[guide](bundle.md)). Its limits:

- A bundle proves that its bytes are the ones its manifest lists and that its
  parts agree. It does not prove that it was built from verified inputs; the
  DogBone build checks the `CP-FREEZE` weights hash and records the checkpoint
  identities, and bundle predictions inherit the S14 evidence because they
  reproduce the S14 serving path bit for bit on TRAIN cases.
- The bundle serves one family, one material and the 10-20% strain-controlled
  load of its envelope, on the quarter model. Every other request is refused;
  so is a prediction whose deformation folds (`det F <= 1e-12`) anywhere it is
  evaluated.
- The served confidence bound inherits the S14 indicator's limits: moderate
  ranking and 86.8% HOLDOUT coverage of a nominal 90% quantile. Indicators above
  the calibrated range are flagged, not extrapolated.
- The clean install was checked on Linux x86_64 with Python 3.11 only.

The S10 production run recorded zero natural FEM failures. Failed-case visibility is
therefore exercised by one explicitly non-production canary through the same role
snapshot path; that proves the path represents the controlled failure but is not an
observed production failure. The execution-profile file was reconstructed after the
FEM run from its unique full-hash preimage and persisted before S10 closeout. Its hash
matches the request plan, ledger and report, but the file was not present during the
original solves.

The public commitment proves only that a later reveal matches the committed
coordinates. It cannot prove that the local holder never saw the
ignored private reveal. Git also does not preserve historical file mtimes, so live
ordering must be captured and committed with each artifact batch, while Git first-add
ancestry is verified separately. The project implements deterministic training,
one frozen, TRAIN-selected candidate, its S14 verification and its S15 bundle; it
does not implement certification.
