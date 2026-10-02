# PI-GINOT

PI-GINOT currently includes immutable unit-safe contracts, a local content-addressed
store with explicit retention, and a verified compressible Neo-Hookean physics kernel
with differentiable plane-stress closure. It also includes strict tagged 2D triangle
geometry ingestion for Gmsh v4.1 meshes, deterministic RBF and FFD reference-mesh
morphing, executable family-membership predicates, an invalid-instance ledger, a
sparse nonlinear primary FEM solver for 2D finite-strain plane stress with P1/P2
displacement, independent FEM verification, preregistered DOE, evidence-role,
leakage, replacement, and holdout-commitment contracts, and
[fail-closed S10 dataset contracts](adr/0011-s10-dataset-trust-boundaries.md) for
public request plans, snapshots, extraction, collocation, projection and
normalization, and the
[fixed S11 PI-GINOT operator architecture](adr/0012-s11-operator-architecture.md),
followed by the
[S12 nondimensional objective registry](adr/0013-s12-objective-registry.md).
The first public production DogBone dataset contains 344 successful FEM cases and
88,064 deterministic collocation points. All nine structural G11 checks pass. The
mandatory pre-S12 architecture contingency trained four diagnostic candidates and
then fell back to the S11 baseline under its preregistered guardrails; DEVELOPMENT
remained unopened. That contingency is now discharged, S12 is complete, and all
nine G12 objective and gradient-verification checks pass. The focused S12 suite
records 89 passed with 92.99% branch coverage; the full suite records 1,928 passed
and 2 skipped. S12 opened neither DEVELOPMENT nor HOLDOUT. S13 trained on TRAIN
evidence only: the S11 baseline could not be fine-tuned into an admissible
candidate, and the
[energy-form operator of ADR 0018](adr/0018-s13c-energy-form-operator.md) replaced
it. Its single registered trial passes every frozen S13 guardrail, the amended G11
and G13 gates pass, and `CP-FREEZE` freezes it for S14. S14 measured it on
DEVELOPMENT and once on HOLDOUT under a signed protocol: the HOLDOUT error budget
passes every slice, two DEVELOPMENT peak slices were accepted by the decision owner
([ADR 0021](adr/0021-s14-development-budget-acceptance.md)), and G14 waits only for
the reducer families of `IMPLEMENTATION-v1.0-OSS.md`.

Read the [limitations](limitations.md) before relying on future numerical results.
