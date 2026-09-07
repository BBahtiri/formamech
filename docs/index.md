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
normalization. The first public production DogBone dataset is now materialized for
344 successful FEM cases and 88,064 deterministic collocation points. All seven G10
checks pass, so S11 may begin from that exact committed dataset report.

Read the [limitations](limitations.md) before relying on future numerical results.
