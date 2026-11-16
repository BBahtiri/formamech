# FormaMech

**Start here:** the [user guide](user-guide.md) shows how to predict with the
shipped bundles, run the end-to-end example, and build a model for your own
geometry. The [limitations](limitations.md) say what the numbers do and do not
mean.

## Work with your own CAD or material

The current development version imports supported 2D STEP/DXF profiles,
meshes them, and trains one model per fixed geometry over declared material
and displacement-load ranges. Follow the [own-CAD workflow](campaigns/own-cad-workflow.md)
to inspect and label boundaries, prepare a training plan, compare against FEM
and save a reusable candidate bundle. The [meshing guide](campaigns/own-cad-meshing.md)
explains local refinement and reference-convergence diagnostics.

The installed [campaign compiler](campaigns/s28-compiler-guide.md) supports
explicit execution and resume. Try the [custom-material example](campaigns/s29-custom-material.md)
for complete inputs. New candidates retain `NOT_VERIFIED`; keep their inputs,
weights and comparison reports locally under `.formamech-runs/` or outside the
checkout, and assess suitability for their own geometry and input range.

## Project history

FormaMech currently includes immutable unit-safe contracts, a local content-addressed
store with explicit retention, and a verified compressible Neo-Hookean physics kernel
with differentiable plane-stress closure. It also includes strict tagged 2D triangle
geometry ingestion for Gmsh v4.1 meshes, deterministic RBF and FFD reference-mesh
morphing, executable family-membership predicates, an invalid-instance ledger, a
sparse nonlinear primary FEM solver for 2D finite-strain plane stress with P1/P2
displacement, independent FEM verification, preregistered DOE, evidence-role,
leakage, replacement, and holdout-commitment contracts, and
[fail-closed S10 dataset contracts](https://github.com/BBahtiri/formamech/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0011-s10-dataset-trust-boundaries.md) for
public request plans, snapshots, extraction, collocation, projection and
normalization, and the
[fixed S11 PI-GINOT operator architecture](https://github.com/BBahtiri/formamech/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0012-s11-operator-architecture.md),
followed by the
[S12 nondimensional objective registry](https://github.com/BBahtiri/formamech/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0013-s12-objective-registry.md).
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
[energy-form operator of ADR 0018](https://github.com/BBahtiri/formamech/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0018-s13c-energy-form-operator.md) replaced
it. Its single registered trial passes every frozen S13 guardrail, the amended G11
and G13 gates pass, and `CP-FREEZE` freezes it for S14. S14 measured it on
DEVELOPMENT and once on HOLDOUT under a signed protocol: the HOLDOUT error budget
passes every slice, two DEVELOPMENT peak slices were accepted by the decision owner
([ADR 0021](https://github.com/BBahtiri/formamech/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0021-s14-development-budget-acceptance.md)), and G14 passes with the claim level `HOLDOUT_VERIFIED`, its reducer running over
the twelve families of [ADR 0022](https://github.com/BBahtiri/formamech/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0022-replace-implementation-v1.0-oss.md).
S15 serves it as a hash-closed [bundle](bundle.md)
([ADR 0023](https://github.com/BBahtiri/formamech/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0023-s15-bundle-and-cli.md)) with `load_bundle`, `predict`,
`inspect` and a campaign-plugin command line; G15 passes. S16 applies the twenty
mutants of ADR 0022 as source edits of production code; each fires its primary
gate and the signatures are a committed baseline
([ADR 0024](https://github.com/BBahtiri/formamech/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0024-s16-mutation-suite.md)); G16 passes.

Read the [limitations](limitations.md) before relying on future numerical results.
