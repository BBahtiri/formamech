# ADR 0013: S12 nondimensional objective registry

- Status: Accepted
- Date: 2026-09-12
- Step: S12
- Gate: `artifacts/gates/S12.json`

## Context

S12 turns the S4 plane-stress physics contract and the selected S11 operator into
explicit, differentiable objectives before production training. The S11 baseline
has independent `u` and `P` heads, so `P_served = P_predicted` must remain the
authoritative stress field. Constitutive reconstruction is a consistency oracle,
not an inference-time substitute for the served head.

The objectives must support supervised and physics-only work without changing
their term identities, remain meaningful when the two coordinate axes arrive in
different units, and refuse evidence roles that could leak the sealed evaluation
set. They must also expose the S11 geometry, material, load and boundary-condition
paths to gradient verification. S12 is formula and gradient evidence; it is not a
trained-model or prediction-accuracy result.

## Decision

The registry has exactly three modes: `DATA_ONLY`, `PHYSICS_ONLY` and `HYBRID`.
`HYBRID` is the default. The data terms are displacement and served-stress errors.
The physics terms are one selected equilibrium form plus constitutive consistency,
angular momentum, component-wise Dirichlet and traction-free residuals. All terms
retain stable first-class names and are reduced from nondimensional residuals with
strictly positive declared weights.

Weak-form equilibrium is the default reported equilibrium term. It assembles the
Galerkin residual
`integral(P_served : grad(delta v)) - integral(B . delta v) - integral(T0 . delta v)`.
Strong-form `Div_X(P_served) + B` remains available only through an explicit
registry selection and remains part of verification. Selecting strong form does
not silently change any other objective term.

Every objective uses declared reference scales `L0`, `u0` and `P0`. Displacement
and stress residuals are divided by `u0` and `P0`; strong equilibrium is divided by
`P0/L0`; and the weak residual is divided by `P0 L0` per unit thickness. Coordinate
inputs additionally carry separate positive conversion factors for the x and y
axes. Physical derivatives apply the matching per-axis conversion before
nondimensional reduction; using the x-axis factor for both derivatives is not an
allowed shortcut.

Equilibrium, constitutive consistency, angular momentum and traction-free terms
all consume `P_served` from the independent predicted-stress head. Constitutive
consistency compares only the in-plane `2 x 2` components to the plane-stress
material response. The constructed 33 component is identically zero and is not
included as a residual. Angular momentum remains the separate independent skew
component of `P_served F_3D^T`, and the free-boundary condition remains the
separate first-class residual `P_served N` on declared free boundaries.

The reusable verification harness names parameters without assuming the DogBone
family. The S12 campaign binds them as follows:

| Family-neutral parameter | DogBone source |
|---|---|
| `geometry_axial_extent` | `L_total` |
| `geometry_outer_width` | `W_grip` |
| `geometry_inner_width` | `W_gauge` |
| `geometry_transition_radius` | `R_fillet` |
| `elastic_modulus` | `E` |
| `poisson_ratio` | `nu` |
| `load_amplitude` | `nominal_axial_strain` |
| `prescribed_displacement` | derived from `L_total` and `nominal_axial_strain` |

The derived half-model boundary quantity is
`prescribed_displacement = L_total * nominal_axial_strain / 2`. The first seven
parameters must reach the physics-objective total; the prescribed displacement
must independently reach the Dirichlet path.

Objective reduction is fail-closed by evidence role. In particular, `HOLDOUT` is
rejected before residual type checking or payload access. The `HOLDOUT_ACCESS`
mutant exists only inside verification as a negative control with the production
role guard deliberately removed; it reuses the same required terms and weights
and is not a second production registry path.

## Consequences

The registry can compare data, physics and hybrid configurations without changing
formula semantics, while weak-versus-strong selection is deliberate and visible.
Axis-specific units are part of the residual contract, so anisotropic coordinate
scaling cannot be hidden by a single global conversion. Keeping constitutive,
angular and traction-free terms separate also prevents an in-plane closure
tautology from masking a failed predicted-stress boundary or balance condition.

Perturbation isolation uses six channel-isolated analytic fixtures rather than one
coupled physical state. Each fixture changes only its named residual channel, but
the harness recomputes all six physics residual APIs for every fixture before
checking that exactly the named term fired. This is necessary because perturbing a
shared physical state can legitimately activate several coupled terms and would
not prove objective isolation.

S12 performed no production training and created no trainer, optimizer state or
production checkpoint. It opened neither `DEVELOPMENT` nor `HOLDOUT`, and it did
not access the legacy `experiments/` workstream. The gate therefore makes no
operator-accuracy, conditioning-quality or peak-stress claim.

## Verification

G12 evaluates nine checks and reduces all nine to PASS:

- `G12.exact`: analytic exact-solution fixtures measured zero residual for strong
  equilibrium, weak equilibrium, constitutive, angular, Dirichlet and traction-free
  terms, below the `1e-8` limit;
- `G12.perturbed`: each of the six channel-isolated perturbation fixtures fired
  exactly its named term, with all six residual APIs recomputed for every fixture;
- `G12.fd`: finite-difference checks covered both data terms and all six physics
  terms, found no detached term and had maximum relative error
  `7.036847826726204e-11`, below `1e-5`;
- `G12.reachability`: all eight family-neutral geometry, material, load and BC
  parameters reached their declared loss paths;
- `G12.anisotropy`: x-only coordinate rescaling produced relative change `0`, and
  the `COORDINATE_ANISOTROPY` mutant fired;
- `G12.weak-strong`: the same nonconstant continuous recovered P2 stress field was
  represented cellwise by its quadratic polynomial and evaluated at 224 quadrature
  points. Weak balance and strong balance projected onto the same interior bubble
  test `x(1-x)y(1-y)` differed by `8.576208590949791e-20`. The weak and projected
  strong magnitudes were `1.6375806364921086e-6` and
  `1.6375806364921944e-6`, respectively, and reconstructed-stress variation was
  `0.8109940010363081`; all three exceed the `1e-8` non-vacuity thresholds where
  applicable;
- `G12.constitutive-shape`: the residual shape was exactly `(2, 2)`, and an
  out-of-plane served-stress tensor was rejected;
- `G12.role-isolation`: the production API refused `HOLDOUT`, and the
  verification-local `HOLDOUT_ACCESS` mutant fired; and
- `G12.coverage`: focused objective tests passed at 92.99% branch coverage, above
  the required 85%.

The weak/strong comparison uses the recovered nonconstant P2 field from FEM request
`s8-pure_dirichlet-structured-l2-p2` in both forms. It reuses the tighter
nondimensional `1e-12` threshold carried by S8 `G8.patch` solely as a numerical
same-field sign, scale, quadrature and integration-by-parts consistency tolerance.
It is explicitly not a claim that FEM discretization error is `1e-12`, and agreement
does not prove that the recovered field satisfies equilibrium.

The canonical artifact binds the measured Windows/Python 3.11 values. Fresh
numerical replay is threshold-based rather than expected to reproduce the last
floating bit; observed weak/strong variation across processes was approximately
`1.7e-21`.

Implementation revision `0bc851413e13fb13843ebb2b016c2ca2b87f69cc` produced
canonical gate artifact content hash
`621c48a0903d33ed1af7e32fe2b4bc09a5d0315c96f8a1d19720e4b0836e2dd0`.
The focused suite passed 89 tests with 92.99% branch coverage. The final
repository-wide regression passed 1,928 tests with 2 skipped. With all nine G12
checks PASS, S13 production training is unblocked; no S13 work is part of this
decision.
