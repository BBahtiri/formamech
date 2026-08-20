# ADR 0004: S4 physics domain and plane-stress closure

- Status: Accepted
- Date: 2026-08-20
- Step: S4

## Context

S4 requires a machine-precision compressible Neo-Hookean kernel and a differentiable
plane-stress closure, but `STEPS-v1.0.md` does not define the admitted material
range. The prescribed positive closure root is globally unique only when the first
Lamé parameter is non-negative. In addition, the absolute `|P33| < 1e-10 MPa` gate
is scale-dependent: at `E = 210000 MPa`, `nu = 0.49`, even the nearest representable
float64 `F33` can leave a residual above the gate.

## Decision

S4 v1 uses the existing immutable `MaterialSpec` with:

- model `compressible_neo_hookean`;
- exactly `elastic_modulus` in canonical MPa and `poisson_ratio` dimensionless;
- `1 <= E <= 10000 MPa` and `0 <= nu <= 0.49`.

The constitutive API accepts dense, non-empty float32 or float64 PyTorch tensors.
Scientific G4 evidence uses CPU float64. Three-dimensional kernels require trailing
shape `(3, 3)` and the closure requires `(2, 2)`. Constitutive evaluation fails
closed unless `det(F) > 1e-12`.

The plane-stress closure solves for `log(F33)` with a bounded safeguarded Newton
iteration and returns the adjacent representable value with the smallest physical
closure residual. Its custom autograd backward implements the specified implicit
derivative with respect to `F_2D`. Material properties remain immutable scalar
contract data and are not differentiable tensor inputs in S4.

PyTorch is exposed through a dedicated `physics` optional dependency. The full
three-dimensional consistent tangent is part of S4; finite-element assembly and a
condensed plane-stress element tangent remain outside this step.

## Consequences

- Negative Poisson ratios and materials outside the verified modulus range fail with
  `InvalidMaterialError`; they are not silently extrapolated.
- The project material used by the S1 risk spike (`E = 760 MPa`, `nu = 0.23`) is
  inside the admitted range.
- The exact `nu = 0` branch returns `F33 = 1` and a zero deformation gradient.
- Supporting a wider material range requires a new numerical envelope and gate
  evidence rather than relaxing the S4 threshold after the fact.
- S4 provides constitutive physics only: no geometry, mesh, FEM, or ML workflow is
  introduced.
