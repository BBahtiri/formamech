# ADR 0006: S6 morphing and family-membership boundary

- Status: Accepted
- Date: 2026-08-22
- Step: S6

## Context

S6 must turn a tagged S5 reference mesh into deterministic family instances without
silently repairing invalid geometry or losing rejected attempts. The membership
boundary must reuse S2's unit-safe primitive and derived-quantity contracts and be
serialisable for later bundles.

## Decision

S6 provides two immutable deformation-map specifications. The RBF map uses a Gaussian
kernel with an affine polynomial tail and solves the augmented interpolation system,
so configured control-point displacements are interpolated exactly. The FFD map uses
an arbitrary two-dimensional tensor-product Bernstein lattice over an explicit box.
Both expose analytic Jacobians and bind exact named displacement parameters expressed
as canonical millimetres.

A morph retains the reference connectivity. It transports boundary facets through the
same point map and then passes the result through the S5 canonical mesh boundary. The
validity policy requires all of the following:

- `det(grad(Phi))` is strictly greater than the configured margin at all seven points
  of the fixed degree-5 symmetric triangle rule tabulated by
  [Dunavant](https://doi.org/10.1002/nme.1620210612);
- the versioned topology signature, including hole count, is unchanged;
- semantic tag names, counts, and transported facet incidence are unchanged;
- the physical boundary feature-size floor is met; and
- triangle minimum-angle and aspect-ratio bounds are met.

The quadrature rule and validity-policy content hash are recorded in every report.
The degree-5 rule includes the triangle centroid and catches an interior-fold fixture
whose vertices and former degree-2 sample points remain valid. Quadrature positivity
is still the executable S6 gate predicate, not a claim of global bijectivity or a sign
guarantee between the sampled points.

The family-membership predicate compiles an S2 `GeometryFamilySpec` against one S5
reference mesh and one validity policy. It requires the exact primitive parameter
vocabulary, canonical units and dimensions, finite values, and inclusive bounds.
Derived quantities are evaluated from the validated S2 canonical expression tree; no
textual `eval` path exists. The predicate emits a versioned canonical-JSON form and
rejects unknown or missing fields during deserialisation.

Every supplied attempt produces exactly one immutable ledger record. Accepted records
contain a canonical mesh and no rejection reason. Rejected records contain stable
reason codes and no mesh. Duplicate attempt identifiers fail before execution.

## Consequences

- Equal reference mesh, map specification, validity policy, and parameter bytes yield
  a byte-identical canonical mesh.
- Invalid parameters, non-injective sampled maps, topology or tag changes, insufficient
  feature size, and unacceptable mesh quality fail closed with stable reason codes.
- The family predicate is tied to one reference mesh identity and one policy hash;
  changing either requires recompilation.
- A later stage must add a stronger map-level certificate before describing an inverse
  map or cross-mesh pullback as verified. S10 cannot inherit that claim from S6's
  sampled determinant predicate alone.
- S6 does not create a DOE, heal geometry, remesh, run FEM, evaluate constitutive
  physics, or train a neural operator.
