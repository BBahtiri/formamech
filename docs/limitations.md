# Limitations

PI-GINOT performs numerical verification against finite element method (FEM)
solutions. That is not experimental validation, and it does not show that predictions
agree with measurements from physical specimens.

Verification for a bundle establishes nothing about geometries outside that bundle's
declared input envelope. Inputs outside the envelope must not be treated as covered by
the bundle's results.

Finite verification sampling is evidence about the cases evaluated. It is not a
mathematical proof of a uniform error bound over every admissible input.

At the current S6 stage, PI-GINOT contains contract and local content-storage
infrastructure, a constitutive physics kernel, and strict tagged 2D triangle geometry
ingestion. The verified S4 material envelope is `1 <= E <= 10000 MPa` and
`0 <= nu <= 0.49`; values outside it are unsupported.

S5 accepts explicitly unitized, planar, linear triangle meshes with named line
boundary groups and positive named region groups. It rejects healing, defeaturing,
remeshing, inferred units, quads, high-order elements, mixed area elements, 3D
meshes, unclassified facets, duplicate nodes or cells, inverted elements,
non-manifold edges, and query points outside the mesh when containment is required.
S6 deforms an existing S5 reference mesh with interpolatory Gaussian RBF maps or
tensor-product Bernstein FFD maps. It preserves connectivity and accepts a morph only
when the declared topology and tags remain unchanged, feature-size and mesh-quality
bounds pass, and `det(grad(Phi))` is strictly above the configured margin at every
point of a fixed degree-2 triangle quadrature rule. This finite check is an executable
acceptance predicate; it is not a mathematical proof of global injectivity or of the
Jacobian sign between quadrature points. S6 does not heal or remesh failed instances.

Family membership enforces exact named canonical quantities, inclusive primitive and
derived bounds, the reference mesh identity, and a hashed validity policy. The
invalid-instance ledger records every supplied attempt but does not construct or
sample a DOE. The project does not yet implement geometry construction, FEM,
neural-operator training, verification, or inference.
