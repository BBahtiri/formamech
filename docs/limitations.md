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
S6 deforms an existing S5 reference mesh with interpolatory Gaussian RBF maps or
tensor-product Bernstein FFD maps. It preserves connectivity and accepts a morph only
when the declared topology and tags remain unchanged, feature-size and mesh-quality
bounds pass, and `det(grad(Phi))` is strictly above the configured margin at every
point of a fixed seven-point, degree-5 triangle rule. This finite check is an
executable acceptance predicate; it is not a mathematical proof of global injectivity
or of the Jacobian sign between quadrature points. A later cross-mesh comparison must
not call its pullback verified solely because an S6 report passed. S6 does not heal or
remesh failed instances.

Family membership enforces exact named canonical quantities, inclusive primitive and
derived bounds, the reference mesh identity, and a hashed validity policy. S9 samples
that predicate with deterministic, domain-separated DOE streams and reports a vector
of finite coverage diagnostics. Those diagnostics are evidence about the sampled
population, not a proof that every admissible point is covered. Version-one targeted
guard points cross primitive bounds; derived-bound failures remain visible in the
rejection ledger but are not claimed as dense guard strata.

S9 preregisters immutable case roles and transitive leakage groups, preserves failed
cases through deterministic reserve replacement, and commits to exactly 64 withheld
holdout coordinates with BLAKE2b-256. The public commitment proves only that a later
reveal matches those coordinates. It does not prove that the local holder never saw
the ignored private reveal. Git also does not preserve historical file mtimes, so S9
records live ordering when artifacts are created and separately verifies reproducible
Git first-add ancestry. The project does not yet implement verified datasets,
neural-operator training, certification, or inference.
