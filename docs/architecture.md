# Architecture and Research Lineage

This page documents FormaMech's technical lineage from the foundational
`PI-GINOT-DogBone` study, its physics loss formulations, hardware execution
boundaries, and epistemic verification model.

## 1. Lineage from `PI-GINOT-DogBone` to FormaMech `v0.2.0.dev0`

FormaMech descends from the data-free geometry-informed neural operator lineage
introduced in `PI-GINOT-DogBone` (A. Dean & B. Bahtiri, *PI-GINOT: Data-free
geometry-informed neural operator learning for finite-strain hyperelasticity on
parametric DogBone specimens*, arXiv:2607.23299) and its companion `PI-GINOT`
energy-form study (see [upstream sources](adr/upstream-sources.md)).

| Dimension | `PI-GINOT-DogBone` Baseline | FormaMech (`v0.2.0.dev0`) |
| --- | --- | --- |
| **Domain / Physics** | Finite-strain compressible Neo-Hookean hyperelasticity in 2D plane stress (solid mechanics; zero CFD) | Finite-strain solid mechanics (zero CFD): isotropic & fibre-reinforced hyperelasticity (Neo-Hookean, Mooney–Rivlin, Yeoh, Gent, HGO, GOH, custom symbolic strain-energy formulas) and stateful generalized-Maxwell viscoelasticity |
| **Physics Loss** | Strong-form reference equilibrium `Div(P) = 0`, traction-free residuals `P N = 0`, symmetry residuals, `det(F)` barrier, and integral axial force consistency | Both strong/weak equilibrium (`piginot.objectives.equilibrium`) and served **Deep Ritz / total potential energy** (`piginot.objectives.energy`, `pi-ginot-energy-parametric-film-v1`) + **P1 hierarchical `M0 -> M1` energy** (`piginot.objectives.hierarchical_energy`) using redrawn stratified triangle quadrature and deterministic Dunavant rules with `relu(J_min - J)^2` barrier |
| **Dirichlet Enforcement** | Closed-form DogBone coordinate expressions | Exact affine lifting for quarter specimens plus Sukumar–Srivastava **Approximate Distance Function (ADF)** lifting (`piginot.lifting.adf`) supporting lines and circular arcs on arbitrary 2D CAD profiles |
| **Compute Hardware** | PyTorch neural operator with CUDA GPU support | PyTorch neural operators are device-agnostic (CPU/CUDA tensors); dual execution profiles separate `cpu-deterministic-bitwise` from opt-in `gpu-accelerated-tolerance`; pluggable `SolverBackend` supports serial `scikit-fem`, vectorized PyTorch CPU/GPU `torch-fem`, and optional LGPL-3.0 external plugins (`formamech-dolfinx`) |
| **Verification & Provenance** | Post-hoc independent Abaqus UEL comparisons | Built-in RFC 8785 JCS canonical hashing (`rfc8785`), per-inference **Trust Receipts** (`formamech.trust-receipt-v1`) with orthogonal `VerificationTier`, Tier 0 zero-FEM admissibility, Tier 1 per-CAD split-conformal & DWR bounds, Tier 1.5 test-time physics LoRA, Tier 2 spectrally safe warm-start FEM, and Level 1–4 VVUQ benchmarks |

### Why FormaMech Separates Strong-Form and Energy-Form Physics

The served fixed-geometry and parametric operators (`pi-ginot-energy-parametric-film-v1`,
`pi-ginot-boundary-energy-film-v1`) minimize the total potential energy using
first-order spatial derivatives (`grad u`) rather than second-order stress
divergences (`Div P`), which improves optimization conditioning and naturally
satisfies traction-free Neumann boundaries under displacement-controlled and
conservative dead-traction loading.

Because pure potential-energy minimization breaks down under **non-conservative
follower pressures** (`t = -p n` on open boundaries) and can exhibit
quadrature-driven energy overshooting if integrated carelessly, FormaMech keeps
strong/weak equilibrium and reference-surface traction residuals
(`piginot.objectives.equilibrium`, `piginot.objectives.free_traction`) alongside
stratified redrawn quadrature and multi-order Dunavant consistency checks.

## 2. Compute Architecture and Dependency Boundaries

FormaMech enforces strict package boundaries (`tests/training/test_import_boundary.py`):

- **Core (`formamech`)**: pure Python standard library + `rfc8785` canonical
  serialization only. No NumPy, PyTorch, FEM, or CAD imports on metadata paths.
- **Runtime (`formamech[runtime]`)**: NumPy, PyTorch, and `safetensors` for
  portable bundle loading and fast inference without mesher or FEM dependencies.
- **Verification & FEM (`formamech[verify]`, `formamech[fem]`)**: pluggable FEM
  solver backends (`scikit-fem` default 2D reference oracle, PyTorch vectorized
  `torch-fem` supporting CPU/GPU, 2.5D plane strain/axisymmetric, follower
  pressure, and frictionless rigid-indenter contact; extensible via
  `formamech.fem.backends` entry points for isolated LGPL-3.0 FEniCSx/DOLFINx
  3D/TMC plugins), DWR adjoints, and spectrally safe warm-start Newton serving.
- **UI, Agent & Co-Simulation Extras (`formamech[ui]`, `formamech[mcp]`,
  `formamech[export]`)**: optional Streamlit/Plotly Verification & Trust Inspector
  (`formamech ui`), sandboxed MCP JSON-RPC 2.0 server (`formamech mcp`), and
  ONNX / FMI 3.0 Co-Simulation FMU packaging (`formamech export`).
- **CAD (`formamech[cad]`)**: Gmsh and `ezdxf` for planar STEP/DXF import and
  hierarchical mesh generation, plus canonical 3D volumetric `TET10`/`HEX8`
  contracts (`CanonicalMesh3D` refusing `TET4` for $\nu \ge 0.45$) and linear-complexity
  $O(N \cdot S)$ `Surface3DEnergyOperator` foundations.

## 3. Epistemic Verification Boundaries (ASME V&V 10 / VVUQ Alignment)

FormaMech maintains a strict boundary between four distinct kinds of evidence so
that weaker checks are never conflated into an ambiguous "verified" flag:

1. **Artifact & Syntax Integrity**: RFC 8785 JCS canonicalization and SHA-256 /
   BLAKE2b digests prove exact reproducibility of bytes, meshes, weights, and
   specifications—never physical accuracy.
2. **Physical Admissibility**: Checking `det(F) > 0`, exact Dirichlet
   compliance, and equilibrium/traction residuals flags physically invalid fields
   at zero FEM cost without guaranteeing proximity to the FEM solution.
3. **Statistical Calibration**: Split-conformal prediction intervals provide
   marginal finite-sample coverage guarantees over exchangeable in-envelope
   queries—never pointwise continuum proofs.
4. **Discrete FEM Verification**: DWR adjoint indicators and converged warm-start
   FEM solves verify agreement against the declared discrete FEM model (mesh,
   element order, constitutive model, solver tolerance)—never experimental
   validation against physical specimens (see [limitations](limitations.md)).
