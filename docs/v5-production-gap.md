# FormaMech v0.5 — Closing the Physics AI Production Gap

FormaMech v0.5 adds five pure-MIT production engineering capabilities verified against
published NAFEMS standards (`LE10`, `LE11`, `T1`, `T3`, `T4`) and structured real-world
open-source V&V dataset protocols.

## 1. `<1%`-Overhead 3D UQ, Element-Bound Hotspots & Physics-Regime Guards (V5-1)

- **3D Split-Conformal & AS6983 Conservative Shifted Bounds** (`piginot.bounds.conformal_3d`):
  Calibrates single-pass latent uncertainty on `DEV` (`n >= 19` for `alpha = 0.05`) and
  emits one-sided conservative bounds protecting against peak stress or displacement
  underprediction, verified via Beta-Binomial holdout checks alongside mean, p90, and
  worst-case physical-unit errors (`mm`, `MPa`).
- **3D Linear DWR Adjoint Certification** (`piginot.certify.adjoint_3d`):
  Evaluates goal-oriented dual-weighted residuals for compliance, reaction force,
  strain energy, and smoothed Gauss-point peak von Mises stress in a single linear solve,
  escalating to warm-started 3D FEM only when tolerances or regime guards trip.
- **Element-Bound Stress Hotspots** (`piginot.fem.backend_3d`):
  Preserves unsmoothed quadrature-point peak von Mises stress and max principal strain
  prior to nodal L2 projection (verified on NAFEMS `LE10` within `<= 2.0%` of `-5.38 MPa`).
- **Physics-Regime Guards** (`piginot.certify.regime_guards`):
  Fail-closed monitors for spatial BC-layout shifts, yield-onset crossing, linearized
  buckling proximity, contact-gap closure, and Jacobian folding.

## 2. Component-Level Hybrid Sim + Physical-Test Grounding & Active Test Planner (V5-2)

- **Multi-Fidelity Discrepancy Grounding** (`piginot.validation.hybrid_grounding`):
  Aligns arbitrary 2D/3D DIC, strain-gauge, load-cell, and thermocouple coordinates onto
  canonical meshes via KD-tree interpolation and fits a Kennedy-O'Hagan RBF-GP discrepancy
  model `y_grounded = rho * y_sim + delta_GP` evaluated strictly on blind physical
  `VALIDATION_HOLDOUT` specimens under ASME V&V 10 (`u_val`) without conflating numerical
  `ClaimLevel` with physical `ExperimentalValidationTier`.
- **Active Physical Test-Plan Optimizer** (`piginot.active.test_planner`):
  Ranks next candidate laboratory tests maximizing information gain per unit test cost
  while refusing inadmissible or `VALIDATION_HOLDOUT` candidates, with optional pluggable
  BoTorch (MIT) and BayBE (Apache-2.0) adapters.

## 3. Solver-Agnostic SimOps Fabric, Archive Training & Neutral Benchmarks (V5-3)

- **NAFEMS EDSWG SimOps Schema & Multi-Solver Fabric** (`piginot.ingest.simops_schema`,
  `piginot.ingest.simops_fabric`):
  Native pure-MIT parsers for Abaqus/CalculiX (`.inp` + `.sta`/`.msg` convergence logs),
  Ansys ASCII `.cdb`, LS-DYNA `.k`, Nastran semantic `.bdf`, and SimJEB/DeepJEB `.npz`,
  separating `VALID_PHYSICAL_FAILURE` (preserved with labels) from `QUARANTINED_SOLVER_DEFECT`
  and writing geometry-group-locked `ShardedMeshDataset3D` corpora with zero leakage.
- **Archive-to-Foundation Training, 3D LoRA & `N*` Economics** (`piginot.training.archive_trainer`):
  Trains `MultiTopologyOperator3D` directly from sharded archives, adapts onto new
  component families with `<= 10%` trainable 3D LoRA parameters, and checks pre-flight
  amortized break-even `N*`.
- **License-Tiered External V&V Registry & Industry Benchmarks**
  (`piginot.benchmarks.external_data`, `piginot.benchmarks.industry_suite`):
  Separates Tier A commercial-clean corpora (`NIST AM-Bench`, `NASA PCoE`, `DeepJEB`,
  `ABC`, `MATR.io`) from Tier B ShareAlike eval-only and Tier C opt-in non-commercial
  corpora (excluding GPL-3.0 repositories), and runs multi-scale + BC-shift benchmarks
  emitting canonical JSON and Markdown leaderboards.

## 4. DfM Constraints, Watertight B-Rep STEP, GD&T & PLM / HITL Governance (V5-4)

- **DfM Constraints** (`piginot.optimization.dfm_constraints`):
  Keep-out/keep-in datum zones, minimum member thickness, AM overhang angle, and casting /
  milling pull-direction draft projections integrated into `optimize_3d_topology_simp`.
- **Watertight OpenCASCADE B-Rep STEP & GD&T Robustness** (`piginot.cad.brep_synthesis`):
  Synthesizes editable ISO-10303-21 STEP solids preserving exact cylindrical bolt bores,
  runs automatic OpenCASCADE round-trip watertightness & re-meshed 3D FEM checks, and
  certifies `+/- tol` worst-case GD&T tolerance-band corners.
- **Population-Centric PLM / EU AI Act Annex IV & Veteran-Engineer HITL Signoff**
  (`piginot.governance`):
  Exports JSON-LD, ReqIF 1.2 XML, EU AI Act Annex IV Markdown, and SHA-256 manifests
  gated by the cryptographic `HITL_SIGNOFF_MACHINE` (`DRAFT -> PENDING_HUMAN_REVIEW ->
  APPROVED | REJECTED_WITH_FEEDBACK`), also wired into `ContinualQualificationLoop`.

## 5. Multi-Material 3D Electro-Thermal-Mechanical Solver & Vector-BC Operator (V5-5)

- **Multi-Material 3D ETM FEM** (`piginot.fem.electro_thermal_3d`):
  Coupled electrical conduction + Joule heating (`q_J = sigma_e |grad V|^2`), orthotropic
  steady/transient multi-material heat conduction with interfacial contact conductance
  `h_c` and spatially varying convective CHT maps `h(x), T_amb(x)`, and finite-strain
  multi-material `F = F_m F_th` thermo-mechanical warpage — verified on NAFEMS `T1`, `T3`,
  `T4`, `LE11`, analytical Joule heating, and central-FD consistent tangent checks.
- **Vector-BC Coupled 3D Neural Operator & Guarded Transient Stepper**
  (`piginot.operators.multiphysics_operator_3d`):
  32-channel continuous vector-BC surface tokens + volumetric heat-source slices with
  dual-FiLM conditioning jointly predicting `(T, u)`, paired with residual-guarded
  transient FEM fallback.
- **Parametric Electronics Templates** (`piginot.cad.electronics_templates_3d`):
  `parametric-chiplet-package-3d`, `parametric-cold-plate-3d`, `parametric-busway-3d`,
  and IPC-2581-lite multi-layer JSON stackup importer.
