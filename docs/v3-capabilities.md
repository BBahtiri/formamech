# Industrial 3D Scaling, Interoperability, Multiphysics & Copilot (v0.3)

This guide summarizes the five v0.3 engineering pillars implemented across FormaMech's MIT-licensed core.

> **Epistemic scope (ASME V&V 10):** Numerical verification (`ClaimLevel`, capped at `HOLDOUT_VERIFIED` in the byte-frozen S14 status module) proves discrete agreement with converged FEM only. Physical experimental validation is tracked strictly orthogonally via `ExperimentalValidationTier` (`UNVALIDATED` / `COUPON_CALIBRATED` / `HOLDOUT_EXPERIMENTALLY_VALIDATED`).

## 1. GPU-First Compute & Scalable 3D Iterative Linear Solvers (`V3-1`)

- **3D Sparse Iterative Solvers (`piginot.fem.linear_solvers`)**: `SparseLinearSolver3D` and `solve_linear_system` provide `auto`, `direct`, `pcg-jacobi`, `pcg-ssor`, `pcg-ilu`, `pcg-amg` (optional `.[scale]` extra with `pyamg`), and `gmres-ilu`, governed by the explicit `LINEAR_SOLVER_MACHINE` FSM.
- **Neural Coarse-Space Deflation**: `NeuralCoarsePreconditioner` supports symmetric two-level coarse-space acceleration inside `TorchFemBackend3D` and `ScikitFemBackend3D`.
- **Compute Runtime Governance (`piginot.training.compute`)**: `ComputeRuntimeConfig` and `ComputeRuntimeSession` enforce explicit replay tiers (`cpu-strict-deterministic` vs `gpu-accelerated-tolerance`), optional AMP forward with full-precision energy integration, mathematically exact additive chunked volumetric quadrature (`quadrature_chunk_size`), and distributed strategies (`SingleProcessStrategy`, `TorchDDPStrategy`, `SimulatedDistributedStrategy`).

## 2. Geometry-Generalizing 3D Operators, 3D LoRA & Foundation Hub (`V3-2`)

- **Heterogeneous 3D Multi-Mesh Operators**: `Surface3DEnergyOperator` supports masked token batching (`token_mask`), 16-channel GALE surface descriptors (`build_gale_surface_3d_tensors`), and general-face Rvachev R-conjunction 3D ADF lifting (`GeneralFaceADFLifting3D`).
- **Parametric 3D CAD & OOD Guards (`piginot.cad.parametric_3d`, `piginot.operators.geometry_3d`)**: Built-in OpenCASCADE 3D templates (`parametric-holed-block`, `parametric-l-bracket`, `parametric-mounting-lug`), Halton QMC DOE sampling, Euler characteristic / genus stability checks (`TopologyStabilityGuard3D`), and dimension-normalized shrinkage Mahalanobis `GeometryOODDetector3D`.
- **3D LoRA, Foundation Hub & P5 3D Optimization**: `LowRankAdapter3DConfig` (enforcing `<= 10%` trainable parameter budget with `ADAPTER_3D_MACHINE`), `FoundationRegistry` tamper-checked checkpoints, `Parametric3DDesignBridge` integrated with `run_fem_guarded_design_optimization`, and CLI commands under `formamech cad parametric-3d`.

## 3. Solver-Agnostic CAE Ingestion, VTU/PVD/GLB Export & ASME V&V 10 (`V3-3`)

- **CAE Archive Readers & Screening (`piginot.ingest`)**: Native readers for VTU/XDMF, Abaqus `.inp`, CalculiX `.inp` + `.frd`, and Nastran `.bdf`, plus pluggable `formamech.ingest.readers` entry points. `screen_cae_archive` enforces SHA-256 deduplication, `det(F) > 0` element admissibility, MAD outlier screening, and leak-free `TRAIN` / `DEV` / `HOLDOUT` partitioning (`formamech ingest`).
- **ParaView & Web Export (`piginot.export.vtu_export`)**: `export_fields_to_vtu`, `export_trajectory_to_pvd`, and pure-Python binary glTF 2.0 `export_deformed_glb`.
- **Experimental Inverse Calibration & V&V 10 (`piginot.validation`)**: Differentiable Adam + L-BFGS calibration (`calibrate_material_from_experiments`) for Neo-Hookean, Mooney-Rivlin, Yeoh, and J2 Voce models with Gauss-Newton parameter covariance (`u_input`) and blind holdout ASME V&V 10 budgets (`u_val = sqrt(u_num^2 + u_input^2 + u_exp^2)`) (`formamech validate`).

## 4. 3D Multi-Body Assemblies, Contact, Advanced Plasticity, Thermo-Mechanics & Modal (`V3-4`)

- **3D Assemblies & Contact (`piginot.geometry.assembly_3d`, `piginot.fem.contact_3d`)**: `AssemblyMesh3D`, `merge_3d_parts`, tied interface coupling, Augmented-Lagrangian 3D rigid-sphere contact (`RigidSphereIndenter3D`) with consistent tangent, and pure-PyTorch Third-Medium Contact (`third_medium_energy_density`).
- **3D Advanced Elastoplasticity (`piginot.stateful.j2_advanced`, `piginot.stateful.fem_stepper_3d`)**: `J2CombinedHardeningPlasticity` (Armstrong-Frederick kinematic + Voce isotropic + thermal softening), `FiniteStrainJ2Plasticity` (multiplicative Hencky `F = F_e F_p`), and `solve_incremental_fem_3d_trajectory` with tangent Dirichlet predictor, backtracking line search, BLAKE2b state hashing, and guarded surrogate FEM fallback.
- **Thermo-Mechanics & Modal Analysis (`piginot.fem.thermo_mechanical_3d`, `piginot.fem.modal_3d`)**: Steady/transient 3D heat conduction, multiplicative thermo-mechanical coupling (`F = F_m F_th`), consistent/lumped `TET10` mass matrices, unloaded and prestressed eigenfrequency analysis (`solve_3d_modal_analysis`), and exact analytical eigenfrequency sensitivity (`analytical_eigenfrequency_sensitivity`).

## 5. Interactive 3D WebGL Workbench & Guarded MCP Copilot (`V3-5`)

- **Offline 3D WebGL Component (`frontend/viewer-3d`, `piginot.ui.workbench_3d`)**: React 19 + Three.js + `@streamlit/component-v2-lib` component (`formamech.viewer_3d`) with raycasted B-Rep face BC picking, deformed field overlays, wireframe ghost, X/Y/Z clipping planes, and 3D point probe readouts. Exposed alongside `2D Fixed CAD` via the top-level **Workflow mode** selector in `formamech ui`.
- **Guarded MCP Agentic Copilot (`piginot.mcp_agent_workflows`, `piginot.mcp_server`)**: Expands `FormaMechMcpServer` to 13 JSON-Schema-validated, workspace-sandboxed tools governed by `AGENT_WORKFLOW_MACHINE` (`IDLE -> CAD_IMPORTED -> SETUP_PROPOSED -> COMPILED -> QUALIFIED -> EXPORTED`), hard DOF limits, FEM solve budgets, and fail-closed `InvalidTransitionError` enforcement prior to artifact export.
