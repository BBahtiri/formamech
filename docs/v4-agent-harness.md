# AI-Ready Agent Harness & Production 3D Suite (v0.4)

FormaMech v0.4 turns the entire pure-MIT solid mechanics and physics-AI stack into
an **AI-agent-native engineering platform** while closing the remaining production
physics, generative design, serving, and continual-qualification gaps.

## 1. AI-Ready Universal Agent Harness (`V4-0`)

Autonomous AI agents get uniform, sandboxed, budget-guarded, hash-chained access to
every FormaMech capability across 9 logical domains (`discover`, `cad`, `fem`,
`multiphysics`, `surrogate`, `optimize`, `ingest_validate`, `export`, `governance`):

* **MCP 2024-11-05 Server (`formamech mcp serve`):** 18 tools (including progressive
  discovery `discover_capabilities`, `describe_capability`, `invoke_capability`,
  `session_status`, `describe_state_machines`), 6 read-only resources
  (`formamech://capabilities`, `formamech://state-machines`,
  `formamech://session/status`, `formamech://session/ledger`,
  `formamech://workspace/tree`, `formamech://artifact/{relpath}`), and 5
  engineering prompts (`cad-to-verified-surrogate`, `multiphysics-3d-audit`,
  `experimental-vv10-calibration`, `generative-lightweighting`,
  `diagnose-refusal-or-ood`).
* **Machine-Readable CLI (`formamech agent`):**
  ```bash
  formamech agent capabilities --domain optimize
  formamech agent describe optimize.topology_simp_3d
  formamech agent invoke optimize.topology_simp_3d --workspace ./study --dry-run
  formamech agent verify-ledger --workspace ./study
  formamech agent export-tools --provider openai --mode progressive
  ```
* **Native LLM SDK Tool Exporters:** `export_tool_schemas` emits strict JSON Schema
  tool definitions for OpenAI, Anthropic, Gemini, and MCP in either `progressive`
  (4 meta-tools to avoid context bloat) or `full` mode.
* **Self-Healing Fail-Closed Envelopes & Persistent Sessions:** Every invocation
  returns `HarnessEnvelope` with `ok`, `error_code`, `recovery_hint`,
  `allowed_next_actions`, `receipt_sha256`, and a tamper-evident SHA-256
  hash-chained session ledger (`ledger.jsonl`).

## 2. Live Solver Oracles, CAD Connectors & Production Serving API (`V4-1`)

* **External Solver Oracles (`piginot.fem.external_solvers`):** Live adapters for
  CalculiX (`ccx` `.inp`/`.frd`), Abaqus CLI, PyAnsys MAPDL, and LGPL-isolated
  subprocess DOLFINx — all raising `OracleUnavailableError` fail-closed when the
  external binary is absent.
* **Live CAD Connectors (`piginot.cad.connectors`):** Pluggable connectors for
  OpenCASCADE templates, CadQuery, FreeCAD headless subprocess, and Onshape REST
  STEP export.
* **Verified Serving API (`formamech serve-api`):** Zero-dependency HTTP/JSON
  microservice exposing `/health`, `/metrics` (Prometheus exposition),
  `/v1/capabilities`, `/v1/predict`, `/v1/verify-receipt`, `/v1/harness/invoke`,
  and `/v1/ledger/verify`, backed by an append-only SHA-256 hash-chained
  `SignedAuditLedger`.

## 3. Shipped 3D Foundation Checkpoint & Benchmarks (`V4-2`)

* **Compact Shipped Checkpoint (`bundles/foundation/struct3d-base-v1/`):** Multi-family
  16-channel GALE `Surface3DEnergyOperator` pretrained label-free across holed
  blocks, L-brackets, and mounting lugs with tamper-checked `weights.safetensors`
  and `GeometryOODDetector3D`, ready for budgeted 3D LoRA transfer.
* **Reproducible 3D Benchmark Suite (`formamech bench run`):** Runs B1 solver
  scaling, B2 foundation 3D LoRA transfer, B3 OOD stress testing, and B4
  cross-backend analytical uniaxial parity with canonical JSON + Markdown reports.

## 4. Structural Production Physics Pack (`V4-3`)

* **Shells & Beams (`piginot.fem.shells_beams_3d`):** Shear-locking-free 6-DOF/node
  DKT-CST triangular shells (matching thin plate Kirchhoff benchmarks at
  `t/L = 1/50`) and exact 3D Timoshenko beams.
* **Buckling & Post-Buckling (`piginot.fem.buckling_3d`):** Linearized 3D
  eigen-buckling `(K_0 + lambda K_G) phi = 0` and Crisfield spherical arc-length
  (Riks) continuation tracking snap-through past limit points (`RIKS_SOLVER_MACHINE`).
* **Transient Dynamics (`piginot.fem.dynamics_3d`):** Implicit Newmark-beta,
  HHT-alpha with Rayleigh damping, and Explicit central-difference with automatic
  fail-closed `CflStabilityError` guards.
* **Fatigue & Fracture (`piginot.stateful.fatigue_fracture`):** ASTM E1049-85
  rainflow counting, Goodman/Gerber mean-stress correction, Basquin S-N &
  Coffin-Manson E-N life curves, Palmgren-Miner damage, and staggered AT2 brittle
  phase-field fracture with Miehe spectral tension-compression split.

## 5. Multi-Topology 3D Operators & Verified Generative Design (`V4-4`)

* **Multi-Topology 3D Operators (`piginot.operators.multi_topology_3d`):**
  20-channel surface representation combining GALE features with local wall
  thickness, Laplacian spectral harmonic coordinates, and genus encoding, guarded
  by `MultiTopologyOODDetector3D`.
* **Verified 3D SIMP Topology Optimization (`piginot.optimization.topology_opt_3d`):**
  Exact self-adjoint compliance sensitivities, cone filtering, Heaviside projection,
  Optimality Criteria updates, and mandatory final exact 3D FEM certification
  (`TOPOLOGY_OPT_MACHINE`, RFC 8785 Trust Receipt).
* **Bernstein FFD Shape Optimization (`piginot.optimization.ffd_shape_opt_3d`):**
  Trivariate Bernstein cage morphing enforcing `min_det_j > 0` (`FfdMeshInversionError`)
  coupled to exact 3D FEM displacement bounds.

## 6. Continual Qualification & Fleet Orchestration (`V4-5`)

* **Out-of-Core `ShardedMeshDataset3D` (`piginot.store.dataset_3d`):** Lazy
  one-shard-at-a-time streaming, SHA-256 shard tamper detection, and zero-leakage
  `TRAIN`/`DEV`/`HOLDOUT` split locking (`SplitLeakageError`).
* **Distributed Fleet Workers (`piginot.training.fleet`, `formamech fleet`):**
  `LocalMultiprocessFleet`, `SlurmScriptFleet`, and `KubernetesJobFleet` governed
  by `FLEET_JOB_MACHINE`.
* **Fail-Closed `ContinualQualificationLoop` (`piginot.training.continual_clpm`,
  `formamech continual`):** Ingests live `SignedAuditLedger` fallbacks into a
  replay-protected adaptation batch, evaluates isolated shadow checkpoints against
  frozen blind `HOLDOUT` cases, and atomically promotes only on zero regression
  (otherwise rolling back deterministically).
