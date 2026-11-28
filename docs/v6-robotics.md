# FormaMech v0.6 — Robotics Physics AI & Real-Time Certified Surrogates

FormaMech v0.6 extends the pure-MIT, ASME V&V 10 & AS6983 cryptographically verified
solid-mechanics platform into **soft robotics, continuum surgical robots, tactile
elastomer sensing, nonlinear transient dynamics, and real-time safety-certified
Model Predictive Control (MPC)**.

All new modules preserve FormaMech's non-negotiable invariants:
- **Zero modifications to byte-frozen S4/S5 core modules** (`verification/status.py`, `operators/energy_operator.py`, `operators/architecture.py`),
- **Exact continuum similitude** (`piginot.fem._similitude`) so ultra-soft biological tissues (`85 Pa`), tactile silicones (`0.3–0.9 MPa`), and stiff Nitinol/metals (`58–210 GPa`) execute natively through the verified compressible Neo-Hookean kernel without breaking its validated `[1, 10000] MPa` bounds,
- **ASME V&V 10 epistemic discipline** separating numerical FEM verification (`ClaimLevel`) from blind physical experimental validation (`ExperimentalValidationTier`),
- **Pure-MIT core & zero raw third-party dataset bundling** (datasets are fetched on demand into gitignored `.formamech-runs/datasets/robotics/`).

---

## Capability Matrix (`V6-R1` – `V6-R6`)

| Track | Module(s) | Engineering & Robotics Capability |
| --- | --- | --- |
| **V6-R1** | `piginot.fem.follower_3d`, `piginot.fem.tendon_3d`, `piginot.objectives.virtual_work_3d` | 3D Nanson's-law follower pressure with exact unsymmetric load stiffness `K_L = -df_ext/du` & polytropic cavity coupling; 3D solid routed-cable FEM with Capstan friction; exact finite-rotation `SE(3)` Cosserat 9-disk continuum backbone solver (disk poses, FBG microstrains, tendon strokes, base 6D wrench); differentiable 3D weak-form virtual-work residual. |
| **V6-R2** | `piginot.fem.diff_contact_3d`, `piginot.robotics.tactile` | Smooth $C^\infty$ general 3D SDF rigid-indenter contact (`SPHERE`, `FLAT_PUNCH`, `CYLINDER_PIN`, `BOX`, `WEDGE_EDGE`) with exact variational contact Hessian, PSD curvature globalization, Augmented Lagrangian enforcement, regularized Coulomb stick-slip friction, DiffMJX straight-through "Contacts From Distance" (CFD) backward gradients, and `< 2 ms` tactile elastomer taxel surrogate with AS6983 bounds. |
| **V6-R3** | `piginot.fem.nonlinear_dynamics_3d`, `piginot.operators.temporal_bundle_3d`, `piginot.robotics.port_hamiltonian` | Large-deformation 3D `TET4`/`TET10` implicit Newmark-$\beta$ & HHT-$\alpha$ transient dynamics with Rayleigh damping and energy-drift telemetry; TB-Transolver multi-step temporally bundled operator (`TemporallyBundledOperator3D`) with PGRD velocity-residual integration; strictly passive `PortHamiltonianSurrogate` (Cholesky SPD $M^{-1}(q)$, PSD $D(q) \ge 0$, RK4 integration, discrete linearization). |
| **V6-R4** | `piginot.robotics.mpc`, `piginot.robotics.cbf`, `piginot.export.onnx_export_3d` | `< 10 ms` (`>= 100 Hz` CPU) horizon-condensed box-constrained `CondensedQpMpcController` (adaptive-restart FISTA); AS6983 `ConformalCbfSafetyFilter` governed by registered FSM `CONTROL_SAFETY_MACHINE` (`EVALUATING -> NOMINAL | CBF_MODIFIED | SAFE_HOLD_OOD`); 3D frozen runtime exporter (`export_surface_3d_runtime`, ONNX/TorchScript with strict parity verification and parameter-Jacobian hook). |
| **V6-R5** | `piginot.operators.surface_operator_3d` | Backward-compatible scaling upgrades on `Surface3DConfig`: Transolver++ adaptive slice temperature (`adaptive_temperature`), AB-UPT anchor subsampling (`anchor_count`), low-rank query compression (`low_rank_query_dim`), and chunked out-of-core decoding (`decode_chunk_size`). |
| **V6-R6** | `piginot.robotics.bridges`, `piginot.ingest.robotics_streams`, `piginot.benchmarks.external_data`, `piginot.harness.catalog` | Simulator/hardware 6D wrench trace import (`import_simulator_wrench_trace`), MuJoCo MJCF compliance export (`export_mujoco_compliance_plugin`), ROS 2 Humble/Jazzy manifest export (`export_ros2_node_manifest`), multi-rate sensor alignment & leak-free episode splitting (`split_episodes_leak_free` with fail-closed `EpisodeLeakageError`), 13 registered real robotics datasets, and 3 new AI-agent harness capabilities (`fem.solve_follower_pressure_3d`, `fem.solve_tendon_continuum_3d`, `fem.solve_sdf_contact_3d`). |

---

## Quickstarts

### 1. 3D General SDF Tactile Contact FEM (Incompressible `TET10`)

```python
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.diff_contact_3d import (
    RigidSdfIndenter3D,
    SdfPrimitiveKind,
    solve_3d_sdf_contact,
)
from piginot.geometry.mesh3d import make_brick_tet10_mesh

mesh = make_brick_tet10_mesh(18.0, 24.0, 4.0, nx=3, ny=4, nz=1)
gel = MaterialSpec(
    name="gelsight-gel",
    model="compressible_neo_hookean",
    properties={
        "elastic_modulus": Quantity(0.30, Unit.MPA).to_canonical(),
        "poisson_ratio": Quantity(0.48, Unit.ONE).to_canonical(),
    },
)
indenter = RigidSdfIndenter3D(
    name="sphere-probe",
    kind=SdfPrimitiveKind.SPHERE,
    contact_face_tags=("z_max",),
    initial_center_mm=(9.0, 12.0, 7.05),
    center_mm=(9.0, 12.0, 6.45),
    radius_mm=3.0,
    penalty_mpa_per_mm=40.0,
    friction_mu=0.35,
    tangential_shift_mm=(0.15, 0.0, 0.0),
)
result = solve_3d_sdf_contact(
    mesh,
    gel,
    dirichlet_bcs=(
        DirichletBC3D("base", ("z_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
    ),
    indenter=indenter,
)
assert result.converged
```

### 2. Exact Finite-Rotation `SE(3)` Cosserat 9-Disk Tendon Continuum Backbone

```python
from piginot.fem.tendon_3d import (
    CosseratBackboneSpec,
    solve_3d_tendon_continuum_backbone,
)

spec = CosseratBackboneSpec(
    length_mm=540.0,
    num_disks=9,
    youngs_modulus_mpa=58000.0,
    poisson_ratio=0.33,
    outer_radius_mm=1.25,
    tendon_offsets_mm=((18.75, 0.0), (0.0, 18.75), (-18.75, 0.0), (0.0, -18.75)),
)
sol = solve_3d_tendon_continuum_backbone(
    spec,
    tendon_tensions_n=(8.0, 2.0, 0.0, 0.0),
    external_forces_n={9: (0.0, -0.25, 0.0)},
)
print(
    "Tip XYZ (mm):",
    sol.disk_positions_mm[-1],
    "Peak FBG (ue):",
    abs(sol.fbg_microstrain).max(),
)
```

### 3. Real-Time Port-Hamiltonian QP-MPC + AS6983 Conformal CBF Safety Filter

```python
import numpy as np
from piginot.robotics.cbf import ConformalBarrierSpec, ConformalCbfSafetyFilter
from piginot.robotics.mpc import CondensedQpMpcConfig, CondensedQpMpcController
from piginot.robotics.port_hamiltonian import (
    PortHamiltonianConfig,
    PortHamiltonianSurrogate,
)

ph = PortHamiltonianSurrogate(
    PortHamiltonianConfig(state_dim=5, control_dim=5, hidden_dim=48)
)
x0_next, A_d, B_d = ph.linearize_discrete(np.zeros(10), np.zeros(5), dt_s=0.02)

mpc = CondensedQpMpcController(
    CondensedQpMpcConfig(state_dim=10, control_dim=5, horizon=10, dt_s=0.02),
    linearize_fn=lambda x, u, _dt: (A_d @ x + B_d @ u + x0_next, A_d, B_d),
)
cbf = ConformalCbfSafetyFilter(
    barriers=(
        ConformalBarrierSpec(
            "q1_wall", safety_limit=0.145, conformal_shift_margin=0.015
        ),
    ),
    u_min=np.full(5, -0.85),
    u_max=np.full(5, 0.85),
    safe_hold_action=np.zeros(5),
)
```

### 4. Autonomous AI-Agent Harness Invocation

```bash
formamech agent describe fem.solve_sdf_contact_3d
formamech agent describe fem.solve_tendon_continuum_3d
formamech agent describe fem.solve_follower_pressure_3d
```

See [Real-Data Robotics Validation Studies (Studies R1–R5)](robotics-studies.md) for complete physical benchmarks, datasets, D3.js + custom SVG figures, and live animations.
