# FormaMech v0.7 — Adaptive Hysteresis, Online Tissue ID, Operator CLIK, Two-Way Coupling & 1M-Node Scaling

FormaMech v0.7 closes the exact physical, statistical, and systems gaps surfaced by the
v0.6 real-data validation studies (`R1`–`R5`) — rate-dependent viscoelastic hysteresis,
static split-conformal conservatism under payload shift, unlabeled anatomical tissue
heterogeneity, open-loop continuum actuation, weak jamming recall, one-way simulator
exports, and unprofiled industrial mesh scaling — while preserving **zero edits to the
byte-frozen S4/S5 core** (`src/piginot/physics/`, `verification/status.py`,
`operators/energy_operator.py`, `operators/architecture.py`).

---

## Capability Matrix (`V7-R1` – `V7-R6`)

| Track | Module(s) | Gap Closed & Engineering Capability |
| --- | --- | --- |
| **V7-R1** | `piginot.fem.visco_hysteresis_3d` | **3D Rate-Dependent Viscoelasticity & Mullins Hysteresis:** Finite-strain `TET4`/`TET10` multi-branch Prony viscoelastic + Ogden–Roxburgh Mullins-damage solver (`solve_3d_visco_hysteresis`) with exact rank-1 Mullins consistent tangent, `TractionBC3D` & `FollowerPressureBC3D` support, thermodynamic dissipation certificate ($\dot{\mathcal{D}}_k \ge 0$ & `min_det_f > 0`), plus PGRD/PhysCon-Deform `VelocityResidualHysteresisOperator`. |
| **V7-R2** | `piginot.robotics.adaptive_cbf` | **Online Adaptive Conformal Quantile CBF (ACP-CQR):** Upgrades static split-conformal CBF with state-dependent asymmetric pinball-loss quantiles (`ConformalQuantileRegressor`) and Gibbs–Candès online miscoverage adaptation ($\alpha_{t+1} = \mathrm{clip}(\alpha_t + \gamma(\alpha - \mathrm{err}_t))$). Tightens braking margins in low-uncertainty regimes while expanding dynamically during payload/contact shift before tripping `SAFE_HOLD_OOD`. |
| **V7-R3** | `piginot.robotics.clik_3d`, `piginot.operators.constitutive_op` | **Operator-Based Continuum CLIK & Online PANO/CANO Tissue Fingerprinting:** (1) Infinite-dimensional shape-space damped least-squares CLIK (`OperatorClikController`) with nullspace spherical-obstacle repulsion and optional ACP-CQR CBF safety filter across `Surface3DEnergyOperator`, Cosserat backbones, and distilled neural operators; (2) `< 1 ms` 2-probe `PanoConstitutiveIdentifier` inferring patient-specific Ogden shear modulus $(\mu, \alpha)$ and sub-millimeter shallow tumor localization (`localize_shallow_inclusion`). |
| **V7-R4** | `piginot.robotics.contact_diagnostics` | **Energy-Residual Port-Hamiltonian Contact & Jamming Estimator:** Combines passive `PortHamiltonianSurrogate` power-balance monitoring ($\dot{H} - u^\top y$ and 36D joint-vs-EE mechanical dissipation $\tau_J^\top \dot{q} - (F_c^\top v + \tau_c^\top \omega)$), causal multi-scale wrench-velocity temporal encoding, Coulomb friction-cone slip invariants ($\Vert F_t \Vert / (\mu |F_n|)$), and leak-free episode threshold calibration (`calibrate_threshold_for_max_fpr`). |
| **V7-R5** | `piginot.robotics.live_coupling` | **Live Two-Way Simulator Coupling:** Zero-copy `ctypes.CFUNCTYPE` C-ABI and Python step callback (`LiveSimulatorCoupler.step_coupled`, `step_coupled_torch`, `make_c_abi_callback`) returning 6D reaction wrenches and $6\times 6$ linearized tangent stiffness matrices $K_t$ into MuJoCo/MJWarp/Newton ADMM loops each control step, with `stop_grad_residual=True` Jacobian isolation for stable differentiable policy training. |
| **V7-R6** | `piginot.benchmarks.scaling_3d` | **Large-Mesh `10k -> 1M`-Node Scaling & Memory Harness:** Deterministic benchmark (`run_surface_3d_scaling_benchmark`) sweeping `Surface3DEnergyOperator` (AB-UPT + Transolver++ + LRQ) up to `1,000,000` surface nodes across anchor and `decode_chunk_size` budgets, logging OS Peak RSS, `tracemalloc`, P50/P99 latency, chunk consistency drift, and TorchScript/ONNX export parity inside a SHA-256 receipt. |

---

## Agent Harness Capabilities Added in v0.7

Three new capabilities are registered in `piginot.harness.catalog` (discoverable via
`formamech agent capabilities` and MCP `discover_capabilities`):

- `fem.solve_visco_hysteresis_3d` — 3D finite-strain Prony viscoelastic + Mullins hysteresis solve with thermodynamic admissibility certification ($\dot{\mathcal{D}} \ge 0$, `min_det_f > 0`).
- `validate.identify_constitutive_pano` — Online `< 1 ms` PANO/CANO shear-modulus identification from sparse palpation strain/stress probes.
- `verify.surface_3d_scaling_benchmark` — Reproducible `Surface3DEnergyOperator` scaling, memory, chunk-drift, and runtime export parity receipt.

---

## Quickstarts

### 1. 3D Prony Viscoelastic + Mullins Hysteresis FEM (`V7-R1`)

```python
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.follower_3d import FollowerPressureBC3D
from piginot.fem.visco_hysteresis_3d import (
    MullinsDamageSpec,
    PronyBranch,
    ViscoHysteresis3DConfig,
    solve_3d_visco_hysteresis,
)
from piginot.geometry.mesh3d import make_brick_tet10_mesh

mesh = make_brick_tet10_mesh(24.0, 12.0, 5.0, nx=2, ny=2, nz=1)
mat = MaterialSpec(
    name="dragon-skin-30-visco",
    model="compressible_neo_hookean",
    properties={
        "elastic_modulus": Quantity(0.95, Unit.MPA).to_canonical(),
        "poisson_ratio": Quantity(0.46, Unit.ONE).to_canonical(),
    },
)
res = solve_3d_visco_hysteresis(
    mesh,
    mat,
    dirichlet_bcs=(
        DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
    ),
    load_history=(0.0, 0.5, 1.0, 0.5, 0.0),
    config=ViscoHysteresis3DConfig(
        prony_branches=(PronyBranch(0.22, 0.45), PronyBranch(0.14, 2.2)),
        mullins=MullinsDamageSpec(enabled=True, r_param=2.6, m_energy_density=0.0015),
        dt_s=0.12,
    ),
    follower_bcs=(FollowerPressureBC3D("cavity", ("z_max",), pressure_mpa=0.0018),),
)
assert res.converged and res.thermodynamically_admissible
```

### 2. Online ACP-CQR Adaptive Conformal CBF (`V7-R2`)

```python
import numpy as np
from piginot.robotics.adaptive_cbf import (
    AdaptiveConformalBarrierSpec,
    AdaptiveConformalCbfFilter,
    ConformalQuantileRegressor,
)

cqr = ConformalQuantileRegressor(alpha_target=0.05, n_centers=6)
cqr.fit_and_calibrate(x_train, y_res_train, x_dev, y_res_dev)

filt = AdaptiveConformalCbfFilter(
    barriers=(
        AdaptiveConformalBarrierSpec(
            name="q1_wall",
            safety_limit=0.145,
            base_margin=0.008,
            quantile_model=cqr,
        ),
    ),
    u_min=np.full(5, -0.85),
    u_max=np.full(5, 0.85),
    safe_hold_action=np.array([-0.35, 0.0, 0.0, 0.0, 0.0]),
    soft_ood_threshold=2.0,
    hard_ood_threshold=4.5,
)
```

### 3. Online 2-Probe PANO/CANO Tissue Identification (`V7-R3`)

```python
from piginot.operators.constitutive_op import PalpationProbe, PanoConstitutiveIdentifier

pano = PanoConstitutiveIdentifier()
id_res = pano.identify(
    (
        PalpationProbe(strain=0.02, measured_stress_pa=26.4),
        PalpationProbe(strain=0.05, measured_stress_pa=69.2),
    )
)
print(id_res.shear_modulus_pa, id_res.ogden_alpha, f"{id_res.inference_time_ms:.2f} ms")
```

### 4. Live Zero-Copy Two-Way Simulator Coupling (`V7-R5`)

```python
import numpy as np
from piginot.robotics.live_coupling import ContactGeomState, LiveSimulatorCoupler

coupler = LiveSimulatorCoupler(nodes_mm, elastic_stiffness_n_per_mm=1.55)
geom = ContactGeomState(name="ee", position_mm=(0.0, 0.0, 10.0), radius_mm=50.0)
step_out = coupler.step_coupled(
    np.zeros(6), np.zeros(6), (geom,), stop_grad_residual=True
)
c_callback = coupler.make_c_abi_callback((geom,), stop_grad_residual=True)
```

---

## Validation Evidence

All six v0.7 modules are validated end-to-end across **7 real-data & stress-test studies
(`R3-v2`, `R5-v2`, and `R6`–`R10`)** passing **26 / 26** target gates — documented with
D3.js + custom SVG figures and live animations in
[Real-Data Robotics Validation Studies (R1–R10)](robotics-studies.md).
