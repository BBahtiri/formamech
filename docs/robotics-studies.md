# Real-Data Robotics Validation Studies (Studies R1–R10)

This report documents twelve reproducible, real-data engineering validation and
falsification studies exercising **FormaMech v0.6 (`V6-R1`–`V6-R6`, Studies `R1`–`R5`)**
and **FormaMech v0.7 (`V7-R1`–`V7-R6`, Studies `R3-v2`, `R5-v2`, and `R6`–`R10`)** across
tactile elastomers, dynamic friction-cone slip control, continuum surgical robots &
closed-loop CLIK, soft pneumatic hysteresis & online ACP-CQR CBF, real-to-sim deformable
manipulation & live two-way coupling, online patient-specific surgical tissue
identification, contact-rich industrial assembly jamming detection, and 1M-node scaling.

Every figure and live animation below is rendered directly from the real physical
study outputs using **D3.js + custom SVG + Playwright**
(`scripts/studies_robotics/render_d3_visuals.mjs`,
`scripts/studies_robotics/render_d3_v7_visuals.mjs`, and
`scripts/studies_robotics/render_linkedin_gifs.mjs`).

![FormaMech v0.6 Robotics Physics AI 5-in-1 Showcase — Touch, Continuum, Safe Soft Arm, Brain Tissue, and Robotic Assembly](assets/studies_robotics/linkedin/linkedin_hero_5in1.gif)

> **ASME V&V 10 & License Hygiene Guarantee:** Raw third-party datasets are fetched
> on demand into gitignored `.formamech-runs/datasets/robotics/` via
> `python scripts/studies_robotics/fetch_robotics_datasets.py` and are **never**
> committed to Git or bundled into wheels. Tier B ShareAlike corpora (`SPONGE`,
> `CC-BY-SA-4.0`) are used strictly for local evaluation and aggregate metrics/plots
> (zero raw files or derived weights exported). Numerical verification (`ClaimLevel`)
> and physical experimental validation (`ExperimentalValidationTier`) are reported
> strictly separately — including honest `REJECTED` / `COUPON_CALIBRATED` and OOD
> failure outcomes.

---

## Executive Summary Table (All Real Physical Datasets)

| Study | Physical Dataset(s) & License | Core FormaMech v0.6 Capabilities | Key Quantitative Physical Result | Real-Time Speed & Certification |
| --- | --- | --- | --- | --- |
| **R1 — Tactile Elastomer** | `feats-gelsight-mini` STEP (`MIT`) + `glowtact-gelsight-cnc` **3,997 CNC probes** (`CC-BY-4.0`, Tier A) | 3D `TET10` SDF contact FEM (`SPHERE`, `FLAT_PUNCH`, `BOX`, `WEDGE_EDGE`), `TET4` locking refusal, DiffMJX CFD, ASME V&V 10 hybrid, `active.test_planner`, `< 2 ms` tactile surrogate | Hybrid holdout RMSE **`0.687 N`** (vs. `2.266 N` raw FEM, **70% cut**); **`39.7%` fewer CNC probes** (`14` vs. `23.2` random); honest OOD tip shift (`2.62 -> 7.35 N`) | **`1.26 ms` (`796 Hz`, `1,493×` speedup)** · **ASME V&V 10 `VALIDATED`** · **AS6983 CERTIFIED (`95.0%` upper)** |
| **R2 — Continuum Robot** | `crl-ctcr-pose` **100,000 EM `SE(3)` poses** (`MIT`) + Gotelli 9-disk 4-tendon Nitinol geometry (`CC-BY-4.0`, Tier A) | Leak-free contiguous episode split, exact finite-rotation `SE(3)` Cosserat 9-disk tendon solver, 3D solid Capstan tendon FEM, FBG inverse force recovery | Blind holdout (`5,000` poses): **`0.710 mm`** mean 3D tip error (`p95 = 1.443 mm`); FBG inverse tip force error **`0.0003 N`** under `4 µε` noise; `55.8°` finite bend | **`0.53 ms` (`1,898 Hz`, `4,100×` speedup)** · **AS6983 CERTIFIED (`0.986 mm` bound)** |
| **R3 — Soft Pneumatic & MPC/CBF** | Joshi & Paik `0–200 kPa` × 3 reps (`CC-BY-4.0`, Tier A) + `SPONGE` 5-DoF hardware (`CC-BY-SA-4.0`, Tier B Eval-Only) | 3D `TET10` Nanson follower-pressure FEM, strictly passive `PortHamiltonianSurrogate`, `< 10 ms` `CondensedQpMpcController`, HOCBF `ConformalCbfSafetyFilter` | Joshi hybrid RMSE `3.23 -> 1.30 N` (honest `REJECTED / COUPON_CALIBRATED`); SPONGE blind 0.2s RMSE **`0.73°`** (`0.58°–1.53°` across unseen `50–200g` & `45°–90°`); **`0` CBF violations** vs. `24` unfiltered | **`0.47 ms` (`2,142 Hz` CPU — `45×` faster than `47 Hz` PINN-NMPC)** · **Passivity Verified** · `31` CBF + `7` `SAFE_HOLD_OOD` |
| **R4 — Surgical Brain Tissue** | Hyperelastic Human Brain 1–7 (`Zenodo 8095559 + 21793225`, **182 tissue cylinders**, 9 regions `85.9–434 Pa`, `CC-BY-4.0`, Tier A) | Ultra-soft kPa continuum similitude 3D `TET10` FEM, `V6-R5` AB-UPT (`anchors=32`) + Transolver++ (`adaptive_temperature=True`) `Surface3DEnergyOperator`, TorchScript runtime export | 182 human brain specimens parsed across 7 donors; honest donor-split `REJECTED / COUPON_CALIBRATED` (`210.5 Pa` hybrid vs. `237.1 Pa` raw across unlabeled regions); TorchScript parity **`0.0`** | **`1.88 ms` (`346×` faster than `652 ms` FEM)** · **AS6983 CERTIFIED (`100%` upper coverage)** |
| **R5 — NIST Assembly Co-Design** | `REASSEMBLE` NIST Task Board #1 (**148,444 30 Hz Franka frames**, 15 episodes, 499 segments, `CC-BY-4.0`, Tier A) | Leak-free episode jamming detector, 6D wrench bridge, FEM-certified 3D SIMP topology optimization, watertight OpenCASCADE B-Rep STEP, MuJoCo MJCF + ROS 2 exports | Blind episode jamming/failure detection: **AUROC `0.619`**, Recall `56.3%`; compliant finger under real `35.32 N` NIST peak load **FEM-CERTIFIED** (`0.104 mm` tip disp); **watertight STEP** (`0` open edges) | **3D SIMP `CERTIFIED` (`SINGLE_POINT_VERIFIED`)** · **Watertight B-Rep STEP + MJCF + ROS 2 `120 Hz`** |

---

## Study R1 — GelSight Mini 3D `TET10` Tactile Elastomer Contact Digital Twin

- **Real datasets:** `feats-ai/feats` exact `gelsight_mini.step` (`MIT`, `25.25 × 4.0 × 20.75 mm`, `57` B-Rep faces) and `yxma/glowtact-gelsight-tactile` (`CC-BY-4.0`, **3,997 real CNC `(x, y, z, F)` physical probes** across `round`, `quad`, `triangle`, and `five_star` indenters plus `14,737` 85 Hz load-cell baseline samples).
- **Physical verification highlights:**
  - FormaMech's incompressibility locking guard refuses linear `TET4` at $\nu = 0.48$ and requires quadratic `TET10`.
  - All 4 rigid 3D SDF indenter primitives (`SPHERE` with Coulomb friction $\mu = 0.35$, `FLAT_PUNCH`, `BOX`, `WEDGE_EDGE`) converge cleanly via exact variational contact Hessians and PSD curvature projection; DiffMJX straight-through Contacts-From-Distance produces `0.0 N` hard forward force on separated nodes with `+17.65 N/mm` smooth backward gradient.
  - Calibrated elastomer modulus `E_cal = 0.8677 MPa`; ASME V&V 10 hybrid grounding reaches **`VALIDATED` (`HOLDOUT_EXPERIMENTALLY_VALIDATED`)**, reducing blind holdout RMSE from `2.266 N` (raw FEM) to **`0.687 N`**.
  - FormaMech's `active.test_planner` reaches the `0.85 N` target holdout RMSE in **`14` CNC probes vs. `23.2`** for 10-seed random sampling (**`39.7%` physical test reduction**).
  - The `< 2 ms` tactile surrogate runs at **`1.26 ms` median (`795.5 Hz` CPU, `1,493×` faster than 3D FEM)** with **AS6983 CERTIFIED** bounds (`91.7%` two-sided, `95.0%` conservative upper coverage on 120 blind probes), while transparently surfacing cross-geometry OOD shift (`round: 2.62 N` -> `triangle: 7.09 N`, `five_star: 7.35 N`).

![Study R1 — GelSight Mini 3D TET10 Elastomer Contact Digital Twin](assets/studies_robotics/fig_r1_tactile_elastomer.png)

![Study R1 Live Animation — 3D TET10 Indentation, Tangential Shear Slip & 8x8 Taxel Map](assets/studies_robotics/study_r1_tactile_live.gif)

---

## Study R2 — Continuum & Tendon-Driven Robot Digital Twin

- **Real datasets:** `CRL-Dataset-CTCR-Pose` (`MIT`, **100,000 real EM-tracked `SE(3)` poses** of a 3-tube Nitinol concentric-tube continuum robot) and Gotelli et al. (`CC-BY-4.0`, 9-disk 4-tendon Nitinol continuum robot hardware geometry `L = 540 mm`, `9` spacer disks, `d_c = 37.5 mm`, 10 hardware STLs).
- **Physical verification highlights:**
  - Partitioned strictly into **50 non-overlapping contiguous trajectory episodes** (`30 train / 10 dev / 10 blind holdout`) via `split_episodes_leak_free` (zero random-frame temporal leakage).
  - On `5,000` blind physical holdout poses, FormaMech achieves **`0.710 mm` mean 3D tip error** (`0.472 mm` mid-tube error, `p95 = 1.443 mm`) at **`0.527 ms` median (`1,898 Hz` CPU)** with **AS6983 CERTIFIED** `0.986 mm` conservative error bound.
  - FormaMech's exact finite-rotation `SE(3)` Cosserat 9-disk 4-tendon solver (`solve_3d_tendon_continuum_backbone`) converges across a full 3D orbital bending sweep up to `55.78°`, and Gauss-Newton FBG inverse contact estimation recovers an unknown `[0.18, -0.24] N` tip force to **`0.0003 N` error** under `4 µε` optical noise.
  - Full 3D solid routed-tendon FEM with Capstan friction (`mu = 0.15`) converges (`18.0 mm` tip displacement, `5.42 mm` tendon stroke, `11.33 N` Capstan-attenuated tip tension).

![Study R2 — Continuum & Tendon-Driven Robot Digital Twin](assets/studies_robotics/fig_r2_continuum_tendon_twin.png)

![Study R2 Live Animation — 9-Disk 4-Tendon SE(3) Cosserat Continuum Backbone & FBG Strain](assets/studies_robotics/study_r2_continuum_live.gif)

---

## Study R3 — Soft Pneumatic Follower-Pressure FEM & Real-Time Port-Hamiltonian QP-MPC + Conformal CBF

- **Real datasets:** Joshi & Paik soft pneumatic actuator pulling force (`Zenodo 3627901`, `CC-BY-4.0`, Tier A, `0–200 kPa × 3` physical replicates) and SPONGE 5-DoF articulated soft pneumatic robot (`RepoLUH 123456789/19854`, `CC-BY-SA-4.0`, **Tier B local evaluation only** — zero raw files or trained weights committed).
- **Physical verification highlights:**
  - Full 3D `TET10` Nanson follower-pressure FEM (`solve_3d_follower_pressure`) converges smoothly across `0–200 kPa` (`15.66 mm` large bellows bending, `min_det_f = 0.959`). Kennedy–O'Hagan hybrid grounding reduces holdout RMSE from `3.233 N` to `1.296 N`, while ASME V&V 10 honestly reports `REJECTED / COUPON_CALIBRATED` because physical bellows hysteresis exceeds the tight 3-replicate spread.
  - Trained strictly on SPONGE `0° / 0g` hardware identification, FormaMech's **strictly passive `PortHamiltonianSurrogate`** (`min eig(D) > 0`, `min eig(M^{-1}) > 0` verified) achieves **`0.729°`** 0.2s rollout RMSE in-distribution and **`0.58°–1.53°`** strictly blind across unseen `50g/100g/200g` payloads and `45°/90°` gravity orientations.
  - Closed-loop **Condensed QP-MPC + High-Order AS6983 Conformal CBF (`q_1 + tau_b qd_1 <= 0.145 rad`)** runs at **`0.467 ms` median (`2,142 Hz` CPU — `45×` faster than the `47 Hz` PINN-NMPC baseline)** with **`0` safety barrier violations** (`31` smooth `CBF_MODIFIED` braking steps + `7` `SAFE_HOLD_OOD` payload disturbance holds), compared to **`24` barrier violations** for unfiltered MPC!

![Study R3 — Soft Pneumatic Follower-Pressure FEM & Port-Hamiltonian QP-MPC + Conformal CBF](assets/studies_robotics/fig_r3_pneumatic_soft_mpc.png)

![Study R3 Live Animation — 5-DoF Soft Arm Kinematics, Conformal CBF Safety Wall & FSM State](assets/studies_robotics/study_r3_soft_mpc_cbf_live.gif)

---

## Study R4 — Ultra-Soft Human Brain 3D Hyperelastic FEM & AB-UPT/Transolver++ Runtime Surrogate

- **Real datasets:** Hyperelastic Human Brain 1–7 (`Zenodo 8095559`, `CC-BY-4.0`, **182 real human brain tissue cylinders** across 7 donors) and published regional Neo-Hookean shear moduli (`Zenodo 21793225`, `CC-BY-4.0`, 9 anatomical regions spanning `85.85 Pa` corpus callosum to `434.00 Pa` cortex).
- **Physical verification highlights:**
  - Ultra-soft kPa-scale (`85–434 Pa`, `nu = 0.48`) 3D `TET10` hyperelastic FEM solves seamlessly via FormaMech's exact continuum similitude helper (`_similitude.py`), matching the experimental median/IQR compression envelope across `182` real brain specimens.
  - On donor-split ASME V&V 10 (`HBE_01–04` calibration vs. `HBE_05–07` blind holdout), FormaMech honestly reports `REJECTED / COUPON_CALIBRATED` (`210.5 Pa` hybrid vs. `237.1 Pa` raw) because unlabeled biological variability spans a `5×` stiffness ratio across brain regions.
  - FormaMech's `V6-R5` **`Surface3DEnergyOperator`** (`adaptive_temperature=True`, `anchor_count=32`, `low_rank_query_dim=16`, `decode_chunk_size=128`) exports cleanly via `export_surface_3d_runtime` (TorchScript parity max diff = **`0.0`**), evaluates in **`1.88 ms`** vs. `652 ms` FEM (**`346×` speedup**), and is **AS6983 CERTIFIED (`100%` conservative upper coverage)** against surgical tissue-tearing thresholds.

![Study R4 — Ultra-Soft Human Brain 3D Hyperelastic FEM & AB-UPT/Transolver++ Surrogate](assets/studies_robotics/fig_r4_surgical_tissue.png)

![Study R4 Live Animation — Human Brain Cortex vs Corpus Callosum Surgical Palpation](assets/studies_robotics/study_r4_surgical_tissue_live.gif)

---

## Study R5 — REASSEMBLE NIST Assembly Jamming Detection & Verified 3D Compliant Gripper Co-Design

- **Real dataset:** `REASSEMBLE` (`robot-lev/reassemble`, `CC-BY-4.0`, Tier A — **148,444 real 30 Hz Franka robot frames** across 15 episodes and 499 NIST Assembly Task Board #1 segments: `434` successful vs. `65` failed/jammed).
- **Physical verification highlights:**
  - Split strictly by `episode_index` (`9 train / 3 dev / 3 blind holdout`) via `split_episodes_leak_free`; a physics-based contact-chatter/torque/gripper detector calibrated strictly on training episodes achieves **AUROC `0.619`** and `56.3%` recall on completely unseen teleoperated assembly episodes.
  - Real peak insertion contact wrenches (`35.32 N`, `3.284 Nm`) are imported through `import_simulator_wrench_trace` (SHA-256 verified) and applied directly to a 3D compliant robotic gripper finger (`54 × 14 × 14 mm`, PA12 nylon).
  - FormaMech's `optimize_3d_topology_simp` finishes in **`CERTIFIED` (`SINGLE_POINT_VERIFIED`)** with exact certified FEM compliance `3.585 mJ` and tip displacement `0.1044 mm`, synthesizes a **watertight OpenCASCADE B-Rep STEP CAD solid** (`10,387.3 mm³`, **`0` open boundary edges**, **`0` non-manifold edges**, preserving an exact cylindrical mounting bore datum), and exports **MuJoCo MJCF compliance** and **ROS 2 `120 Hz` deployment manifests** with cryptographic SHA-256 receipts.

![Study R5 — REASSEMBLE NIST Assembly Jamming Detection & Verified 3D Compliant Gripper Co-Design](assets/studies_robotics/fig_r5_nist_assembly_codesign.png)

![Study R5 Live Animation — 30 Hz NIST Jamming Monitor & 3D SIMP Gripper Topology Evolution](assets/studies_robotics/study_r5_nist_jamming_codesign_live.gif)

---

## FormaMech v0.7 Validation & Stress-Test Scorecard (`R3-v2`, `R5-v2`, `R6`–`R10`)

FormaMech v0.7 directly closes every physical and statistical gap surfaced by `R1`–`R5`
and passes **26 / 26** target gates across seven real-data and adversarial studies:

| Study | Physical Dataset(s) & License | Core FormaMech v0.7 Capabilities | Key Quantitative Physical Result | Gate Verdict & Speed |
| --- | --- | --- | --- | --- |
| **R3-v2 — Hysteresis & ACP-CQR** *(Upgrade)* | Soft Pneumatic Hysteresis (`Zenodo 15584329`, **240,001 samples @ 30 Hz**, `CC-BY-4.0`) + Joshi & Paik (`Zenodo 3627901`) + `SPONGE` (`CC-BY-SA-4.0`, Tier B Eval-Only) | `V7-R1` 3D `TET10` Prony + Mullins FEM ($\dot{\mathcal{D}} \ge 0$), `VelocityResidualHysteresisOperator`; `V7-R2` online `AdaptiveConformalCbfFilter` (`ConformalQuantileRegressor` + Gibbs–Candès ACP) | Blind Cycle-4 hysteresis RMSE cut **`51.7%`** (`0.512 -> 0.247 mm`); Joshi & Paik upgraded **`REJECTED -> VALIDATED`** (`1.134 N`); `SPONGE` `0g->200g` shift cuts braking gap **`33.5%`** at **`0` violations** & **`0` `SAFE_HOLD_OOD` trips** (vs. `70` static) | **5 / 5 PASS** · **`0.47 ms` (`2,118 Hz`)** · **ASME V&V 10 `VALIDATED`** |
| **R5-v2 — Energy-Residual Jamming** *(Upgrade)* | `REASSEMBLE` NIST Task Board #1 (`robot-lev/reassemble`, **428,683 frames @ 30 Hz**, 45 episodes, 1,463 segments, `CC-BY-4.0`, Tier A) | `V7-R4` `EnergyResidualContactEstimator` (36D joint-vs-EE Port-Hamiltonian power dissipation + Whitney moment + Coulomb slip + multi-scale Temporal-Bundle 1D-Conv, strict leak-free episode split) | Blind holdout **AUROC `0.619 -> 0.908`** (`+0.289` on 45-ep; **`0.879`** on identical 15-ep v0.6 split!), **Recall `56.3% -> 80.0%`**, **FPR `36.4% -> 9.96%`**, Precision doubled (`27.3% -> 51.9%`) | **3 / 3 PASS** · **`0.019 ms` / seg** (`0.0047 ms` / 30 Hz frame) |
| **R6 — Dynamic Tactile Slip** | Meta Sparsh TacBench (`facebook/gelsight-force-estimation`, **102,628 frames**, 1,513 trajectories, `CC-BY-NC-4.0`, Tier C Noncommercial Eval-Only) | `V6-R2` 3D `TET10` coupled normal+shear contact FEM (`SPHERE`, `FLAT_PUNCH`, `WEDGE_EDGE`) + `V7-R4` Coulomb friction-cone slip-onset estimator + AS6983 conformal bounds | Blind holdout (`19,727` frames / `302` unseen trajectories): 3-axis force RMSE **`0.0451 N`** (`< 0.5 N`), slip-onset **AUROC `0.9998`** (physics cone alone `0.9945`), **AS6983 `95.18%` coverage** | **4 / 4 PASS** · **`0.421 ms` (`2,376 Hz` CPU)** |
| **R7 — Real-to-Sim Deformable Coupling** | `GeorgiaTech/PhysCoRe` & `PhysTwin` (8 real 3-camera RGB-D tracked manipulation episodes, `MIT` + `CC-BY-4.0`, Tier A) | `V7-R1` 3D `TET10` Prony+Mullins FEM calibration + multi-hand SE(3) viscoelastic spring-mesh twin + `V7-R5` `LiveSimulatorCoupler` (zero-copy C-ABI + `stop_grad_residual`) | Blind holdout mean 3D symmetric Chamfer **`3.635 mm`** (median **`2.853 mm`**, `-34.1%` vs. elastic `5.517 mm`); two-way vs. one-way tool divergence **`18.1 mm` mean / `335.8 mm` max**; C-ABI parity **`0.0` exact** | **3 / 3 PASS** · **`4.79 ms` / step** · `stop_grad` verified |
| **R8 — Online Surgical Palpation** | Hyperelastic Human Brain 1–7 (`Zenodo 8095559 + 21793225`, **181 real specimens**, 7 donors, `CC-BY-4.0`, Tier A) + 3D `TET10` liver phantom | `V7-R3` online 2-probe `PanoConstitutiveIdentifier` + multiplicative log-stress AS6983 tearing bounds + `localize_shallow_inclusion` | Blind unseen donors (`HBE_05–07`, `n=76`): stress RMSE slashed **`90.45%` (`210.5 -> 20.11 Pa`**, hybrid V&V 10 **`13.73 Pa`**), upgraded **`REJECTED -> VALIDATED`**, **`100.0%` AS6983 coverage**; liver tumor localized to **`0.09 mm`** | **4 / 4 PASS** · **`0.463 ms` 2-probe ID** (`1.012 ms` P95) |
| **R9 — Closed-Loop Continuum CLIK** | Gotelli 9-disk 2-section 5-DOF Nitinol geometry (`L=540 mm`, `CC-BY-4.0`) + `CRL-CTCR` workspace (`MIT`) | `V7-R3` `OperatorClikController` (Cosserat-FEM-distilled neural shape Jacobian + nullspace obstacle repulsion) + `V7-R2` ACP-CQR CBF under `4 µε` FBG noise & `0.25 N` unknown tip load | Closed-loop 3D figure-8 tip tracking RMSE **`0.231 mm`** (max `0.702 mm`, `< 1.5 mm` gate), **`0` backbone collisions** (`min clearance 15.02 mm`) | **3 / 3 PASS** · **`0.985 ms` (`1,015 Hz` CPU)** |
| **R10 — Adversarial Stress & 1M Scaling** | Injected `det(F)<=0`, `96%` crush, `100×` follower/visco bursts, stiff impact, cross-domain OOD, `10k -> 1,000,000`-node 3D surface meshes | `V7-R6` `run_surface_3d_scaling_benchmark` (`Surface3DEnergyOperator` AB-UPT + Transolver++ + LRQ), `onnx_export_3d`, fail-closed FEM & ACP-CQR refusal contracts | **`5/5 = 100%`** adversarial FEM refusal; **`20/20 = 100%`** severe OOD refusal (`0%` false refusals, `8/8` soft-shift expansions); **`1M` nodes in `831.4 ms`** (`1.20M` nodes/s, Peak RSS **`454.6 MB`**, slope **`0.842`**, TorchScript diff **`0.0`**) | **4 / 4 PASS** · **Chunk drift `2.79e-9`** |

---

### Study R3-v2 — Rate-Dependent Pneumatic Hysteresis (`V7-R1`) & Online ACP-CQR CBF (`V7-R2`)

- **Gap closed:** In v0.6, Study R3 returned an honest `REJECTED / COUPON_CALIBRATED` verdict on Joshi & Paik because elastic follower-pressure FEM cannot model loading/unloading hysteresis, and static split-conformal CBF forced binary `SAFE_HOLD_OOD` trips under `200g` payload shifts.
- **Physical verification highlights:**
  - Full 3D `TET10` Prony viscoelastic + Ogden–Roxburgh Mullins FEM (`solve_3d_visco_hysteresis`) converges across a complete loading/unloading follower-pressure cycle (`min_det_f = 0.9785`, `min_eta = 0.6386`, `1.392 mm` midpoint hysteresis loop gap) with certified nonnegative thermodynamic dissipation ($\dot{\mathcal{D}}_k \ge 0$ at every step, $\sum \Delta \mathcal{D} = 0.47864\ \mathrm{N{\cdot}mm}$).
  - On **Zenodo 15584329** (`240,001` real 30 Hz pneumatic samples across 4 cycles; strict temporal split Cycles 1–2 train / Cycle 3 dev / Cycle 4 blind holdout), `VelocityResidualHysteresisOperator` cuts blind Cycle-4 RMSE **`51.74%`** (`0.5124 -> 0.2473 mm`) and reaches **ASME V&V 10 `VALIDATED` (`HOLDOUT_EXPERIMENTALLY_VALIDATED`)**; **Joshi & Paik (`Zenodo 3627901`)** upgrades from `REJECTED` (`1.296 N`) to **`VALIDATED` (`1.1344 N`)**.
  - On **SPONGE 5-DoF** under a `0g -> 200g` step payload shift (Tier B local evaluation only), `AdaptiveConformalCbfFilter` (`ConformalQuantileRegressor` + Gibbs–Candès ACP) tightens calm-regime margin to `0.0088 rad` (vs. static `0.036 rad`) and dynamically expands to `0.0189 rad` upon payload shift — cutting conservative braking distance by **`33.46%`** at **`0` barrier violations** (vs. `63` unfiltered) and **`0` `SAFE_HOLD_OOD` trips** (vs. `70` for static CBF) at **`0.472 ms` (`2,118 Hz`)**.

![Study R3-v2 — Pneumatic Prony/Mullins Hysteresis (V7-R1) & Online ACP-CQR CBF (V7-R2)](assets/studies_robotics/fig_r3_v2_hysteresis_acp_cbf.png)

![Study R3-v2 Live Animation — Cycle-4 Hysteresis Loop & Online ACP-CQR Margin Expansion Under 0g->200g Shift](assets/studies_robotics/study_r3_v2_hysteresis_acp_cbf_live.gif)

---

### Study R5-v2 — Port-Hamiltonian Power-Residual & Temporal-Bundle Jamming Estimator (`V7-R4`)

- **Gap closed:** v0.6's 5-scalar chatter threshold achieved `0.619` AUROC and `56.3%` recall at `36.4%` FPR because raw force/torque spikes confuse nominal snap-fit seating with mechanical jamming.
- **Physical verification highlights:**
  - Unpacks the full 36D Franka proprioceptive state (`joint_pos`, `joint_vel`, `joint_efforts`, `gripper`, `ee_pose`, `ee_twist`) + 6-axis F/T + commanded `action` across **3 shards (`428,683` 30 Hz frames, `45` physical episodes, `1,463` segments)** of `robot-lev/reassemble` (`CC-BY-4.0`).
  - Computes exact segment-tared joint-vs-EE mechanical power dissipation $r_P(t) = \tau_J^\top \dot{q} - (F_c^\top v + \tau_c^\top \omega)$, Whitney peg-in-hole jamming moment ratios $\Vert M_{xy} \Vert / |F_z|$, Coulomb friction-cone ratios $\Vert F_t \Vert / (\mu |F_n|)$, and causal phase+tail wrench-velocity-power bundles inside a multi-scale 1D-Conv + Port-Hamiltonian ensemble calibrated strictly on DEV episodes.
  - On the leak-free **45-episode split (`27 train / 9 dev / 9 blind holdout`, 296 holdout segments)**, blind holdout **AUROC jumps from `0.6193 -> 0.9084` (`+0.289`)**, **Recall lifts `56.3% -> 80.0%`**, and **FPR drops `3.65×` (`36.4% -> 9.96%`)** at `0.019 ms`/segment (`0.0047 ms`/frame for live 30 Hz PHNN monitoring). Evaluated on the **identical 15-episode `file-000` v0.6 split (`148,444` frames)**, blind holdout AUROC jumps **`0.6193 -> 0.8788` (`+0.260`)**.

![Study R5-v2 — REASSEMBLE Port-Hamiltonian & Multi-Scale Temporal-Bundle Jamming Estimator (V7-R4)](assets/studies_robotics/fig_r5_v2_contact_diagnostics.png)

---

### Study R6 — Dynamic 3-Axis Tactile Force & Friction-Cone Slip-Onset Control (`V6-R2` + `V7-R4`)

- **Real dataset:** Meta Sparsh TacBench (`facebook/gelsight-force-estimation`, `CC-BY-NC-4.0`, **Tier C noncommercial local evaluation only** — `102,628` frames across `1,513` coupled normal+shear trajectories covering `sphere`, `flat`, and `sharp` indenters; zero raw files or trained weights committed).
- **Physical verification highlights:**
  - All `9` 3D `TET10` coupled normal+shear contact FEM probes (`SPHERE`, `FLAT_PUNCH`, `WEDGE_EDGE`) converge with Coulomb stick-slip traction tracking.
  - On a strict leak-free trajectory split (`908 train / 303 dev / 302 blind holdout = 19,727 unseen frames`), FormaMech achieves **`0.0451 N` 3-axis force RMSE** (`Fx 0.0457 N`, `Fy 0.0490 N`, `Fz 0.0401 N` — `11×` better than the `< 0.50 N` gate), **`0.9998` blind slip-onset AUROC** (`99.96%` recall @ `2.05%` FPR; pure Coulomb cone ratio alone achieves `0.9945`), **`95.18%` AS6983 upper coverage** (`+0.0763 N` shift), at **`0.421 ms` median latency (`2,376 Hz` CPU)**.

![Study R6 — Dynamic 3-Axis Tactile Force & Friction-Cone Slip-Onset Control](assets/studies_robotics/fig_r6_dynamic_tactile_slip.png)

![Study R6 Live Animation — Live Coulomb Friction Cone & Incipient Slip Detection at 2,376 Hz](assets/studies_robotics/study_r6_tactile_slip_live.gif)

---

### Study R7 — Real-to-Sim Deformable Manipulation & Live Two-Way Coupling (`V7-R1` + `V7-R5`)

- **Real dataset:** `GeorgiaTech/PhysCoRe` & `PhysTwin` (`MIT` + `CC-BY-4.0`, Tier A — 8 real 3-camera RGB-D tracked 3D deformable manipulation episodes spanning elastomers, ropes, cloth, and plush toys).
- **Physical verification highlights:**
  - Calibrates a 3D `TET10` Prony viscoelastic + Mullins FEM stiffness baseline (`min_det_f = 0.9994`, $\dot{\mathcal{D}} \ge 0$) and fits a multi-hand `SE(3)` Kabsch + Prony viscoelastic spring-mesh twin strictly on the `[0:70%]` episode prefix.
  - Across all 8 real manipulation episodes, blind holdout 3D symmetric Chamfer error averages **`3.635 mm`** (median **`2.853 mm`**, `< 5.0 mm` gate; **`1.34 mm`** on `double_stretch_plastic`, **`1.94 mm`** on `double_squeeze_plastic`, **`1.99 mm`** on `single_push_rope`) — **`34.11%` better** than the undamped pure-elastic baseline (`5.517 mm`).
  - Exercises `LiveSimulatorCoupler` (`V7-R5`) directly on real PhysCoRe manipulation trajectories, explicitly quantifying two-way reaction+stiffness coupled vs. one-way trajectory divergence (**`18.08 mm` mean / `335.78 mm` max**) while verifying **exact `0.0` zero-copy C-ABI callback parity** and **`stop_grad_residual=True` Jacobian isolation**.

![Study R7 — PhysCoRe Deformable Manipulation & Live Two-Way Coupling](assets/studies_robotics/fig_r7_physcore_twoway_coupling.png)

![Study R7 Live Animation — 3D Deformable Elastomer Twin & Two-Way Reaction-Wrench Divergence](assets/studies_robotics/study_r7_physcore_twoway_live.gif)

---

### Study R8 — Online Adaptive 2-Probe Surgical Palpation & Regional Tissue ID (`V7-R3`)

- **Real dataset:** Hyperelastic Human Brain 1–7 (`Zenodo 8095559` + `21793225`, `CC-BY-4.0`, Tier A — **181 real human brain specimens** across 7 donors and 9 anatomical regions spanning `85.85–434.0 Pa`) plus a 3D `TET10` liver phantom with a shallow stiff tumor inclusion.
- **Physical verification highlights:**
  - Resolves Study R4's honest `REJECTED` donor split (`210.5 Pa` global modulus RMSE across unlabeled brain regions) by running **2 online palpation probes** (`2%` & `5%` strain) through `PanoConstitutiveIdentifier` in **`0.463 ms` median (`1.012 ms` P95, `< 10 ms` gate)** to infer patient- and site-specific Ogden parameters $(\mu, \alpha)$ spanning `78.6–2,460.5 Pa`.
  - On the strictly blind unseen donors (`HBE_05–HBE_07`, `n = 76` specimens), 10%-strain stress RMSE drops **`90.45%` (`210.5 -> 20.11 Pa`**, full-curve RMSE **`10.78 Pa`**), **ASME V&V 10 upgrades from `REJECTED` to `VALIDATED` (`HOLDOUT_EXPERIMENTALLY_VALIDATED`, `13.73 Pa`)**, and multiplicative log-stress **AS6983 tearing-bound coverage reaches `100.0%` (`76/76`)**.
  - On the 3D `TET10` liver phantom (`converged=True`, `min_det_f = 0.948`), robotic 5×5 palpation localizes the embedded shallow tumor to **`0.09 mm`** accuracy (`2.82×` stiffness contrast).

![Study R8 — Online 2-Probe PANO/CANO Surgical Tissue Identification (V7-R3)](assets/studies_robotics/fig_r8_adaptive_surgical_palpation.png)

---

### Study R9 — Closed-Loop Soft Continuum Shape Control: Operator CLIK + Obstacle CBF (`V7-R3` + `V7-R2`)

- **Real geometry & workspace:** Gotelli 9-disk 2-section redundant 5-DOF Nitinol continuum robot (`L = 540 mm`, `CC-BY-4.0`, anchored to exact `SE(3)` Cosserat FEM) cross-checked against `CRL-Dataset-CTCR-Pose` (`MIT`, `189.8 mm` workspace span).
- **Physical verification highlights:**
  - Distills exact Cosserat FEM equilibrium solutions (`2.225 s` / solve) into a smooth neural shape operator (`1.282 mm` open-loop blind tip RMSE) and closes the shape-space Jacobian loop with `OperatorClikController` + nullspace spherical-obstacle repulsion (`optic_nerve`, `aorta_branch`) + `AdaptiveConformalCbfFilter`.
  - Under simulated `4 µε` FBG optical noise (`0.102 mm` equivalent tip noise) and an unmodeled `0.25 N` lateral tip disturbance, closed-loop 3D figure-8 tip tracking achieves **`0.231 mm` RMSE** (`0.702 mm` peak, `< 1.5 mm` gate), **`0` mid-backbone obstacle collisions** (`min clearance 15.02 mm`), at **`0.985 ms` median latency (`1,015 Hz` CPU, `< 2 ms` gate)**.

![Study R9 — Continuum Operator CLIK + Obstacle CBF Control (V7-R3 + V7-R2)](assets/studies_robotics/fig_r9_continuum_clik_cbf.png)

![Study R9 Live Animation — 9-Disk 2-Section Continuum Backbone Tracking Figure-8 Around Obstacles at 1,015 Hz](assets/studies_robotics/study_r9_continuum_clik_live.gif)

---

### Study R10 — Adversarial Stress, OOD Refusal & 1M-Node Scaling Audit (`V7-R6`)

- **Purpose:** Dedicated falsification and industrial-scaling audit proving that FormaMech fails safely (`REFUSED` / `SAFE_HOLD_OOD`) rather than silently extrapolating, and documenting exact CPU latency and peak RSS scaling up to **1,000,000 3D surface nodes**.
- **Audit highlights:**
  - **Adversarial FEM falsification (`5 / 5 = 100%` refusal recall):** Cleanly refuses inverted elements (`det(F) <= 0`), `96%` extreme crush, `100×` follower-pressure burst, visco-hysteresis inadmissible burst, and stiff through-penetration contact impact.
  - **Cross-domain OOD refusal (`20 / 20 = 100%` recall, `0.0%` false refusal):** Refuses all 20 severe OOD queries via fail-closed `SAFE_HOLD_OOD`, accepts all 20 nominal ID cases, and dynamically expands margins on `8 / 8` soft-shift regimes.
  - **10k → 1,000,000-node `Surface3DEnergyOperator` scaling (8-thread CPU):** Scales from `17.01 ms` (`309.1 MB` RSS @ `10k`) → `88.01 ms` (`344.4 MB` @ `100k`) → `407.49 ms` (`367.7 MB` @ `500k`) → **`831.42 ms` (`1.20M` nodes/s, Peak RSS `454.6 MB` @ `1,000,000` nodes)** — adding only **`+145.5 MB` RSS for a `100×` mesh scale increase** (sub-linear log-log latency slope **`0.842`**, chunk consistency drift **`2.79e-9`**, TorchScript runtime parity **`0.0` exact**).

![Study R10 — Adversarial Stress, OOD Refusal & 1M-Node Surface3DEnergyOperator Scaling Audit (V7-R6)](assets/studies_robotics/fig_r10_stress_ood_scaling.png)

---

## Reproducing All v0.6 & v0.7 Studies & D3.js Visuals

```bash
# 1. Fetch all real open-source physical datasets on demand (into gitignored .formamech-runs/)
python scripts/studies_robotics/fetch_robotics_datasets.py

# 2. Run v0.6 studies (R1-R5) and v0.7 studies (R3-v2, R5-v2, R6-R10) on CPU
python scripts/studies_robotics/study_r1_tactile_elastomer.py
python scripts/studies_robotics/study_r2_continuum_tendon_twin.py
python scripts/studies_robotics/study_r3_pneumatic_soft_mpc.py
python scripts/studies_robotics/study_r4_surgical_tissue.py
python scripts/studies_robotics/study_r5_nist_assembly_codesign.py
python scripts/studies_robotics/study_r3_v2_hysteresis_acp_cbf.py
python scripts/studies_robotics/study_r5_v2_contact_diagnostics.py
python scripts/studies_robotics/study_r6_dynamic_tactile_slip.py
python scripts/studies_robotics/study_r7_physcore_twoway_coupling.py
python scripts/studies_robotics/study_r8_adaptive_surgical_palpation.py
python scripts/studies_robotics/study_r9_continuum_clik_cbf.py
python scripts/studies_robotics/study_r10_stress_ood_scaling_audit.py

# 3. Render publication PNGs and animated GIFs via D3.js + custom SVG + Playwright
node scripts/studies_robotics/render_d3_visuals.mjs
python scripts/studies_robotics/build_gifs.py
node scripts/studies_robotics/render_d3_v7_visuals.mjs
python scripts/studies_robotics/build_v7_gifs.py
```

Canonical scrubbed result summaries are tracked under `docs/assets/studies_robotics/study_r*_summary.json`.
