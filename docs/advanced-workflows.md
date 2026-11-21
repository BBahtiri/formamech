# Advanced workflows (P1–P6)

FormaMech provides six end-to-end physics-informed and FEM-guarded workflows
layered cleanly on top of its verified bundle, CAD, and compiler contracts. All
newly generated candidates, adapters, design reports, and trajectories carry
explicit `claim_level = "NOT_VERIFIED"` provenance until formally qualified.

## P1 — Hierarchical CAD mesh convergence & multi-level energy training

1. Import a 2D STEP or DXF profile and emit a verified `M0/M1/M2` `MeshHierarchy`:

```text
formamech cad import part.dxf --unit mm --mesh-size-mm 2.0 --hierarchy --output .formamech-runs/part-proj
formamech cad hierarchy .formamech-runs/part-proj --output .formamech-runs/part-hierarchy.json
```

2. Evaluate distinct-remesh geometric convergence across multiple mesh sizes:

```text
formamech cad convergence part.dxf --unit mm --mesh-sizes 3.0 1.5 0.75 \
  --output .formamech-runs/part-geom-conv.json
```

3. Activate hierarchical `M0 -> M1` energy training in a compiler config by
setting optional fields on `objective.energy` (`cross_mesh_weight`,
`local_physics_weight`, `quadrature_degree`, `use_deterministic_quadrature`).
Omitting them preserves exact v1 config hashes.

## P2 — Multi-CAD geometry tokens, OOD gate & low-rank adaptation

Adapt a pretrained `BoundaryEnergyOperator` checkpoint to a new CAD project
strictly within a budgeted low-rank step ceiling (`<= 10%` of from-scratch
steps by default), guarded by a fitted `GeometryOODDetector`:

```text
formamech cad adapt \
  --project .formamech-runs/part-proj \
  --checkpoint .formamech-runs/base-boundary-op \
  --config adapt-config.json \
  --ood-detector .formamech-runs/ood-detector.json \
  --output .formamech-runs/part-adapter
```

Unsupported topologies are refused with exit code `2`; accepted adapters are
saved as tamper-checked `adapter.safetensors` + canonical `adapter.json`.

## P3 — Certificate-driven active FEM & multi-fidelity learning

Rank and select the next `TRAIN`-only FEM acquisition batch deterministically
(pure standard library; refuses any `DEVELOPMENT` or `HOLDOUT` candidate and
forbids claiming `label_free` whenever acquired FEM targets supervise training):

```text
formamech active select \
  --plan examples/advanced_workflows/active_plan.json \
  --pool examples/advanced_workflows/active_train_pool.json \
  --output .formamech-runs/selected-batch.json
```

Programmatic multi-round orchestration with `FEMCostLedger` and
`multifidelity_energy_loss` is exposed via `formamech.active.run_active_fem_campaign`.

## P4 — CAD-level numerical trust report & live DWR adjoints

Attach a versioned `CADTrustReport` separating geometry, discretization,
discrete-surrogate, and statistical calibration status to any bundle prediction.
In `--mode certified` or `--mode exact`, live dual-weighted residual (DWR)
adjoints (`evaluate_cad_trust`) are evaluated automatically on the bundle's FEM
linearization:

```text
formamech predict bundles/notched-v1-cp-freeze \
  --request bundles/requests/notched-in-domain.json \
  --mode certified \
  --trust-report \
  --output .formamech-runs/notched-certified-trust.json
```

Pass `--trust-sidecar <path>` to bind persisted geometry/discretization
convergence reports; without qualified convergence and calibration evidence,
overall status fails closed to `DIAGNOSTIC_ONLY` (or `OUT_OF_ENVELOPE`).

## P5 — FEM-guarded differentiable design optimization

Run trust-region design optimization directly against any verified FormaMech
bundle and its FEM recipe — surrogate explores inside the executable envelope,
periodic FEM finite differences check gradient sign consistency, and the final
design is always independently verified by converged P2 FEM:

```text
formamech optimize bundles/notched-v1-cp-freeze \
  --problem examples/advanced_workflows/notched_design_problem.json \
  --output .formamech-runs/notched-opt-report.json
```

## P6 — Stateful finite-strain viscoelasticity & guarded rollout

Run a hash-bound trajectory rollout with thermodynamic admissibility checks and
tamper-checked `safetensors` checkpoints (`--checkpoint-out` / `--resume`):

```text
formamech stateful rollout \
  --artifact .formamech-runs/visco-artifact \
  --load-path examples/advanced_workflows/viscoelastic_load_path.json \
  --checkpoint-out .formamech-runs/visco-ckpt \
  --output .formamech-runs/visco-rollout.json
```

## Three-Tier Verification Funnel, Trust Receipts & Pluggable Backends

Every `predict` call can emit a canonical RFC 8785 Trust Receipt
(`formamech.trust-receipt-v1`) separating orthogonal `VerificationTier` from
release `ClaimLevel`, while enabling Tier 0 zero-FEM admissibility checks,
Tier 1 per-CAD conformal calibration, Tier 1.5 test-time physics LoRA, and
Tier 2 spectrally safe warm-start FEM across any registered `SolverBackend`
(`scikit-fem` or `torch-fem`):

```text
formamech calibrate bundles/notched-v1-cp-freeze \
  --scores cal-scores.json --alpha 0.05 \
  --output .formamech-runs/notched-cal.json

formamech predict bundles/notched-v1-cp-freeze \
  --request bundles/requests/notched-in-domain.json \
  --mode certified \
  --backend torch-fem \
  --tier0 \
  --cad-calibration .formamech-runs/notched-cal.json \
  --spectral-repair \
  --escalate \
  --receipt \
  --output .formamech-runs/verified-with-receipt.json
```

For portable user-owned CAD bundles, `--mode certified` and `--mode exact`
construct the discrete FEM problem directly from the bundle's embedded
`PortableProblem` specification without requiring an external recipe file.

## Verification & Trust Inspector UI, MCP Server & ONNX / FMI 3.0 Export

- **Interactive Inspector (`formamech[ui]`)**: launch the Streamlit/Plotly field,
  admissibility, FEM-diff, and downloadable dossier inspector:

```text
formamech ui --bundle bundles/notched-v1-cp-freeze --port 8501
```

- **MCP Server (`formamech mcp`)**: run the workspace-sandboxed JSON-RPC 2.0
  Model Context Protocol server exposing `inspect_bundle`, `inspect_cad_project`,
  `predict_verified`, `run_design_optimization`, and `verify_receipt`:

```text
formamech mcp list-tools --workspace .
formamech mcp serve --workspace . --max-dofs 100000
```

- **ONNX & FMI 3.0 Co-Simulation FMU Export (`formamech export`)**:

```text
formamech export fmu bundles/notched-v1-cp-freeze --output .formamech-runs/notched.fmu
formamech export onnx bundles/notched-v1-cp-freeze --output .formamech-runs/notched.onnx
```

## VVUQ Nonlinear Benchmark Suite (Levels 1–4)

Run the backend-agnostic ASME V&V 10 / NAFEMS nonlinear benchmark suite
(Level 1 MMS, Level 2 Cook's Membrane, Level 3a HGO arterial strip, Level 3b
non-conservative follower pressure cantilever, Level 3c axisymmetric pressurized
thick-walled cylinder, and Level 4 3D `TET10` / OOD conformal shift contracts):

```text
python -m campaigns.vvuq_suite
```

## Diagnostic CPU pilot harness

Run the reproducible CPU-sized pilot benchmark across all six features (writes
strictly to gitignored `.formamech-runs/pilots/`):

```text
python scripts/run_cpu_pilots.py
```
