# AGENTS.md — Operating Guide for Autonomous AI Agents Working with FormaMech

FormaMech is a pure-MIT, cryptographically verified solid mechanics and physics-AI platform designed for **direct autonomous AI-agent operation**.

## 1. Quickest Entry Points for Agents

### Option A — Model Context Protocol (MCP 2024-11-05 over stdio)
```bash
formamech mcp serve --workspace /path/to/workspace --max-dofs 150000
```
- **Tools:** 18 tools including `discover_capabilities`, `describe_capability`, `invoke_capability`, `session_status`, `describe_state_machines`, plus direct CAD/FEM/optimization/verification tools.
- **Resources:** `formamech://capabilities`, `formamech://state-machines`, `formamech://session/status`, `formamech://session/ledger`, `formamech://workspace/tree`, `formamech://artifact/{relpath}`.
- **Prompts:** `cad-to-verified-surrogate`, `multiphysics-3d-audit`, `experimental-vv10-calibration`, `generative-lightweighting`, `diagnose-refusal-or-ood`.

### Option B — CLI Machine-Readable Agent Harness
```bash
formamech agent capabilities
formamech agent describe fem.solve_3d_solid
formamech agent invoke cad.generate_sample_3d --workspace ./study --args-json '{"template":"box"}'
formamech agent invoke fem.solve_3d_solid --workspace ./study --args-json '{"youngs_modulus_mpa":100.0}'
formamech agent session-status --workspace ./study
formamech agent verify-ledger --workspace ./study
formamech agent export-tools --provider openai --mode progressive
```

### Option C — Python SDK (`FormaMechAgentHarness`)
```python
from piginot.harness import FormaMechAgentHarness, HarnessBudget, export_tool_schemas

harness = FormaMechAgentHarness("./study", budget=HarnessBudget(max_fem_solves=16))
caps = harness.invoke("discover.capabilities", {"domain": "cad"})
env = harness.invoke("cad.generate_sample_3d", {"template": "box", "mesh_size_mm": 7.0})
assert env.ok
```

## 2. Progressive Discovery Pattern (Prevent Context Bloat)
1. Call `discover.capabilities` (filtered by `domain` or `search`) to locate the capability name.
2. Call `discover.describe` with `{"name": "<capability>"}` to inspect the exact `input_schema`, `cost_class`, and preconditions.
3. Optionally pass `dry_run=True` to validate arguments and budget impact before spending FEM solves or training epochs.
4. Execute via `invoke_capability`; inspect `ok`, `receipt_sha256`, `artifact_paths`, and `allowed_next_actions`.
5. On `ok=False`, read `error_code` and `recovery_hint` — never blindly retry identical invalid inputs.

## 3. Non-Negotiable Engineering & Repository Invariants
- **ASME V&V 10 Epistemic Discipline:** Numerical verification (`ClaimLevel <= HOLDOUT_VERIFIED`) proves discrete agreement with converged FEM only — never claim physical experimental validation unless `validate.experimental_vv10` passes blind physical coupon holdouts (`ExperimentalValidationTier`).
- **Workspace Sandboxing:** Never pass paths escaping `workspace_root`.
- **Byte-Frozen Modules (DO NOT MODIFY):** `src/piginot/verification/status.py`, `src/piginot/operators/energy_operator.py`, `src/piginot/operators/architecture.py`.
- **Package Isolation:** Keep `src/piginot/physics/` strictly free of FEM/mesh imports and `import piginot.training` free of eager `piginot.fem`/`skfem` imports.

## 4. Verification Commands Before Committing
```bash
ruff format --check .
ruff check .
mypy --strict src/
pytest tests/training/test_import_boundary.py tests/test_s5_anti_scope.py
mkdocs build --strict -f mkdocs.yml
```
