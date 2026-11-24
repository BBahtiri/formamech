---
name: formamech-agent
description: Operate FormaMech's full verified solid mechanics, 3D FEM, multiphysics, neural operator, generative design, and ASME V&V 10 platform via the Universal Agent Harness (MCP, CLI, or Python SDK).
---

# FormaMech Agent Harness Skill

Use this skill whenever an engineering task requires 2D/3D CAD import, finite-strain FEM, multiphysics (shells, buckling, dynamics, plasticity, thermo-mechanics, modal, contact), neural surrogate training/serving, generative topology/shape optimization, CAE archive ingestion, or experimental ASME V&V 10 validation.

## Standard Autonomous Workflow

1. **Discover capabilities progressively:**
   - MCP: call `discover_capabilities` (optionally with `domain` in `discover|cad|fem|multiphysics|surrogate|optimize|ingest_validate|export|governance`).
   - CLI: `formamech agent capabilities --domain <domain>`.
2. **Inspect exact schema before invoking:**
   - MCP: `describe_capability({"name": "<capability>"})`.
   - CLI: `formamech agent describe <capability>`.
3. **Dry-run costly solves/training when uncertain:**
   - Pass `"dry_run": true` to verify argument types and remaining FEM/epoch budget before execution.
4. **Execute & Chain Verified Steps:**
   - Every invocation appends a SHA-256 hash-chained entry to `.formamech-agent/ledger.jsonl` and returns a structured `HarnessEnvelope` with `receipt_sha256`, `artifact_paths`, `budget`, and `allowed_next_actions`.
5. **Verify Before Reporting:**
   - Call `governance.verify_session_ledger` (and `governance.verify_receipt` when serving bundles) before presenting final engineering numbers or exported `.vtu`/`.glb`/`.fmu` artifacts.
   - Always state epistemic scope explicitly: numerical FEM verification (`ClaimLevel`) vs. physical coupon validation (`ExperimentalValidationTier`).
