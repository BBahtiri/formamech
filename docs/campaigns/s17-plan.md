# S17 plan — examples, benchmarks, docs and the second family

- Status: In progress. Decisions 1, 7 and 8 of
  [ADR 0025](../adr/0025-s17-examples-and-second-family.md) were taken by the
  decision owner; the private-reveal backup was signed off on 2026-10-05.
- Date: 2026-10-05
- Base: `codex/s16-mutation` (G16 PASS)
- Gate: `G17` (`STEPS-v1.0.md` §S17)

## 1. What S17 must deliver

| Check | Threshold |
| --- | --- |
| `G17.example1` | fixed-mesh sweep runs end to end on CPU in CI in under 10 minutes |
| `G17.example2-3` | DogBone and notched plate reproduce committed benchmarks within tolerance from pinned seeds |
| `G17.non-dogbone` | the notched plate reaches a verified bundle through the same API as the DogBone, with no framework code branching on family |
| `G17.refusal` | every refusal case returns its expected reason code |
| `G17.fixture-names` | no `dogbone`, `gauge`, `fillet` or `grip` in `src/` |
| `G17.docs` | mkdocs builds; all internal links resolve; `limitations.md` linked above the fold in the README |
| `G17.adrs` | six ADRs: served-stress authority, soft-BC default, single architecture in v1, preregistration over sealed audit, retention tiers, independence vectors |

## 2. The notched-plate campaign

The S1 centre-notched plate (ADR 0025 decision 1) runs the DogBone evidence
chain end to end through the framework APIs. Its artifacts and its
campaign-scoped gates live in `artifacts/campaigns/notched-v1/`.

| Step | Module | Evidence |
| --- | --- | --- |
| S6 | `campaigns/notched_v1/pilot.py`, `s6_gate.py` | `family-validation.json`, `fem-pilot.json`, `gates/S6.json` |
| S9 | `proposal.py`, `s9.py`, `s9_gate.py` | `CP-ENVELOPE.json`, `public-case-manifest.json`, `gates/S9.json` |
| S10 | `s10.py`, `s10_assist.py` | `gates/S10.json` |
| S13 | `s13.py`, `s13_close.py` | `S13-PROTOCOL.json`, `S13-TRAIN-SCREEN.json`, `gates/S11-ENERGY.json`, `gates/S13.json`, `CP-FREEZE.json` |
| S14 | `s14.py`, `s14_envelope.py` | `S14-TRAIN-REHEARSAL.json`, `S14-PROTOCOL.json`, `gates/S14.json` |
| S15 | `s15.py` | `gates/S15.json` |

Framework additions (ADR 0025 decision 2): the `BlendedNotchMap` morph map
and its bundle decoder entry, and `AbsoluteSelectionGuardrails` for a campaign
without an incumbent. The DogBone S10 runner takes the production `run_id` as
a parameter.

## 3. Examples, benchmarks, refusals, documentation

- **Example 1** (`examples/fixed_mesh_sweep/`): one notched geometry on a
  coarse mesh, a strain sweep, FEM, a small operator, held-out verification,
  calibration and a bundle; CI job `example1`.
- **Benchmarks** (`campaigns/benchmarks.py`): one table per family from the
  committed artifacts, reproduced by re-scoring the frozen candidate on its
  TRAIN-screen cases and by replaying the first training segment from the
  pinned seeds.
- **Refusal suite** (`campaigns/refusal_suite.py`): every fixture of both
  families through its envelope checker, and every request-form fixture
  through `predict` on its bundle.
- **Documentation**: the mkdocs site (`mkdocs build --strict -f docs/mkdocs.yml`)
  with every ADR and plan, the examples and the benchmark tables; a link
  check over the README and every page.
- **ADRs**: [0026](../adr/0026-served-stress-authority.md),
  [0027](../adr/0027-soft-dirichlet-default.md),
  [0028](../adr/0028-single-architecture-in-v1.md),
  [0029](../adr/0029-independence-vectors.md), with the existing
  [0003](../adr/0003-s3-content-store-retention.md) and
  [0009](../adr/0009-s9-preregistration-over-sealed-audit.md).

## 4. Results

Recorded when the campaign closes.
