# S17 plan — examples, benchmarks, docs and the second family

- Status: Complete; G17 fails on `G17.non-dogbone` (the notched bundle is
  `NOT_VERIFIED`), as ADR 0025 decision 7 foresaw. Decisions 1, 7 and 8 of
  [ADR 0025](../adr/0025-s17-examples-and-second-family.md) were taken by the
  decision owner; the private-reveal backup was signed off on 2026-10-05 and
  the notched S14 protocol on 2026-10-06.
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

### 4.1 The notched-plate campaign

The notched plate ran the full evidence chain through the framework APIs but
**misses the S14 tolerances on DEVELOPMENT and on HOLDOUT**. The decision
owner accepted the DEVELOPMENT budget ([ADR 0030](../adr/0030-notched-development-budget-acceptance.md)),
which opened HOLDOUT under the unchanged rule. HOLDOUT failed, so G14 passes
at `DEVELOPMENT_VERIFIED` and the bundle serves that level.

| Step | Gate | Result |
| --- | --- | --- |
| S6 | G6 PASS | 500/500 public shapes morph validly; FEM pilot 18/18 |
| S9 | G9 PASS | 8/8 checks; the sealed design's fill distance 0.124 exceeds the 0.10 target (limitation) |
| S10 | G10 PASS | 200/200 FEM cases, no failure; 200 byte-identical replays; mutants fire; coverage 85.5% |
| S11 | G11-ENERGY PASS | exact lifting over 64 vectors; coverage 90.4% |
| S13 | G13 PASS | one registered trial, 2,400 steps; replay bitwise; TRAIN screen ELIGIBLE |
| S14 | G14 PASS, `DEVELOPMENT_VERIFIED` | DEVELOPMENT budget FAIL as computed, accepted by signed waiver (ADR 0030); HOLDOUT budget FAIL |
| S15 | G15 PASS | bundle `notched-v1-cp-freeze` serves `DEVELOPMENT_VERIFIED` through the same API |

TRAIN screen of the frozen candidate (32 cases, surrogate error only): median
peak 1.21% (p90 3.44%), stress field 1.66%, displacement 0.17%, no folded
point.

The S14 per-case budget adds the FEM discretisation and projection terms to
the surrogate error. Its p90 over all cases, against 5% (stress field), 5%
(peak) and 2% (displacement):

| Population | Stress field | Peak | Displacement | Gated rows failing |
| --- | --- | --- | --- | --- |
| TRAIN rehearsal (32 cases) | 5.32% | 6.11% | 0.37% | 20 of 51 |
| DEVELOPMENT (64 cases) | 5.24% | 5.34% | 0.58% | 19 of 51 (accepted, ADR 0030) |
| HOLDOUT (64 cases) | 6.01% | 6.40% | 0.53% | 22 of 51 |

- **Where it fails.** The sharpest notches (radius bin 0, HOLDOUT: stress
  field 7.4%, peak 9.9%), the widest plates (width bin 3: 7.4%, 9.7%) and the
  two small strain strata (6 cases each, up to 10.7%). The narrow plates and
  the blunter notches pass (for example width bin 0: 3.9%, 3.8%).
- **Why a better surrogate alone would not close the gap.** The FEM
  discretisation term is about as large as the surrogate error: on HOLDOUT its
  median is 1.34% for the stress field and 1.08% for the peak, against
  surrogate medians of 1.71% and 1.28%; its maximum reaches 5.4% and 4.8%. A
  notched v2 needs a finer FEM mesh at the notch as well as a stronger
  surrogate.
- **What passes.** Refusals 216/216 with no false refusal over 700 in-domain
  requests; pooled Spearman of the confidence indicator 0.625 on DEVELOPMENT
  and 0.609 on HOLDOUT (threshold 0.6); the DEVELOPMENT-calibrated 90% bound
  covers 87.7% of HOLDOUT vertices; u-and-P inference 1,410× faster than FEM
  on one thread; the twelve-family reducer; coverage 91.4%.
- **Order of decisions.** The decision owner signed off the protocol
  (`S14-PROTOCOL-SIGNOFF.json`) knowing the TRAIN-rehearsal shortfall. After
  DEVELOPMENT failed, HOLDOUT first stayed closed (the holdout phase refused
  before any consumption record). The decision owner then accepted the
  DEVELOPMENT result (`S14-PROTOCOL-AMENDMENT-1.json`, committed before the
  consumption record `S14-HOLDOUT-CONSUMED.json`), and HOLDOUT was consumed
  once.

### 4.2 Examples, benchmarks, refusals, documentation

- **Example 1** passes end to end in 128 s on one CPU thread (FEM 75 s,
  training 51 s, verification and bundle 2 s). Held-out strains 0.11, 0.15
  and 0.19: stress field 1.07–1.34%, peak at most 0.94%, displacement at most
  0.14% (`artifacts/benchmarks/EXAMPLE1-SUMMARY.json`).
- **Benchmarks** ([table](../benchmarks.md),
  `artifacts/benchmarks/BENCHMARKS.json`). Both families reproduce: replaying
  the first 400 optimizer steps from the pinned seeds, interrupted and
  resumed at step 200, gives bit-identical model, state and manifest files,
  and re-scoring the frozen candidate on its TRAIN-screen cases reproduces
  every committed metric within 1e-9 (largest difference 1.3e-14 DogBone,
  0 notched).
- **Refusal suite** (`artifacts/benchmarks/REFUSAL-SUITE.json`): 840 of 840
  refused with the expected reason code. That is 240 DogBone and 216 notched
  fixtures through the envelope checkers, and 204 and 180 request-form
  fixtures through `predict` on each family's bundle.
- **Documentation**: the strict mkdocs build and the link check pass. The
  limitations page gains the S17 limits, and the README summarises S17 under
  the limitations banner.
- **Tests**: the full suite passes at the S17 close (3,039 passed, 1 skipped,
  branch coverage of `src/piginot` 91%).

### 4.3 G17

G17 fails on one required check (`artifacts/gates/S17.json`):

| Check | Status | Measured |
| --- | --- | --- |
| `G17.example1` | PASS | 128 s end to end; CI job `example1` declared |
| `G17.example2-3` | PASS | both replays bitwise; re-scoring within 1e-9 |
| `G17.non-dogbone` | **FAIL** | the notched bundle passes G15 but serves `NOT_VERIFIED`, not `HOLDOUT_VERIFIED`; the framework changes since `8a87cd5` are exactly the five declared files (no family branch) |
| `G17.refusal` | PASS | 840/840, both bundles |
| `G17.fixture-names` | PASS | no family vocabulary in `src/` |
| `G17.docs` | PASS | strict mkdocs build; links and README checks |
| `G17.adrs` | PASS | the six ADRs exist and are accepted |

Turning `G17.non-dogbone` into a PASS needs a notched v2 campaign (finer
reference mesh at the notch, then a new S9 ceremony, S10, S13 and S14), not a
change to the tolerances.
