# DogBone v1 public family artifacts

These files contain no campaign secret, private role-allocation preimage, or holdout
coordinate. Public PILOT, TRAIN and DEVELOPMENT labels are intentionally present in
the S10 closeout report.

- `compiled-family.json` is the canonical JSON preimage returned by
  `campaigns.dogbone_v1.family.compiled_family()`. Its runtime reference mesh is
  represented by the structural `mesh_id` in the membership predicate; recomputable
  quality diagnostics are deliberately outside the identity preimage.
- `family-validation.json` is the canonical result of
  `campaigns.dogbone_v1.pilot.run_family_validation_sweep()`. Its four reported
  floating-point diagnostics are rounded to 12 decimal places only after all raw
  validity decisions have been made.
- `fem-solvability-pilot.json` is the canonical public report from
  `campaigns.dogbone_v1.fem_pilot.run_fem_pilot()`. It retains the ordered outcome,
  request/result identities, convergence diagnostics, canonical result-payload size,
  timing, and runtime environment for 12 geometry sentinels selected before solving.
  It does not contain the full FEM result payloads.
- `fem-strain-envelope-pilot.json` is the canonical public report from
  `campaigns.dogbone_v1.strain_pilot.run_strain_pilot()`. It repeats the 12 FEM
  sentinels at 10% and 20% nominal axial strain, computing prescribed right-grip
  displacement as `nominal_axial_strain * L_total / 2`, with 16 continuation
  steps.
- `fem-strain-envelope-qualification.json` repeats that fixed population and load
  rule with 32 continuation steps after the first screen exposed
  continuation-sensitive failures.
- `fem-strain-envelope-qualified.json` is the final uniform 64-step qualification
  of the same 10%/20% population. Its exact policy and evidence hash were later
  owner-ratified for the first production campaign.
- `s9-strain-load-proposal-dry-run.json` is the canonical, secret-free replay of the
  load-aware DOE proposal. It is not a production design or commitment.
- The S10 request materializer builds active public FEM requests from the committed
  `CP-ENVELOPE.json` and `public-case-manifest.json`. The large request plan remains
  in the ignored production run root; the committed closeout report binds its exact
  content hash and execution-profile hash.
- `s10-production-dataset.json` is the canonical public closeout report for all 344
  active requests. It records role snapshots, source-verified extraction and
  projection evidence, deterministic collocation replay, physical-integral
  cross-mesh comparisons, TRAIN-only normalization and the controlled
  failure-visibility canary. It contains no private reveal material.
- `S11-CONTINGENCY-PROTOCOL.json` is the pre-label four-candidate architecture
  screen protocol. It fixes the TRAIN fit/screen split, one-shot DEVELOPMENT
  policy, optimizer budget, model initialization, metrics and selection guardrails.
- `S11-CONTINGENCY-EVIDENCE.json` is the exact canonical aggregate evidence from
  the completed screen. It binds the source, normalization, shared sample schedule,
  trained model states and safe model-file hashes without embedding field tensors.
  No DEVELOPMENT metric appears because the TRAIN decision fell back to baseline.
- `S11-CONTINGENCY-SCREEN.svg` visualizes the four frozen TRAIN-screen metric sets
  and the single guardrail that prevented candidate C from being selected.
- `S13-TRAIN-SCREEN-FAILED.json`, `S13B-TRAIN-PILOT.json` and
  `S13B-TRAIN-INVALID-POINT.json` retain the failed S13 six-trial screen and the
  failed S13b peak pilot (ADR 0015).
- `S13C-PROTOCOL.json` is the canonical S13c preregistration (ADR 0018): the
  energy-form operator, potential-energy objective, single registered trial,
  sampler seed and the unchanged frozen S13 guardrails. Training refuses to run
  unless the rebuilt protocol equals these bytes.
- `S13C-TRAIN-SCREEN.json` is the TRAIN-only screen of the final registered S13c
  checkpoint: frozen guardrail verdict, per-case peak, field, peak-location and
  FEM energy-gap diagnostics, the orientation count over all 256 TRAIN cases and
  the comparison with the S11 incumbent, S13 trial 0 and the S13b peak arm.
- `S13C-G13-REPLAY-PROBE.json` compares an independent replay of the registered
  trial, interrupted and resumed half-way, with the registered checkpoint bytes.

Pinned identities:

- compiled family: `a8143a75e62c614d40f52049cfc88c1cda74d38c9bd488cc3bed4c5678d973a4`
- reference mesh: `3bfc1043f5aa9e989469adacb0ee1d0b483825ffabcd5dfac4bf9caca580d82c`
- morph map: `e992a1998dec38e4ac2407db12fa6d37bf839f99904ddf296319f1b5c8e25716`
- membership predicate: `e816b6eb1e6de25aaf86265b8ad9bb5f5ff005282f43c590971d0785bb96a873`
- validity policy: `b3c2df59eed939f1aa14a69d05a6fd2dc57be571ffba33c6355982713cc6931d`
- validation evidence: `4dfac913a1fdbc5d752e83e5053e62fe45cada814b90638cdb89f05169aeb92d`
- FEM solvability evidence: `e2aa1e0e2cd9d20b31043d9c6cb066c138c9b3cb00c7432111ddd38e6f156928`
- FEM strain-load evidence: `4ff6dba3f311ff949d5cf662b2d68dd358e5b6f2592157677797f6173d0f14d2`
- FEM strain-load 32-step evidence: `fa5f031b761101987d8af642b86e2b893babf4d0e6e9289b27846bd12479f325`
- FEM strain-load 64-step evidence: `2f6cfcb9ed56e072a25ed3b09ade2de9666ade5c7be15fcd33489eb67e4502a0`
- S9 strain-load proposal dry run: `42cb8f7ee8a16f89663289ecf39cc959d656f3fbba152c9826c469f0a920be34`
- S10 production dataset: `ed9a7758a87ce9fddab1ec365fc3529372b49c181406a09723573a91a96c9ef2`
- S11 contingency protocol: `f16a4f63c3a8cfc3ee321bc4e15162802cbc6f106ed807f5aa4bc59eceac81df`
- S11 contingency evidence: `aae56fcb3ab6b6b2c693c5b3dbdbce89a1716fd90086dd1786217b189957ec18`
- S11 contingency evidence-file BLAKE2b-256:
  `260eca6c345ee1edea17b9098859baaa6de4a35e27de65c6ccdead0cad831f36`
- S11 contingency screen canonical-LF SVG SHA-256:
  `d92255515dbfaaa1d646eaf74253364211d028c5590727d712e36f90a5eaefae`

The FEM pilot fixes the pinned source values `E = 760 MPa`, `nu = 0.23`, and
prescribed grip displacement `1 mm`, then uses P2 displacement, eight load steps, and
the verified S7 scikit-fem backend. All 12 sentinels succeeded on the recorded Windows
host. Total measured solve time was 261.094054 s, median per-case time was 21.019403 s,
the observed maximum was 25.600437 s, and the 12 uncompressed canonical FEM result
payloads totalled 52,117,983 bytes. Each P2 solve had 7,458 scalar displacement DOFs.
These measurements size one candidate campaign on one host; they do not estimate a
domain-wide failure probability or portable runtime.

The fixed `1 mm` displacement from the pinned source is not a ratified production
load because it couples nominal strain to specimen length. The strain-load pilot
evaluates the replacement primitive but does not approve a final load envelope:
21 of 24 attempts succeeded, with `wide_gauge_long` failing at both 10% and 20% and
`arc_margin` failing at 20%. Successful paired cases departed from linear right
reaction scaling by 5.46% to 5.70%.

The 32-step qualification improved the same population to 23 of 24 successes. Its
only remaining failure was `wide_gauge_long` at 20%, which succeeded in a targeted
64-step continuation diagnostic. Neither finite run is hidden or relabelled as a
pass; the complete population was then rerun uniformly at 64 steps.

That final uniform sweep succeeded in all 24 attempts. All 12 paired geometries
showed right-reaction scaling 5.46% to 5.70% below the linear 2:1 reference; the
minimum recorded deformation Jacobian was 0.9909092600273233. Median solve time was
108.960118 s and p95 was 133.426422 s on the recorded host. These results support
the separately recorded owner ratification of the 10% to 20% nominal-strain envelope
with the 64-step solver recipe; the finite sweep is evidence, not self-approval.

The JSON artifacts are generated mechanically from typed contracts and verified by
`tests/campaigns/test_dogbone_v1_family.py` and
`tests/campaigns/test_dogbone_fem_pilot.py` plus
`tests/campaigns/test_dogbone_strain_pilot.py` and
`tests/campaigns/test_dogbone_campaign_proposal.py`. The S10 public request boundary
is verified by `tests/campaigns/test_dogbone_s10_public_boundary.py`, and the closed
production gate is checked by
`tests/campaigns/test_dogbone_s10_gate_artifact.py`.

## Local S10 production execution

`campaigns.dogbone_v1.s10_production` runs only the 344 active public cases. It
persists each visible outcome atomically, stores every successful canonical FEM
result in `ObjectStore`, recovers a success written immediately before an
interruption, and refuses resume after a request-plan, execution-profile, backend,
or source-revision change. The large local state lives under the ignored
`.piginot-runs/` directory; only later compact evidence belongs in this tracked
artifact directory.

Launches must bind this worktree explicitly and pin numerical thread counts to the
single thread declared by the execution profile. Run `--dry-run` first, then one
new production attempt as a smoke check, and finally resume without the limit:

```powershell
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python -m campaigns.dogbone_v1.s10_production --dry-run
python -m campaigns.dogbone_v1.s10_production --max-new-attempts 1
python -m campaigns.dogbone_v1.s10_production
```

The command requires a clean committed worktree. The recorded production run
completed all 344 requests successfully. Closeout materialized 88,064 collocation
points, 344 projection/error records, 343 physical-integral comparisons and 256
TRAIN-normalized samples. The canonical report above and `artifacts/gates/S10.json`
record the resulting G10 PASS evidence.

The exact execution profile was recovered after FEM completion from the unique full
256-bit request/ledger hash preimage and persisted before closeout. Its canonical
hash is
`d3ede9186f3365dab5b53f510c5c973ed368c1decd08e56461bf4fc41caf675f`.
This recovery closes the report binding but does not claim that the profile file was
present during the original solves.

## S11 architecture contingency outcome

Implementation revision `22fea865a74bf69a28dc845e874a008ba31ef650` trained the
four preregistered candidates for 1,500 Adam steps each on the same 192-case TRAIN
fit population and the same case/query schedule. The first frozen comparison used
the disjoint 64-case TRAIN screen. Candidate C, the multiscale local-boundary and
residual stress-trunk design, reduced median absolute peak error from 8.09% to
4.05%, absolute median signed bias from 7.76% to 2.36%, p90 peak error from 13.63%
to 8.34%, and median stress-field error from 19.20% to 17.79%.

Candidate C nevertheless increased median displacement-field error from 3.73% to
3.86%, a 3.48% relative regression against the preregistered maximum of 2%. It
therefore failed exactly one guardrail. The deterministic decision fell back to
candidate A, and the DEVELOPMENT population remained unopened. This is not a
license to relax the threshold after observing the screen: S12 inherits the
baseline architecture, while C remains diagnostic evidence that local stress
capacity addresses the known stress bias.
