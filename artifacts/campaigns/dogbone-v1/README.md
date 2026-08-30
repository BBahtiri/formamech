# DogBone v1 public family artifacts

These files contain no campaign secret, role allocation, or holdout coordinate.

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
64-step continuation diagnostic. Neither finite run is hidden or called a pass; a
uniform 64-step population must qualify before the load proposal can be recommended.

The JSON artifacts are generated mechanically from typed contracts and verified by
`tests/campaigns/test_dogbone_v1_family.py` and
`tests/campaigns/test_dogbone_fem_pilot.py` plus
`tests/campaigns/test_dogbone_strain_pilot.py`.
