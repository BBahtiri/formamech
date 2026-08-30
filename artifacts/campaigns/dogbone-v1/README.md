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

Pinned identities:

- compiled family: `a8143a75e62c614d40f52049cfc88c1cda74d38c9bd488cc3bed4c5678d973a4`
- reference mesh: `3bfc1043f5aa9e989469adacb0ee1d0b483825ffabcd5dfac4bf9caca580d82c`
- morph map: `e992a1998dec38e4ac2407db12fa6d37bf839f99904ddf296319f1b5c8e25716`
- membership predicate: `e816b6eb1e6de25aaf86265b8ad9bb5f5ff005282f43c590971d0785bb96a873`
- validity policy: `b3c2df59eed939f1aa14a69d05a6fd2dc57be571ffba33c6355982713cc6931d`
- validation evidence: `4dfac913a1fdbc5d752e83e5053e62fe45cada814b90638cdb89f05169aeb92d`
- FEM solvability evidence: `ecc8057de75414fe7931695b9e9d589c83d50254aa548554c260fcadb7eba9ec`

The FEM pilot fixes the pinned source values `E = 760 MPa`, `nu = 0.23`, and
prescribed grip displacement `1 mm`, then uses P2 displacement, eight load steps, and
the verified S7 scikit-fem backend. All 12 sentinels succeeded on the recorded Windows
host. Total measured solve time was 261.094054 s, median per-case time was 21.019403 s,
the observed maximum was 25.600437 s, and the 12 uncompressed canonical FEM result
payloads totalled 52,117,983 bytes. These measurements size one candidate campaign on
one host; they do not estimate a domain-wide failure probability or portable runtime.

The JSON artifacts are generated mechanically from typed contracts and verified by
`tests/campaigns/test_dogbone_v1_family.py` and
`tests/campaigns/test_dogbone_fem_pilot.py`.
