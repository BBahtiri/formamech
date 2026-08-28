# DogBone v1 public family artifacts

These files contain no campaign secret, role allocation, or holdout coordinate.

- `compiled-family.json` is the canonical JSON preimage returned by
  `campaigns.dogbone_v1.family.compiled_family()`.
- `family-validation.json` is the canonical result of
  `campaigns.dogbone_v1.pilot.run_family_validation_sweep()`.

Pinned identities:

- compiled family: `f5bac9413c0f602b7ad5e9605ffca9b998fe1875118009e68b6a296c6724b35d`
- reference mesh: `3bfc1043f5aa9e989469adacb0ee1d0b483825ffabcd5dfac4bf9caca580d82c`
- morph map: `e992a1998dec38e4ac2407db12fa6d37bf839f99904ddf296319f1b5c8e25716`
- membership predicate: `e816b6eb1e6de25aaf86265b8ad9bb5f5ff005282f43c590971d0785bb96a873`
- validity policy: `b3c2df59eed939f1aa14a69d05a6fd2dc57be571ffba33c6355982713cc6931d`
- validation evidence: `c83e09c38a01fe87765a258310071ceb03e08ca6ef4618d169b973c8a8bad75a`

The JSON artifacts are generated mechanically from the typed contracts and verified
byte-for-byte by `tests/campaigns/test_dogbone_v1_family.py`.
