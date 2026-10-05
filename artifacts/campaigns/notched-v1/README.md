# Notched-plate v1 public campaign artifacts

The second production family (ADR 0025): the S1 centre-notched plate. These
files contain no campaign secret, private role-allocation preimage or HOLDOUT
coordinate; the private reveal lives outside Git
(`.piginot-private/notched-v1/reveal-v1.json` and its backups).

| File | Step | Content |
| --- | --- | --- |
| `family-validation.json` | S6 | 500-shape public morph sweep (`campaigns.notched_v1.pilot validation`) |
| `fem-pilot.json` | S6 | FEM pilot of the reference shape and the eight box corners at 10% and 20% strain, production recipe |
| `s9-proposal-dry-run.json` | S9 | public-seed replay of the DOE proposal; not a production design |
| `executable-envelope.json` | S9 | the sealed envelope `notched-v1-strain-controlled-10-20-v1` |
| `public-case-manifest.json` | S9 | 216 public cases (200 active, 16 reserves); no HOLDOUT case |
| `replacement-policy.json` | S9 | ordered reserves per role |
| `CP-ENVELOPE.json` | S9 | the commitment, including the blake2b HOLDOUT commitment |
| `live-commitment-order.json` | S9 | creation-time order evidence, committed after `CP-ENVELOPE.json` |
| `S13-PROTOCOL.json` | S13 | the preregistered training and selection protocol |
| `S13-TRAIN-SCREEN.json` | S13 | TRAIN-screen evaluation and selection of the registered trial |
| `S13-G13-REPLAY-PROBE.json` | S13 | independent replay of the first 400 steps |
| `CP-FREEZE.json` | S13 | the frozen candidate |
| `S14-TRAIN-REHEARSAL.json` | S14 | the TRAIN-only rehearsal |
| `S14-PROTOCOL.json`, `S14-PROTOCOL-SIGNOFF.json` | S14 | the frozen protocol and its sign-off |
| `S14-DEVELOPMENT.json`, `S14-HOLDOUT.json` | S14 | the DEVELOPMENT and HOLDOUT results |
| `S15-VALIDATION.json` | S15 | the G15 measurements of the bundle |
| `gates/` | S6–S15 | the campaign-scoped gate artifacts (G6, G9, G10, G11, G13, G14, G15) |

The framework gates G2–G5, G7, G8 and G12 are shared with the DogBone campaign
(`artifacts/gates/`); the S14 reducer reads the campaign-scoped artifacts for
the other five families (`campaigns.notched_v1.s14.CAMPAIGN_FAMILIES`).
