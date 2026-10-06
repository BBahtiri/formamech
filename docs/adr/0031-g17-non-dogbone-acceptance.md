# ADR 0031: The decision owner accepts G17.non-dogbone at DEVELOPMENT_VERIFIED

- Status: Accepted — decided by the user (decision owner) on 2026-10-06
- Date: 2026-10-06
- Step: S17 (gate `G17`)
- Record: `artifacts/gates/S17-ACCEPTANCE.json`
- Related: [ADR 0025](0025-s17-examples-and-second-family.md) decisions 7
  and 8, [ADR 0030](0030-notched-development-budget-acceptance.md)

## Context

`G17.non-dogbone` asks that the second family reach a `HOLDOUT_VERIFIED`
bundle through the same API as the DogBone, with no framework code branching
on family. The centre-notched plate ran the whole evidence chain through the
framework calls, and the framework changes since the S16 base are exactly the
declared additions. Its accuracy misses the S14 tolerances:

| Population | Stress field | Peak | Displacement | Gated rows failing |
| --- | --- | --- | --- | --- |
| DEVELOPMENT (64 cases) | 5.24% | 5.34% | 0.58% | 19 of 51 (accepted, ADR 0030) |
| HOLDOUT (64 cases) | 6.01% | 6.40% | 0.53% | 22 of 51 |

(p90 over all cases of the per-case budget, tolerance 5%, 5% and 2%; the
worst HOLDOUT slice reaches 10.7%.) With the ADR 0030 waiver, G14 passes at
`DEVELOPMENT_VERIFIED` and the bundle passes G15 and serves that level, so
`G17.non-dogbone` computes FAIL.

Asked whether to record an acceptance so that G17 reads PASS, or to keep G17
FAIL, the decision owner chose the acceptance ("The error is fine i want to
make this as a PASS").

## Decision

1. **`G17.non-dogbone` is accepted at `DEVELOPMENT_VERIFIED`.** The signed
   record names the check, the accepted claim level, the notched G15 gate it
   accepts (LF-normalised SHA-256) and the bundle manifest. The G17 gate
   passes the check only when all of them match, and records "FAIL as
   computed; accepted by the decision owner (ADR 0031)" with the measured
   values.
2. **The structural half of the check is not waivable.** A failing G15 or a
   framework change outside the declared additions fails the check whatever
   the record says.
3. **Claim levels are unchanged.** The notched bundle serves
   `DEVELOPMENT_VERIFIED`, not `HOLDOUT_VERIFIED`. Its limitations name the
   waived DEVELOPMENT budget and the failed HOLDOUT budget with their numbers.

## Consequences

- G17 passes on the strength of the decision owner's acceptance for one of its
  seven checks. Every page that reports G17 says so.
- The notched predictions are not verified to the S14 tolerances on unseen
  cases. A notched v2 campaign (a finer reference mesh at the notch, a new S9
  ceremony, S10, S13 and S14) is the route to `HOLDOUT_VERIFIED`.
- A new notched bundle or G15 gate needs a new acceptance; this record does
  not carry over.
