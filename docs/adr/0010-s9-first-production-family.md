# ADR 0010: First S9 production-family decision

- Status: Accepted — exact production family ratified
- Date: 2026-08-27
- Last reviewed: 2026-08-29
- Proposal author: Codex
- Decision owner: Betim Bahtiri
- Accepted: 2026-08-29T13:00:53Z by explicit task-thread ratification
- Recorded basis: repeated user direction to use the pinned DogBone repository as the
  reference source, followed by explicit owner ratification on 2026-08-29 of the
  compiled family, reference mesh, and validity-policy hashes listed in this ADR;
  selection as the first ceremony is not a recorded claim that its model is proven
- Step: S9

## Context

The user repeatedly directed the project to use the DogBone repository as its
reference implementation, then ratified the exact mesh, morph recipe, membership
bounds, and compiled hash recorded here. This ADR selects it as the first ceremony
decision because a no-hole quarter specimen de-risks the identity, validation, and
preregistration path before a genuinely different topology is attempted. The basis
is ceremony risk reduction, not a claim that the pinned neural-operator
implementation is proven.

The tracked framework did not yet contain a production reference mesh or a compiled
family for that geometry: S6's gate fixtures exercised only generic rectangular RBF
and FFD morphs. Reusing the S1 spike implementation would violate its permanent
`LEGACY` boundary, while substituting that rectangle would not implement the requested
reference direction.

The designated source is
[`BBahtiri/PI-GINOT-DogBone`](https://github.com/BBahtiri/PI-GINOT-DogBone) at commit
`9f6794141beca9d3e66433465c3b07039a7e41d2`. It defines a no-hole quarter specimen
with four physical dimensions: total length, outer width, reduced width, and circular
transition radius. The published primitive ranges are 40–70 mm, 16–26 mm, 6–14 mm,
and 8–20 mm respectively.

## Decision

Use recipe `dogbone-quarter-no-hole-v1`, implemented outside the family-agnostic
package surface in `campaigns/dogbone_v1`. The reusable package addition is an exact
`CircularTransitionMap`; it contains no fixture-specific name or PI-GINOT model code.

The reference shape is `(54, 20, 10, 12) mm`. Its deterministic structured mesh has
969 nodes and 1,792 triangles, is aligned to the reduced/transition interface, and has
canonical mesh identity
`3bfc1043f5aa9e989469adacb0ee1d0b483825ffabcd5dfac4bf9caca580d82c`.

For reference coordinate `(x, y)`, the map sends the reduced and transition intervals
to their target intervals with separate positive affine longitudinal scales. It then
uses

`y_target = y * target_profile(x_target) / reference_profile(x)`.

The circular profile is the same analytic construction as the pinned reference
repository. The inverse uses the reciprocal interval and profile scales. Within each
interface-aligned subdomain, the determinant is the positive longitudinal scale times
the positive profile-height ratio. This gives this specific map an analytic,
piecewise global bijection; it does not upgrade the generic RBF/FFD sampled-Jacobian
claim in ADR 0006.

The primitive bounds retain the four published source ranges. Four dimensionless
derived axes are compiled through S2 expressions and evaluated by S6:

- `grip_to_gauge_ratio = W_grip / W_gauge`;
- `fillet_to_gauge_ratio = R_fillet / W_gauge`;
- `length_to_gauge_ratio = L_total / W_gauge`;
- `arc_feasibility_ratio = 2 * R_fillet / (W_grip - W_gauge)`.

The proposed lower bound on `arc_feasibility_ratio` is 1.35. The source repository
requires only a value greater than 1, but its mesh-free geometry can represent nearly
tangent transitions that the fixed-connectivity FEM mesh cannot retain above the 5°
minimum-angle policy. A public calibration sweep located failures only at ratios up to
1.29. Raising mesh resolution did not remove the angular collapse, so the envelope
margin is explicit instead of relaxing the mesh-quality gate.

The validity policy is Jacobian `> 0.01`, minimum boundary feature size 0.05 mm,
minimum triangle angle 5°, and maximum aspect ratio 20. Its hash is
`b3c2df59eed939f1aa14a69d05a6fd2dc57be571ffba33c6355982713cc6931d`.

Compiled-family schema 1.1 retains the full reference mesh at runtime but excludes it
from the outer canonical preimage. The membership predicate binds the structural
`mesh_id`, and construction refuses any runtime mesh whose identity differs. This is
the same structural boundary already used by S5: nodes, cells, facets, tags, regions,
unit, and orientation determine `mesh_id`; recomputable quality diagnostics do not.
The public validation summary rounds its four diagnostic measurements to 12 decimal
places only after raw S6 validity decisions. This prevents platform `libm` variation
from changing public evidence without moving an acceptance threshold.
No production checkpoint references the superseded schema-1.0 proposal hash, so this
pre-ratification correction needs no stored-object migration or compatibility shim.

The geometry membership above is only one part of an S9 executable envelope.
Material and load bounds remain an explicit owner decision; values in the pinned
reference implementation are evidence for that decision, not approval to encode a
production envelope.

## Evidence

The disjoint public validation stream
`dogbone-family-validation-20260828-v1` attempted 527 candidates to obtain 500
membership-accepted shapes. Twenty-seven candidates were transparently rejected by
the derived bound. All 500 accepted shapes passed S5 canonicalization, tag/topology
transport, 12,544 degree-5 Jacobian checks per shape, feature size, and mesh quality.

Worst accepted measurements were:

- minimum Jacobian determinant: 0.1751;
- minimum boundary feature size: 0.1303 mm;
- minimum triangle angle: 5.4977°;
- maximum aspect ratio: 8.2758.

The canonical evidence is
`artifacts/campaigns/dogbone-v1/family-validation.json`, with content hash
`4dfac913a1fdbc5d752e83e5053e62fe45cada814b90638cdb89f05169aeb92d`.
The accepted compiled-family hash for later `CP-ENVELOPE` use is
`a8143a75e62c614d40f52049cfc88c1cda74d38c9bd488cc3bed4c5678d973a4`.

## Consequences

- The requested source direction now has an owner-ratified, exact, reproducible
  production-family preimage.
- The source parameter box is retained, with one visible coupled exclusion needed by
  the FEM mesh-quality contract.
- No DogBone executable-envelope preimage or hash is created until material and load
  bounds are approved.
- Family-specific names remain outside `src/`, and
  `tests/test_no_fixture_names.py` enforces that hygiene property. This supports
  `G17.fixture-names` only. The 548-line circular-transition provider is generic in
  naming but currently has only the DogBone campaign as a consumer, so
  `G17.non-dogbone` remains `NOT_RUN`; the notched-plate example must still reach a
  verified bundle through the same API with no framework branch.
- The original PI-GINOT architecture is not copied or replaced by this change. Model
  architecture remains an S11 decision; this ADR only establishes geometry identity.
- This evidence does not approve material/load bounds, DOE populations, leakage
  tolerance, reserves, or the production commitment. G9 and S10 remain blocked.
