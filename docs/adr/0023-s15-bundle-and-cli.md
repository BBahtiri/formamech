# ADR 0023: S15 bundle format, inference closure and command line

- Status: Proposed — implemented as the defaults below; awaiting the decision
  owner's confirmation of S15-D1 to S15-D5
- Date: 2026-10-03
- Step: S15 (gate `G15`)
- Builds on: ADR 0018 (constitutive served stress), ADR 0019 (refusal and
  confidence), ADR 0021 (DEVELOPMENT waiver), ADR 0022 (reducer families)

## Context

S15 must ship the frozen S13c candidate as a self-contained bundle with
`load_bundle`, `predict` and `inspect`, and a command line covering the
framework and campaign commands of STEPS §S15. `G15` asks that the bundle load
and predict in a clean environment with runtime dependencies only, that every
served field and its provenance be returned, that `P_material` never be served
in place of `P_served`, that out-of-envelope requests refuse with a reason
code, that the manifest detect any changed byte, and that every report carry
the required limitation paragraph.

## Decisions

1. **S15-D1 Bundle format.** A directory with `bundle.json` and exactly seven
   files: `spec/reference-mesh.json`, `spec/morph-map.json`,
   `spec/envelope.json` (the S9 executable envelope, which carries the
   membership predicate), `spec/model.json`, `weights/model.safetensors`,
   `report/verification.json` and `report/limitations.json`. The manifest lists
   the SHA-256 and size of every file and its own SHA-256 over the RFC 8785
   form of everything else. Loading checks the manifest bytes are canonical,
   every hash and size, that no file is unlisted and no path is a symbolic
   link, and only then decodes anything. Every decoder rebuilds the typed
   object and requires its canonical bytes to equal the stored bytes; the mesh
   must also carry its own content identity. Loading then checks that the
   parts agree: the membership names the bundled mesh and topology, every
   model input is bounded by the envelope, every boundary condition and QoI
   names a mesh tag, the manifest claim equals the verification report's, a
   verified claim has a passing report, every outstanding item has a status
   and a reason, and the limitations hold the required paragraph. A bundle is
   written to a staging directory, loaded, and only then renamed into place.
2. **S15-D2 Runtime dependencies.** `numpy`, `torch`, `safetensors` and
   `rfc8785` (the new `runtime` extra). The inference closure imports no FEM,
   mesh-generation or symbolic package; a test runs it with those imports
   blocked.
3. **S15-D3 Confidence.** The frozen S14 `patch-mean` indicator plus the
   boundary-traction term is computed at every mesh vertex; a query point takes
   the largest indicator of the vertices of every cell whose closure holds it.
   The bound served is the DEVELOPMENT-calibrated 90% quantile of the pointwise
   in-plane stress error, in MPa, and indicators above the calibrated range are
   flagged `beyond_calibration`. The indicator needs second derivatives at
   three points per cell, so `predict --no-confidence` serves the fields alone
   about ten times faster.
4. **S15-D4 Claim level.** The bundle carries the G14 claim
   (`HOLDOUT_VERIFIED`); a prediction carries it only when the request is in
   the envelope. A refused request, or a bundle that fails to load, reports
   `NOT_VERIFIED`.
5. **S15-D5 Command line.** `bundle verify`, `predict`, `inspect` and
   `store prune` are framework commands. `family compile`, `pilot`, `campaign`,
   `simulate`, `train`, `verify`, `holdout-reveal` and `bundle build` dispatch
   to a campaign plugin named by `--campaign`: an entry point of the
   `piginot.campaigns` group, a `module:attribute` path, or
   `campaigns.<name>.cli:COMMANDS`. The DogBone plugin
   (`campaigns/dogbone_v1/cli.py`) calls the registered script of each step
   unchanged; HOLDOUT is consumed only by `holdout-reveal`. No family name
   enters `src/`.
6. **Served-stress authority.** `spec/model.json` declares `served_stress`.
   Each registered architecture provides exactly one authority; the energy-form
   operator provides `constitutive-plane-stress-neo-hookean` (ADR 0018), and
   its weights must carry the architecture's module digest. A declared
   authority that the architecture does not provide refuses the bundle
   (`BUNDLE_STRESS_AUTHORITY_MISMATCH`); nothing is substituted. `P_material`,
   the material law at the served deformation, is computed separately and
   reported only as `constitutive_gap = max |P_served − P_material| / E`.
   The `SERVE_P_MATERIAL` mutant is detected with a two-head stand-in
   architecture whose served stress differs from the material law.
7. **Inadmissible predictions refuse.** Where S14 scoring recorded NaN stress
   at a vertex with `det F ≤ 1e-12`, `predict` refuses with
   `PREDICTION_INADMISSIBLE_DEFORMATION` instead of serving a partial field.
8. **Refusal scope.** `predict` takes parameters and query points only; the
   bundle always morphs its own reference mesh. The 36 S14 fixtures that submit
   an explicit mutated mesh therefore have no bundle request form and are
   reported as not applicable; the other 204 fixtures and the 844 in-domain
   cases run through `predict`.
9. **Anti-scope boundary.** The S5 test that rejected the `simulate`, `train`
   and `bundle` commands now requires that each needs a campaign plugin, the
   same way earlier steps advanced it.

## Consequences

- The DogBone bundle reproduces the S14 vertex displacements, served stresses
  and confidence indicators bit for bit on TRAIN cases.
- On one CPU thread a DogBone prediction takes 0.44 s for the fields alone and
  4.9 s with the confidence bound at every vertex.
- Bundles are directories, not archives. Packing one into a single file is
  left to the release step.
- Only the energy-form architecture is registered for serving.
