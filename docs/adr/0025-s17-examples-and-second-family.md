# ADR 0025: S17 examples and the centre-notched plate as the second family

- Status: Accepted — decisions 1, 7 and 8 were taken by the decision owner on
  2026-10-05, who also signed off the private-reveal backup (cloud primary and
  backup, and a copy in `pi_ginot_private_reveals/notched-v1/`) and, on
  2026-10-06, the notched S14 protocol; decisions 2–6 and 9–11 were accepted
  as implemented on 2026-10-06, at the release sign-off (ADR 0032)
- Date: 2026-10-05
- Step: S17 (gate `G17`)

## Context

`G17` needs three reference examples (a fixed-mesh sweep that runs in CI in
under ten minutes, the DogBone, and a notched plate), committed benchmarks that
reproduce from pinned seeds, a refusal suite, a documentation site and six
ADRs. `G17.non-dogbone` asks that the notched plate reach a verified bundle
through the same API as the DogBone, with no framework code branching on
family. The second-family spike (a holed plate) showed that the operator,
lifting and trainer transfer, with a median field error of 5.05%.

## Decisions

1. **The second family is the S1 centre-notched plate** (decided). A 50 mm
   strip of width `W` with a C2-blended double-edge notch of depth `d` and root
   radius `r` at mid-length: profile `h(x) = W/2 − d (1 − (x/b)²)³` for
   `x < b = sqrt(6 d r)`, else `W/2`. Inputs `width` 16–30 mm, `notch_radius`
   2–8 mm, `notch_depth` 1–5 mm (the S1 spike ranges) and
   `nominal_axial_strain` 0.10–0.20; the material is the DogBone material
   (E = 760 MPa, ν = 0.23). The quarter model `[0, 25] × [0, W/2]` has the
   DogBone boundary conditions; the notch root lies on the symmetry line.
2. **Two framework additions, no family branch.** `BlendedNotchMap`
   (`src/piginot/morphing/blended_notch.py`) is a new morph-map type built like
   the circular transition: a piecewise-linear longitudinal map with its knot
   at the blend length and a height-ratio scaling; the bundle decoder registry
   gains one entry for it. `AbsoluteSelectionGuardrails`
   (`src/piginot/training/selection.py`) lets a selection policy carry
   absolute limits instead of the DogBone incumbent (decision 6); the DogBone
   guardrails, their hashes and the selection code paths are unchanged. The
   DogBone S10 runner takes the production `run_id` as a parameter so the
   notched campaign keeps its own ledger identity. Every other framework module
   is used unchanged.
3. **The evidence chain is the DogBone chain.** S6 validity sweep and FEM
   pilots, the S9 ceremony with a HOLDOUT commitment and a private reveal, S10
   FEM production, S13 training on TRAIN geometry only, the S14 signed
   protocol with DEVELOPMENT and then HOLDOUT once, and an S15 bundle. Its
   artifacts and campaign-specific gates (G9, G10, G11, G13, G14, G15) live in
   `artifacts/campaigns/notched-v1/`; the framework gates G2–G8 and G12 are
   shared, because they verify family-agnostic code.
4. **Budget.** 8 PILOT, 128 TRAIN (96 fit, 32 screen), 64 DEVELOPMENT and 64
   HOLDOUT cases, 16 reserves. FEM uses the DogBone S10 recipe (P2, 64 load
   steps); the hardest corner (W 16, r 2, d 5, 20% strain) converges in 108 s.
5. **Training.** The S13c recipe unchanged — architecture, objective,
   schedule, seeds — with the four conditioning inputs of this family and a
   z-score fitted on the TRAIN-fit coordinates. Training reads geometry only,
   so it binds to the TRAIN request plan and can run while FEM production
   solves the labelled cases.
6. **Selection.** One registered trial, admitted by preregistered absolute
   guardrails (no folded point, TRAIN-screen median field and peak errors at
   most the S14 tolerances), because the DogBone guardrails compare with a
   DogBone incumbent that this family does not have.
7. **Tolerances** (decided): the S14 tolerances unchanged — p90 field 5%,
   peak 5%, displacement 2%. If the notched plate misses them its bundle is
   `NOT_VERIFIED` and `G17.non-dogbone` fails; the result is reported as is.
8. **Claim target** (decided): `HOLDOUT_VERIFIED` through the full protocol,
   with the decision owner's sign-off on the private-reveal backup and on the
   S14 protocol before DEVELOPMENT opens.
9. **Example 1, the fixed-mesh sweep.** The notched reference geometry with a
   single-point geometry envelope, a coarse mesh and a strain sweep: FEM, a
   small operator, verification on held-out strains and a bundle, end to end
   in CI in under ten minutes.
10. **Benchmarks and refusals.** A committed benchmark table per family,
    reproduced from pinned seeds by deterministic replay and re-evaluation;
    the refusal suite is every family's fixtures through its envelope checker
    and every request-form fixture through `predict`. The notched set has 216
    fixtures and no derived-bound category, because the family's derived
    bounds sit on corners of the primitive box and no in-box request violates
    them.
11. **ADRs.** ADR 0003 (retention tiers) and ADR 0009 (preregistration over
    sealed audit) already cover two of the six topics; four new ADRs record
    the served-stress authority, the soft-BC default, the single architecture
    in v1 and the independence vectors.

## Outcome (2026-10-06)

Decision 7 applied, then the decision owner accepted the results. The notched
plate misses the S14 tolerances on TRAIN (rehearsal all-cases p90: stress
field 5.32%, peak 6.11%) and on DEVELOPMENT (5.24%, 5.34%; 19 of 51 gated rows
fail). The decision owner signed off the protocol knowing the rehearsal
result, then accepted the DEVELOPMENT budget in a signed waiver
([ADR 0030](0030-notched-development-budget-acceptance.md)), which opened
HOLDOUT under the unchanged rule. HOLDOUT fails (6.01%, 6.40%; 22 of 51 rows),
so G14 passes at `DEVELOPMENT_VERIFIED` and the bundle passes G15 at that
level. `G17.non-dogbone` computes FAIL and the decision owner accepted it
([ADR 0031](0031-g17-non-dogbone-acceptance.md)), so G17 passes on that
acceptance.

About half of each case's budget is the FEM discretisation term of the
24 + 24 × 16 reference mesh, so a notched v2 needs a finer reference mesh at
the notch as well as a stronger surrogate. The framework changes since the S16
base are exactly the additions of decision 2.
