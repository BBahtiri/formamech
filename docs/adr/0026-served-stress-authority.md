# ADR 0026: The served-stress authority

- Status: Accepted — records the decisions of ADR 0018 (decision 2, ratified
  2026-10-02) and ADR 0023 (decision 6) as one standing rule for v1
- Date: 2026-10-05
- Step: S17 (`G17.adrs`)

## Context

A neural operator can produce a stress in two ways: from an independent stress
head, or from the material law applied to the predicted deformation. The two
disagree whenever the network is not exact. STEPS-v1.0 §S11 first fixed
`P_served = P_predicted` (two heads). The S13 diagnostics (ADR 0016) showed what
that costs: the S11 operator's displacement was accurate to about 1.6% while the
stress recomputed from its own kinematics was 268% off, and the field, peak and
physics residuals pulled against each other. ADR 0018 replaced that operator
with the energy-form PI-GINOT, whose stress is constitutive.

Whatever the architecture, a user of a bundle must know exactly which stress is
served, and nothing may silently serve a different one.

## Decision

1. **One authority per architecture.** Each registered architecture declares
   the single stress it serves. The energy-form operator
   (`pi-ginot-energy-parametric-film-v1`) serves
   `constitutive-plane-stress-neo-hookean`: `P_served = P_NH(I + grad u_pred)`
   through the S4 plane-stress kernel. There is no independent stress head.
2. **The bundle declares it.** `spec/model.json` carries `served_stress`. At
   load time the inference closure compares it with the authority of the
   registered architecture; a mismatch refuses the bundle with
   `BUNDLE_STRESS_AUTHORITY_MISMATCH`. Nothing is substituted.
3. **The material law is reported, never served.** `P_material`, the material
   law at the served deformation, is computed separately and reported only as
   `constitutive_gap = max |P_served − P_material| / E`. For a constitutive
   authority the gap is zero up to rounding; for a two-head architecture it
   would measure the head's inconsistency.
4. **Every downstream number uses the served stress.** S14 scores
   `P_served`, the peak QoI is the largest Frobenius norm of `P_served` over the
   root vertices, and the confidence indicator is the equilibrium residual of
   `P_served`.

## Consequences

- Field, gradient, stress and peak of a prediction are one consistent object;
  `L_constitutive` is identically zero for the energy-form operator.
- `G11.heads` is amended for this architecture (ADR 0018): the check is that
  the served stress equals the energy gradient, not that an independent head
  exists.
- A future architecture with its own stress head must register its authority
  and pass G11 and G15 before a bundle can declare it.

## Verification

- `G15.substitution`: a re-signed bundle that declares another authority is
  refused with `BUNDLE_STRESS_AUTHORITY_MISMATCH` (DogBone and notched bundles).
- The `SERVE_P_MATERIAL` mutant of the S16 suite fires its primary gate.
- `G11.heads` measures the relative difference between the served stress and
  the energy gradient (DogBone and notched G11 artifacts).
