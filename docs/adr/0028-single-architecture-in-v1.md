# ADR 0028: A single served architecture in v1

- Status: Accepted
- Date: 2026-10-05
- Step: S17 (`G17.adrs`)

## Context

The repository contains three operator architectures: the S11 boundary
cross-attention operator (`pi-ginot-s11-boundary-cross-attention-v1`), the
S11c local-stress candidate of the pre-S12 screen, and the S13c energy-form
operator (`pi-ginot-energy-parametric-film-v1`, ADR 0018). Only the last one
passed the S13 guardrails, G11 (amended) and G13, and only it was verified
under the S14 protocol.

Every served architecture multiplies the evidence a release needs: its own
G11 checks, its own stress authority (ADR 0026), its own bundle loader and its
own mutation coverage. The second family of S17 also has to show that the
framework does not branch on family; adding architectures per family would
undo that.

## Decision

1. **One registered architecture.** The bundle registry
   (`piginot.bundle.serving.ARCHITECTURES`) holds exactly one entry,
   `pi-ginot-energy-parametric-film-v1`. A bundle naming any other
   architecture is refused.
2. **Every family uses it.** The DogBone and notched-plate campaigns, and
   Example 1, use the same architecture with the same objective; only the
   conditioning inputs, the frame and the lifting displacement differ, and they
   are data in `spec/model.json`.
3. **Size is a configuration, not an architecture.** The operator
   configuration (widths, token count, layers) is part of the bundle and of the
   module digest check; Example 1 uses a smaller configuration of the same
   architecture.
4. **The other architectures stay as research code.** The S11 and S11c
   operators remain in `piginot.operators` with their tests and gate history,
   but no bundle can serve them.

## Consequences

- The release evidence is one G11, one stress authority and one loader.
- An architecture change after v1 requires a new registry entry, its G11,
  G13 and G15 evidence and a new ADR.

## Verification

- `tests/bundle` refuses an unregistered architecture identifier.
- `G17.non-dogbone`: the notched-plate bundle is built and served through the
  same `build_bundle`, `load_bundle` and `predict` as the DogBone bundle.
