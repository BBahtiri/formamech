# Examples

Three reference examples cover the framework. All of them run on CPU.

## Example 1 — fixed-mesh strain sweep

One notched-plate geometry (width 23 mm, notch radius 5 mm, notch depth 3 mm)
on a coarse structured mesh of 225 vertices, under a nominal axial strain from
10% to 20%. The example compiles a single-point geometry family and its
envelope, solves the P2 FEM reference at six training and three held-out
strains, trains a small energy-form operator with the framework trainer,
verifies it against FEM on the held-out strains, calibrates the confidence
indicator, builds a bundle, serves a held-out strain through `predict` and
checks that a strain outside the envelope is refused.

```bash
OMP_NUM_THREADS=1 python -m examples.fixed_mesh_sweep.sweep --output example1-output
```

It takes about three minutes on one thread; it fails if it takes more than ten.
CI runs it on every push. Its bundle claims `NOT_VERIFIED`, because the S14
protocol is not part of the example.

## Example 2 — DogBone campaign

The first production campaign (`campaigns/dogbone_v1/`): a quarter dog-bone
specimen with four geometry inputs (total length, grip width, gauge width,
fillet radius) and the nominal axial strain. It runs every step of the evidence
chain: the S9 preregistration with a sealed HOLDOUT commitment, 344 S10 FEM
cases, the S13c energy-form operator selected on TRAIN evidence only, the
signed S14 protocol with DEVELOPMENT and one HOLDOUT evaluation, and the S15
bundle with claim level `HOLDOUT_VERIFIED`. The campaign plug-in exposes the
steps on the command line:

```bash
piginot verify --campaign dogbone_v1 -- development --run-root OUT ...
piginot bundle verify BUNDLE
piginot predict BUNDLE --request request.json --output prediction.json
```

The plans record every step: [S13c](campaigns/s13c-enhanced-model-plan.md),
[S14](campaigns/s14-verification-plan.md), [S15](campaigns/s15-bundle-plan.md)
and [S16](campaigns/s16-mutation-plan.md).

## Example 3 — centre-notched plate campaign

The second family (`campaigns/notched_v1/`, [ADR 0025](adr/0025-s17-examples-and-second-family.md)):
a 50 mm plate of width `W` with a C2-blended double-edge notch of depth `d` and
root radius `r` at mid-length, inputs width 16–30 mm, notch radius 2–8 mm,
notch depth 1–5 mm and nominal axial strain 10–20%. It runs the same evidence
chain through the same framework calls as the DogBone; the only framework
additions are the `BlendedNotchMap` morph map and absolute selection
guardrails. Its artifacts and its campaign-scoped gates are in
`artifacts/campaigns/notched-v1/`, and the [S17 plan](campaigns/s17-plan.md)
lists the module of every step.

```bash
python -m campaigns.notched_v1.s9 ceremony --created-utc T --private-backup PATH
python -m campaigns.notched_v1.s10 produce --run-root RUN
python -m campaigns.notched_v1.s13 train --run-root TRAINING
python -m campaigns.notched_v1.s14 development --s10-run-root RUN --training-run-root TRAINING --run-root S14
```

The notched plate misses the S14 tolerances: its DEVELOPMENT budget fails
(all-cases p90: stress field 5.24%, peak 5.34% against 5%), so HOLDOUT stays
closed and its bundle serves `NOT_VERIFIED` through the same API as the
DogBone ([results](campaigns/s17-plan.md#4-results)). The
[benchmarks](benchmarks.md) page gives the measured numbers of both campaigns
and how they are reproduced.
