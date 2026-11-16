# Example 1 — fixed-mesh strain sweep

One notched-plate geometry (width 23 mm, notch radius 5 mm, notch depth 3 mm)
on a coarse structured mesh (225 vertices), loaded by a nominal axial strain
between 10% and 20%. The example runs the whole framework path on CPU in a few
minutes:

1. compile a single-point geometry family and a strain envelope;
2. solve the P2 FEM reference at six training and three held-out strains;
3. train a small energy-form operator on the training strains with the
   framework trainer (no FEM label enters the loss);
4. verify against FEM on the held-out strains;
5. calibrate the confidence indicator on the training strains, build a
   bundle, serve a held-out strain through `formamech.bundle.predict` and check
   that a strain outside the envelope is refused.

```bash
OMP_NUM_THREADS=1 python -m examples.fixed_mesh_sweep.sweep --output example1-output
```

The bundle claims `NOT_VERIFIED`: the example does not run the S14 protocol
(FEM reference error terms, DEVELOPMENT and a one-time HOLDOUT). The two
campaigns in `campaigns/` show that protocol end to end.

CI runs the example on every push (`.github/workflows/ci.yml`, job
`example1`) with a ten-minute limit; it prints `summary.json`, which records
the held-out errors and the time of each stage.
