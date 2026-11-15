# Train a model with a custom material

This tutorial uses the complete inputs in `examples/custom_material/` from
the source checkout. It declares an isotropic strain-energy formula, a small
public TRAIN split and an explicit training budget. The resulting candidate
uses its own material and input envelope and retains `NOT_VERIFIED`.

## Prepare and inspect a plan

Install the current source package in a virtual environment:

```text
python -m pip install -e "."
python -I examples/custom_material/prepare.py --output .piginot-runs/material-plan-inputs
piginot compile --spec .piginot-runs/material-plan-inputs/problem.json --config .piginot-runs/material-plan-inputs/config.json --output .piginot-runs/material-plan --plan-only
```

The preparation script reads `problem.json`, `source.json` and
`config-template.json` beside it. It copies the fixed problem and public
cases, records installed adapter/input identities, and writes
`preparation.json`. An existing output directory is refused.
Metadata-only preparation records an unprobed CPU capability; it performs
no training, FEM solve or inference. Inspect `plan.json` before execution.

Keep the prepared inputs, results and checkpoints in `.piginot-runs/`, which
Git ignores, or choose a working directory outside the checkout.

## Capture an execution profile and train

Install the numerical dependencies and prepare a separate input directory:

```text
python -m pip install -e ".[verify]"
python -I examples/custom_material/prepare.py --output .piginot-runs/material-inputs --execution-profile
piginot compile --spec .piginot-runs/material-inputs/problem.json --config .piginot-runs/material-inputs/config.json --output .piginot-runs/material-run
```

Preparation captures the installed numerical libraries, platform and CPU
capability. The compiler checks that profile again, then performs material
and lifting checks before its configured training and reference solves.
Review the template's training budget and solver settings before starting.
The example's small budget demonstrates execution; it is not a recommended
accuracy configuration for another material or geometry.

A custom material declares its energy, parameter units, symmetry and
small-strain moduli in [ProblemSpec](../problem-spec.md#materials). Material
admission checks those declarations before fitting. Changing the formula,
geometry, parameter ranges or budget requires new inputs and a separate run.

## Read, resume and reuse

Read both `report.md` and `report.json`. They distinguish execution errors,
failed/missing references, FEM comparisons and claim level. Exit code `3`
indicates scientific refusal or incomplete evidence; check the stage
outcomes before deciding what completed. `candidate-bundle/` is present
only when export succeeded.

Resume the same inputs and budget explicitly:

```text
piginot compile --spec .piginot-runs/material-inputs/problem.json --config .piginot-runs/material-inputs/config.json --output .piginot-runs/material-run --resume
```

An exported candidate supports uncalibrated fast prediction. The
[compiler guide](s28-compiler-guide.md) describes run integrity and refusal
codes; [bundle reuse](own-cad-workflow.md#save-and-reuse-a-candidate-bundle)
describes moving and checking an entire candidate directory. Keep the
candidate's comparison reports with its inputs when evaluating suitability.
