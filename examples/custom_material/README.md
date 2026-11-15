# Custom material: prepare, plan and run a small example

This development example uses a user-defined isotropic strain-energy formula
on the registered tiny DogBone mesh. It has two public TRAIN-fit cases and
one separate TRAIN-screen case. Training uses one optimizer step, seed 2901,
and no FEM labels. The reference uses cold solves at refinement levels 1 and 2
with at most two workers. It is a software example with `NOT_VERIFIED` outputs;
one training step establishes no accuracy claim.

The formula in `problem.json` describes a compressible Neo-Hookean material
using the named parameters `E` and `nu` and invariants `I1` and `J`:

```text
E/(4*(1+nu))*(I1-3)-E/(2*(1+nu))*log(J)
+E*nu/(2*(1+nu)*(1-2*nu))*log(J)^2
```

It also declares the parameter units, isotropic symmetry and small-strain
moduli. A different energy requires matching material declarations and must
pass all eight admission checks before training. A useful trained model needs
its own declared training budget and scientific verification.

## Inspect a plan from the source checkout

Run these commands from the repository root. The tutorial inputs are included
in `examples/custom_material/`; prepared inputs and results stay in the
Git-ignored `.piginot-runs/` directory:

```text
python -m pip install -e "."
python -I examples/custom_material/prepare.py --output .piginot-runs/material-plan-inputs
piginot compile --spec .piginot-runs/material-plan-inputs/problem.json --config .piginot-runs/material-plan-inputs/config.json --output .piginot-runs/material-plan --plan-only
```

Preparation creates a new input folder and refuses an existing folder or any
overlap with the tutorial assets. It records the installed adapter identity,
available distribution versions and the exact input, template and profile
hashes in `preparation.json`. The default `METADATA_ONLY` profile has an
unprobed CPU capability. Preparing and inspecting this plan needs no numerical
imports, training, inference or FEM solves. `plan.json` records
`DECLARED_NOT_NUMERICALLY_PROBED` and `NOT_VERIFIED`.

## Capture an execution profile and run explicitly

Install the numerical extra, then prepare a separate input folder:

```text
python -m pip install -e ".[verify]"
python -I examples/custom_material/prepare.py --execution-profile --output .piginot-runs/material-run-inputs
piginot compile --spec .piginot-runs/material-run-inputs/problem.json --config .piginot-runs/material-run-inputs/config.json --output .piginot-runs/material-run
```

`--execution-profile` captures actual library versions and the CPU capability
before execution. The receipt says `ENVIRONMENT_CAPTURED_NOT_EXECUTED`; the
compiler still checks the runtime and enforces one numerical thread. Preserve
the input files and receipt before starting. Changing operating systems,
library versions or the CPU capability requires a new profile and run folder.
Preparation itself never starts the compiler.

The explicit compile command checks material and lifting, trains the fixed
one-step model, solves the two references and writes `report.json`, `report.md`
and a candidate bundle when export succeeds. The example configures tolerances of
2% displacement, 5% stress field and 5% peak error. This tiny model can fail
those screens and return exit code 3. Read the reports: an expected scientific
refusal is useful evidence only when the engineering stages and references
completed. Unexpected errors or incomplete references must be investigated.
The candidate remains `NOT_VERIFIED`; it inherits no calibration or certificate
from the shipped family models.

To continue after interruption, use identical inputs, budget and execution
profile:

```text
piginot compile --spec .piginot-runs/material-run-inputs/problem.json --config .piginot-runs/material-run-inputs/config.json --output .piginot-runs/material-run --resume
```

The geometry is the registered tiny mesh, with fixed dimensions and material
parameters and strains from 1% to 1.2%. Arbitrary meshes, new geometry families,
DEVELOPMENT, sealed data and an accuracy search are outside this example.
