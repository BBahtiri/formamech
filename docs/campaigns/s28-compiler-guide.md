# Campaign compiler

The installed `piginot compile` command plans and executes a declared problem
with packaged DogBone/notched adapters or a prepared fixed CAD profile. It
performs material and lifting checks, training, FEM comparisons and candidate
export. Exported candidates retain `NOT_VERIFIED` and their own input envelope.

## Install and inspect a plan

From the source checkout, install the `verify` extra for numerical execution:

```text
python -m pip install -e ".[verify]"
```

Keep run inputs and outputs outside the checkout or in its ignored
`.piginot-runs/` directory. From the working folder containing your inputs:

```text
piginot compile --spec problem.json --config config.json --output preview --plan-only
```

The base wheel is sufficient for planning. `--plan-only` checks the problem,
configuration, public manifest and registered adapter, then writes `plan.json`.
It performs no training, FEM solve, material-admission solve or inference.
Adapter discovery and resources come from the installed package. No historical
campaign checkout is needed.

The problem uses `piginot.problem-spec-v1`; the configuration uses
`piginot.campaign-config-v1`. The [custom-material tutorial](s29-custom-material.md)
creates complete sample inputs and captures the installed execution profile.
For a fixed STEP/DXF profile, [CAD preparation](own-cad-workflow.md) generates
the problem, source manifest and configuration from your geometry/settings.
Capture a new execution profile when changing platforms or numerical libraries.

The configuration records the following separately:

- The installed adapter ID and implementation SHA-256.
- The public manifest path and raw SHA-256, with disjoint TRAIN-fit,
  TRAIN-screen and explicitly requested DEVELOPMENT case IDs.
- The operator shape, ordered conditioning parameters, TRAIN-fit
  normalization, potential-energy objective and fixed training budget.
- The reference initializer, refinement levels and worker limit.
- The training and reference execution profiles, including actual library
  versions, platform fingerprint, precision and seeds.

The manifest path is relative to the configuration file. ProblemSpec remains
authoritative for parameter ranges, material, boundary conditions, lifting,
solver settings and requested tolerances. Unknown fields, conflicting
identities and unsupported capabilities refuse before training. Public
planning does not read sealed data.

## Execute and resume

Use a new output directory for execution:

```text
piginot compile --spec problem.json --config config.json --output run
piginot compile --spec problem.json --config config.json --output run --resume
```

Run in a clean, committed source checkout or from a wheel carrying a clean,
verified build identity. Numerical execution checks the installed source and
adapter identity against the plan. A dirty development wheel can exercise
planning; numerical execution refuses it.

The foreground command owns the run lock. Admission and lifting verification
precede training. Training uses the existing deterministic energy trainer;
its objective and normalization do not consume FEM labels. The configured
final checkpoint is selected. Reference outcomes retain the complete fixed
roster, including failures. Execution uses one numerical thread and at most
the declared two FEM workers.

After interruption, invoke `--resume` with identical inputs, budget and
execution profile. Published checkpoints and request outcomes are verified
before reuse. Recorded numerical failures remain failures; unfinished work
continues. A changed input or corrupt/mixed cache requires a separate run.
The command does not silently change solver settings or retry a terminal
failure. An operating-system shutdown also requires an explicit resume.

## Read the result

`report.json` and `report.md` report the same stage outcomes, requested
tolerances and per-case evidence. `candidate-bundle/`, when produced, contains
the final checkpoint's weights, portable ProblemSpec, material-admission
identity and lifting evidence. Publication is atomic and the bundle is loaded
and checked before it is accepted as an export. Resume verifies the entire
expected bundle manifest.

Reference entries contain status, request/result digests and an `evidence`
record pointing to the full bound outcome under `references/`. Its path is
relative to the run directory; its SHA-256 and binding identify the raw file.
Full request/result JSON stays in that file rather than being copied into
every aggregate report. Resume checks the report entries against the raw
outcomes. FEM comparison uses the same 512-point inference chunks as bundle
serving, so network working memory does not grow with the entire query mesh.

Discrete-FEM comparisons and uncalibrated mesh-refinement diagnostics are
reported separately. They do not establish a total-error bound. Missing
references prevent scientific acceptance and dependent DEVELOPMENT.
The current candidate path retains `NOT_VERIFIED`; matching scientific
verification is required before model promotion. It inherits no calibration,
CP-FREEZE, FEM recipe or certificate from the shipped models. Candidate
serving supports uncalibrated `fast` predictions and explicitly refuses
`certified`, `exact`, calibrated bounds and historical recipe sidecars.

Compiler exit codes are `0` for success, `2` for invalid inputs or unavailable
capabilities, `3` for scientific refusal or incomplete evidence, and `1` for
unexpected execution errors. Inspect the report as well as the shell status;
a completed smoke run may deliberately return `3` and export `NOT_VERIFIED`.

Supported geometry uses installed adapters or a hash-bound fixed CAD project.
The [own-CAD guide](own-cad-workflow.md) lists supported STEP/DXF profiles and
boundary conditions. Neumann loads, 3D, path-dependent materials and harmonic
lifting are outside this interface. A new candidate inherits no claim from a
previous geometry or trained model.
