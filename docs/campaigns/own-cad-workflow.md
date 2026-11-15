# Train a model for your own 2D CAD profile

This workflow trains one model per imported STEP or DXF profile over declared
material and displacement-load ranges. It covers import, training, FEM
comparison and portable candidate export. Each candidate retains
`NOT_VERIFIED`; assess its predictions for the intended geometry and inputs.

## Import and inspect

From the current source checkout, install the CAD and numerical extras:

```text
python -m pip install -e ".[cad,verify]"
```

Run the following commands in a working folder outside the source checkout,
or inside its ignored `.piginot-runs/` directory. Put your CAD file and
settings there; keep generated meshes, reports and weights there too.
Relative paths below refer to that working folder.

```text
piginot cad import profile.step --unit mm --mesh-size-mm 1 --output geometry
piginot cad inspect geometry
```

Use the same commands with a `.stp` or `.dxf` file. Import produces a new
geometry project, a preview and hash-bound meshes at levels 1, 2 and 4.
Inspection lists each boundary identifier, its line/curve kind and endpoint
segments. Currently users identify edges from those coordinates and enter
their identifiers in `physics.json`; there is no graphical boundary selector.
Select the boundaries explicitly; the software does not guess which edges
are clamped or displaced.

Current format limits:

- STEP: one connected planar face parallel to XY, with optional holes.
  Embedded STEP units are authoritative and convert to mm. The declared
  `--unit` is recorded but applies no additional STEP scaling.
- DXF: LINE, ARC, CIRCLE and closed zero-width LWPOLYLINE profiles, with
  nonintersecting holes. For unitless files, `--unit` supplies units; a
  nonzero DXF unit header must agree with it. Unsupported entities, open
  loops, solids and nonplanar profiles refuse rather than being discarded.
- Essential boundaries currently require straight line groups. Curved free
  boundaries are allowed. The reference ladder subdivides the same linear
  polygon; it does not estimate error against the exact curved CAD boundary.
- Force/Neumann loading and full 3D solids are outside this implementation.

The CAD extra installs the Gmsh and ezdxf Python packages. Some Linux systems
also require the OS OpenGL/GLU libraries used by Gmsh; an unavailable backend
is reported explicitly. The [Gmsh manual](https://gmsh.info/doc/texinfo/)
and [ezdxf documentation](https://ezdxf.readthedocs.io/en/stable/) describe
the backends. They are not needed for a prepared project's metadata inspection
or for runtime-only prediction from an exported fixed-geometry bundle.

## Define material, boundaries and ranges

Create `physics.json` with schema `piginot.cad-settings-v1`. It contains:

- `name`, `parameters` with material/load roles, and `material`, using the
  [ProblemSpec](../problem-spec.md) formats. At least one range must vary.
- `boundary_conditions` covering every inspected boundary explicitly:
  Dirichlet displacement, symmetry where valid, or traction-free.
- `qois`, `verification` tolerances and a positive `displacement_scale`
  expression. Do not assign a peak quantity to an unsupported singularity.
- The complete `training` configuration, including the optimizer-step budget
  and seed, plus `sampling` counts `fit_cases` and `screen_cases`.
- Optional model, objective, reference, FEM, constants and constraints.

For example, a load parameter named `u` with unit `mm` can prescribe a
selected edge's x displacement with `{"expression":"u","unit":"mm"}`.
A clamped edge can prescribe zero x and y displacement. The selected region
identifiers must come from your actual geometry inspection.

For an illustrative project, suppose inspection identifies the clamped edge
as `edge-000001` and the displaced edge as `edge-000002`. The following is a
boundary-condition excerpt, not a complete settings file:

```json
[
  {
    "name": "clamp",
    "region": "edge-000001",
    "kind": "dirichlet",
    "components": ["x", "y"],
    "values": [
      {"expression": "0", "unit": "mm"},
      {"expression": "0", "unit": "mm"}
    ]
  },
  {
    "name": "applied_displacement",
    "region": "edge-000002",
    "kind": "dirichlet",
    "components": ["x", "y"],
    "values": [
      {"expression": "ux", "unit": "mm"},
      {"expression": "uy", "unit": "mm"}
    ]
  }
]
```

Declare `ux` and `uy` as load parameters with units and ranges. Add a named
`traction_free` condition for every remaining free boundary, including hole
boundaries. Omitting a group is an error. Boundary identifiers are specific
to an imported project; select the identifiers from your inspection output.

Prepare and inspect the campaign without starting numerical work:

```text
piginot cad prepare --project geometry --settings physics.json --output inputs
piginot compile --spec inputs/problem.json --config inputs/config.json --output preview-run --plan-only
```

Preparation validates the settings and geometry, generates disjoint public
TRAIN-fit and TRAIN-screen cases, and records hashes in `preparation.json`.
It creates no DEVELOPMENT or HOLDOUT data. The current ProblemSpec reserves
one slot for each of those roles, but the receipt marks both ungenerated
and unrequested. The default CPU profile is `NOT_PROBED`; it is for planning.

## Execute a declared run

For local refinement, analytic curve checks and bounded pilot preparation,
follow [CAD meshing and reference convergence](own-cad-meshing.md) before
preparing training inputs. Freeze the final mesh before fitting.

Capture the installed numerical execution profile in a separate new folder:

```text
piginot cad prepare --project geometry --settings physics.json --output execution-inputs --execution-profile
piginot compile --spec execution-inputs/problem.json --config execution-inputs/config.json --output model-run
```

The compiler performs material and lifting checks before the declared
training budget, then compares the result with FEM on the same fixed polygonal
problem. Read the comparison separately from the execution status and check
the requested tolerances for the intended use.

Read `report.md` and `report.json`, including failed or missing references.
An exported candidate remains `NOT_VERIFIED` until separately verified. It
retains its own geometry and material/load envelope and inherits none of the
shipped specimen models' certificates or calibrated bounds.
Training, comparison and export can complete even when an accuracy screen
fails. A compiler exit code of `3` denotes scientific refusal or incomplete
evidence; inspect the stage outcomes before deciding what completed. A
candidate bundle is available only if export succeeded.

To resume, keep every original input and the same execution profile and add
`--resume` to the compile command. CAD, manifest and mesh bytes are part of
the plan binding; changing them requires a new run.

## Save and reuse a candidate bundle

When export succeeds, `model-run/candidate-bundle/` contains the model weights,
fixed reference mesh, input envelope, model/material specification and
verification/limitations records. `bundle.json` binds their hashes. The
compiler creates this directory; no separate specimen-specific bundle build
command is needed.

Check and inspect it:

```text
piginot bundle verify model-run/candidate-bundle
piginot inspect model-run/candidate-bundle --output candidate-inspect.json
```

Create `request.json` using the [bundle request format](../bundle.md#request),
with every input name, unit and value taken from this candidate's envelope.
The shipped DogBone/notched request files describe different models and must
not be substituted for your CAD inputs. Predict with:

```text
piginot predict model-run/candidate-bundle --request request.json --mode fast --no-confidence --output prediction.json
```

To reuse the trained model elsewhere, copy or archive the entire
`candidate-bundle/` directory, preserve its relative paths and bytes, and
install a compatible `piginot[runtime]` package in the destination environment.
Verify and inspect the copied bundle again before prediction. Runtime-only
prediction needs neither the original CAD file nor Gmsh/FEM. Keep the source
CAD, settings, meshes, reports and checkpoints separately if you also want
to reproduce training or continue an interrupted run.

Inspect the output's units, envelope verdict and `NOT_VERIFIED` claim alongside
the FEM report. A successful prediction or bundle integrity check does not
promote its accuracy claim. CAD candidates currently support uncalibrated
`fast` predictions only; `certified`, `exact` and calibrated bounds are
unsupported. FEM comparisons remain separate from runtime serving.
