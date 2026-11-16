# CAD meshing and reference convergence

Use an explicit mesh policy when curved-boundary fidelity or local resolution
matters. The legacy `--mesh-size-mm` import remains available and retains its
fixed-polygon interpretation. A policy import remeshes the original CAD;
midpoint reference refinements then keep that newly prepared polygon fixed.

## Declare and inspect a mesh

Save a policy as strict JSON. Choose sizes and limits for your geometry and
available resources, and replace the boundary ID with one from your own
`cad inspect` output. Evaluate numerical accuracy with separate comparisons.

```json
{
  "schema": "piginot.cad-mesh-policy-v1",
  "global_size_mm": 1.5,
  "minimum_size_mm": 0.1,
  "curve_tolerance_mm": 0.01,
  "curvature_points_per_circle": 32,
  "feature_sizes": [
    {
      "boundary": "edge-000008",
      "size_mm": 0.3,
      "distance_min_mm": 0.5,
      "distance_max_mm": 3.0
    }
  ],
  "point_sizes": [],
  "minimum_angle_deg": 20,
  "maximum_aspect_ratio": 5,
  "maximum_triangles": 120000,
  "maximum_p2_dofs": 500000
}
```

```text
formamech cad import profile.step --unit mm --mesh-policy mesh-policy.json --output geometry
formamech cad inspect geometry
```

The two import sizing options are mutually exclusive. All policy fields are
explicit, including limits on the largest prepared reference level (4).
Selected boundaries use Gmsh Distance/Threshold fields with smooth grading.
Point sizes use radial distance fields without adding CAD boundary entities;
they have `center_mm: [x, y]` instead of `boundary` and the same size/distance
fields. Curvature sizing also contributes. See the
[Gmsh sizing tutorial](https://gmsh.info/doc/texinfo/#t10).

`minimum_size_mm` is the size-field floor, not a guaranteed minimum achieved
edge length. Geometry, curvature constraints and grading can produce smaller
edges. The importer refuses a floor incompatible with the requested circular
chord tolerance. Global/local sizes are meshing targets; inspect the achieved
edge lengths and quality instead of assuming that every edge equals a target.
Aspect ratio here means longest edge divided by its corresponding altitude.

Native Gmsh `Line` and `Circle` curves have analytic checks. Circular nodes
are constrained to enough segments for the declared chord deviation; the
achieved check includes endpoint error and verifies coverage of the source
arc. Other curve types, including an OCC `Ellipse`, currently refuse an
explicit policy as unqualified, even if a particular ellipse looks circular.
Legacy import still accepts its previously supported profiles without making
that geometry claim. These checks bound curve-to-polygon deviation only;
they establish no stress/displacement error bound.

`provenance.mesh_policy` and `provenance.mesh_evidence` in the verified preview
record requested settings, achieved curve deviations, edge lengths, minimum
angle, aspect ratio, and projected reference triangle/P2 DOF counts. Explicit
transfinite boundary counts are checked before Gmsh generation. Full counts
and quality are checked after meshing and before publishing a project or
allocating the reference ladder. These are mesh/solve preparation limits,
not a timeout or a hard allocation limit inside the native Gmsh generator.
Failed preparation publishes no geometry manifest.

## Bounded pilot refinement

The Python API `formamech.cad.adaptation.adapt_mesh` coordinates a finite loop.
Supply a policy, explicit `PilotCase` objects with partition `TRAIN-pilot`,
separate screen IDs, a CAD `remesh(policy)` callback and an
`evaluate(mesh, case)` callback returning `PilotStress` from cell-average
in-plane first-Piola FEM stress. The evaluator owns the actual request,
material, solver and raw FEM evidence. Persist those original requests and
results; returning cell averages alone is not a FEM qualification record.

Declare every limit prospectively: `max_rounds`, `max_solves`,
`indicator_target`, `mark_fraction`, `refinement_factor`, and `max_markers`.
Each round remeshes the original CAD, validates source/policy bindings and
checks the level-4 triangle/DOF limits before any pilot solve. Pilot/screen
overlap is refused. A failed callback is terminal; attempts, policies,
meshes, stresses and hashes remain available on the returned report.

The indicator measures interior traction jumps of cell-average stress. It
omits boundary residuals and body force, and is a marking heuristic rather
than an equilibrium residual or an accuracy bound. `HEURISTIC_TARGET_MET`
does not mean that the model or FEM solution passed an accuracy target.
Round/solve/marker/minimum-size limits and stalled meshes remain explicit.

After a successful preparation, bind its report to the final geometry:

```python
from formamech.cad.project import prepare_geometry
from formamech.contracts import Unit

final_policy = report.attempts[-1].policy
prepare_geometry(
    source,
    output,
    unit=Unit.MM,
    mesh_policy=final_policy,
    adaptation_report=report,
)
```

Publication refuses a failed preparation, changed final mesh or different
policy. The geometry resource binds the report, policy and achieved evidence.
Its existing hash flows into campaign inputs and the plan; changing it
requires a new run. Freeze that mesh before fitting. Old weights do not
inherit a claim for a new mesh.

New candidate bundles also retain the source/resource hashes, policy,
achieved checks and optional adaptation record in `report/verification.json`.
The bundle manifest binds these records with its other files, and the
candidate retains `NOT_VERIFIED`.
Prediction needs no CAD/FEM libraries; reproducible training still needs
the original inputs, raw preparation/FEM evidence and checkpoints.

## Compare FEM fields on a fixed domain

New fixed-CAD compiler reports add `same_polygon_field_changes` to
`reference_diagnostics`. For each adjacent pair in the nested ladder, the
coarse P1/P2 displacement and gradient are evaluated at fine quadrature
points with their actual FE basis. Stress is recomputed with the declared
material and plane-stress closure. Selected recovered vertex peaks use the
same regions/exclusions as model screening. This uses no nearest-node transfer.

The standalone API is:

```python
from formamech.compiler.reference_convergence import compare_fem_pair

changes = compare_fem_pair(
    problem,
    coarse_request,
    coarse_result,
    fine_request,
    fine_result,
    tolerances=None,
)
```

`None` requests measurements only; it produces no convergence PASS. An
explicit tolerance mapping uses the three keys `u_relative_l2`,
`p_relative_l2`, and `absolute_peak_error`. Fix reference tolerances before
qualification; model-error targets are not reference-error targets.
Measured pair changes, even when small, remain uncalibrated and do not
establish an error bound or prove asymptotic convergence from a single pair.
Persist unresolved peaks instead of silently substituting another quantity.

Keep CAD approximation, FEM discretization changes and model error separate
in the reports you retain with a candidate. Changing the mesh defines a new
training and comparison setup; record its inputs and resource limits before
execution.
