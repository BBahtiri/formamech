# User guide

This guide follows the current source development version (0.2.0.dev0).
Read the [limitations](limitations.md) when assessing numerical output.
Predictions belong to a declared geometry, material and input envelope.

| Use | What you provide | What you get |
| --- | --- | --- |
| [Predict with a shipped bundle](#1-predict-with-a-shipped-bundle) | Parameters inside an existing specimen envelope | Fields and quantities from that bundle |
| [Train a small example](#2-train-a-small-model-end-to-end-example-1) | A built-in fixed-geometry configuration | An end-to-end software example |
| [Train for your own CAD](#3-train-for-your-own-2d-cad-profile) | STEP/DXF, boundary settings and material/load ranges | A reusable fixed-geometry candidate and FEM comparisons |
| [Use a custom material](campaigns/s29-custom-material.md) | An admitted strain-energy formula and explicit inputs | A material-specific candidate and its own reports |

## Install

You need Python 3.11, 3.12 or 3.13, and Git. Install from the repository; it
also holds the bundles.

```text
git clone https://github.com/BBahtiri/verified-neural-operators.git
cd verified-neural-operators
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on Linux and macOS,
`.venv\Scripts\activate` on Windows), then install what you need:

```text
python -m pip install -e ".[runtime]"       # predict with bundles
python -m pip install -e ".[verify]"        # compiler training and FEM comparison
python -m pip install -e ".[cad,verify]"    # add STEP/DXF import and meshing
python -m pip install -e ".[dev,geometry,cad,physics,fem,ml,verify]"  # contributors
```

`piginot --version` reports the installed version. The commands above use
`main`, including the CAD/compiler interface. A source checkout also supplies
the specimen bundles and examples; the Python wheel contains the runtime
packages and their adapter resources. CAD dependencies may require additional
OS libraries, as described in the [CAD guide](campaigns/own-cad-workflow.md).

Keep your settings and numerical outputs under the ignored `.piginot-runs/`
directory or outside the clone. This preserves a clean source identity for
compiler execution and keeps experimental results local.

## 1. Predict with a shipped bundle

A bundle is a trained operator together with everything needed to serve it
safely: the reference mesh, the shape map, the allowed input range (the
*envelope*), the verification report and the limitations. Every file is
hash-checked when it is loaded. Two bundles ship in `bundles/`:

| Bundle | Geometry | Inputs | Claim level |
| --- | --- | --- | --- |
| `bundles/dogbone-v1-s13c-cp-freeze` | DogBone tensile specimen | `L_total` 40–70 mm, `W_grip` 16–26 mm, `W_gauge` 6–14 mm, `R_fillet` 8–20 mm | `HOLDOUT_VERIFIED` |
| `bundles/notched-v1-cp-freeze` | centre-notched plate | `width` 16–30 mm, `notch_radius` 2–8 mm, `notch_depth` 1–5 mm | `DEVELOPMENT_VERIFIED` |

Both take `nominal_axial_strain` from 0.10 to 0.20, and are trained for one
material: `elastic_modulus` 760 MPa and `poisson_ratio` 0.23, a compressible
Neo-Hookean solid in 2D plane stress with 1 mm thickness.

### Check and inspect a bundle

```text
piginot bundle verify bundles/notched-v1-cp-freeze
piginot inspect bundles/notched-v1-cp-freeze --output inspect.json
```

`inspect` lists the inputs and their ranges, any derived bounds (ratios such
as the notch depth over the notch radius), the claim level, the verification
verdict, every outstanding item and the limitations.

### Write a request

A request names every input with an explicit unit. Units of the same
dimension are converted, so `2.2 cm` is accepted for `width`. `queries` is
optional: it lists extra points `[x, y]` in mm where you want the fields as
well as at the mesh vertices.

```json
{
  "inputs": {
    "width": {"value": 22.0, "unit": "mm"},
    "notch_radius": {"value": 4.0, "unit": "mm"},
    "notch_depth": {"value": 3.0, "unit": "mm"},
    "elastic_modulus": {"value": 760.0, "unit": "MPa"},
    "poisson_ratio": {"value": 0.23, "unit": "1"},
    "nominal_axial_strain": {"value": 0.15, "unit": "1"}
  },
  "queries": [[2.0, 5.0], [12.0, 9.0]]
}
```

The model works on a quarter of the specimen, using its two symmetry planes:

- **Notched plate:** `x` runs from the notch centre line (0) to the loaded
  end (25 mm), and `y` runs from the centre line (0) to the free edge (`W/2`).
- **DogBone:** `x` runs from the gauge centre (0) to the loaded end (`L/2`),
  and `y` runs from 0 to `W_grip/2`.

Example requests are in `bundles/requests/`.

### Predict

```text
piginot predict bundles/notched-v1-cp-freeze --request bundles/requests/notched-in-domain.json --output prediction.json
```

The exit code tells you the outcome:

| Exit code | Meaning |
| --- | --- |
| 0 | served (`IN_DOMAIN`) |
| 2 | refused: the request is outside the envelope or invalid |
| 1 | the bundle itself was refused (for example a modified file) |

### Read the result

`prediction.json` holds:

| Field | Meaning |
| --- | --- |
| `verdict`, `reason_codes` | `IN_DOMAIN`, or `REFUSED` with stable reason codes |
| `claim_level` | how far the bundle's accuracy was verified (see below); `NOT_VERIFIED` on any refusal |
| `vertices` | at every mesh vertex: `coordinates` (mm), displacement `u` (mm), deformation gradient `F_2D`, `F33` and `J_3D`, first Piola stress `P_served` (MPa), Cauchy stress `sigma` (MPa), and `confidence` |
| `queries` | the same fields at your query points |
| `qois` | quantities of interest, for example the peak stress at the notch root with its value, vertex and location |
| `mesh` | the triangle connectivity, for plotting |
| `units`, `identities`, `limitations` | units of every field, the hashes of the bundle that answered, and the limitation text |

For the request above, the notched bundle gives a peak first Piola stress
magnitude of 230 MPa at the notch root (x ≈ 0.35 mm, y ≈ 8.0 mm), and
answers in about 0.1 s on one CPU thread. The FEM solve takes about 100 s.

`confidence.stress_error_bound` is the calibrated 90% quantile of the
pointwise stress error, given an equilibrium-residual indicator. It ranks
errors only moderately: it covered 87–88% of HOLDOUT vertices for a nominal
90%. `beyond_calibration` flags points outside the calibrated range.

### What the claim level means

| Claim level | Meaning |
| --- | --- |
| `HOLDOUT_VERIFIED` | the error budget passed on unseen HOLDOUT cases: p90 stress-field error ≤ 5%, peak ≤ 5%, displacement ≤ 2%, including the FEM reference error (DogBone: 2.75%, 4.02%, 0.29%) |
| `DEVELOPMENT_VERIFIED` | measured on DEVELOPMENT and HOLDOUT, but HOLDOUT missed the tolerances. Notched: p90 6.01% stress field and 6.40% peak, up to 10% on the sharpest notches and widest plates; the DEVELOPMENT result was accepted by the decision owner (ADR 0030). Treat it as indicative |
| `NOT_VERIFIED` | no accuracy claim (refusals, and Example 1 bundles) |

### Refusals

PI-GINOT refuses rather than extrapolates. Common reason codes:

| Code | Cause |
| --- | --- |
| `PARAMETER_OUT_OF_BOUNDS` | an input is outside its range (for example a strain of 0.30, or a different material) |
| `DERIVED_QUANTITY_OUT_OF_BOUNDS` | the inputs are each in range but their combination is not (for example a DogBone fillet too small for the grip–gauge step) |
| `MISSING_PARAMETER`, `UNKNOWN_PARAMETER`, `PARAMETER_UNIT_MISMATCH` | the request does not name the bundle's inputs with compatible units |
| `QUERY_OUTSIDE_DOMAIN` | a query point lies outside the specimen shape for these inputs |
| `PREDICTION_INADMISSIBLE_DEFORMATION` | the predicted deformation folds somewhere (`det F <= 1e-12`) |
| `BUNDLE_*` | the bundle files changed or are incomplete; re-clone or re-download it |

### Use it from Python

```python
import json

from piginot.bundle import load_bundle, predict
from piginot.cli import parse_request

bundle = load_bundle("bundles/notched-v1-cp-freeze")
with open("bundles/requests/notched-in-domain.json", encoding="utf-8") as handle:
    inputs, queries = parse_request(json.load(handle))
prediction = predict(bundle, inputs, queries)

print(prediction.verdict.value, prediction.claim_level.value)
result = prediction.document()  # the same dictionary as prediction.json
print(result["qois"][0]["value"], "MPa")
```

### Plot a result

Matplotlib is not a dependency; install it separately (`pip install matplotlib`).

```python
import json

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

result = json.load(open("prediction.json", encoding="utf-8"))
xy = np.array(result["vertices"]["coordinates"])
cells = np.array(result["mesh"]["cells"])
stress = np.linalg.norm(np.array(result["vertices"]["P_served"]), axis=(1, 2))

figure, axes = plt.subplots(figsize=(8, 4))
image = axes.tripcolor(
    mtri.Triangulation(xy[:, 0], xy[:, 1], cells), stress, shading="gouraud"
)
figure.colorbar(image, label="|P_served| (MPa)")
axes.set_aspect("equal")
figure.savefig("prediction.png", dpi=150, bbox_inches="tight")
```

![First Piola stress magnitude of the notched plate, width 22 mm, notch radius 4 mm, depth 3 mm, 15% strain](assets/notched-prediction.png)


## Certified and exact predictions (0.2.0.dev, S22)

`piginot predict` takes `--mode` (from the development line; not in release
0.1.0):

| Mode | You get | Needs |
| --- | --- | --- |
| `fast` (default) | the bundle's prediction, as in 0.1.0 | `pip install 'piginot[runtime]'` |
| `certified` | the prediction plus a physics certificate (`eta_energy`, stress change, guards); `--corrected` adds the corrected displacement at the FEM nodes | `pip install 'piginot[verify]'` |
| `exact` | the converged FEM solution, started from the prediction, with the path that produced it | `pip install 'piginot[verify]'` |

These FEM-backed modes use the shipped bundles' own recipe sidecars. New CAD
and compiler candidates support uncalibrated `fast` prediction only; compare
them with FEM through their separate compiler workflow.

```bash
piginot predict bundles/notched-v1-cp-freeze --request bundles/requests/notched-in-domain.json --mode certified --corrected
piginot predict bundles/notched-v1-cp-freeze --request bundles/requests/notched-in-domain.json --mode exact
```

The FEM problem comes from the bundle's recipe in `bundles/fem-recipes/`: the
same mesh, element order and solver settings as the bundle's FEM reference
data. A request the bundle refuses is refused in every mode, before any FEM
work.

**What `certified` measures.** The certificate takes one Newton step of that
FEM problem, starting from the prediction:

| Field | Meaning |
| --- | --- |
| `verdict` | `OK`; or `INDEFINITE` / `INADMISSIBLE` (a guard failed, no estimate is given) |
| `eta_energy` | size of the Newton step in the energy norm |
| `indicators.p_l2_change` | relative L2 change of the first Piola stress along the step |
| `indicators.u_l2_change` | the same for the displacement |
| `guards` | which checks passed: admissible states, positive curvature, energy decrease, positive definite tangent |

On the 64 DEVELOPMENT cases of each family, one step landed on the converged
FEM solution: the step size matched the true distance to it (median ratio
1.000), so `p_l2_change` is the prediction's stress error against that
solution. `--corrected` returns the displacement after the step; its peak
stress was within 0.01% (DogBone) and 0.06% (notched) of the converged
solution on those cases.

**What `exact` returns.** The FEM solution converged from the prediction: warm
Newton at full load, then a short continuation, then the cold load-stepped
solve, then an explicit `solve_failed`. `attempts` lists every rung tried. The
output holds the displacement at the FEM nodes and the total potential energy.
On DEVELOPMENT every case converged on the first rung and matched a cold solve
within 1e-13 (relative).

**What neither measures.** Both modes compare against the FEM solution on the
recipe's mesh, not against the exact solution of the continuum problem. That
mesh has its own discretisation error, which the S14 error budgets include as
the reference term; for the notched plate it is about half of each case's
budget. Neither mode extends a bundle beyond its envelope.

**Escalation.** With `--escalate`, a certified request whose certificate is
not `OK`, or whose `p_l2_change` exceeds `--tolerance`, is solved exactly:

```bash
piginot predict bundles/notched-v1-cp-freeze --request bundles/requests/notched-in-domain.json --mode certified --escalate --tolerance 0.01
```

`--tolerance 0.01` accepts at most a 1% change of the stress field. The
response then has `"escalated": true` and the `exact` result.

**Cost.** On the reference machine (Linux x86_64, AVX2), with three solves in
parallel, a certificate took about 0.5–0.7 s, an exact solve about 5.5 s and a
cold FEM solve about 190 s. Other machines and loads change these numbers.

**Exit codes.** `0` served, `2` refused, `1` an error or `solve_failed`.

From Python:

```python
from piginot.bundle import load_bundle
from piginot.serving import load_recipe, serve

bundle = load_bundle("bundles/notched-v1-cp-freeze")
recipe = load_recipe(bundle, "bundles/notched-v1-cp-freeze")
served = serve(bundle, inputs, mode="certified", recipe=recipe, corrected=True)
print(served.path.value, served.certificate["indicators"]["p_l2_change"])
```

`inputs` comes from `parse_request` as in the example above.

## Calibrated error bounds (0.2.0.dev, S23)

`--bounds` attaches calibrated 90% upper bounds to a served request, in any
mode ([ADR 0104](https://github.com/BBahtiri/verified-neural-operators/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0104-s23-calibrated-bounds.md)). They bound the S14
budgeted error, the quantity the claim levels and their tolerances are about:
the prediction's error against the FEM reference plus that reference's own
discretisation and projection terms.

```bash
piginot predict bundles/dogbone-v1-s13c-cp-freeze --request bundles/requests/dogbone-in-domain.json --bounds
```

| Mode | `bounds.kind` | What is bounded |
| --- | --- | --- |
| `fast` | `budget_total` | the whole budgeted error of the prediction |
| `certified`, `exact` | `reference_terms` | only what the FEM mesh adds; the certificate measures the rest |

The bounds are one number per family and quantity, calibrated on the 64
DEVELOPMENT cases of each bundle:

| Bundle | Bound | Stress field | Peak stress | Displacement |
| --- | --- | --- | --- | --- |
| DogBone | `budget_total` | 2.58% | 4.07% | 0.22% |
| DogBone | `reference_terms` | 1.45% | 2.54% | 0.10% |
| Notched | `budget_total` | 5.41% | 5.47% | 0.63% |
| Notched | `reference_terms` | 3.11% | 3.26% | 0.13% |

Read them as: at least 90% of requests drawn like the calibration cases lie
within the bound. It is a guarantee over requests, not for one particular
request. A request that is inside the bundle's envelope but further from the
training cases than any calibration case is served with
`"status": "OUT_OF_CALIBRATION"` and no bound. The bounds come from the
sidecars in `bundles/calibration/`, bound to each bundle's manifest; they need
only `piginot[runtime]`. Without `--bounds` every output is unchanged.

The bounds were tested once on 64 fresh H1 cases per family (gate G23, PASS):
the stress-field and peak bounds covered 86–91% of the in-calibration cases
(about 91% is expected on average with 64 calibration cases). One bound was not
confirmed: the DogBone `budget_total` displacement bound (0.22%) covered only
46 of 63 cases (73%). `--bounds` withholds it: the quantity is listed under
`not_confirmed` with no number. The H1 results are in `bundles/h1-results/`.

## 2. Train a small model end to end (Example 1)

Example 1 runs the whole method on one fixed notched geometry (width 23 mm,
notch radius 5 mm, depth 3 mm) and a strain sweep, in about 2 minutes on a
CPU:

```text
python -m pip install -e ".[runtime,fem,ml]"
python -m examples.fixed_mesh_sweep.sweep --output example1-output
```

It solves the P2 FEM reference at six training strains and three held-out
strains, then trains a small energy-form operator from the physics alone. It
checks the operator against FEM on the held-out strains, calibrates the
confidence indicator and builds a bundle that serves a held-out strain and
refuses one outside the range. `example1-output/summary.json` holds:

- the timings;
- the held-out errors, about 1.1–1.3% for the stress field;
- the pass or fail result.

Its bundle is `NOT_VERIFIED`, because the full verification protocol is not
run.

To use your own fixed geometry and load range, copy
`examples/fixed_mesh_sweep/sweep.py` and replace its mesh and strain values.
The script shows each framework call in order. See [Examples](examples.md).

## 3. Train for your own 2D CAD profile

The [own-CAD workflow](campaigns/own-cad-workflow.md) imports a supported STEP
or DXF profile, lists its boundary identifiers and prepares a declared run.
You supply the material and parameter ranges, identify clamped/displaced/free
boundaries from the inspected coordinates, and set the mesh, training and
FEM comparison limits. Review a metadata-only plan before execution.

The installed compiler performs material/lifting checks, training and FEM
comparisons, and exports a candidate when those stages permit it. It supports
explicit resume with unchanged inputs. The [compiler guide](campaigns/s28-compiler-guide.md)
explains statuses and reports; the [meshing guide](campaigns/own-cad-meshing.md)
explains local sizing and reference refinement.

Copy the complete candidate directory to reuse a trained model. Check its
hashes and envelope before [fast prediction](campaigns/own-cad-workflow.md#save-and-reuse-a-candidate-bundle).
Own-CAD candidates retain `NOT_VERIFIED` and their own material/load ranges.
Assess their FEM comparisons for the intended use; they inherit no shipped
bundle's calibration or certificates.

## Advanced: define a parametric geometry family

The shipped bundles are tied to their geometry families. To build a new
parametric shape family with a measured claim level, author a Python campaign
and run its declared evidence chain. This extends the fixed-profile workflow
above with a morph map and family-level verification.

What you provide:

1. **A reference mesh:** a planar Gmsh v4.1 triangle mesh with named physical
   groups for the boundaries (loaded end, symmetry lines, free edges). It is
   read with `piginot.geometry.read_gmsh(path, unit=Unit.MM)`, which refuses
   meshes it cannot trust: inverted or degenerate elements, unnamed boundary
   facets, 3D or quad cells.
2. **A shape parametrisation:** a morph map that deforms the reference mesh
   from a few parameters. The framework provides RBF and FFD maps
   (`piginot.morphing`), plus the exact DogBone and notched-plate maps as
   examples of a family-specific map.
3. **An input range:** bounds on every parameter, derived bounds on their
   ratios, and the material and load range.

The steps, with the notched-plate campaign (`campaigns/notched_v1/`) as the
template:

| Step | What runs | Notched module |
| --- | --- | --- |
| Family check | hundreds of morphed shapes are checked for validity; a small FEM pilot at the range corners | `pilot.py` |
| Design and HOLDOUT seal | the design of experiments is fixed and a HOLDOUT set is committed before any training | `proposal.py`, `s9.py` |
| FEM production | P2 FEM solves for every TRAIN and DEVELOPMENT case, about 1–2 minutes each on one core | `s10.py` |
| Training | a preregistered training run of the energy-form operator | `s13.py`, `s13_close.py` |
| Verification | error budgets on DEVELOPMENT, then once on HOLDOUT; refusal fixtures; confidence calibration | `s14.py`, `s14_envelope.py` |
| Bundle | build the bundle and check it from a clean install | `s15.py` |

The notched campaign took about a day of CPU time for 200 FEM cases plus
training. Its result shows that the tolerances are not guaranteed: it reached
`DEVELOPMENT_VERIFIED`, not `HOLDOUT_VERIFIED`, largely because of the FEM
mesh's own error at the notch ([S17 results](https://github.com/BBahtiri/verified-neural-operators/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/campaigns/s17-plan.md#4-results)).
The [S17 plan](https://github.com/BBahtiri/verified-neural-operators/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/campaigns/s17-plan.md) and
[ADR 0025](https://github.com/BBahtiri/verified-neural-operators/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0025-s17-examples-and-second-family.md) describe each step and
its decisions.

## Check the project yourself

```text
python -m pytest                                  # full suite, about 50 minutes on one CPU
python -m pytest tests/test_shipped_bundles.py    # the shipped bundles, a few seconds
python -m mkdocs build --strict -f mkdocs.yml -d ../site
```

Historical specimen gate results are recorded under `artifacts/gates/` and
`artifacts/campaigns/*/gates/`, and the [benchmarks](benchmarks.md) page lists
the measured numbers. Re-measuring the benchmarks needs the FEM datasets,
which are not in the repository; regenerate them with the campaign's FEM
production step.
