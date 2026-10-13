# Problem specification

A problem specification (`piginot.problem-spec-v1`) describes one 2D
finite-strain problem as a JSON file: geometry, parameter ranges, material,
boundary conditions, exact Dirichlet lifting, quantities of interest,
verification tolerances, DOE budget and FEM settings. It is introduced by
S24 ([ADR 0106](adr/0106-s24-problem-spec-and-lifting.md)) and is the
input of the one-command compiler planned for S28. The two shipped families
are written this way in `campaigns/dogbone_v1/problem_spec.json` and
`campaigns/notched_v1/problem_spec.json`.

```python
from piginot.contracts import ProblemSpec

spec = ProblemSpec.load("campaigns/notched_v1/problem_spec.json")
spec.problem_hash  # BLAKE2b-256 identity, bound by training runs
```

Reading validates the file; a refusal is a `ProblemSpecError` whose `rule`
names what is wrong (for example `unknown-key`, `unit`, `range`,
`dirichlet-neumann`, `lifting-coverage`). `spec.check_mesh(mesh)` adds the
rules that need the reference mesh: every region exists, no rigid-body mode
is left free, and a nominal-strain region is perpendicular to its component.

## Sections

| Key | Content |
|---|---|
| `schema`, `name`, `physics` | `piginot.problem-spec-v1`; a lowercase name; `2d-plane-stress-finite-strain` |
| `parameters` | Request inputs: `name`, `role` (`geometry`, `material`, `load`), `lo`, `hi`, `unit` (`mm`, `MPa` or `1`), `distribution` (`uniform`, or `fixed` when `lo = hi`) |
| `constants` | Optional named values with a unit, usable in expressions |
| `constraints` | Optional feasibility bounds: `name`, `expression`, `lo`, `hi`, `unit` |
| `geometry` | `reference_mesh_id`, `morph_map_sha256`, `regions` (boundary tags), `frame` (`origin` and `extent` of the bounding box, as expressions) |
| `material` | `model` (`compressible_neo_hookean` until S25) and the parameter that sets each property |
| `boundary_conditions` | `name`, `region`, `kind` (`dirichlet`, `symmetry`, `traction_free`, `neumann`), `components` (`x`, `y`), `values` |
| `lifting` | `kind` (`quarter` or `adf`), `displacement_scale`, and for `adf` the `pieces` of every Dirichlet and symmetry condition |
| `qois` | `name`, `field` (`P_served`, `sigma`), `reduction` (`max-frobenius`), `region`, `exclude_regions` |
| `verification` | `quantile` and the tolerance of `absolute_peak_error`, `p_relative_l2`, `u_relative_l2` |
| `split` | DOE budget per role: `pilot`, `train`, `development`, `holdout`, `reserve` |
| `fem` | Reference solver settings and the FEM material name |

## Expressions and values

Expressions use names of parameters and constants, numbers, `+ - * /` and
parentheses. They are parsed into a syntax tree and evaluated left to right;
nothing is passed to `eval`. Units are checked: a frame entry is a length, a
constraint has its declared unit, and a bare number takes the unit expected
where it stands.

A Dirichlet value is either `{"expression": "...", "unit": "mm"}` or
`{"nominal_strain": "<parameter>"}`, which prescribes `u_c = strain × x_c` on
a region perpendicular to `c`. A symmetry condition prescribes zero.
`neumann` tractions validate but are not yet supported by the builders.

## Liftings

The operator's displacement is `u = offset + scale ⊙ φ`, exact on the
Dirichlet set for any network output `φ`.

- `quarter`: the lifting of the shipped bundles, for a frame at the origin
  with `u_x` fixed on two regions (one at zero) and `u_y` fixed at zero on
  one.
- `adf`: approximate distance functions over line segments and minor
  circular arcs (Sukumar and Srivastava, 2022), each traversed with the domain
  on its left:

  ```json
  "pieces": {
    "left_x_symmetry": [{"line": {"start": ["0", "0.5 * W"], "end": ["0", "0"]}}],
    "rim_clamp": [{"arc": {"start": ["r", "0"], "end": ["0", "r"], "center": ["0", "0"]}}]
  }
  ```

  A line may be marked `"trimmed": false` when its whole supporting line meets
  the domain only along the piece.

`piginot.problem.verify_lifting` measures a spec's lifting over its parameter
box on the morphed meshes (ADR 0106 decision 4).
