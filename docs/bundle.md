# Bundles

A bundle is a directory that holds everything needed to serve one verified
operator: the family's reference mesh, morph map and executable envelope, the
model specification, the weights, the verification report and the
limitations. `bundle.json` lists the SHA-256 and size of every file and its own
hash, and a bundle is refused if any byte differs
([ADR 0023](https://github.com/BBahtiri/verified-neural-operators/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0023-s15-bundle-and-cli.md)).

## Install for serving

```text
python -m pip install "piginot[runtime]"
```

The runtime needs `numpy`, `torch`, `safetensors` and `rfc8785` only.

## Commands

```text
piginot bundle verify BUNDLE
piginot inspect BUNDLE [--output report.json]
piginot predict BUNDLE --request request.json [--output prediction.json] [--no-confidence]
```

`bundle verify` checks every byte and that the parts agree. `inspect` reports
the envelope, the model, the claim level, the verification verdict, every
outstanding (failed, waived or not-run) item with its reason, and the
limitations. `predict` exits with 0 for an in-domain prediction, 2 for a
refused request and 1 when the bundle itself is refused.

## Request

Inputs are quantities with explicit units; the envelope's units are canonical
(`mm`, `MPa`, `1`), and other units of the same dimension are converted.

```json
{
  "inputs": {
    "L_total": {"value": 54.0, "unit": "mm"},
    "W_grip": {"value": 20.0, "unit": "mm"},
    "W_gauge": {"value": 10.0, "unit": "mm"},
    "R_fillet": {"value": 12.0, "unit": "mm"},
    "elastic_modulus": {"value": 760.0, "unit": "MPa"},
    "poisson_ratio": {"value": 0.23, "unit": "1"},
    "nominal_axial_strain": {"value": 0.15, "unit": "1"}
  },
  "queries": [[5.0, 2.0], [13.5, 4.0]]
}
```

`queries` is optional. Coordinates are in the bundle's reference frame (for the
DogBone bundle, the quarter model `[0, L/2] x [0, W_grip/2]`).

## Prediction

| Field | Meaning |
| --- | --- |
| `verdict` | `IN_DOMAIN` or `REFUSED` |
| `reason_codes` | sorted, stable codes of a refusal (S9, S6, S5 and `PREDICTION_*`) |
| `claim_level` | the bundle's claim for an in-domain request, else `NOT_VERIFIED` |
| `identities` | bundle manifest, weights and model-spec hashes, envelope, mesh |
| `units` | the unit of every field |
| `stress_authority` | where `P_served` comes from |
| `vertices`, `queries` | `coordinates`, `u`, `F_2D`, `F33`, `J_3D`, `P_served`, `sigma` and `confidence` |
| `qois` | each QoI with its value, unit, vertex and location |
| `constitutive_gap` | `max |P_served − P_material| / E` |
| `limitations` | the required limitation paragraph |

`P_served` is the stress the bundle declares; for the energy-form operator it
is the plane-stress Neo-Hookean first Piola stress of the predicted
deformation. `sigma = P_served F_2Dᵀ / J_3D` is its in-plane Cauchy block.
`confidence.stress_error_bound` is the calibrated 90% quantile of the
pointwise in-plane stress error for the confidence indicator at that point;
`beyond_calibration` marks indicators above the calibrated range.

## Building a bundle

Bundles are built by a campaign plugin, for example:

```text
python -m piginot.cli bundle build --campaign dogbone_v1 -- \
    --output BUNDLE --weights CP-FREEZE/model.safetensors --created-utc 2026-10-03T00:00:00Z
```
