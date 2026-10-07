# Bundles

The two trained operators of release 0.1.0, ready to serve with
`piginot predict`. Each bundle is a hash-closed directory: `bundle.json` lists
the SHA-256 of every file, and a bundle whose bytes differ is refused. Do not
edit the files. Git is told not to convert them (`.gitattributes`).

| Bundle | Family | Inputs (envelope) | Claim level |
| --- | --- | --- | --- |
| `dogbone-v1-s13c-cp-freeze` | DogBone tensile specimen | `L_total` 40–70 mm, `W_grip` 16–26 mm, `W_gauge` 6–14 mm, `R_fillet` 8–20 mm, `nominal_axial_strain` 0.10–0.20 | `HOLDOUT_VERIFIED` |
| `notched-v1-cp-freeze` | centre-notched plate | `width` 16–30 mm, `notch_radius` 2–8 mm, `notch_depth` 1–5 mm, `nominal_axial_strain` 0.10–0.20 | `DEVELOPMENT_VERIFIED` |

The material is fixed in both: `elastic_modulus` 760 MPa, `poisson_ratio`
0.23. Derived bounds (for example the DogBone gauge-to-grip ratio, or the notch
depth relative to the width and the radius) also apply. `piginot inspect`
lists them, and `piginot predict` refuses violations with a reason code.

`requests/` holds example requests: one in-domain request per bundle, and one
the notched bundle refuses because the strain is out of range. See the
[user guide](../docs/user-guide.md) for how to run them, and the
[limitations](../docs/limitations.md) before using any number.

The manifests match the S15 gate records: DogBone `009db9d3…`
(`artifacts/campaigns/dogbone-v1/S15-VALIDATION.json`) and notched
`3749f4be…` (`artifacts/campaigns/notched-v1/gates/S15.json`).
