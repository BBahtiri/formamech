# Predict with the shipped bundles

This demo uses the existing family-specific bundles in the source
checkout. It predicts a field and inspects an envelope refusal. The
[limitations](../limitations.md) describe each bundle's supported scope.

## Install and check

From the cloned repository, install runtime dependencies:

```text
python -m pip install -e ".[runtime]"
formamech bundle verify bundles/dogbone-v1-s13c-cp-freeze
formamech bundle verify bundles/notched-v1-cp-freeze
```

The bundles are separate from the Python wheel. If copying them to another
machine, preserve each complete directory and all relative paths. Runtime
prediction does not require the source CAD file, mesher or FEM stack.

## Predict

Create an output directory under the ignored run folder:

```text
python -c "from pathlib import Path; Path('.formamech-runs/demo').mkdir(parents=True, exist_ok=True)"
formamech predict bundles/dogbone-v1-s13c-cp-freeze --request bundles/requests/dogbone-in-domain.json --mode fast --no-confidence --output .formamech-runs/demo/dogbone-prediction.json
formamech predict bundles/notched-v1-cp-freeze --request bundles/requests/notched-in-domain.json --mode fast --no-confidence --output .formamech-runs/demo/notched-prediction.json
```

The requests use each model's own material, shape and load envelope. Check
the response's envelope verdict, units, vertex fields, quantities of interest,
limitations and claim level. An integrity check establishes matching files;
it does not independently establish a prediction's accuracy.

## Inspect a refusal

```text
formamech predict bundles/notched-v1-cp-freeze --request bundles/requests/notched-out-of-envelope.json --mode fast --no-confidence --output .formamech-runs/demo/refused.json
```

This supplied request is outside the bundle envelope. The command returns
exit code `2`, with `REFUSED`, `PARAMETER_OUT_OF_BOUNDS` and `NOT_VERIFIED` in
the response. Investigate other errors rather than treating every nonzero
exit code as this expected refusal.

To use a different fixed CAD profile or material, create its own candidate
with the [CAD workflow](own-cad-workflow.md) or the
[custom-material tutorial](s29-custom-material.md). Such candidates retain
`NOT_VERIFIED` and support uncalibrated fast prediction.
