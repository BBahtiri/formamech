"""Generate specimen field plots and model load-response curves on CPU."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .fields import field_arrays, plot_prediction

REPOSITORY = Path(__file__).resolve().parents[2]
STRAINS = (0.10, 0.125, 0.15, 0.175, 0.20)
FAMILIES = (
    ("dogbone", "DogBone", "dogbone-v1-s13c-cp-freeze"),
    ("notched", "Notched plate", "notched-v1-cp-freeze"),
)


def _plot_response(series: Mapping[str, list[dict[str, float]]], output: Path) -> None:
    figure = Figure(figsize=(12, 4.4), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for (name, label, _), color in zip(FAMILIES, ("#2563eb", "#c2410c"), strict=True):
        points = series[name]
        strain = [100 * row["nominal_axial_strain"] for row in points]
        for axis, key in zip(
            axes, ("max_displacement_mm", "max_piola_MPa"), strict=True
        ):
            axis.plot(
                strain, [row[key] for row in points], "o-", label=label, color=color
            )
    for axis, label in zip(
        axes,
        ("Maximum vertex displacement (mm)", "Maximum vertex first Piola norm (MPa)"),
        strict=True,
    ):
        axis.set(xlabel="Nominal axial strain (%)", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(
        "Load response · Shipped model predictions", fontsize=15, fontweight="bold"
    )
    figure.supxlabel(
        "Fixed specimen geometry and material · Vertex maxima · No FEM comparison in this example",
        fontsize=10,
    )
    figure.savefig(output, dpi=180, facecolor="white")


def _predict_family(
    root: Path, output: Path, name: str, label: str, bundle_name: str
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    from formamech.bundle import load_bundle, predict

    from formamech.cli import parse_request

    bundle = load_bundle(root / "bundles" / bundle_name)
    request = json.loads(
        (root / "bundles/requests" / f"{name}-in-domain.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for strain in STRAINS:
        current = {
            "inputs": {
                **request["inputs"],
                "nominal_axial_strain": {"value": strain, "unit": "1"},
            }
        }
        inputs, _ = parse_request(current)
        document = predict(bundle, inputs, confidence=False).document()
        _, _, displacement, stress = field_arrays(document)
        rows.append(
            {
                "nominal_axial_strain": strain,
                "max_displacement_mm": float(displacement.max()),
                "max_piola_MPa": float(stress.max()),
            }
        )
        if strain == 0.15:
            (output / f"{name}-prediction.json").write_text(
                json.dumps(document, allow_nan=False) + "\n", encoding="utf-8"
            )
            plot_prediction(
                document,
                output / f"{name}-fields.png",
                title=f"{label} · Quarter model · 15% nominal axial strain",
            )
    return rows, {
        "bundle": bundle_name,
        "manifest_sha256": bundle.manifest.manifest_sha256,
        "claim_level": document["claim_level"],
        "base_inputs": request["inputs"],
    }


def run_demo(output: Path, *, root: Path = REPOSITORY) -> dict[str, Any]:
    """Use five public in-envelope loads per shipped model; perform no fitting."""
    import torch

    if output.exists():
        raise ValueError("Choose a new output directory for the demo")
    output.mkdir(parents=True)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        responses, models = {}, {}
        for name, label, bundle_name in FAMILIES:
            responses[name], models[name] = _predict_family(
                root, output, name, label, bundle_name
            )
        _plot_response(responses, output / "load-response.png")
        report = {
            "models": models,
            "response": responses,
            "training_runs": 0,
            "fem_solves": 0,
        }
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return report
    finally:
        torch.set_num_threads(previous_threads)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".formamech-runs/prediction-plots")
    )
    args = parser.parse_args(argv)
    try:
        run_demo(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
