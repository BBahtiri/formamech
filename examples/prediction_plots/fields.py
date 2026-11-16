"""Plot displacement and first-Piola stress from a saved prediction JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation


def field_arrays(document: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    """Read canonical vertex fields without inferring new mesh connectivity."""
    if not isinstance(document, Mapping) or document.get("verdict") != "IN_DOMAIN":
        raise ValueError("Plotting requires an IN_DOMAIN prediction")
    units = document.get("units", {})
    if not isinstance(units, Mapping) or (
        units.get("coordinates", "mm") != "mm"
        or units.get("u") != "mm"
        or units.get("P_served") != "MPa"
    ):
        raise ValueError("Expected coordinates/u in mm and P_served in MPa")
    try:
        vertices = document["vertices"]
        xy = np.array(vertices["coordinates"], dtype=float, copy=True)
        displacement = np.array(vertices["u"], dtype=float, copy=True)
        stress = np.array(vertices["P_served"], dtype=float, copy=True)
        raw_cells = document["mesh"]["cells"]
        cells = np.array(raw_cells, copy=True)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Missing or invalid prediction mesh/vertex fields") from error
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        raise ValueError("coordinates must contain at least three [x, y] vertices")
    if displacement.shape != xy.shape or stress.shape != (len(xy), 2, 2):
        raise ValueError("Expected one displacement vector and 2x2 stress per vertex")
    if not all(np.isfinite(array).all() for array in (xy, displacement, stress)):
        raise ValueError("Prediction coordinates and fields must be finite")
    if cells.ndim != 2 or cells.shape[1] != 3 or len(cells) == 0:
        raise ValueError("mesh.cells must contain triangular connectivity")
    if cells.dtype.kind not in "iu" or any(
        isinstance(value, (bool, np.bool_))
        for value in np.asarray(raw_cells, dtype=object).flat
    ):
        raise ValueError("Triangle indices must be integers, not booleans or floats")
    if cells.min() < 0 or cells.max() >= len(xy):
        raise ValueError("Triangle indices are outside the vertex array")
    a = xy[cells[:, 1]] - xy[cells[:, 0]]
    b = xy[cells[:, 2]] - xy[cells[:, 0]]
    if np.any(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] == 0):
        raise ValueError("Mesh contains a degenerate triangle")
    return (
        xy,
        cells,
        np.linalg.norm(displacement, axis=1),
        np.linalg.norm(stress, axis=(1, 2)),
    )


def plot_prediction(
    document: Mapping[str, Any],
    output: Path,
    *,
    title: str = "FormaMech prediction",
) -> Path:
    """Save two field panels on the original mesh; never fill holes by remeshing."""
    if output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("Choose a PNG, SVG or PDF output")
    xy, cells, displacement, stress = field_arrays(document)
    mesh = Triangulation(xy[:, 0], xy[:, 1], triangles=cells)
    figure = Figure(figsize=(12, 4.4), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, values, name, unit, color in zip(
        axes,
        (displacement, stress),
        ("Displacement magnitude", "In-plane first Piola stress norm"),
        ("mm", "MPa"),
        ("viridis", "magma"),
        strict=True,
    ):
        image = axis.tripcolor(
            mesh,
            values,
            shading="gouraud",
            cmap=color,
            vmin=0,
            vmax=max(float(values.max()), 1e-12),
        )
        axis.set(title=name, xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
        axis.margins(0.03)
        figure.colorbar(image, ax=axis, label=unit, shrink=0.85, pad=0.03)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.supxlabel(
        "Undeformed coordinates · Model prediction · "
        + str(document.get("claim_level", "NOT_VERIFIED")),
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="FormaMech prediction")
    args = parser.parse_args(argv)
    try:
        if args.prediction.resolve() == args.output.resolve() or (
            args.output.exists() and args.prediction.samefile(args.output)
        ):
            raise ValueError("Output cannot overwrite the input prediction")
        document = json.loads(args.prediction.read_text(encoding="utf-8"))
        plot_prediction(document, args.output, title=args.title)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
