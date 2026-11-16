"""Scientific field plots preserve supplied topology and canonical quantities."""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")


@pytest.fixture
def fields():
    return importlib.import_module("examples.prediction_plots.fields")


@pytest.fixture
def document():
    return {
        "verdict": "IN_DOMAIN",
        "claim_level": "NOT_VERIFIED",
        "units": {"coordinates": "mm", "u": "mm", "P_served": "MPa"},
        "mesh": {
            "mesh_id": "a" * 64,
            "cells": [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]],
        },
        "vertices": {
            "coordinates": [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]],
            "u": [[3, 4], [0, 12], [5, 12], [8, 15], [7, 24], [20, 21]],
            "P_served": [
                [[1, 2], [2, 4]],
                [[0, -3], [4, 0]],
                [[6, 0], [0, 8]],
                [[1, 2], [2, -4]],
                [[0, 0], [0, 0]],
                [[3, -4], [0, 0]],
            ],
        },
    }


def test_fields_use_vertex_vector_and_tensor_norms_without_mutating_input(
    fields, document
):
    before = copy.deepcopy(document)
    xy, cells, displacement, stress = fields.field_arrays(document)
    np.testing.assert_array_equal(xy, document["vertices"]["coordinates"])
    np.testing.assert_array_equal(cells, document["mesh"]["cells"])
    np.testing.assert_allclose(displacement, [5, 12, 13, 17, 25, 29])
    np.testing.assert_allclose(stress, [5, 5, 10, 5, 0, 5])
    assert document == before


def test_prepared_cad_units_may_omit_canonical_coordinate_unit(fields, document):
    del document["units"]["coordinates"]
    xy, cells, displacement, stress = fields.field_arrays(document)
    assert xy.shape == (6, 2) and cells.shape == (4, 3)
    assert displacement.shape == stress.shape == (6,)


@pytest.mark.parametrize("bad", [None, [], {}, {"verdict": "REFUSED"}])
def test_missing_and_refused_predictions_are_clear_refusals(fields, bad):
    with pytest.raises(ValueError, match=".+"):
        fields.field_arrays(bad)


@pytest.mark.parametrize("section", ["mesh", "vertices"])
def test_partial_in_domain_predictions_are_clear_refusals(fields, document, section):
    del document[section]
    with pytest.raises(ValueError, match=".+"):
        fields.field_arrays(document)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("units", "coordinates", "m"),
        ("units", "coordinates", None),
        ("units", "u", "cm"),
        ("units", "u", None),
        ("units", "P_served", "Pa"),
        ("units", "P_served", None),
        ("vertices", "coordinates", []),
        ("vertices", "coordinates", [[0, 1, 2]]),
        ("vertices", "coordinates", [[0, float("nan")]] * 6),
        ("vertices", "coordinates", [[0, float("inf")]] * 6),
        ("vertices", "u", [[0, 0]]),
        ("vertices", "u", [[0, 0, 0]] * 6),
        ("vertices", "u", [[0, float("nan")]] * 6),
        ("vertices", "P_served", [[1, 2]] * 6),
        ("vertices", "P_served", [[[0, 0], [0, float("inf")]]] * 6),
        ("mesh", "cells", []),
        ("mesh", "cells", [[0, 1, 2, 3]]),
        ("mesh", "cells", [[0.0, 1.0, 3.0]]),
        ("mesh", "cells", [[True, 2, 3]]),
        ("mesh", "cells", [[-1, 1, 3]]),
        ("mesh", "cells", [[0, 1, 6]]),
        ("mesh", "cells", [[0, 0, 3]]),
    ],
)
def test_noncanonical_or_malformed_fields_are_refused(
    fields, document, section, key, value
):
    document[section][key] = value
    with pytest.raises(ValueError, match=".+"):
        fields.field_arrays(document)


def test_distinct_but_collinear_triangle_is_refused(fields, document):
    document["vertices"]["coordinates"][3] = [1, 0]
    with pytest.raises(ValueError, match=".+"):
        fields.field_arrays(document)


@pytest.mark.parametrize("extension", ["png", "svg", "pdf"])
def test_render_keeps_concave_mesh_connectivity_and_labels(
    fields, document, tmp_path, monkeypatch, extension
):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.tri import Triangulation

    triangulations = []
    figures = []
    original_triangulation = Triangulation.__init__
    original_savefig = Figure.savefig

    def triangulate(self, x, y, triangles=None, mask=None):
        assert triangles is not None, "automatic triangulation would fill the L cutout"
        triangulations.append(np.asarray(triangles).copy())
        original_triangulation(self, x, y, triangles=triangles, mask=mask)

    def savefig(self, *args, **kwargs):
        figures.append(
            {
                "canvas": self.canvas,
                "panels": len(
                    [axis for axis in self.axes if axis.get_label() != "<colorbar>"]
                ),
                "labels": " ".join(
                    text
                    for axis in self.axes
                    for text in (
                        axis.get_xlabel(),
                        axis.get_ylabel(),
                        axis.get_title(),
                    )
                ),
                "titles": [text.get_text() for text in self.texts],
            }
        )
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Triangulation, "__init__", triangulate)
    monkeypatch.setattr(Figure, "savefig", savefig)
    output = tmp_path / "new" / "plots" / f"fields.{extension}"
    assert (
        fields.plot_prediction(document, output, title="Synthetic L profile") == output
    )
    payload = output.read_bytes()
    assert payload
    if extension == "png":
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    elif extension == "svg":
        assert b"<svg" in payload[:1000]
    else:
        assert payload.startswith(b"%PDF-")
    assert triangulations
    expected = {tuple(sorted(cell)) for cell in document["mesh"]["cells"]}
    assert all(
        {tuple(sorted(cell)) for cell in triangles} == expected
        for triangles in triangulations
    )
    assert len(figures) == 1
    assert figures[0]["panels"] == 2
    labels = figures[0]["labels"]
    assert "mm" in labels and "MPa" in labels
    assert "Synthetic L profile" in figures[0]["titles"]
    assert isinstance(figures[0]["canvas"], FigureCanvasAgg)


def _cli_refuses(fields, arguments):
    with pytest.raises((ValueError, SystemExit)) as error:
        fields.main(arguments)
    if isinstance(error.value, SystemExit):
        assert error.value.code == 2
    else:
        assert str(error.value).strip()


def test_json_cli_writes_an_image_and_preserves_its_prediction(
    fields, document, tmp_path
):
    source = tmp_path / "prediction.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    before = source.read_bytes()
    output = tmp_path / "plots" / "cli.png"
    assert (
        fields.main(
            [
                "--prediction",
                str(source),
                "--output",
                str(output),
                "--title",
                "CLI profile",
            ]
        )
        == 0
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert source.read_bytes() == before


@pytest.mark.parametrize("kind", ["json", "refused", "units", "format"])
def test_json_cli_refuses_invalid_inputs_without_publishing_output(
    fields, document, tmp_path, kind
):
    source = tmp_path / "prediction.json"
    if kind == "refused":
        document["verdict"] = "REFUSED"
    elif kind == "units":
        document["units"]["u"] = "m"
    source.write_text("{" if kind == "json" else json.dumps(document), encoding="utf-8")
    output = tmp_path / ("plot.gif" if kind == "format" else "plot.png")
    _cli_refuses(fields, ["--prediction", str(source), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_json_cli_cannot_overwrite_its_input(fields, document, tmp_path, alias):
    source = tmp_path / "prediction.png"
    source.write_text(json.dumps(document), encoding="utf-8")
    before = source.read_bytes()
    output = source if alias == "direct" else tmp_path / "alias.png"
    try:
        if alias == "symlink":
            output.symlink_to(source)
        elif alias == "hardlink":
            os.link(source, output)
    except OSError as error:
        pytest.skip(f"file-link capability unavailable: {error}")
    _cli_refuses(fields, ["--prediction", str(source), "--output", str(output)])
    assert source.read_bytes() == before


@pytest.fixture
def demo():
    return importlib.import_module("examples.prediction_plots.__main__")


@pytest.fixture
def demo_inputs(document, tmp_path, monkeypatch):
    from piginot import bundle as bundle_api

    root = tmp_path / "source"
    requests = root / "bundles" / "requests"
    requests.mkdir(parents=True)
    bundles = {}
    for name, folder, scale in (
        ("dogbone", "dogbone-v1-s13c-cp-freeze", 1),
        ("notched", "notched-v1-cp-freeze", 2),
    ):
        directory = root / "bundles" / folder
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"synthetic unchanged weights")
        (requests / f"{name}-in-domain.json").write_text(
            json.dumps(
                {
                    "inputs": {
                        "nominal_axial_strain": {"value": 0.15, "unit": "1"},
                        "elastic_modulus": {"value": 760, "unit": "MPa"},
                    }
                }
            ),
            encoding="utf-8",
        )
        bundles[folder] = SimpleNamespace(
            name=name,
            scale=scale,
            manifest=SimpleNamespace(manifest_sha256=str(scale) * 64),
        )
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    calls = []

    def predict(bundle, inputs, queries=None, *, confidence=True):
        assert queries is None and confidence is False
        strain = float(inputs["nominal_axial_strain"].value)
        calls.append((bundle.name, strain))
        current = copy.deepcopy(document)
        for field in ("u", "P_served"):
            current["vertices"][field] = (
                np.asarray(current["vertices"][field]) * strain * bundle.scale
            ).tolist()
        return SimpleNamespace(document=lambda: current)

    monkeypatch.setattr(bundle_api, "load_bundle", lambda path: bundles[path.name])
    monkeypatch.setattr(bundle_api, "predict", predict)
    return root, before, calls


def test_demo_uses_ten_fast_predictions_and_preserves_inputs_and_thread_count(
    demo, demo_inputs, tmp_path
):
    import torch

    root, before, calls = demo_inputs
    output = tmp_path / "demo"
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        report = demo.run_demo(output, root=root)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous)
    assert report == json.loads((output / "summary.json").read_text("utf-8"))
    assert report["training_runs"] == report["fem_solves"] == 0
    assert set(path.name for path in output.iterdir()) == {
        "dogbone-fields.png",
        "notched-fields.png",
        "load-response.png",
        "dogbone-prediction.json",
        "notched-prediction.json",
        "summary.json",
    }
    strains = [0.10, 0.125, 0.15, 0.175, 0.20]
    assert calls == [
        (name, strain) for name in ("dogbone", "notched") for strain in strains
    ]
    for name, scale in (("dogbone", 1), ("notched", 2)):
        rows = report["response"][name]
        assert [row["nominal_axial_strain"] for row in rows] == strains
        assert [row["max_displacement_mm"] for row in rows] == pytest.approx(
            [29 * strain * scale for strain in strains]
        )
        assert [row["max_piola_MPa"] for row in rows] == pytest.approx(
            [10 * strain * scale for strain in strains]
        )
        model = report["models"][name]
        assert model["manifest_sha256"] == str(scale) * 64
        assert model["claim_level"] == "NOT_VERIFIED"
        assert model["base_inputs"]["nominal_axial_strain"]["value"] == 0.15
    for path in output.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_demo_refuses_existing_outputs_before_loading_models(
    demo, tmp_path, kind, monkeypatch
):
    from piginot import bundle as bundle_api

    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
        preserved = output / "keep.txt"
    else:
        preserved = output
    preserved.write_bytes(b"preserve this output")
    monkeypatch.setattr(
        bundle_api,
        "load_bundle",
        lambda path: pytest.fail("existing output loaded a model"),
    )
    with pytest.raises(ValueError, match=".+"):
        demo.run_demo(output)
    assert preserved.read_bytes() == b"preserve this output"


def test_demo_refusal_restores_thread_count_and_publishes_no_summary(
    demo, demo_inputs, tmp_path, monkeypatch
):
    import torch

    from piginot import bundle as bundle_api

    root, before, _ = demo_inputs
    output = tmp_path / "refused"
    monkeypatch.setattr(
        bundle_api,
        "predict",
        lambda *args, **kwargs: SimpleNamespace(
            document=lambda: {"verdict": "REFUSED"}
        ),
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with pytest.raises(ValueError, match="IN_DOMAIN"):
            demo.run_demo(output, root=root)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous)
    assert not (output / "summary.json").exists()
    assert not tuple(output.glob("*.png"))
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize("explicit", [False, True])
def test_demo_cli_uses_the_declared_output_directory(
    demo, tmp_path, monkeypatch, explicit
):
    output = tmp_path / "plots" if explicit else Path(".piginot-runs/prediction-plots")
    calls = []
    monkeypatch.setattr(demo, "run_demo", lambda value: calls.append(value))
    assert demo.main(["--output", str(output)] if explicit else []) == 0
    assert calls == [output]


def test_demo_cli_reports_refusals(demo, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    _cli_refuses(demo, ["--output", str(output)])
