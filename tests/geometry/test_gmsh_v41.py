from __future__ import annotations

from collections import Counter
from pathlib import Path

import meshio
import numpy as np
import pytest

from piginot.contracts import Unit
from piginot.geometry import (
    BoundaryTagError,
    GeometryFormatError,
    GeometryReasonCode,
    GeometryValidationError,
    read_gmsh,
    write_gmsh,
)

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "geometry" / "tagged_square_v41.msh"


def _write_variant(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="ascii")
    return path


def _fixture_text() -> str:
    return _FIXTURE.read_text(encoding="ascii")


def _replace_elements(text: str, elements: str) -> str:
    before, remainder = text.split("$Elements\n", maxsplit=1)
    _, after = remainder.split("$EndElements", maxsplit=1)
    return f"{before}$Elements\n{elements}$EndElements{after}"


def _boundary_name_counts(mesh: meshio.Mesh) -> Counter[str]:
    names_by_physical_id = {
        int(raw[0]): name for name, raw in mesh.field_data.items() if int(raw[1]) == 1
    }
    line_physical_ids = np.concatenate(
        [
            physical_ids
            for block, physical_ids in zip(
                mesh.cells, mesh.cell_data["gmsh:physical"], strict=True
            )
            if block.type == "line"
        ]
    )
    return Counter(names_by_physical_id[int(value)] for value in line_physical_ids)


def test_reads_hand_authored_tagged_gmsh_v41_into_golden_canonical_mesh() -> None:
    assert _fixture_text().startswith("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")

    mesh = read_gmsh(_FIXTURE, unit=Unit.MM)

    assert mesh.nodes == (
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )
    assert mesh.cells == ((0, 2, 3), (0, 3, 1))
    assert mesh.cell_type == "triangle"
    assert mesh.boundary_facets == ((0, 2), (1, 0), (2, 3), (3, 1))
    assert dict(mesh.semantic_tags) == {
        "bottom": (0,),
        "fixed": (1,),
        "loaded": (2,),
        "top": (3,),
    }
    assert mesh.region_ids == (21, 21)
    assert mesh.unit is Unit.MM
    assert mesh.orientation == "counterclockwise"
    assert mesh.topology_signature == ("planar-cell-complex-v1:components=1;holes=0")
    assert mesh.quality_metrics["minimum_angle_degrees"] == pytest.approx(45.0)
    assert mesh.quality_metrics["maximum_angle_degrees"] == pytest.approx(90.0)
    assert mesh.quality_metrics["minimum_edge_length_mm"] == pytest.approx(1.0)
    assert mesh.quality_metrics["maximum_edge_length_mm"] == pytest.approx(2.0**0.5)
    assert mesh.quality_metrics["minimum_area_mm2"] == pytest.approx(0.5)
    assert mesh.quality_metrics["maximum_aspect_ratio"] == pytest.approx(
        2.0**0.5 * (2.0 + 2.0**0.5) / (2.0 * 3.0**0.5)
    )
    assert mesh.source_provenance["format"] == "gmsh-v4.1"

    node_references = {
        node_id
        for entities in (*mesh.cells, *mesh.boundary_facets)
        for node_id in entities
    }
    semantic_facet_ids = {
        facet_id for facet_ids in mesh.semantic_tags.values() for facet_id in facet_ids
    }
    raw_gmsh_ids = {31, 47, 59, 83, 97, 101, 205, 309, 413, 901, 902, 903, 904}
    assert node_references <= set(range(len(mesh.nodes)))
    assert semantic_facet_ids == set(range(len(mesh.boundary_facets)))
    assert node_references.isdisjoint(raw_gmsh_ids)
    assert semantic_facet_ids.isdisjoint(raw_gmsh_ids)


def test_writes_v41_that_meshio_reads_and_reimports_exactly(tmp_path: Path) -> None:
    canonical = read_gmsh(_FIXTURE, unit=Unit.MM)
    destination = tmp_path / "roundtrip.msh"
    second_destination = tmp_path / "roundtrip-again.msh"

    write_gmsh(canonical, destination)
    write_gmsh(canonical, second_destination)

    assert destination.read_bytes() == second_destination.read_bytes()
    assert destination.read_text(encoding="ascii").startswith(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
    )
    independent = meshio.read(destination)
    assert np.array_equal(
        independent.points,
        np.column_stack(
            [np.asarray(canonical.nodes), np.zeros(len(canonical.nodes), dtype=float)]
        ),
    )
    assert np.array_equal(
        independent.get_cells_type("line"),
        np.asarray(canonical.boundary_facets, dtype=int),
    )
    assert np.array_equal(
        independent.get_cells_type("triangle"),
        np.asarray(canonical.cells, dtype=int),
    )
    assert _boundary_name_counts(independent) == Counter(
        {name: len(facet_ids) for name, facet_ids in canonical.semantic_tags.items()}
    )
    assert independent.field_data["region_21"].tolist() == [21, 2]

    restored = read_gmsh(destination, unit=Unit.MM)
    assert restored.nodes == canonical.nodes
    assert restored.cells == canonical.cells
    assert restored.boundary_facets == canonical.boundary_facets
    assert dict(restored.semantic_tags) == dict(canonical.semantic_tags)
    assert restored.region_ids == canonical.region_ids


def test_rejects_any_gmsh_version_other_than_v41(tmp_path: Path) -> None:
    path = _write_variant(
        tmp_path,
        "wrong-version.msh",
        _fixture_text().replace("4.1 0 8", "4.0 0 8", 1),
    )

    with pytest.raises(GeometryFormatError) as caught:
        read_gmsh(path, unit=Unit.MM)

    assert caught.value.reason_code is GeometryReasonCode.UNSUPPORTED_GMSH_VERSION


def test_rejects_nonzero_z_coordinates(tmp_path: Path) -> None:
    source = _fixture_text()
    source = source.replace(
        "\n1 1 0\n0 1 0\n$EndNodes",
        "\n1 1 0.25\n0 1 0\n$EndNodes",
        1,
    )
    path = _write_variant(tmp_path, "non-planar.msh", source)

    with pytest.raises(GeometryValidationError) as caught:
        read_gmsh(path, unit=Unit.MM)

    assert caught.value.reason_code is GeometryReasonCode.UNSUPPORTED_MESH_DIMENSION


def test_missing_boundary_name_reports_source_facet_id(tmp_path: Path) -> None:
    source = _fixture_text().replace("$PhysicalNames\n5\n", "$PhysicalNames\n4\n")
    source = source.replace('1 12 "loaded"\n', "", 1)
    path = _write_variant(tmp_path, "missing-boundary-name.msh", source)

    with pytest.raises(BoundaryTagError) as caught:
        read_gmsh(path, unit=Unit.MM)

    assert caught.value.reason_code is GeometryReasonCode.UNCLASSIFIED_BOUNDARY_FACET
    assert caught.value.facet_ids == (1,)


def test_ambiguous_boundary_name_reports_source_facet_id(tmp_path: Path) -> None:
    source = _fixture_text().replace("$PhysicalNames\n5\n", "$PhysicalNames\n6\n")
    source = source.replace(
        '1 12 "loaded"\n',
        '1 12 "loaded"\n1 12 "also_loaded"\n',
        1,
    )
    path = _write_variant(tmp_path, "ambiguous-boundary-name.msh", source)

    with pytest.raises(BoundaryTagError) as caught:
        read_gmsh(path, unit=Unit.MM)

    assert caught.value.reason_code is GeometryReasonCode.AMBIGUOUS_BOUNDARY_TAG
    assert caught.value.facet_ids == (1,)


_LINE_BLOCKS = """\
1 31 1 1
901 101 205
1 47 1 1
902 205 309
1 59 1 1
903 309 413
1 83 1 1
904 413 101
"""


def test_rejects_unsupported_area_cell_type(tmp_path: Path) -> None:
    elements = f"""\
5 5 901 905
{_LINE_BLOCKS}2 97 3 1
905 101 205 309 413
"""
    path = _write_variant(
        tmp_path,
        "quad-only.msh",
        _replace_elements(_fixture_text(), elements),
    )

    with pytest.raises(GeometryValidationError) as caught:
        read_gmsh(path, unit=Unit.MM)

    assert caught.value.reason_code is GeometryReasonCode.UNSUPPORTED_CELL_TYPE


def test_rejects_mixed_area_cell_blocks(tmp_path: Path) -> None:
    elements = f"""\
6 7 901 907
{_LINE_BLOCKS}2 97 2 2
905 101 205 309
906 101 309 413
2 97 3 1
907 101 205 309 413
"""
    path = _write_variant(
        tmp_path,
        "mixed-triangle-quad.msh",
        _replace_elements(_fixture_text(), elements),
    )

    with pytest.raises(GeometryValidationError) as caught:
        read_gmsh(path, unit=Unit.MM)

    assert caught.value.reason_code is GeometryReasonCode.MIXED_CELL_TYPES
