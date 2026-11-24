"""Import one planar STEP/DXF face into a tagged, canonical linear mesh."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any

from piginot.cad._errors import CadImportError
from piginot.cad.mesh_policy import MeshPolicy
from piginot.contracts import CanonicalMesh, Unit


@dataclass(frozen=True, slots=True)
class BoundaryInfo:
    name: str
    kind: str
    source_id: int
    source_label: str

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source_id": self.source_id,
            "source_label": self.source_label,
        }


@dataclass(frozen=True, slots=True)
class ImportedCad:
    mesh: CanonicalMesh
    boundaries: tuple[BoundaryInfo, ...]
    source_sha256: str
    source_format: str
    provenance: Mapping[str, str]
    mesh_policy: MeshPolicy | None = None
    mesh_evidence: Mapping[str, Any] | None = None


def _validate(path: Path, unit: Unit, size: float) -> str:
    if (
        isinstance(size, bool)
        or not isinstance(size, (int, float))
        or not math.isfinite(size)
        or size <= 0
    ):
        raise CadImportError(
            "mesh_size", "mesh_size_mm must be a positive finite number"
        )
    if unit not in (Unit.MM, Unit.CM, Unit.M):
        raise CadImportError("unit", "declare mm, cm or m explicitly")
    if path.is_symlink() or not path.is_file():
        raise CadImportError("source", "CAD input must be a regular file")
    suffix = path.suffix.lower()
    if suffix not in (".step", ".stp", ".dxf"):
        raise CadImportError("format", "supported CAD formats are STEP, STP and DXF")
    return "dxf" if suffix == ".dxf" else "step"


def _backend() -> Any:
    try:
        gmsh = __import__("gmsh")
    except (ImportError, OSError) as error:
        raise CadImportError(
            "dependency", "CAD import requires the cad extra and Gmsh native libraries"
        ) from error
    return gmsh


def _step(gmsh: Any, path: Path) -> tuple[int, Mapping[int, str]]:
    # STEP declares its own units. OCC converts them to canonical mm once;
    # the caller's length unit is never used as an additional scale factor.
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    gmsh.model.occ.importShapes(str(path), highestDimOnly=False, format="step")
    gmsh.model.occ.synchronize()
    faces = gmsh.model.getEntities(2)
    if gmsh.model.getEntities(3) or len(faces) != 1:
        raise CadImportError(
            "topology",
            "STEP requires one planar face, not solids or disconnected faces",
        )
    return int(faces[0][1]), {}


def _planar_boundary(gmsh: Any, surface: int) -> tuple[tuple[int, ...], float]:
    box = gmsh.model.getBoundingBox(2, surface)
    tolerance = 1e-7 * max(1.0, box[3] - box[0], box[4] - box[1])
    # OCC bounding boxes add geometric tolerance even for an exact plane.
    if box[5] - box[2] > 4 * tolerance:
        raise CadImportError(
            "planarity", "only profiles in a plane parallel to XY are supported"
        )
    offset = (box[2] + box[5]) / 2
    if offset != 0:
        gmsh.model.occ.translate([(2, surface)], 0, 0, -offset)
        gmsh.model.occ.synchronize()
    boundary = tuple(
        sorted(
            abs(int(tag))
            for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False)
            if dim == 1
        )
    )
    if not boundary or set(boundary) != {
        int(tag) for _, tag in gmsh.model.getEntities(1)
    }:
        raise CadImportError(
            "topology", "profile contains unused curves or lacks a closed boundary"
        )
    return boundary, offset


def _boundaries(
    gmsh: Any, curves: tuple[int, ...], labels: Mapping[int, str]
) -> tuple[BoundaryInfo, ...]:
    result = tuple(
        BoundaryInfo(
            f"edge-{index:06d}",
            {"Line": "line", "Circle": "arc"}.get(gmsh.model.getType(1, tag), "other"),
            tag,
            labels.get(tag, gmsh.model.getEntityName(1, tag)),
        )
        for index, tag in enumerate(curves, start=1)
    )
    for item in result:
        group = gmsh.model.addPhysicalGroup(1, [item.source_id])
        gmsh.model.setPhysicalName(1, group, item.name)
    return result


def _mesh(
    gmsh: Any,
    surface: int,
    boundaries: tuple[BoundaryInfo, ...],
    provenance: Mapping[str, str],
) -> CanonicalMesh:
    from piginot.geometry.canonicalize import canonicalize_mesh
    from piginot.geometry.models import CellType, MeshPayload

    gmsh.model.addPhysicalGroup(2, [surface], tag=1)
    gmsh.model.mesh.generate(2)
    element_types, _, elements = gmsh.model.mesh.getElements(2, surface)
    if tuple(int(kind) for kind in element_types) != (2,):
        raise CadImportError("mesh", "CAD mesher must produce linear triangles only")
    cells = tuple(
        tuple(int(value) for value in elements[0][i : i + 3])
        for i in range(0, len(elements[0]), 3)
    )
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coordinates = {
        int(tag): tuple(float(value) for value in coords[3 * i : 3 * i + 3])
        for i, tag in enumerate(tags)
    }
    used = sorted({index for cell in cells for index in cell})
    if any(abs(coordinates[tag][2]) > 1e-9 for tag in used):
        raise CadImportError("planarity", "meshed profile is not planar XY")
    indices = {tag: index for index, tag in enumerate(used)}
    nodes = tuple((coordinates[tag][0], coordinates[tag][1]) for tag in used)
    triangles = tuple(tuple(indices[tag] for tag in cell) for cell in cells)

    # Orient generated elements to the canonical XY normal, without changing
    # coordinates, connectivity membership or imported CAD geometry.
    def oriented(cell: tuple[int, ...]) -> tuple[int, ...]:
        a, b, c = (nodes[index] for index in cell)
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return cell if cross > 0 else (cell[0], cell[2], cell[1])

    facets: list[tuple[int, int]] = []
    semantic: dict[str, tuple[int, ...]] = {}
    for boundary in boundaries:
        kinds, _, blocks = gmsh.model.mesh.getElements(1, boundary.source_id)
        if tuple(int(kind) for kind in kinds) != (1,):
            raise CadImportError("mesh", "CAD boundary must contain linear segments")
        start = len(facets)
        facets.extend(
            (indices[int(blocks[0][i])], indices[int(blocks[0][i + 1])])
            for i in range(0, len(blocks[0]), 2)
        )
        semantic[boundary.name] = tuple(range(start, len(facets)))
    mesh = canonicalize_mesh(
        MeshPayload(
            nodes,
            tuple(oriented(cell) for cell in triangles),
            CellType.TRIANGLE,
            tuple(facets),
            semantic,
            (1,) * len(triangles),
            Unit.MM,
            provenance,
        )
    )
    if "components=1;" not in mesh.topology_signature:
        raise CadImportError("topology", "CAD profile must be one connected domain")
    return mesh


def import_cad(
    path: Path,
    *,
    unit: Unit,
    mesh_size_mm: float | None = None,
    mesh_policy: MeshPolicy | None = None,
) -> ImportedCad:
    """Import one XY profile. STEP embedded units prevail; DXF units must agree.

    Gmsh is opened only for this call and never takes over a caller's session.
    Curved CAD boundaries are represented by generated straight mesh edges;
    this is a discrete geometry, not a continuum geometric error certificate.
    """
    if (mesh_size_mm is None) == (mesh_policy is None):
        raise CadImportError(
            "mesh_policy", "supply exactly one of mesh_size_mm and mesh_policy"
        )
    if mesh_policy is not None:
        if not isinstance(mesh_policy, MeshPolicy):
            raise CadImportError("mesh_policy", "mesh_policy must be a MeshPolicy")
        mesh_size_mm = mesh_policy.global_size_mm
    source = Path(path)
    assert mesh_size_mm is not None
    source_format = _validate(source, unit, mesh_size_mm)
    payload = source.read_bytes()
    gmsh = _backend()
    if gmsh.isInitialized():
        raise CadImportError(
            "session", "CAD import requires its own inactive Gmsh session"
        )
    gmsh.initialize([], readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        for option, value in {
            "General.NumThreads": 1,
            "Mesh.MaxNumThreads1D": 1,
            "Mesh.MaxNumThreads2D": 1,
            "Mesh.ElementOrder": 1,
            "Mesh.RecombineAll": 0,
            "Mesh.Algorithm": 6,
            "Mesh.RandomSeed": 1,
            "Mesh.MeshSizeMin": mesh_size_mm,
            "Mesh.MeshSizeMax": mesh_size_mm,
        }.items():
            gmsh.option.setNumber(option, value)
        gmsh.model.add("user-cad-profile")
        if source_format == "step":
            surface, labels = _step(gmsh, source)
        else:
            from piginot.cad.dxf import add_dxf_profile

            surface, labels = add_dxf_profile(gmsh, source, unit)
        curves, offset = _planar_boundary(gmsh, surface)
        boundaries = _boundaries(gmsh, curves, labels)
        if mesh_policy is not None:
            from piginot.cad.mesh_controls import configure

            circles = configure(gmsh, boundaries, mesh_policy)
        provenance = MappingProxyType(
            {
                "generator": "gmsh-occ-planar-profile-v1",
                "format": source_format,
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "declared_unit": unit.value,
                "canonical_unit": "mm",
                "unit_policy": "embedded-step-units-to-mm"
                if source_format == "step"
                else "explicit-dxf-units-to-mm",
                "mesh_size_mm": str(float(mesh_size_mm)),
                "gmsh_version": str(gmsh.__version__),
                "dxf_reader_version": version("ezdxf")
                if source_format == "dxf"
                else "not-used",
                "dxf_endpoint_policy": "join within 1e-9 * max(1 mm, profile extent); no larger-gap repair"
                if source_format == "dxf"
                else "not-used",
                "source_plane_z_offset_mm": str(offset),
                "geometry_policy": "linearized-cad-boundary; no continuum geometric-error claim",
                **(
                    {"mesh_policy_sha256": mesh_policy.sha256}
                    if mesh_policy is not None
                    else {}
                ),
            }
        )
        mesh = _mesh(gmsh, surface, boundaries, provenance)
        evidence = None
        if mesh_policy is not None:
            from piginot.cad.mesh_controls import measure
            from piginot.cad.resource import _freeze

            evidence = _freeze(measure(gmsh, mesh, boundaries, circles, mesh_policy))
        if source.read_bytes() != payload:
            raise CadImportError("source", "CAD input changed during import")
        return ImportedCad(
            mesh,
            boundaries,
            hashlib.sha256(payload).hexdigest(),
            source_format,
            provenance,
            mesh_policy,
            evidence,
        )
    except CadImportError:
        raise
    except Exception as error:
        raise CadImportError(
            "cad", f"CAD backend rejected the profile: {error}"
        ) from error
    finally:
        gmsh.finalize()
