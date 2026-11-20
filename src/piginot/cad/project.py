"""Prepare and inspect portable, hash-bound fixed CAD geometry projects."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import piginot
import piginot_campaigns
from piginot.cad.mesh_policy import MeshPolicy
from piginot.cad.resource import CadResourceError, load_geometry_resource
from piginot.contracts import CanonicalMesh, Unit, canonical_json_bytes

_LEVELS = (1, 2, 4)

if TYPE_CHECKING:
    from piginot.cad.adaptation import AdaptationReport


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _has_symlink(path: Path) -> bool:
    absolute = path.absolute()
    return any(part.is_symlink() for part in (absolute, *absolute.parents))


def _source_bytes(path: Path) -> bytes:
    if _has_symlink(path) or not path.is_file():
        raise ValueError("CAD source must be a regular file without symlinks")
    return path.read_bytes()


def _output_path(source: Path, output: Path) -> Path:
    if _has_symlink(output):
        raise ValueError("project output cannot contain a symlink")
    output = output.resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("project output cannot overlap its CAD source")
    for module in (piginot, piginot_campaigns):
        package = Path(cast(str, module.__file__)).resolve().parent
        if output.is_relative_to(package) or package.is_relative_to(output):
            raise ValueError("project output cannot overlap the installed package")
    if output.exists():
        raise FileExistsError(f"project output already exists: {output}")
    return output


def _import_cad(
    source: Path,
    *,
    unit: Unit,
    mesh_size_mm: float | None = None,
    mesh_policy: MeshPolicy | None = None,
) -> Any:
    from piginot.cad.importing import import_cad

    if mesh_policy is None:
        return import_cad(source, unit=unit, mesh_size_mm=mesh_size_mm)
    return import_cad(source, unit=unit, mesh_policy=mesh_policy)


def _boundary_metadata(
    imported: Any, mesh: CanonicalMesh
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    names = tuple(boundary.name for boundary in imported.boundaries)
    kinds = {boundary.name: boundary.kind for boundary in imported.boundaries}
    if (
        len(set(names)) != len(names)
        or set(names) != set(mesh.semantic_tags)
        or any(kind not in ("line", "arc", "other") for kind in kinds.values())
    ):
        raise ValueError("imported boundary metadata differs from the mesh boundaries")
    sources = {
        boundary.name: {
            "source_id": boundary.source_id,
            "source_label": boundary.source_label,
        }
        for boundary in imported.boundaries
    }
    return kinds, sources


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _preview(
    mesh: CanonicalMesh,
    source_sha256: str,
    resource_sha256: str,
    boundary_kinds: Mapping[str, str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    sources = provenance.get("boundary_sources", {})
    if not isinstance(sources, Mapping) or any(
        not isinstance(entry, Mapping) for entry in sources.values()
    ):
        raise ValueError("CAD boundary source metadata must contain data objects")
    return {
        "schema": "piginot.cad-preview-v1",
        "geometry_resource": {"path": "geometry.json", "sha256": resource_sha256},
        "source": {"sha256": source_sha256, "format": provenance.get("source_format")},
        "units": "mm",
        "bounding_box_mm": {
            "min": [min(point[axis] for point in mesh.nodes) for axis in (0, 1)],
            "max": [max(point[axis] for point in mesh.nodes) for axis in (0, 1)],
        },
        "mesh": {
            "mesh_id": mesh.mesh_id,
            "node_count": len(mesh.nodes),
            "triangle_count": len(mesh.cells),
            "nodes_mm": _plain(mesh.nodes),
            "triangles": _plain(mesh.cells),
        },
        "boundaries": [
            {
                "name": name,
                "kind": boundary_kinds[name],
                "source_id": sources.get(name, {}).get("source_id"),
                "source_label": sources.get(name, {}).get("source_label"),
                "segments_mm": [
                    _plain(
                        tuple(mesh.nodes[node] for node in mesh.boundary_facets[index])
                    )
                    for index in mesh.semantic_tags[name]
                ],
            }
            for name in sorted(boundary_kinds)
        ],
        "provenance": _plain(provenance),
        "claim_level": "NOT_VERIFIED",
    }


def _project_payloads(
    source: Path,
    source_bytes: bytes,
    imported: Any,
    adaptation_report: AdaptationReport | None = None,
) -> tuple[dict[str, bytes], bytes]:
    from piginot.cad.refinement import refine_fixed_mesh

    meshes = {level: refine_fixed_mesh(imported.mesh, level) for level in _LEVELS}
    kinds, sources = _boundary_metadata(imported, meshes[1])
    provenance = {
        **imported.provenance,
        "source_format": imported.source_format,
        "boundary_sources": sources,
        "domain": "fixed-polygonal-approximation",
        "refinement": "midpoint",
        "geometry_accuracy_bound": "NOT_ESTABLISHED",
        "unsupported_essential_boundary_kinds": ["arc", "other"],
        **(
            {
                "mesh_policy": imported.mesh_policy.document(),
                "mesh_evidence": _plain(imported.mesh_evidence),
            }
            if getattr(imported, "mesh_policy", None) is not None
            else {}
        ),
    }
    if adaptation_report is not None:
        accepted = adaptation_report.attempts[-1]
        if (
            accepted.mesh is None
            or imported.mesh_policy != accepted.policy
            or canonical_json_bytes(imported.mesh)
            != canonical_json_bytes(accepted.mesh)
        ):
            raise ValueError(
                "adaptation evidence differs from the imported final mesh/policy"
            )
        provenance["adaptation"] = {
            "sha256": adaptation_report.sha256,
            "report": adaptation_report.document(),
        }
    source_name = "source" + source.suffix.lower()
    files = {
        source_name: source_bytes,
        **{
            f"mesh-{level}.json": canonical_json_bytes(mesh)
            for level, mesh in meshes.items()
        },
    }
    manifest = {
        "schema": "piginot.cad-geometry-v1",
        "source": {"path": source_name, "sha256": _sha(source_bytes)},
        "meshes": {
            str(level): {
                "path": f"mesh-{level}.json",
                "sha256": _sha(files[f"mesh-{level}.json"]),
                "mesh_id": mesh.mesh_id,
            }
            for level, mesh in meshes.items()
        },
        "boundary_kinds": kinds,
        "provenance": provenance,
    }
    payload = canonical_json_bytes(manifest) + b"\n"
    preview = _preview(meshes[1], _sha(source_bytes), _sha(payload), kinds, provenance)
    return {**files, "preview.json": canonical_json_bytes(preview) + b"\n"}, payload


def prepare_geometry(
    source: Path,
    output: Path,
    *,
    unit: Unit,
    mesh_size_mm: float | None = None,
    mesh_policy: MeshPolicy | None = None,
    adaptation_report: AdaptationReport | None = None,
) -> dict[str, Any]:
    """Import once and publish a new project; never overwrite an existing path."""

    if unit not in (Unit.MM, Unit.CM, Unit.M):
        raise ValueError("CAD source unit must be mm, cm or m")
    if adaptation_report is not None:
        from piginot.cad.adaptation import AdaptationReport

        if (
            not isinstance(adaptation_report, AdaptationReport)
            or adaptation_report.status != "HEURISTIC_TARGET_MET"
            or not adaptation_report.attempts
        ):
            raise ValueError(
                "only a completed heuristic preparation can bind adaptation evidence"
            )
    if mesh_policy is not None:
        if mesh_size_mm is not None:
            raise ValueError("supply exactly one of mesh_size_mm and mesh_policy")
        if not isinstance(mesh_policy, MeshPolicy):
            raise ValueError("mesh_policy must be a MeshPolicy")
    elif (
        isinstance(mesh_size_mm, bool)
        or not isinstance(mesh_size_mm, (int, float))
        or not math.isfinite(mesh_size_mm)
        or mesh_size_mm <= 0
    ):
        raise ValueError("mesh size must be a finite positive number in millimetres")
    original = _source_bytes(source)
    source = source.resolve()
    output = _output_path(source, output)
    imported = (
        _import_cad(source, unit=unit, mesh_policy=mesh_policy)
        if mesh_policy is not None
        else _import_cad(source, unit=unit, mesh_size_mm=mesh_size_mm)
    )
    files, manifest = _project_payloads(source, original, imported, adaptation_report)
    if imported.source_sha256 != _sha(original) or _source_bytes(source) != original:
        raise ValueError("CAD source changed during import or mesh preparation")
    output.mkdir(parents=True)
    for name, payload in files.items():
        with (output / name).open("xb") as stream:
            stream.write(payload)
    with (output / "geometry.json").open("xb") as stream:
        stream.write(manifest)
    return {
        "status": "IMPORTED",
        "project": str(output),
        "geometry_resource": str(output / "geometry.json"),
        "geometry_resource_sha256": _sha(manifest),
        "source_sha256": _sha(original),
        "mesh_id": imported.mesh.mesh_id,
        "boundary_names": sorted(boundary.name for boundary in imported.boundaries),
        "claim_level": "NOT_VERIFIED",
    }


def inspect_geometry(project: Path) -> dict[str, Any]:
    """Derive a preview from verified members without loading any CAD backend."""

    path = project / "geometry.json"
    if _has_symlink(path):
        raise CadResourceError("geometry manifest cannot contain a symlink")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CadResourceError(f"cannot read geometry manifest: {error}") from error
    resource = load_geometry_resource(path, _sha(payload))
    return _preview(
        resource.mesh(1),
        resource.source_sha256,
        resource.sha256,
        resource.boundary_kinds,
        resource.provenance,
    )


def build_project_hierarchy(
    project: Path,
    *,
    output: Path | None = None,
    geometry_tolerance_mm: float | None = None,
    quadrature_rule: str = "stratified-v1",
) -> dict[str, Any]:
    """Build and optionally persist a verified ``MeshHierarchy`` (M0/M1/M2) for a CAD project."""
    from piginot.cad.hierarchy import build_mesh_hierarchy

    path = project / "geometry.json"
    if _has_symlink(path):
        raise CadResourceError("geometry manifest cannot contain a symlink")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CadResourceError(f"cannot read geometry manifest: {error}") from error
    resource = load_geometry_resource(path, _sha(payload))
    hierarchy = build_mesh_hierarchy(
        resource.mesh(1),
        cad_source_hash=resource.source_sha256,
        geometry_tolerance_mm=(
            0.05 if geometry_tolerance_mm is None else float(geometry_tolerance_mm)
        ),
        quadrature_rule=quadrature_rule,
    )
    document: dict[str, Any] = {
        **hierarchy.document(),
        "hierarchy_hash": hierarchy.hierarchy_hash,
        "geometry_resource_sha256": resource.sha256,
        "claim_level": "NOT_VERIFIED",
    }
    if output is not None:
        if _has_symlink(output):
            raise ValueError("hierarchy output cannot contain a symlink")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(document) + b"\n")
    return document


def evaluate_project_geometry_convergence(
    source: Path,
    *,
    unit: Unit,
    mesh_sizes_mm: tuple[float, ...],
    max_relative_area_change: float = 0.01,
    max_relative_perimeter_change: float = 0.01,
    max_chord_error_mm: float = 0.05,
    output: Path | None = None,
) -> dict[str, Any]:
    """Remesh a CAD source at multiple distinct mesh sizes and evaluate geometry convergence."""
    from piginot.cad.hierarchy import (
        GeometryConvergenceSample,
        evaluate_geometry_convergence,
    )

    if len(mesh_sizes_mm) < 2:
        raise ValueError("geometry convergence requires at least two mesh sizes")
    original = _source_bytes(source)
    source_sha = _sha(original)
    source = source.resolve()
    samples: list[GeometryConvergenceSample] = []
    for idx, size_mm in enumerate(mesh_sizes_mm):
        imported = _import_cad(source, unit=unit, mesh_size_mm=float(size_mm))
        if imported.source_sha256 != source_sha:
            raise ValueError("CAD source changed during multi-size remeshing")
        chord_err: float | None = None
        evidence = getattr(imported, "mesh_evidence", None)
        if isinstance(evidence, Mapping):
            raw_chord = evidence.get("achieved_max_chord_error_mm")
            if isinstance(raw_chord, (int, float)) and not isinstance(raw_chord, bool):
                chord_err = float(raw_chord)
        samples.append(
            GeometryConvergenceSample.from_mesh(
                f"remesh-{idx}-h{size_mm:g}mm",
                imported.mesh,
                cad_source_hash=source_sha,
                declared_curve_tolerance_mm=(
                    0.05 if max_chord_error_mm is None else float(max_chord_error_mm)
                ),
                achieved_max_chord_error_mm=(
                    0.0 if chord_err is None else float(chord_err)
                ),
            )
        )
    report = evaluate_geometry_convergence(
        samples,
        max_relative_area_change=max_relative_area_change,
        max_relative_perimeter_change=max_relative_perimeter_change,
        max_chord_error_mm=max_chord_error_mm,
    )
    document: dict[str, Any] = {
        **report.document(),
        "report_hash": report.report_hash,
        "mesh_sizes_mm": [float(s) for s in mesh_sizes_mm],
    }
    if output is not None:
        if _has_symlink(output):
            raise ValueError("convergence output cannot contain a symlink")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(document) + b"\n")
    return document
