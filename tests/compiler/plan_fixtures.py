"""Small public metadata fixtures, with no geometry, FEM or model queries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from piginot.compiler.adapters import AdapterDescriptor, GeometryIdentity
from piginot.doe import (
    EvidenceRole,
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from tests.compiler.config_fixtures import config_document
from tests.problem_helpers import plate_document


def inputs(directory: Path, *, development: bool = False):
    spec = plate_document()
    spec["split"] = {
        "pilot": 0,
        "train": 3,
        "development": 1,
        "holdout": 1,
        "reserve": 0,
    }
    values = {
        "width": 12.0,
        "height": 5.0,
        "strain": 0.07,
        "modulus": 760.0,
        "poisson": 0.3,
    }
    cases = tuple(
        PublishedCase(
            case_id=case_id,
            leakage_group_id=f"group-{case_id}",
            role=role,
            primitive_coordinates=values,
            derived_coordinates={"aspect": 2.4},
            strata={"width": 0},
            status=PublishedCaseStatus.ACTIVE,
        )
        for case_id, role in (
            ("fit-001", EvidenceRole.TRAIN),
            ("fit-002", EvidenceRole.TRAIN),
            ("screen-001", EvidenceRole.TRAIN),
            ("development-001", EvidenceRole.DEVELOPMENT),
        )
    )
    manifest = PublishedCaseManifest(
        manifest_id="synthetic-public",
        primitive_axes=tuple(values),
        derived_axes=("aspect",),
        stratum_axes=("width",),
        cases=cases,
    )
    manifest_path = directory / "manifest.json"
    manifest_path.write_bytes(manifest.to_bytes())
    config = config_document()
    config["source"]["path"] = "manifest.json"
    config["source"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config["model"]["parameter_names"] = list(values)
    config["selection"]["development"] = development
    config["source"]["development_case_ids"] = (
        ["development-001"] if development else []
    )
    spec_path, config_path = directory / "problem.json", directory / "config.json"
    spec_path.write_text(json.dumps(spec))
    config_path.write_text(json.dumps(config))
    adapter = AdapterDescriptor(
        adapter_id="unit-adapter",
        version="1.0",
        implementation_sha256="a" * 64,
        geometries=(
            GeometryIdentity(
                spec["geometry"]["reference_mesh_id"],
                spec["geometry"]["morph_map_sha256"],
            ),
        ),
        materials=(
            "compressible_neo_hookean",
            "formula-isotropic",
            "formula-fibre",
            "hgo",
        ),
        liftings=("quarter", "adf"),
        architectures=("pi-ginot-energy-parametric-film-v1",),
        reference_initializations=("cold", "neo-hookean-full-load-hgo"),
    )
    return spec_path, config_path, manifest, adapter
