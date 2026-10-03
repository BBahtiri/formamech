"""Assemble a bundle from its parts and prove that it loads before publishing it."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import rfc8785

from piginot import __version__
from piginot.bundle.inference import (
    ENVELOPE_PATH,
    LIMITATIONS_PATH,
    MODEL_PATH,
    MORPH_MAP_PATH,
    REFERENCE_MESH_PATH,
    VERIFICATION_PATH,
    WEIGHTS_PATH,
    load_bundle,
)
from piginot.bundle.limitations import REQUIRED_LIMITATION_PARAGRAPH
from piginot.bundle.manifest import BundleManifest, write_bundle
from piginot.bundle.specs import ModelSpec
from piginot.contracts import CanonicalMesh, canonical_json_bytes
from piginot.doe.envelope import ExecutableEnvelope


def build_bundle(
    directory: Path,
    *,
    name: str,
    created_utc: str,
    claim_level: str,
    reference_mesh: CanonicalMesh,
    morph_map: object,
    envelope: ExecutableEnvelope,
    model: ModelSpec,
    weights: bytes,
    verification: Mapping[str, Any],
    limitation_items: Sequence[Mapping[str, str]],
    identities: Mapping[str, str],
) -> BundleManifest:
    """Write ``directory`` atomically; it appears only if it loads cleanly.

    The identities record the ``piginot`` version whose inference closure
    built and loaded the bundle.
    """

    if directory.exists():
        raise FileExistsError(f"bundle path already exists: {directory}")
    files = {
        REFERENCE_MESH_PATH: canonical_json_bytes(reference_mesh),
        MORPH_MAP_PATH: canonical_json_bytes(morph_map),
        ENVELOPE_PATH: envelope.to_bytes(),
        MODEL_PATH: model.to_bytes(),
        WEIGHTS_PATH: bytes(weights),
        VERIFICATION_PATH: rfc8785.dumps(dict(verification)),
        LIMITATIONS_PATH: rfc8785.dumps(
            {
                "paragraph": REQUIRED_LIMITATION_PARAGRAPH,
                "items": [dict(item) for item in limitation_items],
            }
        ),
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        target = staging / "bundle"
        manifest = write_bundle(
            target,
            files,
            name=name,
            created_utc=created_utc,
            claim_level=claim_level,
            identities={**identities, "piginot_version": __version__},
        )
        load_bundle(target)
        os.rename(target, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


__all__ = ("build_bundle",)
