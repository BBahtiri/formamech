"""A small synthetic bundle on the DogBone family, built without the checkpoint."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import rfc8785
import torch
from safetensors.torch import save

from campaigns.dogbone_v1.s14_envelope import _family, dogbone_envelope, theta
from piginot.bundle import (
    CONSTITUTIVE_PLANE_STRESS,
    REQUIRED_LIMITATION_PARAGRAPH,
    ConditioningInput,
    ConfidenceSpec,
    ModelSpec,
    QoiSpec,
    ScaledParameter,
    build_bundle,
)
from piginot.bundle.inference import (
    ENVELOPE_PATH,
    LIMITATIONS_PATH,
    MODEL_PATH,
    MORPH_MAP_PATH,
    REFERENCE_MESH_PATH,
    VERIFICATION_PATH,
    WEIGHTS_PATH,
)
from piginot.bundle.manifest import BundleManifest, write_bundle
from piginot.contracts import canonical_json_bytes
from piginot.operators import (
    ENERGY_ARCHITECTURE_ID,
    EnergyFormOperator,
    EnergyOperatorConfig,
    energy_module_digest,
)

NAMES = ("L_total", "W_grip", "W_gauge", "R_fillet", "nominal_axial_strain")
REFERENCE_VALUES = {
    "L_total": 54.0,
    "W_grip": 20.0,
    "W_gauge": 10.0,
    "R_fillet": 12.0,
    "nominal_axial_strain": 0.15,
}
CREATED = "2026-10-03T00:00:00Z"
CLAIM = "HOLDOUT_VERIFIED"


def tiny_config() -> EnergyOperatorConfig:
    return EnergyOperatorConfig(
        parameter_names=NAMES,
        hidden_dim=8,
        latent_token_count=2,
        parameter_hidden_dim=8,
        num_attention_heads=2,
        decoder_layers=1,
        positional_frequencies=2,
    )


def tiny_weights(
    *, output_bias: tuple[float, float] | None = None, digest: str | None = None
) -> bytes:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = EnergyFormOperator(tiny_config()).to(torch.float32)
    if output_bias is not None:
        with torch.no_grad():
            model.output.bias.copy_(torch.tensor(output_bias))
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    metadata = {
        "architecture_id": ENERGY_ARCHITECTURE_ID,
        "architecture_module_digest": digest or energy_module_digest(model),
    }
    return save(tensors, metadata=metadata)


def tiny_model_spec(**overrides: Any) -> ModelSpec:
    config = dataclasses.asdict(tiny_config())
    config["parameter_names"] = list(NAMES)
    values: dict[str, Any] = {
        "architecture_id": ENERGY_ARCHITECTURE_ID,
        "operator_config": config,
        "conditioning": tuple(
            ConditioningInput(name, name, REFERENCE_VALUES[name], 1.0 + i)
            for i, name in enumerate(NAMES)
        ),
        "frame_origin": (ScaledParameter((), 0.0), ScaledParameter((), 0.0)),
        "frame_extent": (
            ScaledParameter(("L_total",), 0.5),
            ScaledParameter(("W_grip",), 0.5),
        ),
        "lifting_kind": "axis-aligned-tension",
        "lifting_displacement": ScaledParameter(
            ("nominal_axial_strain", "L_total"), 0.5
        ),
        "material_name": "synthetic",
        "material_model": "compressible_neo_hookean",
        "material_parameters": {
            "elastic_modulus": "elastic_modulus",
            "poisson_ratio": "poisson_ratio",
        },
        "boundary_conditions": {
            "left_symmetry": ("x",),
            "bottom": ("y",),
            "right_grip": ("x",),
        },
        "served_stress": CONSTITUTIVE_PLANE_STRESS,
        "dtype": "float64",
        "query_chunk": 512,
        "qois": (QoiSpec("peak", "P_served", "right_arc", ("right_grip",), "MPa"),),
        "confidence": ConfidenceSpec(
            "patch-mean", 1.0, 760.0, 0.9, (0.001, 0.01, 0.1), (0.001, 0.002, 0.004)
        ),
        "units": {"u": "mm", "P_served": "MPa"},
    }
    values.update(overrides)
    return ModelSpec(**values)


def verification(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "gate_id": "G14",
        "verdict": "PASS",
        "claim_level": CLAIM,
        "outstanding_work": [
            {"id": "G16", "status": "NOT_RUN", "reason": "mutation suite not run"}
        ],
    }
    document.update(overrides)
    return document


LIMITATION_ITEMS = ({"id": "synthetic", "statement": "a synthetic test bundle"},)


def build(directory: Path, **overrides: Any) -> BundleManifest:
    family = _family()
    arguments: dict[str, Any] = {
        "name": "synthetic",
        "created_utc": CREATED,
        "claim_level": CLAIM,
        "reference_mesh": family.reference_mesh,
        "morph_map": family.morph_map,
        "envelope": dogbone_envelope(),
        "model": tiny_model_spec(),
        "weights": tiny_weights(),
        "verification": verification(),
        "limitation_items": LIMITATION_ITEMS,
        "identities": {"origin": "tests"},
    }
    arguments.update(overrides)
    return build_bundle(directory, **arguments)


@lru_cache(maxsize=1)
def file_contents() -> dict[str, bytes]:
    family = _family()
    return {
        REFERENCE_MESH_PATH: canonical_json_bytes(family.reference_mesh),
        MORPH_MAP_PATH: canonical_json_bytes(family.morph_map),
        ENVELOPE_PATH: dogbone_envelope().to_bytes(),
        MODEL_PATH: tiny_model_spec().to_bytes(),
        WEIGHTS_PATH: tiny_weights(),
        VERIFICATION_PATH: rfc8785.dumps(verification()),
        LIMITATIONS_PATH: rfc8785.dumps(
            {
                "paragraph": REQUIRED_LIMITATION_PARAGRAPH,
                "items": [dict(item) for item in LIMITATION_ITEMS],
            }
        ),
    }


def write_raw(
    directory: Path,
    replace: Mapping[str, bytes | None] | None = None,
    *,
    claim_level: str = CLAIM,
) -> BundleManifest:
    """Write a bundle without the build-time load check, with parts replaced."""

    files = dict(file_contents())
    for path, payload in (replace or {}).items():
        if payload is None:
            files.pop(path, None)
        else:
            files[path] = payload
    return write_bundle(
        directory,
        files,
        name="synthetic",
        created_utc=CREATED,
        claim_level=claim_level,
        identities={"origin": "tests"},
    )


def request(**changes: float) -> dict[str, object]:
    return theta({**REFERENCE_VALUES, **changes})
