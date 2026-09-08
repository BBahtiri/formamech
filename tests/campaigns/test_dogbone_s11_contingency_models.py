from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest
import torch

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s11_contingency import (
    RAW_PARAMETER_NAMES,
    build_architecture_screen_protocol,
)
from campaigns.dogbone_v1.s11_contingency_models import build_candidate_models
from piginot.doe import PublishedCaseManifest
from piginot.operators import module_digest


def _protocol():  # type: ignore[no-untyped-def]
    root = Path(__file__).resolve().parents[2]
    manifest = PublishedCaseManifest.from_bytes(
        (
            root
            / "artifacts"
            / "campaigns"
            / "dogbone-v1"
            / "public-case-manifest.json"
        ).read_bytes()
    )
    return build_architecture_screen_protocol(manifest)


def _models():  # type: ignore[no-untyped-def]
    protocol = _protocol()
    tags = tuple(compiled_family().reference_mesh.semantic_tags)
    return protocol, build_candidate_models(protocol, boundary_tag_names=tags)


def test_candidate_initialization_is_a_controlled_factorial() -> None:
    protocol, models = _models()

    assert len(models) == len(protocol.candidates) == 4
    for candidate, model in zip(protocol.candidates, models, strict=True):
        assert module_digest(model) == candidate.operator_module_digest

    states = tuple(model.state_dict() for model in models)
    for left, right in combinations(states, 2):
        for key in left.keys() & right.keys():
            if left[key].shape == right[key].shape:
                torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0)

    raw_width = len(RAW_PARAMETER_NAMES)
    baseline_weight = states[0]["parameter_encoder.0.weight"]
    derived_weight = states[1]["parameter_encoder.0.weight"]
    torch.testing.assert_close(
        derived_weight[:, :raw_width],
        baseline_weight,
        rtol=0.0,
        atol=0.0,
    )
    assert int(torch.count_nonzero(derived_weight[:, raw_width:]).item()) > 0


def test_candidate_initialization_replays_bit_for_bit() -> None:
    protocol, first = _models()
    tags = tuple(compiled_family().reference_mesh.semantic_tags)
    second = build_candidate_models(protocol, boundary_tag_names=tags)

    for first_model, second_model in zip(first, second, strict=True):
        for key, tensor in first_model.state_dict().items():
            torch.testing.assert_close(
                tensor,
                second_model.state_dict()[key],
                rtol=0.0,
                atol=0.0,
            )


def test_candidate_builder_rejects_noncanonical_boundary_tags() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="boundary_tag_names"):
        build_candidate_models(protocol, boundary_tag_names=("bottom", "bottom"))
