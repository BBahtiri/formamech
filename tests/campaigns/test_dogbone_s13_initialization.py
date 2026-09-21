from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save as save_safetensors

from campaigns.dogbone_v1.s11_contingency_models import (
    build_candidate_models,
    model_state_hash,
)
from campaigns.dogbone_v1.s11_contingency_run import load_committed_protocol
from campaigns.dogbone_v1.s13_initialization import (
    SelectedS11Initialization,
    _load_selected_s11_checkpoint_core,
    load_selected_s11_checkpoint_evidence,
)


def _payload(
    state: dict[str, torch.Tensor],
    *,
    metadata: dict[str, str],
) -> bytes:
    return save_safetensors(
        {name: value.detach().cpu().contiguous() for name, value in state.items()},
        metadata=metadata,
    )


def _metadata(evidence) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {
        "candidate_id": evidence.candidate_id,
        "protocol_content_hash": evidence.protocol_content_hash,
        "source_bindings_hash": evidence.source_bindings_hash,
        "initial_state_hash": evidence.initial_state_hash,
        "final_state_hash": evidence.final_state_hash,
        "operator_module_digest": evidence.operator_module_digest,
    }


def _fixture_checkpoint(tmp_path: Path):  # type: ignore[no-untyped-def]
    protocol = load_committed_protocol()
    evidence = load_selected_s11_checkpoint_evidence()
    baseline = build_candidate_models(
        protocol,
        boundary_tag_names=protocol.boundary_tag_names,
    )[0]
    state_hash = model_state_hash(baseline)
    evidence = replace(
        evidence,
        initial_state_hash=state_hash,
        final_state_hash=state_hash,
    )
    payload = _payload(dict(baseline.state_dict()), metadata=_metadata(evidence))
    evidence = replace(
        evidence,
        byte_count=len(payload),
        file_blake2b256=hashlib.blake2b(payload, digest_size=32).hexdigest(),
    )
    (tmp_path / evidence.file_name).write_bytes(payload)
    return protocol, evidence, baseline


def test_tracked_selected_checkpoint_evidence_is_exactly_pinned() -> None:
    evidence = load_selected_s11_checkpoint_evidence()

    assert evidence.evidence_file_blake2b256 == (
        "260eca6c345ee1edea17b9098859baaa6de4a35e27de65c6ccdead0cad831f36"
    )
    assert evidence.candidate_id == "A-s11-baseline"
    assert evidence.file_name == "A-s11-baseline.safetensors"
    assert evidence.byte_count == 3_113_560
    assert evidence.file_blake2b256 == (
        "c646f8fe58d2759012f2e1392d613395d5256bd6d8947d7728d3352a1bcf7c8e"
    )
    assert evidence.final_state_hash == (
        "51b83fd2468d66d5556b75286feb08a13b2e9ca16d0b7f4cdcb83ff26eaba906"
    )
    assert evidence.normalization.content_hash == (
        "b3f59978629e11a4f7535d4f7be17984ffa7323d3a164128a2b8cd4fbf7420df"
    )


def test_selected_checkpoint_loader_preflights_and_loads_exact_state(
    tmp_path: Path,
) -> None:
    protocol, evidence, expected = _fixture_checkpoint(tmp_path)

    loaded = _load_selected_s11_checkpoint_core(
        tmp_path,
        protocol=protocol,
        evidence=evidence,
    )

    assert type(loaded) is SelectedS11Initialization
    assert loaded.candidate_id == evidence.candidate_id
    assert loaded.normalization == evidence.normalization
    assert loaded.source_bindings_hash == evidence.source_bindings_hash
    assert model_state_hash(loaded.model) == evidence.final_state_hash
    assert all(
        torch.equal(loaded.model.state_dict()[name], value)
        for name, value in expected.state_dict().items()
    )


def test_selected_checkpoint_loader_rejects_file_digest_drift(tmp_path: Path) -> None:
    protocol, evidence, _ = _fixture_checkpoint(tmp_path)
    path = tmp_path / evidence.file_name
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="file digest"):
        _load_selected_s11_checkpoint_core(
            tmp_path,
            protocol=protocol,
            evidence=evidence,
        )


def test_selected_checkpoint_loader_rejects_header_provenance_drift(
    tmp_path: Path,
) -> None:
    protocol, evidence, model = _fixture_checkpoint(tmp_path)
    metadata = _metadata(evidence)
    metadata["source_bindings_hash"] = "f" * 64
    payload = _payload(dict(model.state_dict()), metadata=metadata)
    drifted = replace(
        evidence,
        byte_count=len(payload),
        file_blake2b256=hashlib.blake2b(payload, digest_size=32).hexdigest(),
    )
    (tmp_path / evidence.file_name).write_bytes(payload)

    with pytest.raises(ValueError, match="source_bindings_hash"):
        _load_selected_s11_checkpoint_core(
            tmp_path,
            protocol=protocol,
            evidence=drifted,
        )


def test_selected_checkpoint_loader_rejects_unbound_header_metadata(
    tmp_path: Path,
) -> None:
    protocol, evidence, model = _fixture_checkpoint(tmp_path)
    metadata = _metadata(evidence)
    metadata["unbound_note"] = "not-part-of-the-evidence"
    payload = _payload(dict(model.state_dict()), metadata=metadata)
    drifted = replace(
        evidence,
        byte_count=len(payload),
        file_blake2b256=hashlib.blake2b(payload, digest_size=32).hexdigest(),
    )
    (tmp_path / evidence.file_name).write_bytes(payload)

    with pytest.raises(ValueError, match="metadata keys"):
        _load_selected_s11_checkpoint_core(
            tmp_path,
            protocol=protocol,
            evidence=drifted,
        )


def test_selected_checkpoint_loader_rejects_tensor_schema_drift(
    tmp_path: Path,
) -> None:
    protocol, evidence, model = _fixture_checkpoint(tmp_path)
    state = dict(model.state_dict())
    state.pop(next(iter(state)))
    payload = _payload(state, metadata=_metadata(evidence))
    drifted = replace(
        evidence,
        byte_count=len(payload),
        file_blake2b256=hashlib.blake2b(payload, digest_size=32).hexdigest(),
    )
    (tmp_path / evidence.file_name).write_bytes(payload)

    with pytest.raises(ValueError, match="keys do not match"):
        _load_selected_s11_checkpoint_core(
            tmp_path,
            protocol=protocol,
            evidence=drifted,
        )
