"""S24: compare a 400-step replay with a same-CPU reference (ADR 0106 d. 6).

A replay of a family's registered recipe at the S24 revision must reproduce a
400-step run of the same recipe at the v0.2.0 base revision on the same CPU
(amendment 1 of ADR 0106, ``artifacts/problem/S24-REPLAY-AMENDMENT.json``;
trained weights depend on the CPU instruction set, ADR 0100): every tensor of
``model.safetensors`` and ``state.safetensors`` equal, ``optimizer.json`` and
``adaptive.json`` byte-identical, and every manifest field equal except
``run_hash`` and the two file hashes that cover it (``model_export_hash``,
``tensor_payload_hash``); safetensors metadata may differ only in
``run_hash``. The two run bindings must be equal except
``implementation_revision``, the reference's being the base revision. The
registered AVX512 hashes of ``S24-PROTOCOL.json`` are reported beside the
result. Usage::

    python -m campaigns.s24_replay --family dogbone-v1 \
        --reference <base run root>/checkpoint-step-0400 \
        --replay <replay run root>/checkpoint-step-0400 --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/problem")
FAMILIES = ("dogbone-v1", "notched-v1")
ALLOWED_MANIFEST_DIFFERENCES = ("run_hash", "model_export_hash", "tensor_payload_hash")
ALLOWED_METADATA_DIFFERENCES = ("run_hash",)
TENSOR_FILES = ("model.safetensors", "state.safetensors")
EXACT_FILES = ("optimizer.json", "adaptive.json")
AMENDMENT_PATH = OUTPUT / "S24-REPLAY-AMENDMENT.json"
RUN_BINDING_NAME = "run-binding.json"


def replay_path(family: str) -> Path:
    return OUTPUT / f"S24-REPLAY-{family}.json"


def _tensors(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}, dict(
            handle.metadata() or {}
        )


def compare(reference: Path, replay: Path) -> dict[str, Any]:
    """The comparison record of two checkpoint directories."""

    import torch

    record: dict[str, Any] = {"tensor_files": {}, "exact_files": {}}
    for name in TENSOR_FILES:
        left, left_meta = _tensors(reference / name)
        right, right_meta = _tensors(replay / name)
        differing = sorted(
            key
            for key in set(left) | set(right)
            if key not in left
            or key not in right
            or not torch.equal(left[key], right[key])
        )
        metadata = sorted(
            key
            for key in set(left_meta) | set(right_meta)
            if left_meta.get(key) != right_meta.get(key)
        )
        record["tensor_files"][name] = {
            "tensors": len(left),
            "differing_tensors": differing,
            "differing_metadata": metadata,
        }
    for name in EXACT_FILES:
        record["exact_files"][name] = (reference / name).read_bytes() == (
            replay / name
        ).read_bytes()
    left_manifest = json.loads((reference / "checkpoint.json").read_text("utf-8"))
    right_manifest = json.loads((replay / "checkpoint.json").read_text("utf-8"))
    record["manifest_differences"] = sorted(
        key
        for key in set(left_manifest) | set(right_manifest)
        if left_manifest.get(key) != right_manifest.get(key)
    )
    record["step"] = right_manifest.get("step")
    record["reference_run_hash"] = left_manifest.get("run_hash")
    record["reference_model_state_hash"] = left_manifest.get("model_state_hash")
    record["replay_run_hash"] = right_manifest.get("run_hash")
    record["model_state_hash"] = right_manifest.get("model_state_hash")
    record["passed"] = (
        all(
            not entry["differing_tensors"]
            and set(entry["differing_metadata"]) <= set(ALLOWED_METADATA_DIFFERENCES)
            for entry in record["tensor_files"].values()
        )
        and all(record["exact_files"].values())
        and set(record["manifest_differences"]) <= set(ALLOWED_MANIFEST_DIFFERENCES)
        and left_manifest.get("step") == right_manifest.get("step")
    )
    return record


def _binding(checkpoint: Path) -> dict[str, Any] | None:
    path = checkpoint.parent / RUN_BINDING_NAME
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return document


def judge(
    comparison: Mapping[str, Any],
    reference_binding: Mapping[str, Any] | None,
    replay_binding: Mapping[str, Any] | None,
    family: str,
    amendment: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """The amended verdict of a comparison (ADR 0106 amendment 1)."""

    rule = amendment["families"][family]
    revision = amendment["reference_revision"]

    def recipe(binding: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if binding is None:
            return None
        return {k: v for k, v in binding.items() if k != "implementation_revision"}

    reference_revision = (
        None
        if reference_binding is None
        else reference_binding.get("implementation_revision")
    )
    expected = rule["reference_model_state_hash"]
    record = {
        "reference_revision": reference_revision,
        "reference_is_base_revision": reference_revision == revision,
        "replay_implementation_revision": None
        if replay_binding is None
        else replay_binding.get("implementation_revision"),
        "bindings_equal_except_revision": reference_binding is not None
        and recipe(reference_binding) == recipe(replay_binding),
        "reference_model_state_hash": comparison["reference_model_state_hash"],
        "reference_matches_amendment": expected is None
        or comparison["reference_model_state_hash"] == expected,
        "registered_profile_comparison": {
            "registered_model_state_hash": protocol["model_state_hash"],
            "matches_registered": comparison["model_state_hash"]
            == protocol["model_state_hash"],
            "expected_to_match": False,
            "reason": "the registered run was trained on Linux/AVX512; trained "
            "weights depend on the CPU instruction set (ADR 0100)",
        },
    }
    record["passed"] = bool(
        comparison["passed"]
        and record["reference_is_base_revision"]
        and record["bindings_equal_except_revision"]
        and record["reference_matches_amendment"]
    )
    return record


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare an S24 replay checkpoint.")
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    protocol = json.loads(
        (REPOSITORY / OUTPUT / "S24-PROTOCOL.json").read_text("utf-8")
    )["replays"][options.family]
    amendment = json.loads((REPOSITORY / AMENDMENT_PATH).read_text("utf-8"))
    comparison = compare(options.reference, options.replay)
    verdict = judge(
        comparison,
        _binding(options.reference),
        _binding(options.replay),
        options.family,
        amendment,
        protocol,
    )
    record = {
        "schema_version": "1.0",
        "step": "S24",
        "family": options.family,
        "evidence_role": "TRAIN",
        **comparison,
        **verdict,
        "comparison_passed": comparison["passed"],
    }
    if options.write:
        path = REPOSITORY / replay_path(options.family)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
