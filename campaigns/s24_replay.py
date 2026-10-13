"""S24: compare a 400-step replay with the registered checkpoint (ADR 0106 d. 6).

A replay of a family's registered run at the S24 revision must reproduce the
registered step-400 checkpoint: every tensor of ``model.safetensors`` and
``state.safetensors`` equal, ``optimizer.json`` and ``adaptive.json``
byte-identical, and every manifest field equal except ``run_hash`` and the
two file hashes that cover it (``model_export_hash``,
``tensor_payload_hash``); safetensors metadata may differ only in
``run_hash``. Usage::

    python -m campaigns.s24_replay --family dogbone-v1 \
        --registered <registered run root>/checkpoint-step-0400 \
        --replay <replay run root>/checkpoint-step-0400 --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/problem")
FAMILIES = ("dogbone-v1", "notched-v1")
ALLOWED_MANIFEST_DIFFERENCES = ("run_hash", "model_export_hash", "tensor_payload_hash")
ALLOWED_METADATA_DIFFERENCES = ("run_hash",)
TENSOR_FILES = ("model.safetensors", "state.safetensors")
EXACT_FILES = ("optimizer.json", "adaptive.json")


def replay_path(family: str) -> Path:
    return OUTPUT / f"S24-REPLAY-{family}.json"


def _tensors(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}, dict(
            handle.metadata() or {}
        )


def compare(registered: Path, replay: Path) -> dict[str, Any]:
    """The comparison record of two checkpoint directories."""

    import torch

    record: dict[str, Any] = {"tensor_files": {}, "exact_files": {}}
    for name in TENSOR_FILES:
        left, left_meta = _tensors(registered / name)
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
        record["exact_files"][name] = (registered / name).read_bytes() == (
            replay / name
        ).read_bytes()
    left_manifest = json.loads((registered / "checkpoint.json").read_text("utf-8"))
    right_manifest = json.loads((replay / "checkpoint.json").read_text("utf-8"))
    record["manifest_differences"] = sorted(
        key
        for key in set(left_manifest) | set(right_manifest)
        if left_manifest.get(key) != right_manifest.get(key)
    )
    record["step"] = right_manifest.get("step")
    record["registered_run_hash"] = left_manifest.get("run_hash")
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


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare an S24 replay checkpoint.")
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args(arguments)
    protocol = json.loads(
        (REPOSITORY / OUTPUT / "S24-PROTOCOL.json").read_text("utf-8")
    )["replays"][options.family]
    record = {
        "schema_version": "1.0",
        "step": "S24",
        "family": options.family,
        "evidence_role": "TRAIN",
        **compare(options.registered, options.replay),
    }
    binding = options.replay.parent / "run-binding.json"
    record["replay_implementation_revision"] = (
        json.loads(binding.read_text("utf-8")).get("implementation_revision")
        if binding.is_file()
        else None
    )
    record["registered_matches_protocol"] = (
        record["registered_run_hash"] == protocol["run_hash"]
        and record["model_state_hash"] == protocol["model_state_hash"]
        and record["step"] == protocol["target_step"]
    )
    record["passed"] = record["passed"] and record["registered_matches_protocol"]
    if options.write:
        path = REPOSITORY / replay_path(options.family)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
