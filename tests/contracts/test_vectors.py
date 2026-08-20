from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from piginot.contracts.canonical import canonical_json_bytes, strict_json_loads
from piginot.contracts.hashing import hash_spec

VECTOR_PATH = Path(__file__).parents[1] / "fixtures" / "canonical" / "vectors-v1.json"


def _vectors() -> dict[str, Any]:
    return cast(dict[str, Any], strict_json_loads(VECTOR_PATH.read_bytes()))


def test_committed_canonicalization_vectors_are_byte_identical() -> None:
    vectors = _vectors()

    for vector in vectors["canonicalization_vectors"]:
        assert canonical_json_bytes(vector["input"]) == vector["canonical"].encode(
            "utf-8"
        ), vector["id"]


def test_committed_hash_vectors_close_over_domain_and_canonical_payload() -> None:
    vectors = _vectors()

    for vector in vectors["hash_vectors"]:
        framed = {"domain": vector["domain"], "payload": vector["payload"]}
        assert canonical_json_bytes(framed) == vector["canonical_framed"].encode(
            "utf-8"
        ), vector["id"]
        assert (
            hash_spec(vector["payload"], vector["domain"]) == vector["blake2b_256"]
        ), vector["id"]


def test_committed_hash_vectors_cover_each_domain_once() -> None:
    vectors = _vectors()
    domains = [vector["domain"] for vector in vectors["hash_vectors"]]

    assert domains == ["problem", "recipe", "content", "bundle"]
