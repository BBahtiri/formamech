from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    Dimension,
    LoadSpec,
    Quantity,
    Unit,
)
from piginot.contracts.canonical import canonical_json_bytes, strict_json_loads
from piginot.contracts.hashing import hash_spec

VECTOR_PATH = Path(__file__).parents[1] / "fixtures" / "canonical" / "vectors-v1.json"
TYPED_VECTOR_PATH = (
    Path(__file__).parents[1] / "fixtures" / "canonical" / "vectors-v2.json"
)


def _boundary_condition() -> BoundaryConditionSpec:
    return BoundaryConditionSpec(
        name="edge_action",
        boundary_tag="loaded_edge",
        kind="traction",
    )


def _load() -> LoadSpec:
    return LoadSpec(
        name="edge_action",
        boundary_tag="loaded_edge",
        kind="traction",
    )


def _quantity_mm() -> Quantity:
    return Quantity(1.0, Unit.MM, Dimension.LENGTH)


def _canonical_quantity_mm() -> CanonicalQuantity:
    return CanonicalQuantity(1.0, Unit.MM, Dimension.LENGTH)


def _moment_n_mm() -> CanonicalQuantity:
    return CanonicalQuantity(1000.0, Unit.N_MM, Dimension.MOMENT)


def _moment_from_n_m() -> CanonicalQuantity:
    return Quantity(1.0, Unit.N_M, Dimension.MOMENT).to_canonical()


def _energy_n_mm() -> CanonicalQuantity:
    return CanonicalQuantity(1000.0, Unit.N_MM, Dimension.ENERGY)


def _energy_from_j() -> CanonicalQuantity:
    return Quantity(1.0, Unit.J, Dimension.ENERGY).to_canonical()


_TYPED_CONSTRUCTORS: dict[str, Callable[[], object]] = {
    "boundary_condition": _boundary_condition,
    "load": _load,
    "quantity_mm": _quantity_mm,
    "canonical_quantity_mm": _canonical_quantity_mm,
    "moment_n_mm": _moment_n_mm,
    "moment_from_n_m": _moment_from_n_m,
    "energy_n_mm": _energy_n_mm,
    "energy_from_j": _energy_from_j,
}


def _vectors() -> dict[str, Any]:
    return cast(dict[str, Any], strict_json_loads(VECTOR_PATH.read_bytes()))


def _typed_vectors() -> dict[str, Any]:
    return cast(dict[str, Any], strict_json_loads(TYPED_VECTOR_PATH.read_bytes()))


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


def test_v1_vector_manifest_is_explicitly_historical() -> None:
    assert _vectors()["status"] == "superseded-by-vectors-v2.json"


def test_typed_vector_manifest_is_versioned_and_supersedes_v1() -> None:
    vectors = _typed_vectors()

    assert vectors["schema_version"] == 2
    assert vectors["status"] == "active"
    assert vectors["supersedes"] == "vectors-v1.json"


def test_committed_typed_vectors_have_exact_canonical_bytes_and_hashes() -> None:
    vectors = _typed_vectors()
    kinds = [vector["kind"] for vector in vectors["typed_vectors"]]

    assert kinds == list(_TYPED_CONSTRUCTORS)
    for vector in vectors["typed_vectors"]:
        value = _TYPED_CONSTRUCTORS[vector["kind"]]()
        assert canonical_json_bytes(value) == vector["canonical"].encode(), vector["id"]
        assert hash_spec(value, vector["domain"]) == vector["blake2b_256"], vector["id"]


def test_typed_vectors_lock_distinct_and_equivalent_identities() -> None:
    vectors = _typed_vectors()
    by_kind = {vector["kind"]: vector for vector in vectors["typed_vectors"]}

    assert by_kind["boundary_condition"]["canonical"] != by_kind["load"]["canonical"]
    assert (
        by_kind["boundary_condition"]["blake2b_256"] != by_kind["load"]["blake2b_256"]
    )
    assert (
        by_kind["quantity_mm"]["canonical"]
        != by_kind["canonical_quantity_mm"]["canonical"]
    )
    assert (
        by_kind["quantity_mm"]["blake2b_256"]
        != by_kind["canonical_quantity_mm"]["blake2b_256"]
    )
    assert by_kind["energy_n_mm"]["canonical"] != by_kind["moment_n_mm"]["canonical"]
    assert (
        by_kind["energy_n_mm"]["blake2b_256"] != by_kind["moment_n_mm"]["blake2b_256"]
    )
    assert by_kind["energy_from_j"]["canonical"] == by_kind["energy_n_mm"]["canonical"]
    assert (
        by_kind["energy_from_j"]["blake2b_256"] == by_kind["energy_n_mm"]["blake2b_256"]
    )
    assert (
        by_kind["moment_from_n_m"]["canonical"] == by_kind["moment_n_mm"]["canonical"]
    )
    assert (
        by_kind["moment_from_n_m"]["blake2b_256"]
        == by_kind["moment_n_mm"]["blake2b_256"]
    )
