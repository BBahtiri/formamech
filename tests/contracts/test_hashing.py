from __future__ import annotations

import re

import pytest

from piginot.contracts.errors import InvalidHashDomainError
from piginot.contracts.hashing import hash_spec
from piginot.contracts.units import Quantity, Unit

HASH_DOMAINS = ("problem", "recipe", "content", "bundle")


def test_hash_spec_separates_exactly_the_four_supported_domains() -> None:
    payload = {"family": "plate", "schema_version": 1}

    hashes = {domain: hash_spec(payload, domain) for domain in HASH_DOMAINS}

    assert len(set(hashes.values())) == 4
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values())


@pytest.mark.parametrize(
    "domain",
    ["", "Problem", "model", "dataset", "checkpoint", "problem\x00"],
)
def test_hash_spec_rejects_every_domain_outside_the_four_allowed(domain: str) -> None:
    with pytest.raises(InvalidHashDomainError, match="domain"):
        hash_spec({"schema_version": 1}, domain)


def test_hash_spec_is_independent_of_mapping_insertion_order() -> None:
    first = {"zeta": 3, "alpha": 1, "middle": 2}
    second = {"middle": 2, "zeta": 3, "alpha": 1}

    assert hash_spec(first, "problem") == hash_spec(second, "problem")


def test_raw_unit_dtos_hash_unequal_but_canonical_quantities_hash_equal() -> None:
    one_metre = Quantity(value=1.0, unit=Unit.M)
    one_thousand_millimetres = Quantity(value=1000.0, unit=Unit.MM)

    assert hash_spec(one_metre, "problem") != hash_spec(
        one_thousand_millimetres, "problem"
    )
    assert hash_spec(one_metre.to_canonical(), "problem") == hash_spec(
        one_thousand_millimetres.to_canonical(), "problem"
    )
