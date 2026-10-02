from __future__ import annotations

import pytest

from campaigns.dogbone_v1.s14_envelope import (
    FIXTURE_COUNTS,
    check_fixture,
    check_in_domain,
    dogbone_checker,
    fixture_set_hash,
    in_domain_cases,
    in_domain_queries,
    refusal_fixtures,
    theta,
)
from piginot.verification import GeometryOutcome, RefusalDecision


def test_fixture_set_matches_adr_0019_and_is_reproducible() -> None:
    fixtures = refusal_fixtures()

    assert len(fixtures) == 240
    assert {
        category: sum(f.category == category for f in fixtures)
        for category in FIXTURE_COUNTS
    } == FIXTURE_COUNTS
    assert fixture_set_hash(fixtures) == fixture_set_hash(refusal_fixtures())


@pytest.mark.parametrize("category", sorted(FIXTURE_COUNTS))
def test_each_category_refuses_with_its_expected_code_twice(category: str) -> None:
    sample = [f for f in refusal_fixtures() if f.category == category][::6]
    for fixture in sample:
        first = check_fixture(fixture)
        second = check_fixture(fixture)
        assert not first.accepted, fixture.fixture_id
        assert fixture.expected_code in first.reason_codes, fixture.fixture_id
        assert first == second


def test_in_domain_cases_are_accepted_with_their_queries() -> None:
    cases = in_domain_cases()

    assert len(cases) == 844
    for case in cases[::120]:
        assert check_in_domain(case).accepted, case.case_id


def test_checker_contract_and_query_selection() -> None:
    checker = dogbone_checker()
    values = {
        "L_total": 55.0,
        "W_grip": 21.0,
        "W_gauge": 10.0,
        "R_fillet": 14.0,
        "nominal_axial_strain": 0.15,
    }
    assert checker.check(theta(values)).accepted
    with pytest.raises(ValueError):
        RefusalDecision(True, ("X",))
    with pytest.raises(ValueError):
        RefusalDecision(False)
    assert GeometryOutcome(None, ("A",)).reason_codes == ("A",)
    mesh_free = checker.check(theta(values), queries=())
    assert mesh_free.reason_codes == ("INVALID_QUERY",)
    case = in_domain_cases()[0]
    assert check_in_domain(case).accepted
    assert len(in_domain_queries(_morphed(values))) == 32


def _morphed(values: dict[str, float]):  # type: ignore[no-untyped-def]
    from campaigns.dogbone_v1.s14_envelope import _morph

    outcome = _morph(theta(values))
    assert outcome.mesh is not None
    return outcome.mesh
