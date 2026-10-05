"""Notched refusal fixtures: counts, expected codes and the tag mutation."""

from __future__ import annotations

import pytest

from campaigns.notched_v1 import s14_envelope as envelope


def test_the_fixture_set_has_the_registered_counts_and_unique_ids() -> None:
    fixtures = envelope.refusal_fixtures()
    assert len(fixtures) == sum(envelope.FIXTURE_COUNTS.values()) == 216
    assert len({fixture.fixture_id for fixture in fixtures}) == 216
    assert all(fixture.theta_override is not None for fixture in fixtures)
    assert len(envelope.request_form_fixtures()) == 180


@pytest.mark.parametrize("category", sorted(envelope.FIXTURE_COUNTS))
def test_one_fixture_per_category_is_refused_with_its_code(category: str) -> None:
    fixture = next(f for f in envelope.refusal_fixtures() if f.category == category)
    decision = envelope.check_fixture(fixture)
    assert not decision.accepted
    assert fixture.expected_code in decision.reason_codes


def test_the_renamed_tag_mutation_renames_a_notched_tag() -> None:
    outcome = envelope.morph(envelope.theta(envelope.BASE_POINTS[0]))
    assert outcome.mesh is not None
    even = envelope.mutated_mesh(outcome.mesh, "renamed-tag:0")
    odd = envelope.mutated_mesh(outcome.mesh, "renamed-tag:1")
    assert "loaded_end" in even.semantic_tags and "right_end" not in even.semantic_tags
    assert "top" not in odd.semantic_tags
    decision = envelope.notched_checker().check(
        envelope.theta(envelope.BASE_POINTS[0]), mesh=even
    )
    assert "TAG_IDENTITY_CHANGED" in decision.reason_codes


def test_an_in_domain_public_case_is_accepted() -> None:
    case = envelope.in_domain_cases()[0]
    assert envelope.check_in_domain(case).accepted
