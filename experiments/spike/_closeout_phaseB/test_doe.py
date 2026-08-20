"""RED tests for the S1 closeout Phase B design of experiments."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.spike.doe import (
    PRIMARY_RATIOS,
    coverage_report,
    declare_envelope,
    label_envelope,
    ratio_support,
    stratified_design,
)
from experiments.spike.geometry import derived_ratios, sample_geometry_cases


def _iteration8_training_split() -> list:
    """Reproduce the exact 20-per-family split the incumbent was trained on."""
    cases = sample_geometry_cases(master_seed=20260809, count_per_family=30, nx=32, ny=10)
    train = []
    for family_index, family in enumerate(("dogbone", "centre_notched")):
        family_cases = sorted(
            (case for case in cases if case.family == family),
            key=lambda case: case.geometry_id,
        )
        rng = np.random.default_rng(np.random.SeedSequence([991, family_index]))
        order = rng.permutation(len(family_cases))
        train.extend(family_cases[index] for index in order[:20])
    return train


def _k(case) -> float:
    return derived_ratios(case.family, case.parameters)["net_section_factor"]


# --- the regression test that documents the confound ---------------------------
def test_uniform_box_sampling_leaves_the_net_section_tail_uncovered() -> None:
    train = [c for c in _iteration8_training_split() if c.family == "centre_notched"]
    confirmation = [
        c
        for c in sample_geometry_cases(master_seed=2026081301, count_per_family=25,
                                       nx=32, ny=10)
        if c.family == "centre_notched"
    ]
    train_max = max(_k(c) for c in train)
    confirmation_max = max(_k(c) for c in confirmation)
    assert train_max == pytest.approx(1.56, abs=0.01)
    assert confirmation_max == pytest.approx(2.09, abs=0.01)
    assert confirmation_max > train_max
    beyond = [c for c in confirmation if _k(c) > train_max]
    assert len(beyond) == 5


def test_ratio_support_spans_more_than_the_uniform_training_draw() -> None:
    support = ratio_support("centre_notched", seed=17)
    lower, upper = support["net_section_factor"]
    assert upper > 2.5, support
    assert lower < 1.15, support


def test_stratified_design_covers_the_upper_net_section_range() -> None:
    ledger = stratified_design("centre_notched", count=20, seed=7)
    values = [_k(case) for case in ledger.accepted]
    assert len(ledger.accepted) == 20
    assert max(values) > 2.0, "the design must reach the tail iteration 8 missed"
    assert min(values) < 1.2


def test_stratified_design_beats_uniform_on_tail_coverage() -> None:
    uniform = [c for c in _iteration8_training_split() if c.family == "centre_notched"]
    stratified = stratified_design("centre_notched", count=20, seed=7).accepted
    assert max(_k(c) for c in stratified) > max(_k(c) for c in uniform)


def test_every_attempt_is_accounted_for() -> None:
    for family in ("dogbone", "centre_notched"):
        ledger = stratified_design(family, count=20, seed=3)
        assert len(ledger.accepted) + len(ledger.rejected) == ledger.attempted
        assert all(entry["reason"] for entry in ledger.rejected)


def test_design_is_deterministic_for_a_fixed_seed() -> None:
    first = stratified_design("dogbone", count=12, seed=5).accepted
    second = stratified_design("dogbone", count=12, seed=5).accepted
    assert [c.geometry_id for c in first] == [c.geometry_id for c in second]


def test_primary_ratios_are_declared_for_both_families() -> None:
    assert set(PRIMARY_RATIOS) == {"dogbone", "centre_notched"}
    assert PRIMARY_RATIOS["centre_notched"][0] == "net_section_factor"


def test_envelope_labels_cases_beyond_the_training_range() -> None:
    train = [c for c in _iteration8_training_split() if c.family == "centre_notched"]
    envelope = declare_envelope(train)
    confirmation = [
        c
        for c in sample_geometry_cases(master_seed=2026081301, count_per_family=25,
                                       nx=32, ny=10)
        if c.family == "centre_notched"
    ]
    labels = [label_envelope(c.family, c.parameters, envelope) for c in confirmation]
    out = [lab for lab in labels if lab["status"] == "out_of_envelope"]
    assert len(out) >= 5
    assert all(lab["violations"] for lab in out)
    assert any("net_section_factor" in lab["violations"] for lab in out)


def test_envelope_accepts_its_own_training_cases() -> None:
    train = [c for c in _iteration8_training_split() if c.family == "dogbone"]
    envelope = declare_envelope(train)
    for case in train:
        assert label_envelope(case.family, case.parameters, envelope)["status"] == (
            "in_envelope"
        )


def test_coverage_report_records_range_and_stratum_counts() -> None:
    ledger = stratified_design("centre_notched", count=20, seed=7)
    report = coverage_report(ledger.accepted)
    entry = report["centre_notched"]["net_section_factor"]
    assert entry["min"] < entry["max"]
    assert sum(entry["stratum_counts"]) == 20
    assert entry["empty_strata"] == 0
