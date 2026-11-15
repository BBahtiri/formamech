"""A finite sampling budget preserves accepted sequences and refuses exhaustion."""

from __future__ import annotations

import hashlib

import pytest

from piginot.contracts import ProblemSpec, canonical_json_bytes
from piginot.problem import verification
from piginot_campaigns.adapters import load_problem
from tests.problem_helpers import plate_document


def coupled_equality():
    document = load_problem("dogbone-v1").document()
    for parameter in document["parameters"]:
        if parameter["name"] == "W_grip":
            parameter.update(lo=18.0, hi=22.0, distribution="uniform")
        elif parameter["name"] == "W_gauge":
            parameter.update(lo=9.0, hi=11.0, distribution="uniform")
        else:
            value = 0.5 * (parameter["lo"] + parameter["hi"])
            parameter.update(lo=value, hi=value, distribution="fixed")
    document["constraints"].append(
        {
            "name": "coupled_widths",
            "expression": "W_grip/W_gauge",
            "lo": 2.0,
            "hi": 2.0,
            "unit": "1",
        }
    )
    return ProblemSpec.from_document(document)


def test_feasible_corners_do_not_leave_interior_sampling_unbounded(monkeypatch):
    spec = coupled_equality()
    assert len(verification.box_points(spec, interior=0)) == 3
    calls = []

    class RejectedDraws:
        def uniform(self, lo, hi):
            calls.append((lo, hi))
            if len(calls) > 2000:
                pytest.fail(
                    "interior sampling exceeded its deterministic attempt budget"
                )
            return lo + (hi - lo) * (0.25 if len(calls) % 2 else 0.75)

    monkeypatch.setattr(verification.random, "Random", lambda seed: RejectedDraws())
    with pytest.raises(ValueError, match="1000 attempts") as raised:
        verification.box_points(spec, interior=1, seed=2801)
    assert type(raised.value).__name__ == "ParameterSamplingError"
    assert raised.value.requested == 1 and raised.value.accepted == 0
    assert raised.value.attempts == 1000
    assert len(calls) == 2000


def test_seeded_sampling_refuses_the_actual_zero_volume_feasible_domain():
    with pytest.raises(verification.ParameterSamplingError) as raised:
        verification.box_points(coupled_equality(), interior=1, seed=2801)
    assert raised.value.attempts == 1000
    assert raised.value.requested == 1 and raised.value.accepted == 0


@pytest.mark.parametrize(
    "name,interior,seed,count,digest",
    [
        (
            "dogbone",
            0,
            0,
            25,
            "0549169ad1e548fc6a5bc1f3be1763d332a03be13501be64ddb57e5cb2dad3e3",
        ),
        (
            "dogbone",
            5,
            3,
            30,
            "a4c8b5e3d5b0a637eec1a5a7a63d06f6013e0e8baf6a0b4b56311c068bc93910",
        ),
        (
            "dogbone",
            64,
            2801,
            89,
            "4e7929eb38dc6f36e3f22c45ba8be1a85f15baa2b1ecb27c20ba0a41f9cc3c7e",
        ),
        (
            "notched",
            0,
            0,
            17,
            "e05718d2383d5bb2b94e78386e83653ccca1be89a5647a78ccfb77c2fde2def8",
        ),
        (
            "notched",
            5,
            3,
            22,
            "987344ba03441713dd3e3fad1490023f2e3edd5fee72c9e66fe75e5933983038",
        ),
        (
            "notched",
            64,
            2801,
            81,
            "b779b5f438d0ec6aabdc8bbc7f3130b320ab30dd5196af2cfb0ee5a6a0863a1b",
        ),
        (
            "plate",
            0,
            0,
            5,
            "df2e21b884515e549280a106d40390ccaf87c3c5995d3891714430f5d95bb6ef",
        ),
        (
            "plate",
            5,
            3,
            10,
            "ebe69a89134ce98bd3d2667fe2dcf88efc187cd7eee7465318e1391e366b9058",
        ),
        (
            "plate",
            64,
            2801,
            69,
            "2db67e0d8a871df3367f589e7057a27f443292759836317481dc775db90e7a3a",
        ),
    ],
)
def test_valid_point_sequences_match_prepatch_bytes(
    name, interior, seed, count, digest
):
    spec = (
        ProblemSpec.from_document(plate_document())
        if name == "plate"
        else load_problem(name + "-v1")
    )
    points = verification.box_points(spec, interior=interior, seed=seed)
    assert len(points) == count
    assert hashlib.sha256(canonical_json_bytes(points)).hexdigest() == digest
