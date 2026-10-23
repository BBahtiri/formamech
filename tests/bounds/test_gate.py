"""S23 calibration envelope gate: Mahalanobis distance to TRAIN (ADR 0104)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from piginot.bounds import GateError, MahalanobisGate

AXES = ("width", "radius", "modulus")


def _train(seed: int = 3, cases: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width = rng.uniform(16.0, 30.0, cases)
    radius = 0.2 * width + rng.uniform(0.0, 2.0, cases)
    modulus = np.full(cases, 760.0)
    return np.column_stack([width, radius, modulus])


@dataclass(frozen=True)
class _Quantity:
    value: float


def test_fixed_axes_are_dropped_and_calibration_cases_are_admitted() -> None:
    train = _train()
    calibration = _train(seed=4, cases=32)
    gate = MahalanobisGate.fit(AXES, train.tolist(), calibration.tolist())
    assert gate.axes == ("width", "radius")
    for row in calibration:
        assert gate.admits(dict(zip(AXES, row, strict=True)))
    distances = [
        gate.distance(dict(zip(AXES, row, strict=True))) for row in calibration
    ]
    assert gate.max_distance == max(distances)


def test_the_distance_matches_the_definition() -> None:
    train = _train()
    gate = MahalanobisGate.fit(AXES, train.tolist(), train[:4].tolist())
    kept = train[:, :2]
    inverse = np.linalg.inv(np.cov(kept, rowvar=False))
    point = np.array([25.0, 3.0])
    centred = point - kept.mean(axis=0)
    expected = float(np.sqrt(centred @ inverse @ centred))
    values = {"width": 25.0, "radius": 3.0, "modulus": 760.0}
    assert gate.distance(values) == pytest.approx(expected, rel=1e-12)


def test_a_point_far_from_train_is_not_admitted() -> None:
    train = _train()
    gate = MahalanobisGate.fit(AXES, train.tolist(), train[:16].tolist())
    # inside the box of each axis, against the width-radius correlation
    far = {"width": 16.5, "radius": 8.0, "modulus": 760.0}
    assert gate.distance(far) > gate.max_distance
    assert not gate.admits(far)


def test_quantities_are_read_by_value() -> None:
    train = _train()
    gate = MahalanobisGate.fit(AXES, train.tolist(), train[:4].tolist())
    plain = {"width": 22.0, "radius": 4.0, "modulus": 760.0}
    wrapped = {key: _Quantity(value) for key, value in plain.items()}
    assert gate.distance(plain) == gate.distance(wrapped)


def test_the_gate_survives_a_json_round_trip_exactly() -> None:
    train = _train()
    gate = MahalanobisGate.fit(AXES, train.tolist(), _train(5, 16).tolist())
    copy = MahalanobisGate.from_document(json.loads(json.dumps(gate.document())))
    assert copy == gate
    point = {"width": 21.3, "radius": 5.1, "modulus": 760.0}
    assert copy.distance(point) == gate.distance(point)


def test_a_missing_input_is_an_error() -> None:
    gate = MahalanobisGate.fit(AXES, _train().tolist(), _train()[:4].tolist())
    with pytest.raises(GateError, match="radius"):
        gate.distance({"width": 20.0})
    with pytest.raises(GateError, match="finite"):
        gate.distance({"width": float("nan"), "radius": 1.0})


@pytest.mark.parametrize(
    ("train", "calibration", "message"),
    [
        ([[1.0, 2.0, 3.0]] * 10, [[1.0, 2.0, 3.0]], "varies"),
        ([[float(i), float(i * i), 2.0] for i in range(2)], [[0, 1, 2]], "more cases"),
        ([[1.0, 2.0]] * 10, [[1.0, 2.0]], "array"),
        ([[float("inf"), 1.0, 2.0]] * 10, [[1.0, 1.0, 2.0]], "finite"),
        ([[float(i), 2.0 * i, 3.0] for i in range(10)], [[1.0, 2.0, 3.0]], "singular"),
        ([[float(i), float(i * 7 % 5), 3.0] for i in range(10)], [], "calibration"),
    ],
)
def test_bad_fits_are_refused(
    train: list[list[float]], calibration: list[list[float]], message: str
) -> None:
    with pytest.raises(GateError, match=message):
        MahalanobisGate.fit(AXES, train, calibration)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axes": (), "mean": (), "precision": (), "max_distance": 0.0},
        {"axes": ("a", "a"), "mean": (0, 0), "precision": ((1, 0), (0, 1))},
        {"axes": ("a",), "mean": (0.0, 1.0), "precision": ((1.0,),)},
        {"axes": ("a", "b"), "mean": (0.0, 1.0), "precision": ((1.0,), (1.0,))},
        {"axes": ("a",), "mean": (float("nan"),), "precision": ((1.0,),)},
        {"axes": ("a",), "mean": (0.0,), "precision": ((True,),)},
        {"axes": ("a",), "mean": (0.0,), "precision": ((1.0,),), "max_distance": -1},
    ],
)
def test_malformed_gates_are_refused(kwargs: dict[str, object]) -> None:
    arguments = {"max_distance": 1.0, **kwargs}
    with pytest.raises(GateError):
        MahalanobisGate(**arguments)  # type: ignore[arg-type]


def test_a_malformed_document_is_refused() -> None:
    with pytest.raises(GateError, match="malformed"):
        MahalanobisGate.from_document({"axes": ["a"]})
