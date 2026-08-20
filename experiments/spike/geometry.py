"""Analytic, quarantined geometries for the S1 risk spike.

The centre-notched family uses a fixed 50 mm tensile length and a C2-blended
double-edge notch.  ``notch_radius`` is the exact root radius of curvature;
the protocol does not prescribe numerical bounds or a notch construction, so
both choices are explicit here and are recorded in the spike report.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

DOGBONE_RANGES: Final[dict[str, tuple[float, float]]] = {
    "L_total": (40.0, 70.0),
    "W_grip": (16.0, 26.0),
    "W_gauge": (6.0, 14.0),
    "R_fillet": (8.0, 20.0),
}

# S1 leaves these bounds unspecified.  They maintain a positive ligament and
# keep the smooth notch blend inside the fixed half-length for every sample.
CENTRE_NOTCHED_RANGES: Final[dict[str, tuple[float, float]]] = {
    "width": (16.0, 30.0),
    "notch_radius": (2.0, 8.0),
    "notch_depth": (1.0, 5.0),
}

CENTRE_NOTCHED_LENGTH: Final = 50.0
BOUNDARY_POINT_COUNT: Final = 320
TOPOLOGY_SIGNATURE: Final = "quarter-single-component-no-holes"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SpikeMesh:
    """One independently owned quarter-model mesh."""

    geometry_id: str
    mesh_id: str
    family: str
    seed: int
    parameters: Mapping[str, float]
    points: FloatArray
    cells: IntArray
    boundary_points: FloatArray
    qoi_node_ids: IntArray
    topology_signature: str = TOPOLOGY_SIGNATURE

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return mesh-qualified node identities for split-leakage checks."""
        return tuple(f"{self.mesh_id}:{index}" for index in range(len(self.points)))


def _validate_parameters(
    family: str,
    parameters: Mapping[str, float],
) -> dict[str, float]:
    ranges = _ranges_for(family)
    missing = ranges.keys() - parameters.keys()
    extra = parameters.keys() - ranges.keys()
    if missing or extra:
        raise ValueError(
            f"{family} parameters mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    values = {name: float(parameters[name]) for name in ranges}
    for name, (lower, upper) in ranges.items():
        value = values[name]
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")

    if family == "dogbone":
        height_change = (values["W_grip"] - values["W_gauge"]) / 2.0
        if height_change <= 0.0:
            raise ValueError("W_gauge must be smaller than W_grip")
        if values["R_fillet"] <= height_change:
            raise ValueError("R_fillet must exceed the half-width change")
        fillet_length = math.sqrt(
            height_change * (2.0 * values["R_fillet"] - height_change)
        )
        if values["L_total"] / 2.0 - fillet_length < 2.0:
            raise ValueError("L_total leaves less than 2 mm of gauge")
    else:
        half_width = values["width"] / 2.0
        if values["notch_depth"] >= half_width - 1.0:
            raise ValueError("notch_depth leaves an insufficient ligament")
        blend_length = math.sqrt(6.0 * values["notch_depth"] * values["notch_radius"])
        if blend_length >= CENTRE_NOTCHED_LENGTH / 2.0 - 1.0:
            raise ValueError("notch_radius and notch_depth exceed the half-length")
    return values


def _ranges_for(family: str) -> dict[str, tuple[float, float]]:
    if family == "dogbone":
        return DOGBONE_RANGES
    if family == "centre_notched":
        return CENTRE_NOTCHED_RANGES
    raise ValueError(f"unknown geometry family: {family}")


def _dogbone_profile(
    parameters: Mapping[str, float],
    nx: int,
) -> tuple[FloatArray, FloatArray, float]:
    half_length = parameters["L_total"] / 2.0
    grip_height = parameters["W_grip"] / 2.0
    gauge_height = parameters["W_gauge"] / 2.0
    radius = parameters["R_fillet"]
    height_change = grip_height - gauge_height
    fillet_length = math.sqrt(height_change * (2.0 * radius - height_change))
    transition = half_length - fillet_length

    fillet_columns = max(2, nx // 3)
    gauge_columns = nx - fillet_columns
    x_gauge = np.linspace(0.0, transition, gauge_columns + 1)
    x_fillet = np.linspace(transition, half_length, fillet_columns + 1)[1:]
    x = np.concatenate((x_gauge, x_fillet))

    center_y = gauge_height + radius
    radicand = np.maximum(radius**2 - (x - transition) ** 2, 0.0)
    curved_height = center_y - np.sqrt(radicand)
    height = np.where(x <= transition, gauge_height, curved_height)
    return x, height, transition


def _centre_notched_profile(
    parameters: Mapping[str, float],
    nx: int,
) -> tuple[FloatArray, FloatArray, float]:
    half_length = CENTRE_NOTCHED_LENGTH / 2.0
    half_width = parameters["width"] / 2.0
    depth = parameters["notch_depth"]
    radius = parameters["notch_radius"]
    blend_length = math.sqrt(6.0 * depth * radius)

    notch_columns = max(2, nx // 2)
    outer_columns = nx - notch_columns
    x_notch = np.linspace(0.0, blend_length, notch_columns + 1)
    x_outer = np.linspace(blend_length, half_length, outer_columns + 1)[1:]
    x = np.concatenate((x_notch, x_outer))

    scaled = np.minimum(x / blend_length, 1.0)
    blend = (1.0 - scaled**2) ** 3
    height = half_width - depth * blend
    return x, height, blend_length


def _mapped_mesh(
    x: FloatArray, height: FloatArray, ny: int
) -> tuple[FloatArray, IntArray]:
    eta = np.linspace(0.0, 1.0, ny + 1)
    point_blocks = [
        np.column_stack((np.full_like(eta, x_i), eta * h_i))
        for x_i, h_i in zip(x, height, strict=True)
    ]
    points = np.concatenate(point_blocks, axis=0)

    cells: list[tuple[int, int, int]] = []
    stride = ny + 1
    for i in range(len(x) - 1):
        for j in range(ny):
            lower_left = i * stride + j
            lower_right = (i + 1) * stride + j
            upper_left = lower_left + 1
            upper_right = lower_right + 1
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return points, np.asarray(cells, dtype=np.int64)


def _boundary_polyline(points: FloatArray, nx: int, ny: int) -> FloatArray:
    stride = ny + 1
    bottom = np.asarray([i * stride for i in range(nx + 1)], dtype=np.int64)
    right = np.arange(nx * stride + 1, (nx + 1) * stride, dtype=np.int64)
    top = np.asarray([i * stride + ny for i in range(nx - 1, -1, -1)], dtype=np.int64)
    left = np.arange(ny - 1, 0, -1, dtype=np.int64)
    indices = np.concatenate((bottom, right, top, left))
    return points[indices]


def _resample_closed_polyline(polyline: FloatArray, count: int) -> FloatArray:
    closed = np.vstack((polyline, polyline[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    segment_ids = np.searchsorted(cumulative, targets, side="right") - 1
    segment_ids = np.clip(segment_ids, 0, len(segment_lengths) - 1)
    local = (targets - cumulative[segment_ids]) / segment_lengths[segment_ids]
    return closed[segment_ids] + local[:, None] * (
        closed[segment_ids + 1] - closed[segment_ids]
    )


def _stable_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def generate_geometry(
    family: str,
    parameters: Mapping[str, float],
    seed: int,
    *,
    nx: int = 32,
    ny: int = 10,
) -> SpikeMesh:
    """Generate one deterministic, fixed-topology quarter mesh."""
    if nx < 4 or ny < 2:
        raise ValueError("nx must be >= 4 and ny must be >= 2")
    values = _validate_parameters(family, parameters)

    if family == "dogbone":
        x, height, feature_limit = _dogbone_profile(values, nx)
        feature_extent = float(x[-1] - feature_limit)
        feature_selector = (x >= feature_limit) & (
            x <= feature_limit + 0.75 * feature_extent
        )
    else:
        x, height, feature_limit = _centre_notched_profile(values, nx)
        feature_selector = x <= 0.50 * feature_limit

    points, cells = _mapped_mesh(x, height, ny)
    mesh_payload = (
        np.ascontiguousarray(points).tobytes() + np.ascontiguousarray(cells).tobytes()
    )
    mesh_id = f"mesh-{_stable_digest(mesh_payload)[:16]}"
    identity_payload = json.dumps(
        {"family": family, "parameters": values, "seed": int(seed)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    geometry_id = f"{family}-{_stable_digest(identity_payload)[:16]}"

    top_nodes = np.arange(ny, len(points), ny + 1, dtype=np.int64)
    qoi_node_ids = top_nodes[feature_selector]
    boundary = _resample_closed_polyline(
        _boundary_polyline(points, nx, ny),
        BOUNDARY_POINT_COUNT,
    )
    return SpikeMesh(
        geometry_id=geometry_id,
        mesh_id=mesh_id,
        family=family,
        seed=int(seed),
        parameters=values,
        points=points,
        cells=cells,
        boundary_points=boundary,
        qoi_node_ids=qoi_node_ids,
    )


def sample_geometry_cases(
    *,
    master_seed: int,
    count_per_family: int = 30,
    nx: int = 32,
    ny: int = 10,
) -> tuple[SpikeMesh, ...]:
    """Sample a reproducible campaign with independent per-geometry seeds."""
    if count_per_family < 1:
        raise ValueError("count_per_family must be positive")
    families = ("dogbone", "centre_notched")
    children = np.random.SeedSequence(master_seed).spawn(count_per_family * 2)
    cases: list[SpikeMesh] = []

    for family_index, family in enumerate(families):
        ranges = _ranges_for(family)
        for index in range(count_per_family):
            child = children[family_index * count_per_family + index]
            case_seed = int(child.generate_state(1, dtype=np.uint64)[0])
            rng = np.random.default_rng(child)
            while True:
                parameters = {
                    name: float(rng.uniform(lower, upper))
                    for name, (lower, upper) in ranges.items()
                }
                try:
                    case = generate_geometry(
                        family,
                        parameters,
                        case_seed,
                        nx=nx,
                        ny=ny,
                    )
                except ValueError:
                    continue
                cases.append(case)
                break
    return tuple(cases)
