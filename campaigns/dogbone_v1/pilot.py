"""Public, deterministic family-level validation evidence for the S9 proposal."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from campaigns.dogbone_v1.family import compiled_family, parameter_point
from piginot.contracts import hash_spec
from piginot.morphing import apply_morph

PUBLIC_VALIDATION_SEED = "dogbone-family-validation-20260828-v1"
SAMPLE_GENERATOR_ID = "blake2b-latin-hypercube-four-axis-v1"
_PERSON = b"piginot-dbv1"
_REPORTED_METRIC_DECIMALS = 12
_BOUNDS = (
    (40.0, 70.0),
    (16.0, 26.0),
    (6.0, 14.0),
    (8.0, 20.0),
)
_PARAMETER_NAMES = ("L_total", "W_grip", "W_gauge", "R_fillet")


@dataclass(frozen=True, slots=True)
class FamilyValidationSummary:
    """Canonical public result of one family-level morph validation sweep."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    recipe_id: str
    public_seed: str
    sample_generator_id: str
    candidate_count: int
    attempted: int
    membership_accepted: int
    accepted: int
    membership_rejections: Mapping[str, int]
    morph_rejections: Mapping[str, int]
    minimum_jacobian_determinant: float
    minimum_feature_size_mm: float
    minimum_angle_degrees: float
    maximum_aspect_ratio: float
    quadrature_points_per_shape: int
    reference_mesh_id: str
    validity_policy_hash: str
    compiled_family_hash: str
    worst_case_parameters: Mapping[str, tuple[float, float, float, float]]

    def __post_init__(self) -> None:
        for name in (
            "minimum_jacobian_determinant",
            "minimum_feature_size_mm",
            "minimum_angle_degrees",
            "maximum_aspect_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _reported_metric(getattr(self, name)),
            )
        object.__setattr__(
            self,
            "membership_rejections",
            MappingProxyType(dict(sorted(self.membership_rejections.items()))),
        )
        object.__setattr__(
            self,
            "morph_rejections",
            MappingProxyType(dict(sorted(self.morph_rejections.items()))),
        )
        object.__setattr__(
            self,
            "worst_case_parameters",
            MappingProxyType(dict(sorted(self.worst_case_parameters.items()))),
        )


def _reported_metric(value: float) -> float:
    rounded = round(value, _REPORTED_METRIC_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _digest(seed_key: bytes, domain: str, index: int) -> bytes:
    payload = domain.encode("utf-8") + b"\0" + index.to_bytes(8, "big")
    return hashlib.blake2b(
        payload,
        key=seed_key,
        digest_size=16,
        person=_PERSON,
    ).digest()


def _uniform(seed_key: bytes, domain: str, index: int) -> float:
    integer = int.from_bytes(_digest(seed_key, domain, index)[:8], "big")
    return (integer + 0.5) / 2**64


def _permutation(seed_key: bytes, axis_name: str, count: int) -> tuple[int, ...]:
    def key(index: int) -> tuple[bytes, int]:
        return _digest(seed_key, f"permutation/{axis_name}", index), index

    return tuple(sorted(range(count), key=key))


def _samples(count: int) -> tuple[tuple[float, float, float, float], ...]:
    seed_key = hashlib.blake2b(
        PUBLIC_VALIDATION_SEED.encode("utf-8"), digest_size=32
    ).digest()
    permutations = tuple(
        _permutation(seed_key, axis_name, count) for axis_name in _PARAMETER_NAMES
    )
    rows: list[tuple[float, float, float, float]] = []
    for row in range(count):
        values = tuple(
            lower
            + (upper - lower)
            * (
                permutations[axis_index][row]
                + _uniform(seed_key, f"jitter/{axis_name}", row)
            )
            / count
            for axis_index, (axis_name, (lower, upper)) in enumerate(
                zip(_PARAMETER_NAMES, _BOUNDS, strict=True)
            )
        )
        rows.append((values[0], values[1], values[2], values[3]))
    return tuple(rows)


def run_family_validation_sweep(
    *,
    required_accepted: int = 500,
    candidate_count: int = 900,
) -> FamilyValidationSummary:
    """Validate a disjoint public LHS without using production campaign secrets."""

    if required_accepted <= 0 or candidate_count < required_accepted:
        raise ValueError("candidate_count must cover a positive accepted population")
    family = compiled_family()
    membership_rejections: Counter[str] = Counter()
    morph_rejections: Counter[str] = Counter()
    minima = {
        "jacobian": float("inf"),
        "feature": float("inf"),
        "angle": float("inf"),
    }
    maximum_aspect = 0.0
    worst: dict[str, tuple[float, float, float, float]] = {}
    accepted = 0
    membership_accepted = 0
    attempted = 0
    quadrature_points = 0

    for values in _samples(candidate_count):
        attempted += 1
        theta = parameter_point(*values)
        decision = family.membership.evaluate(theta)
        if not decision.accepted:
            membership_rejections.update(
                reason.value for reason in decision.reason_codes
            )
            continue
        membership_accepted += 1
        result = apply_morph(
            family.reference_mesh,
            family.morph_map,
            family.membership,
            theta,
        )
        if not result.accepted:
            morph_rejections.update(reason.value for reason in result.reason_codes)
            continue
        accepted += 1
        report = result.report
        quadrature_points = report.quadrature_points_checked
        measures = {
            "jacobian": report.minimum_jacobian_determinant,
            "feature": report.minimum_feature_size_mm,
            "angle": report.minimum_angle_degrees,
        }
        for name, value in measures.items():
            if value is None:
                raise RuntimeError("accepted morph omitted a validity measure")
            if value < minima[name]:
                minima[name] = value
                worst[name] = values
        if report.maximum_aspect_ratio is None:
            raise RuntimeError("accepted morph omitted its aspect ratio")
        if report.maximum_aspect_ratio > maximum_aspect:
            maximum_aspect = report.maximum_aspect_ratio
            worst["aspect"] = values
        if accepted == required_accepted:
            break

    if accepted != required_accepted:
        raise RuntimeError("public validation stream did not reach its accepted target")
    return FamilyValidationSummary(
        recipe_id=family.recipe_id,
        public_seed=PUBLIC_VALIDATION_SEED,
        sample_generator_id=SAMPLE_GENERATOR_ID,
        candidate_count=candidate_count,
        attempted=attempted,
        membership_accepted=membership_accepted,
        accepted=accepted,
        membership_rejections=membership_rejections,
        morph_rejections=morph_rejections,
        minimum_jacobian_determinant=minima["jacobian"],
        minimum_feature_size_mm=minima["feature"],
        minimum_angle_degrees=minima["angle"],
        maximum_aspect_ratio=maximum_aspect,
        quadrature_points_per_shape=quadrature_points,
        reference_mesh_id=family.reference_mesh.mesh_id,
        validity_policy_hash=family.membership.validity_policy_hash,
        compiled_family_hash=hash_spec(family, "content"),
        worst_case_parameters=worst,
    )


__all__ = (
    "FamilyValidationSummary",
    "PUBLIC_VALIDATION_SEED",
    "SAMPLE_GENERATOR_ID",
    "run_family_validation_sweep",
)
