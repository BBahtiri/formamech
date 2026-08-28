from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

import pytest

from piginot.contracts import (
    CanonicalQuantity,
    Dimension,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.doe.design import (
    AxisSource,
    AxisSpec,
    CoupledDesignSpec,
    DesignPoint,
    DifficultRegion,
    InteractionSpec,
    RejectedProposal,
    generate_coupled_design,
    normalized_coordinates,
    weighted_distance,
)
from piginot.doe.leakage import LeakagePolicy, bind_leakage_groups


@dataclass(frozen=True, slots=True)
class _Bound:
    name: str
    lower_bound: CanonicalQuantity
    upper_bound: CanonicalQuantity


@dataclass(frozen=True, slots=True)
class _Decision:
    accepted: bool
    derived_values: MappingProxyType[str, CanonicalQuantity]
    reason_codes: tuple[str, ...] = ()


class _CoupledPredicate:
    """Small public-protocol double for the S6 predicate."""

    def __init__(self) -> None:
        self.primitive_bounds = (
            _bound("x", 0.0, 1.0),
            _bound("y", 0.0, 1.0),
        )
        self.derived_bounds = (_bound("governing_ratio", 0.12, 0.88),)
        self.evaluated: list[MappingProxyType[str, CanonicalQuantity]] = []

    def evaluate(self, theta: dict[str, CanonicalQuantity]) -> _Decision:
        frozen = MappingProxyType(dict(theta))
        self.evaluated.append(frozen)
        x = theta["x"].value
        y = theta["y"].value
        ratio = (x + 2.0 * y) / 3.0
        accepted = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.12 <= ratio <= 0.88
        derived = (
            {"governing_ratio": _quantity(ratio)}
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            else {}
        )
        reasons = ()
        if not accepted:
            reasons = (
                "PARAMETER_OUT_OF_BOUNDS"
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
                else "DERIVED_QUANTITY_OUT_OF_BOUNDS",
            )
        return _Decision(accepted, MappingProxyType(derived), reasons)


def _quantity(value: float) -> CanonicalQuantity:
    return CanonicalQuantity(value, Unit.ONE, Dimension.DIMENSIONLESS)


def _bound(name: str, lower: float, upper: float) -> _Bound:
    return _Bound(name, _quantity(lower), _quantity(upper))


def _spec(*, validation_count: int = 40) -> CoupledDesignSpec:
    return CoupledDesignSpec(
        seed=b"public-doe-seed-v1",
        sample_count=30,
        reserve_count=5,
        candidate_count=320,
        validation_count=validation_count,
        validation_candidate_count=240,
        guard_band_count=6,
        guard_band_fraction=0.04,
        axes=(
            AxisSpec(
                "x",
                AxisSource.PRIMITIVE,
                weight=1.0,
                strata=3,
                minimum_per_stratum=2,
                boundary_width=0.12,
                minimum_per_boundary=1,
            ),
            AxisSpec(
                "y",
                AxisSource.PRIMITIVE,
                weight=1.0,
                strata=3,
                minimum_per_stratum=2,
                boundary_width=0.12,
                minimum_per_boundary=1,
            ),
            AxisSpec(
                "governing_ratio",
                AxisSource.DERIVED,
                weight=2.0,
                strata=3,
                minimum_per_stratum=2,
                boundary_width=0.12,
                minimum_per_boundary=1,
            ),
        ),
        leakage_policy=LeakagePolicy(near_duplicate_tolerance=0.0),
        interactions=(
            InteractionSpec(
                "x-by-governing-ratio",
                axes=("x", "governing_ratio"),
                minimum_per_cell=1,
            ),
        ),
        difficult_regions=(
            DifficultRegion(
                "high-ratio-low-x",
                normalized_bounds={
                    "x": (0.0, 0.45),
                    "governing_ratio": (0.70, 1.0),
                },
                minimum_count=2,
            ),
        ),
    )


def test_design_is_deterministic_and_validation_stream_is_domain_separated() -> None:
    first = generate_coupled_design(_CoupledPredicate(), _spec())
    replay = generate_coupled_design(_CoupledPredicate(), _spec())
    longer_validation = generate_coupled_design(
        _CoupledPredicate(),
        replace(
            _spec(),
            validation_count=50,
            validation_candidate_count=260,
        ),
    )

    assert first == replay
    assert first.selected_points == longer_validation.selected_points
    assert first.reserve_points == longer_validation.reserve_points
    assert first.candidate_stream_id != first.validation_stream_id
    assert {point.case_id for point in first.selected_points}.isdisjoint(
        point.case_id for point in first.validation_points
    )


def test_design_binds_approved_leakage_policy_before_role_allocation() -> None:
    predicate = _CoupledPredicate()
    policy = LeakagePolicy(near_duplicate_tolerance=1.0)
    spec = replace(_spec(), leakage_policy=policy)

    design = generate_coupled_design(predicate, spec)
    replayed = bind_leakage_groups(design.leakage_descriptors, policy=policy)
    campaign_points = (*design.selected_points, *design.reserve_points)
    descriptor_by_id = {
        descriptor.case_id: descriptor for descriptor in design.leakage_descriptors
    }

    assert design.leakage_policy == policy
    assert len(set(replayed.values())) == 1
    assert all(
        point.leakage_group_id == replayed[point.case_id] for point in campaign_points
    )
    assert all(
        descriptor_by_id[point.case_id].normalized_coordinates
        == normalized_coordinates(predicate, spec.axes, point)
        for point in campaign_points
    )


def test_design_spec_has_stable_unambiguous_canonical_seed_identity() -> None:
    text = replace(_spec(), seed="same-seed")
    text_replay = replace(_spec(), seed="same-seed")
    binary = replace(_spec(), seed=b"same-seed")

    text_bytes = canonical_json_bytes(text)
    binary_bytes = canonical_json_bytes(binary)

    assert text_bytes == canonical_json_bytes(text_replay)
    assert hash_spec(text, "bundle") == hash_spec(text_replay, "bundle")
    assert text_bytes != binary_bytes
    assert hash_spec(text, "bundle") != hash_spec(binary, "bundle")
    assert b'"encoding":"text-utf8-base64url-v1"' in text_bytes
    assert b'"encoding":"bytes-base64url-v1"' in binary_bytes

    text_design = generate_coupled_design(_CoupledPredicate(), text)
    text_replay_design = generate_coupled_design(_CoupledPredicate(), text_replay)
    binary_design = generate_coupled_design(_CoupledPredicate(), binary)
    assert text_design == text_replay_design
    assert text_design.candidate_stream_id != binary_design.candidate_stream_id
    assert "same-seed" not in repr(text)


def test_design_requires_all_variable_primitive_and_derived_governing_axes() -> None:
    predicate = _CoupledPredicate()
    without_derived = replace(
        _spec(),
        axes=tuple(
            axis for axis in _spec().axes if axis.source is AxisSource.PRIMITIVE
        ),
        interactions=(),
        difficult_regions=(),
    )
    with pytest.raises(ValueError, match="derived governing axes"):
        generate_coupled_design(predicate, without_derived)

    without_primitive = replace(
        _spec(),
        axes=tuple(axis for axis in _spec().axes if axis.name != "y"),
        interactions=(),
        difficult_regions=(),
    )
    with pytest.raises(ValueError, match="primitive axes"):
        generate_coupled_design(predicate, without_primitive)


def test_candidates_use_public_membership_evaluate_and_canonical_quantities() -> None:
    predicate = _CoupledPredicate()

    result = generate_coupled_design(predicate, _spec())

    assert predicate.evaluated
    assert all(
        isinstance(value, CanonicalQuantity)
        for theta in predicate.evaluated
        for value in theta.values()
    )
    assert all(
        set(point.derived_coordinates) == {"governing_ratio"}
        for point in result.selected_points
    )


def test_rarest_quota_first_selection_covers_declared_difficult_region() -> None:
    result = generate_coupled_design(_CoupledPredicate(), _spec())

    difficult = [
        point
        for point in result.selected_points
        if point.coordinates["x"] <= 0.45
        and (point.derived_coordinates["governing_ratio"] - 0.12) / 0.76 >= 0.70
    ]

    assert len(difficult) >= 2
    assert result.selection_trace[0].startswith("quota:")


def test_guard_band_cases_are_negative_and_kept_separate_from_active_design() -> None:
    predicate = _CoupledPredicate()

    result = generate_coupled_design(predicate, _spec())

    assert len(result.guard_band_points) == 6
    assert all(point.guard_band for point in result.guard_band_points)
    assert all(not point.guard_band for point in result.selected_points)
    assert result.points == (*result.selected_points, *result.guard_band_points)
    assert result.guard_band_policy_id == "primitive-bound-negative-v1"
    assert len(result.guard_band_rejections) == len(result.guard_band_points)
    assert all(
        rejection.reason_codes == ("PARAMETER_OUT_OF_BOUNDS",)
        for rejection in result.guard_band_rejections
    )
    for point in result.guard_band_points:
        assert any(value < 0.0 or value > 1.0 for value in point.coordinates.values())


def test_membership_rejections_are_visible_and_stream_accounting_is_complete() -> None:
    spec = _spec()
    result = generate_coupled_design(_CoupledPredicate(), spec)

    assert result.candidate_rejections
    assert result.validation_rejections
    assert result.candidate_accepted_count + len(result.candidate_rejections) == (
        spec.candidate_count
    )
    assert result.validation_accepted_count + len(result.validation_rejections) == (
        spec.validation_candidate_count
    )
    assert {
        reason
        for rejection in (*result.candidate_rejections, *result.validation_rejections)
        for reason in rejection.reason_codes
    } == {"DERIVED_QUANTITY_OUT_OF_BOUNDS"}


def test_design_records_reject_incoherent_case_and_rejection_evidence() -> None:
    point = DesignPoint(
        case_id="case",
        leakage_group_id="group",
        coordinates={"x": 0.2},
        derived_coordinates={"ratio": 0.3},
        strata={"x": 0, "ratio": 0},
    )
    rejection = RejectedProposal(
        case_id="rejected",
        stream_domain="candidate-v1",
        coordinates={"x": 1.1},
        reason_codes=("OUT_OF_BOUNDS",),
    )

    assert point.all_coordinates == {"ratio": 0.3, "x": 0.2}
    with pytest.raises(ValueError, match="case-level"):
        replace(point, case_id="case:node")
    with pytest.raises(ValueError, match="coordinates must not be empty"):
        replace(point, coordinates={})
    with pytest.raises(ValueError, match="disjoint"):
        replace(point, derived_coordinates={"x": 0.3})
    with pytest.raises(ValueError, match="strata"):
        replace(point, strata={"x": -1})
    with pytest.raises(ValueError, match="boolean"):
        replace(point, guard_band=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coordinates must not be empty"):
        replace(rejection, coordinates={})
    with pytest.raises(ValueError, match="stable reason"):
        replace(rejection, reason_codes=())


def test_design_contract_rejects_leakage_group_or_descriptor_drift() -> None:
    design = generate_coupled_design(_CoupledPredicate(), _spec())
    changed_point = replace(
        design.selected_points[0],
        leakage_group_id="caller-substituted-group",
    )

    with pytest.raises(ValueError, match="leakage grouping policy"):
        replace(
            design,
            selected_points=(changed_point, *design.selected_points[1:]),
        )
    with pytest.raises(ValueError, match="selected and reserve"):
        replace(design, leakage_descriptors=design.leakage_descriptors[1:])


def test_spec_rejects_undeclared_or_nonpositive_design_controls() -> None:
    with pytest.raises(ValueError, match="weights"):
        replace(_spec(), axes=(AxisSpec("x", AxisSource.PRIMITIVE, weight=0.0),))
    with pytest.raises(ValueError, match="declared axis"):
        replace(
            _spec(),
            interactions=(InteractionSpec("bad", ("x", "missing")),),
        )
    with pytest.raises(ValueError, match="normalized"):
        DifficultRegion("bad", {"x": (-0.1, 0.5)}, minimum_count=1)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AxisSpec("", AxisSource.PRIMITIVE),
        lambda: AxisSpec("x", AxisSource.PRIMITIVE, weight=float("nan")),
        lambda: AxisSpec("x", AxisSource.PRIMITIVE, strata=0),
        lambda: AxisSpec("x", AxisSource.PRIMITIVE, minimum_per_stratum=-1),
        lambda: AxisSpec("x", AxisSource.PRIMITIVE, boundary_width=0.0),
        lambda: AxisSpec("x", AxisSource.PRIMITIVE, minimum_per_boundary=-1),
        lambda: InteractionSpec("one-axis", ("x",)),
        lambda: InteractionSpec("duplicate-axis", ("x", "x")),
        lambda: InteractionSpec("negative", ("x", "y"), minimum_per_cell=-1),
        lambda: DifficultRegion("empty", {}),
        lambda: DifficultRegion("wrong-shape", {"x": (0.2, 0.3, 0.4)}),
        lambda: DifficultRegion("negative", {"x": (0.2, 0.3)}, minimum_count=-1),
    ],
)
def test_declarations_fail_closed_on_invalid_values(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": b""},
        {"sample_count": 0},
        {"reserve_count": -1},
        {"candidate_count": 30},
        {"validation_count": 0},
        {"validation_candidate_count": 39},
        {"guard_band_count": -1},
        {"guard_band_fraction": 0.0},
        {"axes": ()},
        {"leakage_policy": object()},
        {"axes": (_spec().axes[0], _spec().axes[0])},
        {"interactions": (InteractionSpec("same", ("x", "y")),) * 2},
        {
            "difficult_regions": (
                DifficultRegion("same", {"x": (0.1, 0.2)}),
                DifficultRegion("same", {"y": (0.1, 0.2)}),
            )
        },
        {"difficult_regions": (DifficultRegion("unknown", {"missing": (0.1, 0.2)}),)},
    ],
)
def test_design_spec_rejects_invalid_population_or_declaration(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        replace(_spec(), **changes)


def test_distance_requires_declared_positive_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        weighted_distance({}, {}, ())
