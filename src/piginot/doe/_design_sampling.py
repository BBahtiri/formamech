"""Private deterministic sampling mechanics for the public S9 DOE contract."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from piginot.contracts import CanonicalQuantity, Dimension, Unit
from piginot.doe.design import (
    AxisSource,
    AxisSpec,
    CoupledDesign,
    CoupledDesignSpec,
    DesignPoint,
    DoeSeed,
    FamilyMembershipPredicateLike,
    RejectedProposal,
    _BoundLike,
    _DecisionLike,
    _finite,
    _freeze,
    _text,
    quota_requirements,
    requirement_ids_for_point,
    weighted_distance,
)
from piginot.doe.leakage import LeakageDescriptor, LeakagePolicy, bind_leakage_groups

_CANDIDATE_DOMAIN = "piginot.s9.doe.candidate-lhs.v1"
_VALIDATION_DOMAIN = "piginot.s9.doe.validation-lhs.v1"
_GUARD_DOMAIN = "piginot.s9.doe.guard-band.v1"
_PERSON = b"piginot-s9-doe"
_SEED_PERSON = b"piginot-s9-seed"


@dataclass(frozen=True, slots=True)
class _CompiledAxis:
    spec: AxisSpec
    lower: float
    upper: float

    def normalize(self, value: float) -> float:
        if self.upper == self.lower:
            return 0.5
        return (value - self.lower) / (self.upper - self.lower)


@dataclass(frozen=True, slots=True)
class _Candidate:
    point: DesignPoint
    normalized: Mapping[str, float]
    leakage_descriptor: LeakageDescriptor


def generate_coupled_design_impl(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
) -> CoupledDesign:
    if not isinstance(spec, CoupledDesignSpec):
        raise TypeError("spec must be a CoupledDesignSpec")
    compiled = _compile_axes(predicate, spec)
    candidates, candidate_rejections = _evaluate_lhs_proposals(
        predicate,
        spec,
        compiled,
        count=spec.candidate_count,
        domain=_CANDIDATE_DOMAIN,
        case_prefix="candidate",
    )
    required = spec.sample_count + spec.reserve_count
    if len(candidates) < required:
        raise ValueError(
            "coupled membership accepted fewer candidates than samples and reserves"
        )

    requirements = quota_requirements(spec)
    selected, remaining, trace, counts = _select_rarest_quota_first(
        candidates,
        spec.sample_count,
        spec,
        requirements,
    )
    reserves = _select_maximin(
        remaining,
        spec.reserve_count,
        selected,
        spec.axes,
    )
    selected_points, reserve_points, leakage_descriptors = _bind_selected_leakage(
        selected,
        reserves,
        spec,
    )
    validation_pool, validation_rejections = _evaluate_lhs_proposals(
        predicate,
        spec,
        compiled,
        count=spec.validation_candidate_count,
        domain=_VALIDATION_DOMAIN,
        case_prefix="validation",
    )
    if len(validation_pool) < spec.validation_count:
        raise ValueError("independent validation stream has too few accepted points")
    validation = validation_pool[: spec.validation_count]
    guards, guard_rejections = _guard_band_points(predicate, spec, compiled)
    unresolved = tuple(
        quota_id
        for quota_id, target in requirements.items()
        if counts.get(quota_id, 0) < target
    )
    return CoupledDesign(
        selected_points=selected_points,
        reserve_points=reserve_points,
        guard_band_points=guards,
        validation_points=tuple(candidate.point for candidate in validation),
        candidate_rejections=candidate_rejections,
        validation_rejections=validation_rejections,
        guard_band_rejections=guard_rejections,
        candidate_accepted_count=len(candidates),
        validation_accepted_count=len(validation_pool),
        selection_trace=trace,
        unresolved_quota_ids=unresolved,
        candidate_stream_id=_stream_id(spec.seed, _CANDIDATE_DOMAIN),
        validation_stream_id=_stream_id(spec.seed, _VALIDATION_DOMAIN),
        leakage_policy=spec.leakage_policy,
        leakage_descriptors=leakage_descriptors,
    )


def _bind_selected_leakage(
    selected: Sequence[_Candidate],
    reserves: Sequence[_Candidate],
    spec: CoupledDesignSpec,
) -> tuple[
    tuple[DesignPoint, ...],
    tuple[DesignPoint, ...],
    tuple[LeakageDescriptor, ...],
]:
    campaign_candidates = (*tuple(selected), *tuple(reserves))
    descriptors = tuple(
        candidate.leakage_descriptor for candidate in campaign_candidates
    )
    groups = bind_leakage_groups(descriptors, policy=spec.leakage_policy)

    def with_bound_group(candidate: _Candidate) -> DesignPoint:
        return replace(
            candidate.point,
            leakage_group_id=groups[candidate.point.case_id],
        )

    return (
        tuple(with_bound_group(candidate) for candidate in selected),
        tuple(with_bound_group(candidate) for candidate in reserves),
        descriptors,
    )


def normalized_coordinates_impl(
    predicate: FamilyMembershipPredicateLike,
    axes: Sequence[AxisSpec],
    point: DesignPoint,
) -> Mapping[str, float]:
    temporary = CoupledDesignSpec(
        seed=b"normalization-contract",
        sample_count=1,
        candidate_count=1,
        validation_count=1,
        validation_candidate_count=1,
        axes=tuple(axes),
        leakage_policy=LeakagePolicy(near_duplicate_tolerance=0.0),
    )
    compiled = _compile_axes(predicate, temporary)
    combined = point.all_coordinates
    values = {
        axis.spec.name: axis.normalize(combined[axis.spec.name]) for axis in compiled
    }
    return _freeze(values)


def _compile_axes(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
) -> tuple[_CompiledAxis, ...]:
    primitive = _bound_table(predicate.primitive_bounds, "primitive")
    derived = _bound_table(predicate.derived_bounds, "derived")
    declarations = {axis.name: axis for axis in spec.axes}
    variable_primitive = {
        name for name, (lower, upper, _bound) in primitive.items() if upper > lower
    }
    missing_primitive = variable_primitive - set(declarations)
    if missing_primitive:
        raise ValueError(
            "DOE is missing variable primitive axes: "
            + ", ".join(sorted(missing_primitive))
        )
    missing_derived = set(derived) - set(declarations)
    if missing_derived:
        raise ValueError(
            "DOE is missing derived governing axes: "
            + ", ".join(sorted(missing_derived))
        )
    known = set(primitive) | set(derived)
    unknown = set(declarations) - known
    if unknown:
        raise ValueError("DOE contains axes absent from the membership predicate")

    compiled = []
    for axis in spec.axes:
        table = primitive if axis.source is AxisSource.PRIMITIVE else derived
        if axis.name not in table:
            raise ValueError(
                f"axis {axis.name!r} has the wrong primitive/derived source"
            )
        lower, upper, _bound = table[axis.name]
        if lower == upper and axis.strata != 1:
            raise ValueError("fixed predicate axes must declare exactly one stratum")
        compiled.append(_CompiledAxis(axis, lower, upper))
    return tuple(compiled)


def _bound_table(
    bounds: Sequence[_BoundLike],
    label: str,
) -> dict[str, tuple[float, float, _BoundLike]]:
    table: dict[str, tuple[float, float, _BoundLike]] = {}
    for bound in tuple(bounds):
        name = _text(getattr(bound, "name", None), f"{label} bound name")
        if name in table:
            raise ValueError(f"duplicate {label} membership bound")
        lower = _finite(bound.lower_bound.value, f"{name} lower bound")
        upper = _finite(bound.upper_bound.value, f"{name} upper bound")
        if lower > upper:
            raise ValueError(f"{name} membership bounds are reversed")
        table[name] = (lower, upper, bound)
    if label == "primitive" and not table:
        raise ValueError("membership predicate has no primitive bounds")
    return table


def _evaluate_lhs_proposals(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
    compiled: Sequence[_CompiledAxis],
    *,
    count: int,
    domain: str,
    case_prefix: str,
) -> tuple[tuple[_Candidate, ...], tuple[RejectedProposal, ...]]:
    primitive = _bound_table(predicate.primitive_bounds, "primitive")
    permutations = {
        name: _permutation(spec.seed, f"{domain}/permutation/{name}", count)
        for name in primitive
    }
    accepted: tuple[_Candidate, ...] = ()
    rejected: tuple[RejectedProposal, ...] = ()
    for row in range(count):
        raw = {
            name: lower
            + (upper - lower)
            * (
                permutations[name][row]
                + _uniform(spec.seed, f"{domain}/jitter/{name}", row)
            )
            / count
            for name, (lower, upper, _bound) in primitive.items()
        }
        evaluated = _evaluate_candidate(
            predicate,
            compiled,
            raw,
            case_id=f"{case_prefix}-{row:06d}",
            stream_domain=domain,
            morph_seed=_proposal_seed(spec.seed, domain, row),
        )
        if isinstance(evaluated, _Candidate):
            accepted = (*accepted, evaluated)
        else:
            rejected = (*rejected, evaluated)
    return accepted, rejected


def _evaluate_candidate(
    predicate: FamilyMembershipPredicateLike,
    compiled: Sequence[_CompiledAxis],
    raw: Mapping[str, float],
    *,
    case_id: str,
    stream_domain: str,
    morph_seed: int,
) -> _Candidate | RejectedProposal:
    primitive = _bound_table(predicate.primitive_bounds, "primitive")
    theta: dict[str, object] = {}
    for name, value in raw.items():
        bound = primitive[name][2]
        dimension = bound.lower_bound.dimension
        if not isinstance(bound.lower_bound.unit, Unit) or not isinstance(
            dimension, Dimension
        ):
            raise ValueError("membership bounds must expose canonical units")
        theta[name] = CanonicalQuantity(value, bound.lower_bound.unit, dimension)
    decision = predicate.evaluate(theta)
    if not isinstance(decision.accepted, bool):
        raise ValueError("membership decision accepted flag must be boolean")
    if not decision.accepted:
        return RejectedProposal(
            case_id=case_id,
            stream_domain=stream_domain,
            coordinates=raw,
            reason_codes=_stable_reason_codes(decision),
        )
    expected_derived = {bound.name for bound in predicate.derived_bounds}
    missing = expected_derived - set(decision.derived_values)
    if missing:
        raise ValueError("accepted membership decision omitted derived coordinates")
    derived = {
        name: _finite(value.value, f"derived coordinate {name}")
        for name, value in decision.derived_values.items()
    }
    combined = {**raw, **derived}
    normalized = {
        axis.spec.name: axis.normalize(combined[axis.spec.name]) for axis in compiled
    }
    if any(value < 0.0 or value > 1.0 for value in normalized.values()):
        raise ValueError("accepted membership point lies outside declared axis bounds")
    strata = {
        axis.spec.name: _stratum(normalized[axis.spec.name], axis.spec.strata)
        for axis in compiled
    }
    return _Candidate(
        point=DesignPoint(
            case_id=case_id,
            leakage_group_id=f"leakage-{_coordinate_id(raw)}",
            coordinates=raw,
            derived_coordinates=derived,
            strata=strata,
        ),
        normalized=_freeze(normalized),
        leakage_descriptor=LeakageDescriptor(
            case_id=case_id,
            geometry_lineage_id=f"geometry-{_coordinate_id(raw)}",
            morph_seed=morph_seed,
            normalized_coordinates=normalized,
        ),
    )


def _select_rarest_quota_first(
    candidates: Sequence[_Candidate],
    count: int,
    spec: CoupledDesignSpec,
    requirements: Mapping[str, int],
) -> tuple[
    tuple[_Candidate, ...],
    tuple[_Candidate, ...],
    tuple[str, ...],
    Mapping[str, int],
]:
    selected: tuple[_Candidate, ...] = ()
    remaining = tuple(candidates)
    trace: tuple[str, ...] = ()
    counts: Mapping[str, int] = _freeze({key: 0 for key in requirements})
    coverage = {
        candidate.point.case_id: requirement_ids_for_point(candidate.normalized, spec)
        for candidate in candidates
    }
    for _ in range(count):
        unmet = tuple(
            quota_id
            for quota_id, target in requirements.items()
            if counts[quota_id] < target
        )
        reachable = tuple(
            (
                sum(
                    quota_id in coverage[candidate.point.case_id]
                    for candidate in remaining
                ),
                quota_id,
            )
            for quota_id in unmet
        )
        reachable = tuple(item for item in reachable if item[0] > 0)
        target = min(reachable)[1] if reachable else None
        eligible = (
            tuple(
                candidate
                for candidate in remaining
                if target in coverage[candidate.point.case_id]
            )
            if target is not None
            else remaining
        )
        chosen = _maximin_choice(eligible, selected, spec.axes, coverage, unmet)
        chosen_ids = coverage[chosen.point.case_id]
        counts = _freeze(
            {
                quota_id: value + int(quota_id in chosen_ids)
                for quota_id, value in counts.items()
            }
        )
        selected = (*selected, chosen)
        remaining = tuple(
            candidate
            for candidate in remaining
            if candidate.point.case_id != chosen.point.case_id
        )
        trace = (*trace, f"quota:{target}" if target is not None else "maximin")
    return selected, remaining, trace, counts


def _select_maximin(
    candidates: Sequence[_Candidate],
    count: int,
    anchors: Sequence[_Candidate],
    axes: Sequence[AxisSpec],
) -> tuple[_Candidate, ...]:
    selected: tuple[_Candidate, ...] = ()
    remaining = tuple(candidates)
    for _ in range(count):
        chosen = _maximin_choice(
            remaining,
            (*tuple(anchors), *selected),
            axes,
            {},
            (),
        )
        selected = (*selected, chosen)
        remaining = tuple(
            candidate
            for candidate in remaining
            if candidate.point.case_id != chosen.point.case_id
        )
    return selected


def _maximin_choice(
    candidates: Sequence[_Candidate],
    selected: Sequence[_Candidate],
    axes: Sequence[AxisSpec],
    coverage: Mapping[str, frozenset[str]],
    unmet: Sequence[str],
) -> _Candidate:
    if not candidates:
        raise ValueError("maximin selection exhausted its candidate population")
    unmet_set = frozenset(unmet)

    def key(candidate: _Candidate) -> tuple[float, int, str]:
        distance = (
            min(
                weighted_distance(candidate.normalized, other.normalized, axes)
                for other in selected
            )
            if selected
            else 0.0
        )
        extra = len(coverage.get(candidate.point.case_id, frozenset()) & unmet_set)
        return (-distance, -extra, candidate.point.case_id)

    return min(candidates, key=key)


def _guard_band_points(
    predicate: FamilyMembershipPredicateLike,
    spec: CoupledDesignSpec,
    compiled: Sequence[_CompiledAxis],
) -> tuple[tuple[DesignPoint, ...], tuple[RejectedProposal, ...]]:
    """Create primitive-bound negative cases; derived OOD guards are not implied."""

    primitive = _bound_table(predicate.primitive_bounds, "primitive")
    names = tuple(primitive)
    guards: tuple[DesignPoint, ...] = ()
    rejections: tuple[RejectedProposal, ...] = ()
    for index in range(spec.guard_band_count):
        target = names[index % len(names)]
        upper_side = (index // len(names)) % 2 == 1
        raw = {
            name: lower
            + (upper - lower)
            * _uniform(spec.seed, f"{_GUARD_DOMAIN}/interior/{name}", index)
            for name, (lower, upper, _bound) in primitive.items()
        }
        lower, upper, _bound = primitive[target]
        span = upper - lower
        offset = spec.guard_band_fraction * (span if span > 0.0 else 1.0)
        raw[target] = upper + offset if upper_side else lower - offset

        theta: dict[str, object] = {}
        for name, value in raw.items():
            bound = primitive[name][2]
            dimension = bound.lower_bound.dimension
            if not isinstance(bound.lower_bound.unit, Unit) or not isinstance(
                dimension, Dimension
            ):
                raise ValueError("membership bounds must expose canonical units")
            theta[name] = CanonicalQuantity(value, bound.lower_bound.unit, dimension)
        decision = predicate.evaluate(theta)
        if decision.accepted:
            raise ValueError(
                "guard-band OOD point was accepted by membership predicate"
            )
        rejection = RejectedProposal(
            case_id=f"guard-{index:06d}",
            stream_domain=_GUARD_DOMAIN,
            coordinates=raw,
            reason_codes=_stable_reason_codes(decision),
        )
        derived = {
            name: _finite(value.value, f"guard derived coordinate {name}")
            for name, value in decision.derived_values.items()
        }
        guards = (
            *guards,
            DesignPoint(
                case_id=f"guard-{index:06d}",
                leakage_group_id=f"guard-{index:06d}",
                coordinates=raw,
                derived_coordinates=derived,
                strata={axis.spec.name: axis.spec.strata for axis in compiled},
                guard_band=True,
            ),
        )
        rejections = (*rejections, rejection)
    return guards, rejections


def _stable_reason_codes(decision: _DecisionLike) -> tuple[str, ...]:
    raw_reasons = getattr(decision, "reason_codes", ())
    if not isinstance(raw_reasons, Sequence) or isinstance(
        raw_reasons, (str, bytes, bytearray)
    ):
        raise ValueError("membership rejection reasons must be a sequence")
    stable = tuple(_stable_reason_value(reason) for reason in raw_reasons)
    return stable or ("UNSPECIFIED_MEMBERSHIP_REJECTION",)


def _stable_reason_value(reason: object) -> str:
    value = getattr(reason, "value", reason)
    if not isinstance(value, str) or not value:
        raise ValueError("membership rejection reasons must have stable text values")
    return value


def _stratum(normalized: float, strata: int) -> int:
    return min(strata - 1, max(0, int(normalized * strata)))


def _seed_key(seed: DoeSeed | bytes | str) -> bytes:
    material = DoeSeed.from_input(seed).key_material()
    return hashlib.blake2b(material, digest_size=32, person=_SEED_PERSON).digest()


def _digest(seed: DoeSeed | bytes | str, domain: str, index: int) -> bytes:
    payload = domain.encode("utf-8") + b"\0" + index.to_bytes(8, "big")
    return hashlib.blake2b(
        payload,
        key=_seed_key(seed),
        digest_size=16,
        person=_PERSON,
    ).digest()


def _uniform(seed: DoeSeed | bytes | str, domain: str, index: int) -> float:
    integer = int.from_bytes(_digest(seed, domain, index)[:8], "big")
    return (integer + 0.5) / 2**64


def _permutation(
    seed: DoeSeed | bytes | str, domain: str, count: int
) -> tuple[int, ...]:
    return tuple(
        sorted(range(count), key=lambda index: (_digest(seed, domain, index), index))
    )


def _stream_id(seed: DoeSeed | bytes | str, domain: str) -> str:
    return hashlib.blake2b(
        domain.encode("utf-8"),
        key=_seed_key(seed),
        digest_size=16,
        person=_PERSON,
    ).hexdigest()


def _coordinate_id(coordinates: Mapping[str, float]) -> str:
    payload = "\0".join(
        f"{name}={value.hex()}" for name, value in sorted(coordinates.items())
    ).encode("ascii")
    return hashlib.blake2b(payload, digest_size=12, person=_PERSON).hexdigest()


def _proposal_seed(seed: DoeSeed | bytes | str, domain: str, index: int) -> int:
    """Return one canonical-JSON-safe seed identity for a DOE proposal."""

    return int.from_bytes(
        _digest(seed, f"{domain}/morph-seed", index)[:6],
        "big",
    )
