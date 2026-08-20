"""Dependency-free SVG summaries for the frozen S1 confirmation campaign."""

from __future__ import annotations

import hashlib
import html
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Final

_FILENAMES: Final = (
    "error_ecdf.svg",
    "peak_identity.svg",
    "parameter_error.svg",
    "confidence_intervals.svg",
    "peak_refinement.svg",
    "inference_distribution.svg",
)
_COLORS: Final = ("#1565c0", "#d84315", "#2e7d32", "#6a1b9a", "#00838f")
_LEFT: Final = 92.0
_RIGHT: Final = 920.0
_TOP: Final = 62.0
_BOTTOM: Final = 510.0


@dataclass(frozen=True)
class _Record:
    geometry_id: str
    family: str
    p_error: float
    u_error: float
    fem_peak: float
    predicted_peak: float
    inference_seconds: float
    parameters: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _Estimate:
    family: str
    metric: str
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class _Level:
    nx: float
    ny: float
    peak: float | None
    converged: bool


@dataclass(frozen=True)
class _Refinement:
    geometry_id: str
    family: str
    family_index: int
    relative_change: float | None
    levels: tuple[_Level, ...]


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _validate_finite_tree(value: object, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{context} must contain only finite numbers")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(item, f"{context}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{context}[{index}]")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a mapping with string keys")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{context} must be a sequence")
    return value


def _number(
    value: object,
    context: str,
    *,
    nonnegative: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{context} must be non-negative")
    return result


def _text_field(mapping: Mapping[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _normalise_records(records: Sequence[Mapping[str, object]]) -> tuple[_Record, ...]:
    if not records:
        raise ValueError("records must not be empty")
    output: list[_Record] = []
    for index, value in enumerate(records):
        record = _mapping(value, f"records[{index}]")
        parameters = _mapping(record.get("parameters"), f"records[{index}].parameters")
        if not parameters:
            raise ValueError(f"records[{index}].parameters must not be empty")
        parsed_parameters = tuple(
            (name, _number(number, f"records[{index}].parameters.{name}"))
            for name, number in sorted(parameters.items())
            if name
        )
        if len(parsed_parameters) != len(parameters):
            raise ValueError("parameter names must be non-empty strings")
        output.append(
            _Record(
                geometry_id=_text_field(record, "geometry_id", f"records[{index}]"),
                family=_text_field(record, "family", f"records[{index}]"),
                p_error=_number(record.get("relative_l2_P"), "relative_l2_P"),
                u_error=_number(record.get("relative_l2_u"), "relative_l2_u"),
                fem_peak=_number(
                    record.get("fem_peak_stress_MPa"), "fem_peak_stress_MPa"
                ),
                predicted_peak=_number(
                    record.get("predicted_peak_stress_MPa"),
                    "predicted_peak_stress_MPa",
                ),
                inference_seconds=_number(
                    record.get("inference_seconds"), "inference_seconds"
                ),
                parameters=parsed_parameters,
            )
        )
    geometry_ids = [record.geometry_id for record in output]
    if len(set(geometry_ids)) != len(geometry_ids):
        raise ValueError("record geometry_id values must be unique")
    return tuple(sorted(output, key=lambda record: (record.family, record.geometry_id)))


def _normalise_statistics(
    statistics: Mapping[str, object], families: Sequence[str]
) -> tuple[_Estimate, ...]:
    output: list[_Estimate] = []
    for family in families:
        family_stats = _mapping(statistics.get(family), f"statistics.{family}")
        for metric in ("relative_l2_P", "relative_l2_u"):
            values = _mapping(family_stats.get(metric), f"statistics.{family}.{metric}")
            estimate = _number(values.get("estimate"), f"{family}.{metric}.estimate")
            interval = _sequence(
                values.get("bootstrap_95_ci"), f"{family}.{metric}.bootstrap_95_ci"
            )
            if len(interval) != 2:
                raise ValueError("bootstrap_95_ci must contain two values")
            lower = _number(interval[0], f"{family}.{metric}.ci.lower")
            upper = _number(interval[1], f"{family}.{metric}.ci.upper")
            if not lower <= estimate <= upper:
                raise ValueError("bootstrap confidence interval must contain estimate")
            output.append(_Estimate(family, metric, estimate, lower, upper))
    return tuple(output)


def _normalise_regularity(
    regularity: Mapping[str, object], families: Sequence[str]
) -> tuple[_Refinement, ...]:
    if not isinstance(regularity.get("well_posed"), bool):
        raise ValueError("regularity.well_posed must be boolean")
    samples = _sequence(regularity.get("samples"), "regularity.samples")
    output: list[_Refinement] = []
    for index, value in enumerate(samples):
        sample = _mapping(value, f"regularity.samples[{index}]")
        family = _text_field(sample, "family", f"regularity.samples[{index}]")
        if family not in families:
            raise ValueError("regularity sample family must occur in records")
        family_index = sample.get("family_index")
        if isinstance(family_index, bool) or not isinstance(family_index, int):
            raise ValueError("regularity family_index must be an integer")
        change_value = sample.get("relative_change_32_to_64")
        change = (
            None
            if change_value is None
            else _number(change_value, "relative_change_32_to_64")
        )
        levels: list[_Level] = []
        for level_index, level_value in enumerate(
            _sequence(sample.get("levels"), "regularity sample levels")
        ):
            level = _mapping(level_value, f"sample level {level_index}")
            converged = level.get("converged")
            if not isinstance(converged, bool):
                raise ValueError("regularity level converged must be boolean")
            peak_value = level.get("peak_stress_MPa")
            peak = (
                None
                if peak_value is None and not converged
                else _number(peak_value, "regularity peak_stress_MPa")
            )
            levels.append(
                _Level(
                    _number(level.get("nx"), "regularity nx"),
                    _number(level.get("ny"), "regularity ny"),
                    peak,
                    converged,
                )
            )
        if not levels:
            raise ValueError("each regularity sample must contain levels")
        output.append(
            _Refinement(
                _text_field(sample, "geometry_id", "regularity sample"),
                family,
                family_index,
                change,
                tuple(sorted(levels, key=lambda level: (level.nx, level.ny))),
            )
        )
    if not output:
        raise ValueError("regularity.samples must not be empty")
    return tuple(sorted(output, key=lambda item: (item.family, item.family_index)))


def _bounds(
    values: Sequence[float], *, include_zero: bool = False
) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if include_zero:
        lower = min(0.0, lower)
    if lower == upper:
        padding = max(abs(lower) * 0.08, 1.0)
    else:
        padding = (upper - lower) * 0.08
    return lower - padding, upper + padding


def _xy(
    x: float, y: float, x_bounds: tuple[float, float], y_bounds: tuple[float, float]
) -> tuple[float, float]:
    px = _LEFT + (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * (_RIGHT - _LEFT)
    py = _BOTTOM - (y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) * (_BOTTOM - _TOP)
    return px, py


def _axes(
    title: str,
    x_label: str,
    y_label: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> list[str]:
    output = [f'<text class="title" x="480" y="31">{_escape(title)}</text>']
    for index in range(6):
        fraction = index / 5
        x = _LEFT + fraction * (_RIGHT - _LEFT)
        y = _BOTTOM - fraction * (_BOTTOM - _TOP)
        x_value = x_bounds[0] + fraction * (x_bounds[1] - x_bounds[0])
        y_value = y_bounds[0] + fraction * (y_bounds[1] - y_bounds[0])
        output.extend(
            (
                f'<line class="grid" x1="{x:.2f}" y1="{_TOP}" '
                f'x2="{x:.2f}" y2="{_BOTTOM}"/>',
                f'<line class="grid" x1="{_LEFT}" y1="{y:.2f}" '
                f'x2="{_RIGHT}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{x:.2f}" y="532">{x_value:.3g}</text>',
                f'<text class="ytick" x="84" y="{y + 4:.2f}">{y_value:.3g}</text>',
            )
        )
    output.extend(
        (
            f'<line class="axis" x1="{_LEFT}" y1="{_BOTTOM}" '
            f'x2="{_RIGHT}" y2="{_BOTTOM}"/>',
            f'<line class="axis" x1="{_LEFT}" y1="{_TOP}" '
            f'x2="{_LEFT}" y2="{_BOTTOM}"/>',
            f'<text class="label" x="506" y="570">{_escape(x_label)}</text>',
            f'<text class="label vertical" x="22" y="286">{_escape(y_label)}</text>',
        )
    )
    return output


def _document(title: str, body: Sequence[str]) -> str:
    return "\n".join(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="590" '
            f'viewBox="0 0 960 590" role="img" aria-label="{_escape(title)}">',
            f"<title>{_escape(title)}</title>",
            "<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700;text-anchor:middle}.label{font-size:13px;text-anchor:middle}.vertical{transform:rotate(-90deg);transform-origin:22px 286px}.tick{font-size:11px;text-anchor:middle}.ytick{font-size:11px;text-anchor:end}.grid{stroke:#e2e6ea;stroke-width:1}.axis{stroke:#5f6368;stroke-width:1.5}.legend{font-size:11px}.series{fill:none;stroke-width:2}.point{stroke:#fff;stroke-width:1;opacity:.88}</style>",
            *body,
            "</svg>",
            "",
        )
    )


def _point(
    x: float,
    y: float,
    color: str,
    record: _Record,
    metric: str,
    detail: str,
) -> str:
    return (
        f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="4.2" '
        f'fill="{color}" data-case="{_escape(record.geometry_id)}" '
        f'data-family="{_escape(record.family)}" data-metric="{metric}">'
        f"<title>{_escape(record.geometry_id)} | {_escape(detail)}</title></circle>"
    )


def _error_ecdf(records: Sequence[_Record]) -> str:
    values = [value for record in records for value in (record.p_error, record.u_error)]
    x_bounds = (0.0, max(max(values) * 1.08, 0.16))
    y_bounds = (0.0, 1.04)
    body = _axes("Per-case error ECDF", "relative L2 error", "ECDF", x_bounds, y_bounds)
    for threshold, label in ((0.05, "5% target"), (0.15, "15% S1 gate")):
        x, _ = _xy(threshold, 0.0, x_bounds, y_bounds)
        body.append(
            f'<line x1="{x:.2f}" y1="{_TOP}" x2="{x:.2f}" y2="{_BOTTOM}" stroke="#616161" stroke-dasharray="5 4"/>'
        )
        body.append(f'<text class="legend" x="{x + 4:.2f}" y="76">{label}</text>')
    for metric, color, getter in (
        ("P", _COLORS[0], lambda record: record.p_error),
        ("u", _COLORS[1], lambda record: record.u_error),
    ):
        ranked = sorted(
            records, key=lambda record: (getter(record), record.geometry_id)
        )
        points: list[str] = []
        for rank, record in enumerate(ranked, start=1):
            value = getter(record)
            x, y = _xy(value, rank / len(ranked), x_bounds, y_bounds)
            points.append(f"{x:.2f},{y:.2f}")
            body.append(_point(x, y, color, record, metric, f"{metric}={value:.4f}"))
        body.append(
            f'<polyline class="series" stroke="{color}" points="{" ".join(points)}"/>'
        )
    return _document("Per-case error ECDF", body)


def _peak_identity(records: Sequence[_Record]) -> str:
    values = [
        value
        for record in records
        for value in (record.fem_peak, record.predicted_peak)
    ]
    bounds = _bounds(values, include_zero=True)
    body = _axes(
        "Peak-stress identity",
        "FEM peak stress (MPa)",
        "predicted peak stress (MPa)",
        bounds,
        bounds,
    )
    start = _xy(bounds[0], bounds[0], bounds, bounds)
    end = _xy(bounds[1], bounds[1], bounds, bounds)
    body.append(
        f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" stroke="#424242" stroke-dasharray="6 4"/><text class="legend" x="820" y="84">y = x</text>'
    )
    families = sorted({record.family for record in records})
    for record in records:
        x, y = _xy(record.fem_peak, record.predicted_peak, bounds, bounds)
        color = _COLORS[families.index(record.family) % len(_COLORS)]
        body.append(
            _point(
                x,
                y,
                color,
                record,
                "peak",
                f"FEM={record.fem_peak:.4g} MPa; predicted={record.predicted_peak:.4g} MPa",
            )
        )
    return _document("Peak-stress identity", body)


def _parameter_error(records: Sequence[_Record]) -> str:
    groups: dict[tuple[str, str], list[float]] = {}
    for record in records:
        for name, value in record.parameters:
            groups.setdefault((record.family, name), []).append(value)
    series = sorted(groups)
    y_bounds = (0.0, max(record.p_error for record in records) * 1.08 or 1.0)
    body = _axes(
        "Stress error versus geometry",
        "normalized geometry parameter",
        "relative L2 P error",
        (-0.04, 1.04),
        y_bounds,
    )
    for record in records:
        for name, value in record.parameters:
            values = groups[(record.family, name)]
            normalized = (
                0.5
                if min(values) == max(values)
                else (value - min(values)) / (max(values) - min(values))
            )
            x, y = _xy(normalized, record.p_error, (-0.04, 1.04), y_bounds)
            color = _COLORS[series.index((record.family, name)) % len(_COLORS)]
            body.append(
                _point(
                    x,
                    y,
                    color,
                    record,
                    _escape(name),
                    f"{name}={value:.4g}; normalized={normalized:.3f}; P={record.p_error:.4f}",
                )
            )
    return _document("Stress error versus normalized geometry parameters", body)


def _confidence_intervals(estimates: Sequence[_Estimate]) -> str:
    maximum = max(item.upper for item in estimates)
    x_bounds = (0.0, max(maximum * 1.1, 0.16))
    y_bounds = (-0.75, len(estimates) - 0.25)
    body = _axes(
        "Bootstrap confidence intervals",
        "relative L2 error",
        "family / metric",
        x_bounds,
        y_bounds,
    )
    for index, item in enumerate(estimates):
        y_value = len(estimates) - index - 1
        lower, y = _xy(item.lower, y_value, x_bounds, y_bounds)
        upper, _ = _xy(item.upper, y_value, x_bounds, y_bounds)
        estimate, _ = _xy(item.estimate, y_value, x_bounds, y_bounds)
        color = _COLORS[index % len(_COLORS)]
        label = f"{item.family} {'P' if item.metric.endswith('_P') else 'u'}"
        body.extend(
            (
                f'<line x1="{lower:.2f}" y1="{y:.2f}" x2="{upper:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="5"><title>{_escape(label)} 95% CI [{item.lower:.4f}, {item.upper:.4f}]</title></line>',
                f'<circle cx="{estimate:.2f}" cy="{y:.2f}" r="6" fill="{color}"/>',
                f'<text class="legend" x="{_LEFT + 5}" y="{y - 9:.2f}">{_escape(label)}</text>',
            )
        )
    return _document("Bootstrap confidence intervals", body)


def _peak_refinement(refinements: Sequence[_Refinement]) -> str:
    peaks = [
        level.peak
        for item in refinements
        for level in item.levels
        if level.peak is not None
    ]
    y_bounds = _bounds(peaks or [0.0, 1.0], include_zero=True)
    x_values = sorted({level.nx for item in refinements for level in item.levels})
    x_bounds = _bounds(x_values)
    body = _axes(
        "FEM peak-stress refinement",
        "mesh nx",
        "FEM peak stress (MPa)",
        x_bounds,
        y_bounds,
    )
    for index, item in enumerate(refinements):
        color = _COLORS[index % len(_COLORS)]
        points: list[str] = []
        for level in item.levels:
            if level.peak is None:
                continue
            x, y = _xy(level.nx, level.peak, x_bounds, y_bounds)
            points.append(f"{x:.2f},{y:.2f}")
            change = (
                "n/a" if item.relative_change is None else f"{item.relative_change:.3%}"
            )
            body.append(
                f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" data-sample="{_escape(item.geometry_id)}"><title>{_escape(item.geometry_id)} | {level.nx:g}x{level.ny:g} | {level.peak:.4g} MPa | 32-to-64 change {change}</title></circle>'
            )
        if points:
            body.append(
                f'<polyline class="series" stroke="{color}" points="{" ".join(points)}"/>'
            )
    return _document("FEM peak-stress refinement", body)


def _inference_distribution(records: Sequence[_Record]) -> str:
    families = sorted({record.family for record in records})
    times = [record.inference_seconds * 1000.0 for record in records]
    x_bounds = _bounds(times, include_zero=True)
    y_bounds = (-0.5, len(families) - 0.5)
    body = _axes(
        "Inference-time distribution",
        "inference time (ms)",
        "family",
        x_bounds,
        y_bounds,
    )
    for record in records:
        family_index = families.index(record.family)
        jitter = ((sum(record.geometry_id.encode("utf-8")) % 11) - 5) / 45.0
        x, y = _xy(
            record.inference_seconds * 1000.0, family_index + jitter, x_bounds, y_bounds
        )
        body.append(
            _point(
                x,
                y,
                _COLORS[family_index % len(_COLORS)],
                record,
                "inference",
                f"{record.inference_seconds * 1000.0:.4g} ms",
            )
        )
    return _document("Inference-time distribution", body)


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_summary_plots(
    records: Sequence[Mapping[str, object]],
    statistics: Mapping[str, object],
    regularity: Mapping[str, object],
    output_dir: Path,
) -> Mapping[str, object]:
    """Validate confirmation inputs and write the six deterministic SVG summaries."""
    _validate_finite_tree(records, "records")
    _validate_finite_tree(statistics, "statistics")
    _validate_finite_tree(regularity, "regularity")
    parsed_records = _normalise_records(records)
    families = tuple(sorted({record.family for record in parsed_records}))
    estimates = _normalise_statistics(statistics, families)
    refinements = _normalise_regularity(regularity, families)
    documents = {
        "error_ecdf.svg": _error_ecdf(parsed_records),
        "peak_identity.svg": _peak_identity(parsed_records),
        "parameter_error.svg": _parameter_error(parsed_records),
        "confidence_intervals.svg": _confidence_intervals(estimates),
        "peak_refinement.svg": _peak_refinement(refinements),
        "inference_distribution.svg": _inference_distribution(parsed_records),
    }
    if tuple(documents) != _FILENAMES:
        raise AssertionError("summary plot manifest changed unexpectedly")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, content in documents.items():
        _atomic_write(destination / filename, content)
    return {
        "directory": str(destination.resolve()),
        "count": len(documents),
        "sha256": {
            filename: hashlib.sha256((destination / filename).read_bytes()).hexdigest()
            for filename in sorted(documents)
        },
    }
