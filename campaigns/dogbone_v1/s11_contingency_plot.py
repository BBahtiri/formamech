"""Render the sealed S11 contingency TRAIN-screen result as deterministic SVG."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from piginot.contracts import strict_json_loads

_EVIDENCE_FILE_BLAKE2B256 = (
    "260eca6c345ee1edea17b9098859baaa6de4a35e27de65c6ccdead0cad831f36"
)
_CANDIDATES = (
    "A-s11-baseline",
    "B-derived-ratio-film",
    "C-local-stress-enrichment",
    "D-derived-local-stress",
)
_SHORT_LABELS = ("A: baseline", "B: derived ratios", "C: local stress", "D: combined")
_COLORS = ("#64748b", "#f59e0b", "#0ea5e9", "#8b5cf6")
_METRIC_FIELDS = (
    ("median_absolute_peak_error", "Median peak"),
    ("p90_absolute_peak_error", "P90 peak"),
    ("median_p_relative_l2", "Median P field"),
    ("median_u_relative_l2", "Median u field"),
)


@dataclass(frozen=True, slots=True)
class ScreenMetric:
    """The four displayed error statistics for one registered candidate."""

    candidate_id: str
    median_signed_peak_error: float
    median_absolute_peak_error: float
    p90_absolute_peak_error: float
    median_p_relative_l2: float
    median_u_relative_l2: float


def load_screen_metrics(evidence_path: Path) -> tuple[ScreenMetric, ...]:
    """Load only the exact tracked contingency evidence used by this plot."""

    if not isinstance(evidence_path, Path):
        raise TypeError("evidence_path must be a pathlib.Path")
    raw = evidence_path.read_bytes()
    observed = hashlib.blake2b(raw, digest_size=32).hexdigest()
    if observed != _EVIDENCE_FILE_BLAKE2B256:
        raise ValueError("contingency evidence differs from the sealed result")
    document = _mapping(strict_json_loads(raw), "evidence")
    decision = _mapping(document.get("train_screen_decision"), "TRAIN decision")
    selection = _mapping(decision.get("selection"), "selection")
    if (
        selection.get("fallback_to_baseline") is not True
        or selection.get("selected_candidate_id") != _CANDIDATES[0]
        or document.get("development_confirmation") is not None
    ):
        raise ValueError("plot requires the sealed baseline-fallback result")
    raw_metrics = decision.get("metrics")
    if not isinstance(raw_metrics, list):
        raise ValueError("TRAIN decision metrics must be a JSON array")
    metrics = tuple(_screen_metric(item) for item in raw_metrics)
    if tuple(item.candidate_id for item in metrics) != _CANDIDATES:
        raise ValueError("TRAIN metrics differ from the registered candidate order")
    return metrics


def render_screen_svg(metrics: tuple[ScreenMetric, ...]) -> bytes:
    """Render a deterministic grouped bar chart and the decisive guardrail."""

    if type(metrics) is not tuple or tuple(item.candidate_id for item in metrics) != (
        _CANDIDATES
    ):
        raise ValueError("metrics must follow the exact registered candidate order")
    width, height = 1280, 760
    plot_x, plot_y, plot_w, plot_h = 70.0, 125.0, 790.0, 500.0
    max_percent = 28.0
    baseline = metrics[0]
    local = metrics[2]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">S11 contingency: frozen TRAIN-screen errors</title>',
        '<desc id="description">Four architecture candidates compared on 64 '
        "held-aside TRAIN cases. Lower bars are better. Candidate C improves every "
        "stress metric but exceeds the displacement regression guardrail, so the "
        "formal result falls back to candidate A.</desc>",
        '<rect width="1280" height="760" fill="#ffffff"/>',
        '<text x="70" y="52" font-family="sans-serif" font-size="28" '
        'font-weight="700" fill="#0f172a">S11 contingency: frozen TRAIN-screen '
        "errors</text>",
        '<text x="70" y="82" font-family="sans-serif" font-size="15" '
        'fill="#475569">64 cases; lower is better; all candidates trained for '
        "1,500 Adam steps on the same schedule</text>",
    ]
    for tick in range(0, 26, 5):
        y = plot_y + plot_h - tick / max_percent * plot_h
        lines.extend(
            (
                f'<line x1="{plot_x:.1f}" y1="{y:.1f}" '
                f'x2="{plot_x + plot_w:.1f}" y2="{y:.1f}" '
                'stroke="#e2e8f0" stroke-width="1"/>',
                f'<text x="{plot_x - 12:.1f}" y="{y + 5:.1f}" '
                'text-anchor="end" font-family="sans-serif" font-size="12" '
                f'fill="#64748b">{tick}%</text>',
            )
        )
    group_width = plot_w / len(_METRIC_FIELDS)
    bar_width = 31.0
    bar_gap = 7.0
    bars_width = len(metrics) * bar_width + (len(metrics) - 1) * bar_gap
    for group_index, (field_name, display_name) in enumerate(_METRIC_FIELDS):
        group_left = plot_x + group_index * group_width
        first_x = group_left + (group_width - bars_width) / 2.0
        for candidate_index, item in enumerate(metrics):
            value = 100.0 * cast(float, getattr(item, field_name))
            bar_height = value / max_percent * plot_h
            x = first_x + candidate_index * (bar_width + bar_gap)
            y = plot_y + plot_h - bar_height
            lines.extend(
                (
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                    f'height="{bar_height:.1f}" rx="3" fill="{_COLORS[candidate_index]}"/>',
                    f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" '
                    'text-anchor="middle" font-family="sans-serif" font-size="10" '
                    f'fill="#334155">{value:.1f}</text>',
                )
            )
        center = group_left + group_width / 2.0
        lines.append(
            f'<text x="{center:.1f}" y="{plot_y + plot_h + 28:.1f}" '
            'text-anchor="middle" font-family="sans-serif" font-size="13" '
            f'fill="#334155">{html.escape(display_name)}</text>'
        )
    lines.extend(
        (
            f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" '
            f'y2="{plot_y + plot_h:.1f}" stroke="#334155" stroke-width="1.5"/>',
            f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" '
            f'x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}" '
            'stroke="#334155" stroke-width="1.5"/>',
            '<text x="18" y="390" transform="rotate(-90 18 390)" '
            'font-family="sans-serif" font-size="13" fill="#334155">Relative '
            "error</text>",
        )
    )
    _append_legend(lines)
    _append_outcome(lines, baseline, local)
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_screen_svg(payload: bytes, target: Path) -> None:
    """Create the plot once, allowing only an identical repeated write."""

    if not isinstance(payload, bytes) or not payload.startswith(b"<svg "):
        raise ValueError("payload must be an SVG byte document")
    if not isinstance(target, Path):
        raise TypeError("target must be a pathlib.Path")
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise ValueError(f"refusing to replace a different plot: {target}")
        return
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError("plot parent must be an existing real directory")
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def _append_legend(lines: list[str]) -> None:
    x, y = 900.0, 130.0
    lines.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        'font-size="16" font-weight="700" fill="#0f172a">Candidates</text>'
    )
    for index, (label, color) in enumerate(zip(_SHORT_LABELS, _COLORS, strict=True)):
        row_y = y + 30.0 + index * 27.0
        lines.extend(
            (
                f'<rect x="{x:.1f}" y="{row_y - 13:.1f}" width="16" '
                f'height="16" rx="3" fill="{color}"/>',
                f'<text x="{x + 26:.1f}" y="{row_y:.1f}" '
                'font-family="sans-serif" font-size="14" fill="#334155">'
                f"{html.escape(label)}</text>",
            )
        )


def _append_outcome(
    lines: list[str], baseline: ScreenMetric, local: ScreenMetric
) -> None:
    x, y = 900.0, 300.0
    reductions = (
        (
            "Signed peak bias",
            100.0
            * (
                1.0
                - abs(local.median_signed_peak_error)
                / abs(baseline.median_signed_peak_error)
            ),
            True,
        ),
        (
            "Median peak",
            100.0
            * (
                1.0
                - local.median_absolute_peak_error / baseline.median_absolute_peak_error
            ),
            True,
        ),
        (
            "P90 peak",
            100.0
            * (1.0 - local.p90_absolute_peak_error / baseline.p90_absolute_peak_error),
            True,
        ),
        (
            "Median P field",
            100.0 * (1.0 - local.median_p_relative_l2 / baseline.median_p_relative_l2),
            True,
        ),
        (
            "Median u field",
            100.0 * (local.median_u_relative_l2 / baseline.median_u_relative_l2 - 1.0),
            False,
        ),
    )
    lines.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        'font-size="16" font-weight="700" fill="#0f172a">C versus baseline</text>'
    )
    for index, (label, value, improvement) in enumerate(reductions):
        row_y = y + 29.0 + index * 27.0
        suffix = "reduction" if improvement else "regression"
        color = "#15803d" if improvement else "#b91c1c"
        lines.append(
            f'<text x="{x:.1f}" y="{row_y:.1f}" font-family="sans-serif" '
            f'font-size="14" fill="{color}">{html.escape(label)}: '
            f"{value:.1f}% {suffix}</text>"
        )
    lines.extend(
        (
            f'<rect x="{x:.1f}" y="{y + 180:.1f}" width="320" height="105" '
            'rx="8" fill="#fff7ed" stroke="#fdba74"/>',
            f'<text x="{x + 16:.1f}" y="{y + 210:.1f}" font-family="sans-serif" '
            'font-size="15" font-weight="700" fill="#9a3412">Formal outcome: '
            "fallback to A</text>",
            f'<text x="{x + 16:.1f}" y="{y + 238:.1f}" font-family="sans-serif" '
            'font-size="13" fill="#9a3412">C failed only the +2% displacement '
            "guardrail</text>",
            f'<text x="{x + 16:.1f}" y="{y + 262:.1f}" font-family="sans-serif" '
            'font-size="13" fill="#9a3412">Observed u regression: +3.48%</text>',
        )
    )


def _screen_metric(value: object) -> ScreenMetric:
    item = _mapping(value, "screen metric")
    return ScreenMetric(
        candidate_id=_text(item.get("candidate_id"), "candidate_id"),
        median_signed_peak_error=_signed_number(
            item.get("median_signed_peak_error"), "median_signed_peak_error"
        ),
        median_absolute_peak_error=_number(
            item.get("median_absolute_peak_error"), "median_absolute_peak_error"
        ),
        p90_absolute_peak_error=_number(
            item.get("p90_absolute_peak_error"), "p90_absolute_peak_error"
        ),
        median_p_relative_l2=_number(
            item.get("median_p_relative_l2"), "median_p_relative_l2"
        ),
        median_u_relative_l2=_number(
            item.get("median_u_relative_l2"), "median_u_relative_l2"
        ),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if result < 0.0 or result >= float("inf"):
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _signed_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not -float("inf") < result < float("inf"):
        raise ValueError(f"{label} must be finite")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    payload = render_screen_svg(load_screen_metrics(options.evidence))
    write_screen_svg(payload, options.output)
    print(f"wrote {options.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "ScreenMetric",
    "load_screen_metrics",
    "main",
    "render_screen_svg",
    "write_screen_svg",
)
