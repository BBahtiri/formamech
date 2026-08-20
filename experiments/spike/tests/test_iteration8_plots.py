from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from experiments.spike.iteration8_plots import write_summary_plots


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "geometry_id": f'{family}-case-{index}<&"'
            if index == 0
            else f"{family}-{index}",
            "family": family,
            "relative_l2_P": 0.06 + 0.01 * index,
            "relative_l2_u": 0.02 + 0.005 * index,
            "fem_peak_stress_MPa": 20.0 + 5.0 * index + family_index,
            "predicted_peak_stress_MPa": 19.0 + 5.5 * index + family_index,
            "inference_seconds": 0.004 + 0.001 * index,
            "parameters": {
                "length_mm": 80.0 + 10.0 * index,
                'width<&"': 20.0 + family_index + index,
            },
        }
        for family_index, family in enumerate(("dogbone", "centre_notched"))
        for index in range(3)
    )


def _statistics() -> dict[str, object]:
    return {
        family: {
            "relative_l2_P": {
                "estimate": 0.09 + 0.01 * family_index,
                "bootstrap_95_ci": [0.075, 0.12 + 0.01 * family_index],
            },
            "relative_l2_u": {
                "estimate": 0.03 + 0.005 * family_index,
                "bootstrap_95_ci": [0.02, 0.05 + 0.005 * family_index],
            },
        }
        for family_index, family in enumerate(("dogbone", "centre_notched"))
    }


def _regularity() -> dict[str, object]:
    return {
        "well_posed": True,
        "samples": [
            {
                "geometry_id": f"{family}-refine-{case_index}<&",
                "family": family,
                "family_index": case_index,
                "relative_change_32_to_64": 0.03 + 0.01 * case_index,
                "levels": [
                    {
                        "nx": nx,
                        "ny": ny,
                        "peak_stress_MPa": (
                            20.0 + family_index + case_index + level_index
                        ),
                        "converged": True,
                    }
                    for level_index, (nx, ny) in enumerate(
                        ((16, 5), (32, 10), (64, 20))
                    )
                ],
            }
            for family_index, family in enumerate(("dogbone", "centre_notched"))
            for case_index in range(2)
        ],
    }


def test_write_summary_plots_emits_exact_deterministic_svg_set(tmp_path: Path) -> None:
    output = tmp_path / "summary"

    metadata = write_summary_plots(_records(), _statistics(), _regularity(), output)

    expected = {
        "error_ecdf.svg",
        "peak_identity.svg",
        "parameter_error.svg",
        "confidence_intervals.svg",
        "peak_refinement.svg",
        "inference_distribution.svg",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert metadata == {
        "directory": str(output.resolve()),
        "count": 6,
        "sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in sorted(expected)
        },
    }
    for name in expected:
        root = ET.parse(output / name).getroot()
        assert root.tag.endswith("svg")
        assert root.attrib["role"] == "img"

    repeated = tmp_path / "repeated"
    repeated_metadata = write_summary_plots(
        _records(), _statistics(), _regularity(), repeated
    )
    assert repeated_metadata["sha256"] == metadata["sha256"]


def test_summary_plots_show_every_case_and_escape_xml(tmp_path: Path) -> None:
    records = _records()
    output = tmp_path / "summary"

    write_summary_plots(records, _statistics(), _regularity(), output)

    for name in (
        "error_ecdf.svg",
        "peak_identity.svg",
        "parameter_error.svg",
        "inference_distribution.svg",
    ):
        svg = (output / name).read_text(encoding="utf-8")
        visible_text = "".join(ET.fromstring(svg).itertext())
        for record in records:
            assert str(record["geometry_id"]) in visible_text
        assert '<&"' not in svg

    ecdf = (output / "error_ecdf.svg").read_text(encoding="utf-8")
    assert "5% target" in ecdf
    assert "15% S1 gate" in ecdf
    assert ecdf.count('data-metric="P"') == len(records)
    assert ecdf.count('data-metric="u"') == len(records)

    identity = (output / "peak_identity.svg").read_text(encoding="utf-8")
    assert "y = x" in identity
    assert identity.count('data-case="') == len(records)

    parameters = (output / "parameter_error.svg").read_text(encoding="utf-8")
    assert "normalized geometry parameter" in parameters
    assert parameters.count('data-case="') == 2 * len(records)

    refinement = (output / "peak_refinement.svg").read_text(encoding="utf-8")
    assert refinement.count('data-sample="') == 4 * 3


@pytest.mark.parametrize(
    ("location", "bad_value"),
    (
        ("record", float("nan")),
        ("statistics", float("inf")),
        ("regularity", -float("inf")),
    ),
)
def test_summary_plots_reject_nonfinite_inputs_before_writing(
    tmp_path: Path,
    location: str,
    bad_value: float,
) -> None:
    records = list(_records())
    statistics = _statistics()
    regularity = _regularity()
    if location == "record":
        records[0] = {
            **records[0],
            "parameters": {"invalid": bad_value},
        }
    elif location == "statistics":
        statistics["dogbone"]["relative_l2_P"]["estimate"] = bad_value
    else:
        regularity["samples"][0]["levels"][0]["peak_stress_MPa"] = bad_value
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="finite"):
        write_summary_plots(records, statistics, regularity, output)

    assert not output.exists()
    assert not math.isfinite(bad_value)
