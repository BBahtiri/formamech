from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import rfc8785

from campaigns.dogbone_v1.s11_closeout import (
    s11_gate_payload,
    write_s11_gate_artifact,
)
from campaigns.dogbone_v1.s11_evidence import run_dogbone_s11_evidence
from piginot.operators import (
    GateStatus,
    S11GateMeasurements,
    evaluate_s11_gate,
)

_REVISION = "1" * 40


def _payload() -> dict[str, object]:
    return s11_gate_payload(
        run_dogbone_s11_evidence(code_coverage_percent=89.6),
        implementation_source_revision=_REVISION,
        branch="codex/s11-operator-architecture",
        timestamp_utc="2026-09-07T00:00:00Z",
        focused_test_result="86 passed; total coverage 89.60%",
    )


def _measurements(payload: dict[str, object]) -> S11GateMeasurements:
    measurements = payload["measurements"]
    assert isinstance(measurements, dict)
    return S11GateMeasurements(**measurements)


def test_s11_gate_payload_reduces_all_nine_checks_to_pass() -> None:
    payload = _payload()
    checks = payload["checks"]
    assert isinstance(checks, list)

    evaluation = evaluate_s11_gate(_measurements(payload))

    assert payload["schema_version"] == "1.0"
    assert payload["step"] == "S11"
    assert payload["gate_id"] == "G11"
    assert payload["implementation_source_revision"] == _REVISION
    assert payload["verdict"] == "PASS"
    assert payload["verdict_reason"] == "all nine required G11 checks passed"
    assert evaluation.verdict is GateStatus.PASS
    assert tuple(check["id"] for check in checks) == tuple(
        check.check_id for check in evaluation.checks
    )
    for recorded, reduced in zip(checks, evaluation.checks, strict=True):
        assert recorded["status"] == reduced.status.value
        assert recorded["measured"] == reduced.measured
        assert recorded["threshold"] == reduced.threshold
        assert recorded["comparison"] == reduced.comparison
        assert recorded["reason"] == reduced.reason


def test_s11_gate_payload_preserves_anti_scope_and_contingency_boundaries() -> None:
    payload = _payload()
    scope = payload["scope"]
    architecture = payload["architecture"]
    upstream = payload["upstream_reference"]
    assert isinstance(scope, dict)
    assert isinstance(architecture, dict)
    assert isinstance(upstream, dict)

    assert scope["training_run"] is False
    assert scope["production_checkpoint_created"] is False
    assert scope["post_hoc_field_scaling"] is False
    assert scope["http_or_ui_surface"] is False
    assert scope["served_stress_source"] == (
        "P_served = P_predicted from the independent P head"
    )
    assert "s12_unblocked" not in scope
    assert scope["next_step"] == (
        "run the ADR 0001 S11-S12 architecture iteration before S12"
    )
    assert "ADR 0001" in str(scope["s12_contingency_requirement"])
    assert architecture["output_heads"] == {
        "u": {"shape": "(batch, query_points, 2)"},
        "P": {"shape": "(batch, query_points, 2, 2)"},
    }
    assert upstream["repository"] == "https://github.com/BBahtiri/PI-GINOT-DogBone"
    assert upstream["revision"] == "9f6794141beca9d3e66433465c3b07039a7e41d2"


def test_s11_gate_writer_is_canonical_and_refuses_divergent_rewrites(
    tmp_path: Path,
) -> None:
    payload = _payload()
    target = tmp_path / "S11.json"

    write_s11_gate_artifact(payload, target)
    write_s11_gate_artifact(payload, target)

    assert target.read_bytes() == rfc8785.dumps(cast(Any, payload))
    changed = dict(payload)
    changed["verdict"] = "FAIL"
    with pytest.raises(ValueError, match="refusing to replace"):
        write_s11_gate_artifact(changed, target)
