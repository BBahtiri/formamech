from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from campaigns.dogbone_v1 import s13c_closeout, s13c_trial
from campaigns.dogbone_v1.s13c_closeout import (
    build_parser,
    cp_freeze_payload,
    g13_gate_payload,
)
from campaigns.dogbone_v1.s13c_protocol import (
    build_dogbone_s13c_protocol,
    build_dogbone_s13c_sampler_plan,
    build_dogbone_s13c_trainer_config,
    build_initial_operator,
    search_space,
)
from campaigns.dogbone_v1.s13c_trial import build_s13c_binding
from piginot.doe import EvidenceRole

_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def screen() -> dict[str, Any]:
    return s13c_closeout._load_json(s13c_closeout.SCREEN_PATH)


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    return s13c_closeout._load_json(s13c_closeout.PROBE_PATH)


def _bound_run_root(path: Path) -> Path:
    protocol = build_dogbone_s13c_protocol()
    binding = build_s13c_binding(
        protocol,
        model=build_initial_operator(),
        config=build_dogbone_s13c_trainer_config(protocol),
        plan=build_dogbone_s13c_sampler_plan(protocol),
        implementation_revision=_REVISION,
    )
    s13c_trial._bind_run_root(path, binding)
    return path


def test_identity_changes_for_data_config_code_and_profile(tmp_path: Path) -> None:
    check = s13c_closeout._identity_check(_bound_run_root(tmp_path))

    assert check["status"] == "PASS"
    measured = check["measured"]
    assert isinstance(measured, dict)
    assert measured["distinct_run_hashes"] == 5


def test_search_space_authorizes_only_the_registered_trial(
    screen: dict[str, Any],
) -> None:
    result = s13c_closeout._screen_result(screen)
    check = s13c_closeout._search_space_check(result.trial_id)

    assert check["status"] == "PASS"
    assert check["measured"] == {
        "registered_trials": 1,
        "maximum_trials": 1,
        "registered_trial_authorized": True,
        "registered_trial_matches_screen": True,
        "out_of_space_trials_refused": 3,
        "out_of_space_trials_tested": 3,
    }
    assert s13c_closeout._search_space_check("trial-other")["status"] == "FAIL"


def test_selection_matches_the_screen_and_detects_the_nonranked_mutant(
    screen: dict[str, Any],
) -> None:
    result = s13c_closeout._screen_result(screen)
    check = s13c_closeout._selection_check(screen, result)

    assert check["status"] == "PASS"
    measured = check["measured"]
    assert isinstance(measured, dict)
    assert measured["nonranked_mutant_detected"] is True
    assert measured["selected_trial_id"] == result.trial_id
    tampered = {**screen, "failed_guardrails": ["p_relative_l2"]}
    assert s13c_closeout._selection_check(tampered, result)["status"] == "FAIL"


class _Source:
    def __init__(self, *, refuse: bool) -> None:
        self.refuse = refuse
        self.development_opened = False

    def iter_train_geometry(self, case_ids: tuple[str, ...]) -> Iterator[object]:
        if self.refuse:
            raise PermissionError(f"{case_ids[0]} is not a registered TRAIN case")
        yield object()


def test_holdout_is_unreachable_and_the_holdout_mutant_fires(
    screen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = s13c_closeout._screen_result(screen)
    monkeypatch.setattr(
        s13c_closeout,
        "open_selective_s10_source",
        lambda root, development_access_record: _Source(refuse=True),
    )
    check = s13c_closeout._holdout_check(Path("unused"), screen, result)
    assert check["status"] == "PASS"
    measured = check["measured"]
    assert isinstance(measured, dict)
    assert measured["production_selection_refusals"] == 2
    assert measured["holdout_public_case_count"] == 0

    monkeypatch.setattr(
        s13c_closeout,
        "open_selective_s10_source",
        lambda root, development_access_record: _Source(refuse=False),
    )
    assert s13c_closeout._holdout_check(Path("unused"), screen, result)["status"] == (
        "FAIL"
    )
    assert result.role is EvidenceRole.TRAIN


def test_determinism_and_resume_bind_the_probe_to_the_run_root(
    probe: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = SimpleNamespace(
        model_export_hash=probe["registered_weights_hash"],
        model_state_hash=probe["registered_model_state_hash"],
    )
    monkeypatch.setattr(s13c_closeout, "_manifest", lambda root, step: registered)
    determinism, resume = s13c_closeout._determinism_and_resume(
        Path("unused"), probe, probe["run_hash"]
    )
    assert (determinism["status"], resume["status"]) == ("PASS", "PASS")

    other = SimpleNamespace(model_export_hash="0" * 64, model_state_hash="0" * 64)
    monkeypatch.setattr(s13c_closeout, "_manifest", lambda root, step: other)
    determinism, resume = s13c_closeout._determinism_and_resume(
        Path("unused"), probe, probe["run_hash"]
    )
    assert (determinism["status"], resume["status"]) == ("FAIL", "FAIL")


def test_lineage_resolves_every_identity(
    screen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = s13c_closeout._screen_result(screen)
    lineage = result.lineage
    binding = SimpleNamespace(
        run_hash=lineage.run_hash,
        dataset_snapshot_hash=lineage.dataset_snapshot_hash,
        execution_profile_hash=lineage.execution_profile_hash,
        implementation_revision=screen["training_revision"],
    )
    manifest = SimpleNamespace(
        run_hash=lineage.run_hash,
        manifest_hash=lineage.checkpoint_hash,
        dataset_snapshot_hash=lineage.dataset_snapshot_hash,
        execution_profile_hash=lineage.execution_profile_hash,
        model_export_hash=lineage.weights_hash,
        model_state_hash=lineage.model_state_hash,
    )
    monkeypatch.setattr(
        s13c_closeout,
        "load_registered_checkpoint",
        lambda protocol, path: (None, binding, manifest),
    )
    monkeypatch.setattr(s13c_closeout, "_is_ancestor", lambda revision: True)
    assert s13c_closeout._lineage_check(Path("unused"), screen, result)["status"] == (
        "PASS"
    )
    monkeypatch.setattr(s13c_closeout, "_is_ancestor", lambda revision: False)
    assert s13c_closeout._lineage_check(Path("unused"), screen, result)["status"] == (
        "FAIL"
    )


def test_head_is_its_own_ancestor() -> None:
    assert s13c_closeout._is_ancestor("HEAD") is True


def _passing_checks() -> list[dict[str, object]]:
    return [
        s13c_closeout._check(check_id, True, True, True)
        for check_id in (
            "G13.determinism",
            "G13.resume",
            "G13.search-space",
            "G13.holdout",
            "G13.identity",
            "G13.lineage",
            "G13.selection",
            "G13.coverage",
        )
    ]


def test_g13_payload_requires_the_complete_ordered_check_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g11 = tmp_path / "S11-ENERGY.json"
    g11.write_text('{"verdict":"PASS"}', "utf-8")
    monkeypatch.setattr(s13c_closeout, "G11_ENERGY_PATH", g11)
    checks = _passing_checks()
    payload = g13_gate_payload(
        checks,
        implementation_source_revision=_REVISION,
        branch="codex/s13-closeout",
        timestamp_utc="2026-10-02T06:00:00Z",
        implementation_evidence={"full_test_result": "all passed"},
    )
    assert payload["verdict"] == "PASS"
    assert s13c_closeout._coverage_check(79.9)["status"] == "FAIL"
    checks[-1] = s13c_closeout._coverage_check(79.9)
    failing = g13_gate_payload(
        checks,
        implementation_source_revision=_REVISION,
        branch="codex/s13-closeout",
        timestamp_utc="2026-10-02T06:00:00Z",
        implementation_evidence={},
    )
    assert failing["verdict"] == "FAIL"
    with pytest.raises(ValueError, match="STEPS order"):
        g13_gate_payload(
            list(reversed(checks)),
            implementation_source_revision=_REVISION,
            branch="b",
            timestamp_utc="t",
            implementation_evidence={},
        )


def test_cp_freeze_requires_passing_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g11 = tmp_path / "S11-ENERGY.json"
    g13 = tmp_path / "S13.json"
    g11.write_text(json.dumps({"verdict": "PASS"}), "utf-8")
    g13.write_text(json.dumps({"verdict": "FAIL"}), "utf-8")
    monkeypatch.setattr(s13c_closeout, "G11_ENERGY_PATH", g11)
    monkeypatch.setattr(s13c_closeout, "G13_PATH", g13)
    with pytest.raises(RuntimeError, match="passing amended G11 and G13"):
        cp_freeze_payload(
            run_root=tmp_path,
            implementation_source_revision=_REVISION,
            timestamp_utc="2026-10-02T06:00:00Z",
        )


def test_search_space_is_the_single_registered_trial() -> None:
    assert len(search_space().trials) == 1


def test_parser_offers_g13_and_freeze() -> None:
    parser = build_parser()
    assert parser.parse_args(["freeze", "--run-root", "r"]).command == "freeze"
    with pytest.raises(SystemExit):
        parser.parse_args(["g13", "--run-root", "r"])
