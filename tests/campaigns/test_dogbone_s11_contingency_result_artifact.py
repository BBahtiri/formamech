from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import rfc8785

from piginot.contracts import strict_json_loads

_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "S11-CONTINGENCY-EVIDENCE.json"
)
_FILE_BLAKE2B256 = "260eca6c345ee1edea17b9098859baaa6de4a35e27de65c6ccdead0cad831f36"
_IMPLEMENTATION_REVISION = "22fea865a74bf69a28dc845e874a008ba31ef650"
_PROTOCOL_HASH = "f16a4f63c3a8cfc3ee321bc4e15162802cbc6f106ed807f5aa4bc59eceac81df"
_SOURCE_BINDINGS_HASH = (
    "e32148e4efde6c54fe02ca25b86124d1a43a620eecddb235122919f1649461e3"
)
_SCHEDULE_HASH = "ab8514e93dcd2ef514c00b13362cbaf5198b3649c7fbfbeec86189380a7bc9db"
_CANDIDATE_IDS = (
    "A-s11-baseline",
    "B-derived-ratio-film",
    "C-local-stress-enrichment",
    "D-derived-local-stress",
)


def _document() -> dict[str, Any]:
    raw = _ARTIFACT.read_bytes()
    parsed = strict_json_loads(raw)
    assert isinstance(parsed, dict)
    return cast(dict[str, Any], parsed)


def test_contingency_result_is_the_exact_canonical_run_evidence() -> None:
    raw = _ARTIFACT.read_bytes()
    document = _document()

    assert hashlib.blake2b(raw, digest_size=32).hexdigest() == _FILE_BLAKE2B256
    assert rfc8785.dumps(document) == raw
    assert document["$type"] == "piginot.campaigns.ContingencyEvidence"
    assert document["implementation_revision"] == _IMPLEMENTATION_REVISION
    assert document["protocol_content_hash"] == _PROTOCOL_HASH
    assert document["training_schedule"]["schedule_hash"] == _SCHEDULE_HASH
    assert document["source_bindings"]["$type"] == (
        "piginot.campaigns.S10SourceBindings"
    )


def test_contingency_result_records_the_registered_fallback_without_development() -> None:
    document = _document()
    decision = document["train_screen_decision"]
    selection = decision["selection"]

    assert selection == {
        "$type": "piginot.campaigns.ArchitectureSelection",
        "eligible_candidate_ids": [],
        "fallback_to_baseline": True,
        "selected_candidate_id": "A-s11-baseline",
    }
    assert document["final_candidate_id"] == "A-s11-baseline"
    assert document["development_confirmation"] is None
    assert decision["source_bindings_hash"] == _SOURCE_BINDINGS_HASH
    assert decision["training_schedule_hash"] == _SCHEDULE_HASH
    assert tuple(item["candidate_id"] for item in decision["metrics"]) == (
        _CANDIDATE_IDS
    )
    assert tuple(item["candidate_id"] for item in document["model_files"]) == (
        _CANDIDATE_IDS
    )
    assert all(item["optimizer_steps"] == 1_500 for item in document["training_summaries"])


def test_local_stress_candidate_improves_stress_but_fails_only_u_guardrail() -> None:
    document = _document()
    metrics = {
        item["candidate_id"]: item
        for item in document["train_screen_decision"]["metrics"]
    }
    baseline = metrics["A-s11-baseline"]
    candidate = metrics["C-local-stress-enrichment"]

    guardrails = {
        "signed_peak_bias": abs(candidate["median_signed_peak_error"])
        <= 0.80 * abs(baseline["median_signed_peak_error"]),
        "median_peak": candidate["median_absolute_peak_error"]
        <= 0.90 * baseline["median_absolute_peak_error"],
        "p90_peak": candidate["p90_absolute_peak_error"]
        <= baseline["p90_absolute_peak_error"],
        "u_field": candidate["median_u_relative_l2"]
        <= 1.02 * baseline["median_u_relative_l2"],
        "p_field": candidate["median_p_relative_l2"]
        <= 1.02 * baseline["median_p_relative_l2"],
    }

    assert guardrails == {
        "signed_peak_bias": True,
        "median_peak": True,
        "p90_peak": True,
        "u_field": False,
        "p_field": True,
    }
    assert candidate["median_absolute_peak_error"] < (
        0.51 * baseline["median_absolute_peak_error"]
    )
    assert candidate["median_p_relative_l2"] < baseline["median_p_relative_l2"]


def test_tracked_result_contains_aggregates_and_model_hashes_not_field_payloads() -> None:
    document = _document()
    serialized = _ARTIFACT.read_text(encoding="utf-8")

    assert len(document["model_files"]) == 4
    assert all(
        item["file_name"].endswith(".safetensors")
        and len(item["file_blake2b256"]) == 64
        and item["byte_count"] > 0
        for item in document["model_files"]
    )
    assert "target_u" not in serialized
    assert "target_P" not in serialized
    assert "displacement" not in serialized
    assert "recovered_first_piola_stress" not in serialized
