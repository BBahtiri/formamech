from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from campaigns.dogbone_v1 import s14_protocol
from campaigns.dogbone_v1.s14_protocol import (
    EVALUATION_FILES,
    build_protocol,
    file_digests,
    load_signed_protocol,
    protocol_bytes,
    text_sha256,
    verify_frozen_files,
)
from campaigns.dogbone_v1.s14_rehearsal import GATED_SLICES
from campaigns.dogbone_v1.s14_validate import (
    build_parser,
    holdout_blockers,
    run_development,
)
from piginot.verification import forbidden_wording

_ROOT = Path(__file__).resolve().parents[2]


def _rehearsal(**changes: object) -> bytes:
    document: dict[str, object] = {
        "evidence_role": "TRAIN",
        "development_opened": False,
        "case_count": 64,
        "budget": {
            "gated_status": "PASS",
            "gated": [
                {"quantity": "p_relative_l2", "statistic": 0.02},
                {"quantity": "p_relative_l2", "statistic": 0.03},
            ],
        },
        "confidence_pilot": {
            "selected_variant": "patch-mean",
            "pooled_spearman": {"pointwise": 0.5, "patch-mean": 0.7},
            "median_per_case_spearman": {"pointwise": 0.4, "patch-mean": 0.6},
        },
        "refusal": {"fixture_set_hash": "f" * 64, "in_domain_count": 844},
    }
    document.update(changes)
    return json.dumps(document).encode()


def test_protocol_reproduces_adr_0019_and_the_pilot_choice() -> None:
    document = build_protocol(
        rehearsal_bytes=_rehearsal(), revision="r", created_utc="t"
    )
    budget = document["budget"]
    assert {q: v["tolerance"] for q, v in budget["quantities"].items()} == {  # type: ignore[index]
        "p_relative_l2": 0.05,
        "absolute_peak_error": 0.05,
        "u_relative_l2": 0.02,
    }
    assert budget["gated_slices"] == list(GATED_SLICES)  # type: ignore[index]
    assert len(budget["reported_slices"]) == 14  # type: ignore[index]
    assert document["confidence"]["indicator"] == "patch-mean"  # type: ignore[index]
    assert document["confidence"]["threshold"] == 0.6  # type: ignore[index]
    assert document["train_rehearsal"]["summary"][  # type: ignore[index]
        "largest_gated_slice_statistic"
    ] == {"p_relative_l2": 0.03}
    assert set(document["evaluation_code"]["file_sha256"]) == set(EVALUATION_FILES)  # type: ignore[index]
    assert forbidden_wording(protocol_bytes(document).decode()) == ()
    for bad in (
        {"evidence_role": "DEVELOPMENT"},
        {"development_opened": True},
        {"case_count": 63},
    ):
        with pytest.raises(ValueError):
            build_protocol(
                rehearsal_bytes=_rehearsal(**bad), revision="r", created_utc="t"
            )
    pilot = json.loads(_rehearsal())["confidence_pilot"]
    with pytest.raises(ValueError):
        build_protocol(
            rehearsal_bytes=_rehearsal(
                confidence_pilot={**pilot, "selected_variant": "other"}
            ),
            revision="r",
            created_utc="t",
        )


def test_frozen_files_detect_any_change(tmp_path: Path) -> None:
    document = {"evaluation_code": {"file_sha256": file_digests()}}
    assert verify_frozen_files(document) == ()
    for name in EVALUATION_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_ROOT / name, tmp_path / name)
    assert verify_frozen_files(document, tmp_path) == ()
    changed = tmp_path / EVALUATION_FILES[0]
    changed.write_text(changed.read_text() + "\n# edit\n")
    assert verify_frozen_files(document, tmp_path) == (EVALUATION_FILES[0],)
    crlf = tmp_path / EVALUATION_FILES[1]
    crlf.write_bytes(crlf.read_bytes().replace(b"\n", b"\r\n"))
    assert EVALUATION_FILES[1] not in verify_frozen_files(document, tmp_path)


def test_sign_off_must_name_the_protocol_digest(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(
        protocol_bytes(
            build_protocol(rehearsal_bytes=_rehearsal(), revision="r", created_utc="t")
        )
    )
    signoff = tmp_path / "signoff.json"
    with pytest.raises(PermissionError):
        load_signed_protocol(protocol, signoff)
    signoff.write_text(
        json.dumps({"protocol_sha256": "0" * 64, "signed_off_by": "Betim Bahtiri"})
    )
    with pytest.raises(PermissionError):
        load_signed_protocol(protocol, signoff)
    digest = text_sha256(protocol)
    signoff.write_text(
        json.dumps({"protocol_sha256": digest, "signed_off_by": "someone"})
    )
    with pytest.raises(PermissionError):
        load_signed_protocol(protocol, signoff)
    signoff.write_text(
        json.dumps({"protocol_sha256": digest, "signed_off_by": "Betim Bahtiri"})
    )
    document, loaded = load_signed_protocol(protocol, signoff)
    assert loaded == digest and document["protocol_id"] == s14_protocol.PROTOCOL_ID


def test_step_three_refuses_to_open_development_without_sign_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s14_protocol, "SIGNOFF_PATH", tmp_path / "missing.json")
    run_root = tmp_path / "run"
    with pytest.raises(PermissionError):
        run_development(
            s10_run_root=tmp_path, checkpoint_path=tmp_path, run_root=run_root
        )
    assert not run_root.exists()
    options = build_parser().parse_args(["gate", "--run-root", str(tmp_path)])
    assert options.phase == "gate" and options.reducer_status is None


def test_holdout_stays_closed_when_development_fails() -> None:
    def development(status: str, rho: float) -> dict[str, object]:
        return {
            "budget": {"gated_status": status},
            "confidence": {"pooled_spearman": rho},
        }

    assert holdout_blockers(development("PASS", 0.65)) == ()
    assert holdout_blockers(development("FAIL", 0.65)) == ("G14.budget",)
    assert holdout_blockers(development("NOT_RUN", 0.5)) == (
        "G14.budget",
        "G14.confidence",
    )


def test_committed_protocol_matches_the_frozen_code_and_rehearsal() -> None:
    document = json.loads(s14_protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
    rehearsal = s14_protocol.REHEARSAL_PATH.read_bytes()

    assert verify_frozen_files(document) == ()
    assert (
        document["train_rehearsal"]["sha256"] == hashlib.sha256(rehearsal).hexdigest()
    )
    assert (
        document["confidence"]["indicator"]
        == json.loads(rehearsal)["confidence_pilot"]["selected_variant"]
    )
    assert document["status"] == "FROZEN_PENDING_USER_SIGN_OFF"
