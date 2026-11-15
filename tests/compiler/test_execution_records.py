"""Synthetic terminal outcomes exercise recovery without any numerical work."""

from pathlib import Path

import pytest

from piginot.compiler.execution_records import (
    CaseJob,
    OutcomeStore,
    digest,
    load_bound,
)


def test_pending_tail_survives_two_interruptions_and_old_failure(tmp_path: Path):
    calls = []
    stop = {"at": "c"}

    def solve(case_id):
        calls.append(case_id)
        if case_id == stop["at"]:
            raise KeyboardInterrupt()
        return {"status": "FAIL" if case_id == "b" else "PASS", "case": case_id}

    jobs = tuple(CaseJob(name, name * 64, lambda n=name: solve(n)) for name in "abcd")
    store = OutcomeStore(tmp_path, "1" * 64)
    with pytest.raises(KeyboardInterrupt):
        store.run(jobs)
    previous = {p.name: p.read_bytes() for p in tmp_path.glob("outcome-*.json")}
    stop["at"] = "d"
    with pytest.raises(KeyboardInterrupt):
        store.run(jobs)
    stop["at"] = "never"
    outcomes = store.run(jobs)
    assert calls == ["a", "b", "c", "c", "d", "d"]
    assert [item["status"] for item in outcomes] == ["PASS", "FAIL", "PASS", "PASS"]
    assert all(
        (tmp_path / name).read_bytes() == data for name, data in previous.items()
    )
    assert store.run(jobs) == outcomes


def test_corruption_is_checked_before_any_pending_job(tmp_path: Path):
    calls = []
    jobs = (CaseJob("a", "a" * 64, lambda: {"status": "PASS"}),)
    store = OutcomeStore(tmp_path, "1" * 64)
    store.run(jobs)
    outcome = next(tmp_path.glob("outcome-*.json"))
    outcome.write_bytes(outcome.read_bytes().replace(b'"PASS"', b'"FAIL"'))
    with pytest.raises(ValueError, match="digest"):
        store.run(jobs)
    assert calls == []


def test_terminal_exception_is_retained_and_does_not_strand_tail(tmp_path: Path):
    calls = []

    def fail():
        calls.append("bad")
        raise RuntimeError("solver exploded")

    jobs = (
        CaseJob("bad", "a" * 64, fail),
        CaseJob("tail", "b" * 64, lambda: {"status": "PASS"}),
    )
    store = OutcomeStore(tmp_path, "1" * 64)
    results = store.run(jobs)
    assert results[0]["status"] == "ERROR"
    assert results[0]["error"] == {
        "category": "RuntimeError",
        "message": "solver exploded",
    }
    assert results[1]["status"] == "PASS"
    assert store.run(jobs) == results
    assert calls == ["bad"]


@pytest.mark.parametrize("change", ["binding", "request", "roster"])
def test_changed_binding_request_or_roster_refuses(tmp_path: Path, change: str):
    job = CaseJob("a", "a" * 64, lambda: {"status": "PASS"})
    OutcomeStore(tmp_path, "1" * 64).run((job,))
    store = OutcomeStore(tmp_path, "2" * 64 if change == "binding" else "1" * 64)
    jobs = (CaseJob("a", "b" * 64, job.run),) if change == "request" else (job,)
    if change == "roster":
        jobs += (CaseJob("b", "b" * 64, job.run),)
    with pytest.raises(ValueError, match="binding|roster"):
        store.run(jobs)


def test_invalid_result_is_an_error_not_pass(tmp_path: Path):
    result = OutcomeStore(tmp_path, "1" * 64).run(
        (CaseJob("bad", "a" * 64, lambda: {"status": "PENDING"}),)
    )
    assert result[0]["status"] == "ERROR"


def test_duplicate_roster_and_symlink_refuse(tmp_path: Path):
    job = CaseJob("a", "a" * 64, lambda: {"status": "PASS"})
    with pytest.raises(ValueError, match="unique"):
        OutcomeStore(tmp_path, "1" * 64).run((job, job))
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        OutcomeStore(link, "1" * 64).run((job,))


def test_invalid_envelope_payload_and_evidence_symlink_refuse(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="envelope"):
        load_bound(path, "a" * 64)
    body = {"binding_sha256": "a" * 64, "payload": []}
    from piginot.contracts import canonical_json_bytes

    path.write_bytes(canonical_json_bytes({**body, "content_sha256": digest(body)}))
    with pytest.raises(ValueError, match="object"):
        load_bound(path, "a" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        load_bound(link, "a" * 64)


def test_readonly_validation_requires_original_roster_and_real_directory(tmp_path):
    job = CaseJob("a", "a" * 64, lambda: {"status": "PASS"})
    store = OutcomeStore(tmp_path, "a" * 64)
    with pytest.raises(ValueError, match="roster"):
        store.validate((job,))
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        OutcomeStore(link, "a" * 64).validate((job,))
