import json
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

from piginot.ui.workbench.job_runner import JobRecord, WorkbenchJobRunner

BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "notched-v1-cp-freeze"


def test_finished_job_archives_its_own_report_across_shared_output_reuse(tmp_path):
    runner = WorkbenchJobRunner(tmp_path / "jobs", cwd=tmp_path)
    output = tmp_path / "qualification"
    job = runner.start_qualify(bundle_path=BUNDLE, output_dir=output, sample_dev=0)
    deadline = monotonic() + 60
    try:
        while not job.is_terminal and monotonic() < deadline:
            sleep(0.1)
            job = runner.poll(job.job_id)
        assert job.state == "REFUSED_SCIENTIFIC", runner.read_log_tail(job.job_id)
        published = output / "qualification-report.md"
        original = published.read_bytes()
        assert runner.report_for(job) == original
        published.write_text(
            "Different run in the same output directory", encoding="utf-8"
        )
        reopened = WorkbenchJobRunner(runner.jobs_dir, cwd=tmp_path)
        assert reopened.report_for(reopened.load(job.job_id)) == original
        assert reopened.report_for(replace(job, job_id="different-job")) is None
        reopened._save(replace(job, state="RUNNING", finished_utc=None, exit_code=None))
        recovered = reopened.poll(job.job_id)
        assert recovered.state == "REFUSED_SCIENTIFIC"
        assert recovered.exit_code == 3
        archive = runner.jobs_dir / "evidence" / job.evidence_id / "report.md"
        archive.write_text("corrupted archived report", encoding="utf-8")
        assert reopened.report_for(job) is None
    finally:
        if not job.is_terminal:
            runner.cancel(job.job_id)


def test_unobserved_legacy_exit_does_not_inherit_another_jobs_outcome(tmp_path):
    runner = WorkbenchJobRunner(tmp_path / "jobs", cwd=tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "report.json").write_text('{"exit_code": 0}', encoding="utf-8")
    old = JobRecord(
        job_id="old",
        kind="compile",
        command=(),
        pid=None,
        state="RUNNING",
        exit_code=None,
        started_utc="2026-11-23T00:00:00Z",
        finished_utc=None,
        log_path=str(tmp_path / "old.log"),
        output_dir=str(output),
    )
    runner._save(old)
    runner._save(replace(old, job_id="new", state="SUCCEEDED", exit_code=0))
    recovered = runner.poll(old.job_id)
    assert recovered.state == "FAILED"
    assert recovered.exit_code is None
    assert runner.report_for(recovered) is None


def test_poll_reads_exit_published_during_pid_probe(tmp_path, monkeypatch):
    from piginot.ui.workbench import job_runner

    runner = WorkbenchJobRunner(tmp_path / "jobs")
    token = "c" * 32
    record = JobRecord(
        job_id="finishing",
        kind="compile",
        command=(),
        pid=101,
        state="RUNNING",
        exit_code=None,
        started_utc="2026-11-23T00:00:00Z",
        finished_utc=None,
        log_path=str(tmp_path / "finishing.log"),
        output_dir=str(tmp_path / "run"),
        evidence_id=token,
    )
    runner._save(record)
    evidence = runner.jobs_dir / "evidence" / token
    evidence.mkdir(parents=True)
    (evidence / "owner.json").write_text(
        json.dumps(
            {
                "job_id": record.job_id,
                "kind": record.kind,
                "output_dir": record.output_dir,
            }
        ),
        encoding="utf-8",
    )

    def finish(_pid):
        (evidence / "exit.txt").write_text("0", encoding="ascii")
        return False

    monkeypatch.setattr(job_runner, "_pid_alive", finish)
    assert runner.poll(record.job_id).state == "SUCCEEDED"
