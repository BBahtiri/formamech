from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from formamech.ui import WorkbenchWorkspace

PLATE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cad" / "plate-with-hole.step"
)


def test_failed_import_preserves_same_named_source_and_mesh(tmp_path):
    study = WorkbenchWorkspace(tmp_path / "study")
    preview = study.import_cad_path(PLATE)
    with pytest.raises(ValueError, match="CAD"):
        study.import_cad_bytes(b"bad CAD", PLATE.name)
    assert (study.source_dir / PLATE.name).read_bytes() == PLATE.read_bytes()
    assert study.load_preview() == preview


def test_isolated_imports_in_different_workspaces(tmp_path):
    studies = [WorkbenchWorkspace(tmp_path / str(i)) for i in range(2)]

    def mesh(study):
        return study.import_cad_bytes(PLATE.read_bytes(), PLATE.name, isolated=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(mesh, studies))
    assert all(preview["mesh"]["triangle_count"] > 0 for preview in results)
    assert [study.load_preview() for study in studies] == results


def test_replacement_rechecks_jobs_and_consumes_review(tmp_path):
    from piginot.ui.workbench.job_runner import JobRecord, WorkbenchJobRunner

    study = WorkbenchWorkspace(tmp_path / "study")
    study.import_cad_path(PLATE)
    review = study.replacement_review()
    runner = WorkbenchJobRunner(study.jobs_dir, cwd=study.root)
    runner._save(
        JobRecord(
            job_id="pending-run",
            kind="compile",
            command=(),
            pid=None,
            state="PENDING",
            exit_code=None,
            started_utc="2026-11-23T00:00:00Z",
            finished_utc=None,
            log_path=str(study.jobs_dir / "pending-run.log"),
            output_dir=str(study.run_dir),
        )
    )
    payload = PLATE.read_bytes()
    with pytest.raises(RuntimeError, match="Finish or cancel"):
        study.import_cad_bytes(payload, PLATE.name, expected_study=review)
    assert study.replacement_review() == review
    runner.cancel("pending-run")
    study.import_cad_bytes(payload, PLATE.name, expected_study=review)
    with pytest.raises(RuntimeError, match="study changed"):
        study.import_cad_bytes(payload, PLATE.name, expected_study=review)
    other = WorkbenchWorkspace(tmp_path / "other")
    with pytest.raises(RuntimeError, match="study changed"):
        other.import_cad_bytes(payload, PLATE.name, expected_study=review)
