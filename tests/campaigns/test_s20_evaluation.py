"""S20 records: the committed harness and burned registries match a rebuild."""

from __future__ import annotations

import json
from pathlib import Path

from campaigns import s20_evaluation as s20
from piginot.evaluation import BurnedCaseRegistry


def test_the_committed_records_match_a_rebuild(tmp_path: Path) -> None:
    root = s20.REPOSITORY
    committed = json.loads((root / s20.OUTPUT / "S20-HARNESS.json").read_text("utf-8"))
    assert committed == json.loads(json.dumps(s20.harness_report(root)))
    assert committed["reproduced"] is True
    assert {r["mismatches"] for r in committed["records"]} == {0}
    for campaign, document in s20.burned_registries(root).items():
        path = root / s20.OUTPUT / f"BURNED-HOLDOUT-{campaign}.json"
        assert json.loads(path.read_text("utf-8")) == document
        assert len(BurnedCaseRegistry.from_document(document).case_ids) == 64


def test_main_writes_and_reports(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(
        s20.REPOSITORY / "artifacts" / "campaigns", tmp_path / "artifacts" / "campaigns"
    )
    written = s20.write(tmp_path)
    assert [p.name for p in written] == [
        "S20-HARNESS.json",
        "BURNED-HOLDOUT-dogbone-v1.json",
        "BURNED-HOLDOUT-notched-v1.json",
    ]
    assert s20.main([]) == 0
