from pathlib import Path

import pytest
from formamech.ui import WorkbenchWorkspace

PLATE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cad" / "plate-with-hole.step"
)


@pytest.fixture
def workspace(tmp_path):
    study = WorkbenchWorkspace(tmp_path / "study")
    study.import_cad_path(PLATE)
    return study


def _event(payload, assignment, event_id="apply-1"):
    return {
        "workspace_id": payload["workspace_id"],
        "geometry_id": payload["geometry_id"],
        "draft_revision": payload["draft_revision"],
        "event_id": event_id,
        "region": assignment["region"],
        "assignment": assignment,
    }


def test_apply_acknowledges_and_deduplicates(workspace):
    from piginot.ui.workbench.boundary_picker import apply_picker_event, picker_payload

    before = picker_payload(workspace)
    assignment = {**before["assignments"][0], "name": "bottom_free"}
    event = _event(before, assignment)
    result = apply_picker_event(workspace, event)
    after = picker_payload(workspace, result=result)
    assert result["status"] == "accepted"
    assert (
        result["draft_revision"] == after["draft_revision"] != before["draft_revision"]
    )
    assert after["assignments"][0] == assignment
    assert after["last_apply"] == result
    assert str(workspace.root) not in after["workspace_id"]
    saved = workspace.state_path.read_bytes()
    assert apply_picker_event(WorkbenchWorkspace(workspace.root), event) == result
    assert workspace.state_path.read_bytes() == saved


@pytest.mark.parametrize("field", ["region", "name", "x", "components", "kind"])
def test_invalid_event_returns_field_error_without_saving(workspace, field):
    from piginot.ui.workbench.boundary_picker import apply_picker_event, picker_payload

    payload = picker_payload(workspace)
    assignment = {**payload["assignments"][1]}
    bad = {
        "region": {"region": "missing"},
        "name": {"name": "3 invalid"},
        "x": {"values": [{"unit": "mm", "expression": "unrecognised + 2"}]},
        "components": {"components": ["x", "x"]},
        "kind": {"kind": "unknown"},
    }
    proposal = {**assignment, **bad[field]}
    saved = workspace.state_path.read_bytes()
    result = apply_picker_event(workspace, _event(payload, proposal))
    assert result["status"] == "invalid"
    assert field in result["errors"]
    assert workspace.state_path.read_bytes() == saved
    assert picker_payload(workspace)["assignments"][1] == assignment


@pytest.mark.parametrize("change", ["draft", "geometry"])
def test_other_session_cannot_be_overwritten(workspace, change):
    from piginot.ui.workbench.boundary_picker import apply_picker_event, picker_payload

    old = picker_payload(workspace)
    old_event = _event(old, {**old["assignments"][0], "name": "too_late"})
    other = WorkbenchWorkspace(workspace.root)
    if change == "draft":
        edit = _event(
            old, {**old["assignments"][0], "name": "new_name"}, event_id="other-tab"
        )
        assert apply_picker_event(other, edit)["status"] == "accepted"
    else:
        other.import_cad_path(PLATE)
    saved = workspace.state_path.read_bytes()
    result = apply_picker_event(workspace, old_event)
    assert result["status"] == "stale"
    assert result["reload"] is True
    assert workspace.state_path.read_bytes() == saved


def test_one_edge_can_be_edited_in_an_incomplete_study(workspace):
    from piginot.ui.workbench.boundary_picker import apply_picker_event, picker_payload

    for i, index in enumerate((1, 3)):
        payload = picker_payload(workspace)
        current = payload["assignments"][index]
        assignment = {key: current[key] for key in ("name", "region")}
        assignment["kind"] = "traction_free"
        result = apply_picker_event(workspace, _event(payload, assignment, str(i)))
        assert result["status"] == "accepted"
    payload = picker_payload(workspace)
    assert any("at least one" in message for message in payload["readiness_errors"])
    assert payload["assignments"][3]["kind"] == "traction_free"
