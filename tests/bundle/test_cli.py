from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from piginot import cli
from piginot.bundle import REQUIRED_LIMITATION_PARAGRAPH
from piginot.bundle.inference import MODEL_PATH
from tests.bundle import helpers


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("cli") / "synthetic"
    helpers.build(directory)
    return directory


def _request(path: Path, **changes: object) -> Path:
    inputs: dict[str, object] = {
        "L_total": {"value": 5.4, "unit": "cm"},
        "W_grip": {"value": 20.0, "unit": "mm"},
        "W_gauge": {"value": 10.0, "unit": "mm"},
        "R_fillet": {"value": 12.0, "unit": "mm"},
        "elastic_modulus": {"value": 760.0, "unit": "MPa"},
        "poisson_ratio": {"value": 0.23, "unit": "1"},
        "nominal_axial_strain": {"value": 0.15, "unit": "1"},
    }
    inputs.update(changes)
    path.write_text(json.dumps({"inputs": inputs, "queries": [[5.0, 2.0]]}))
    return path


def test_predict_inspect_and_verify(
    bundle_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "prediction.json"
    request = _request(tmp_path / "request.json")
    code = cli.main(
        [
            "predict",
            str(bundle_path),
            "--request",
            str(request),
            "--output",
            str(output),
        ]
    )
    document = json.loads(output.read_text())
    assert code == cli.EXIT_OK
    assert document["verdict"] == "IN_DOMAIN"
    assert document["limitations"] == REQUIRED_LIMITATION_PARAGRAPH
    assert len(document["queries"]["u"]) == 1

    assert cli.main(["inspect", str(bundle_path)]) == cli.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["limitations"]["paragraph"] == REQUIRED_LIMITATION_PARAGRAPH

    assert cli.main(["bundle", "verify", str(bundle_path)]) == cli.EXIT_OK
    verified = json.loads(capsys.readouterr().out)
    assert verified["verdict"] == "VERIFIED"
    assert verified["limitations"] == REQUIRED_LIMITATION_PARAGRAPH


def test_refusals_and_tampering(
    bundle_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for change, code in (
        (
            {"nominal_axial_strain": {"value": 0.3, "unit": "1"}},
            "PARAMETER_OUT_OF_BOUNDS",
        ),
        ({"L_total": {"value": 54.0, "unit": "furlong"}}, "INVALID_PARAMETER_VALUE"),
        ({"L_total": 54.0}, "INVALID_PARAMETER_VALUE"),
    ):
        request = _request(tmp_path / "r.json", **change)
        exit_code = cli.main(
            ["predict", str(bundle_path), "--request", str(request), "--no-confidence"]
        )
        document = json.loads(capsys.readouterr().out)
        assert exit_code == cli.EXIT_REFUSED
        assert document["reason_codes"] == [code]
        assert document["claim_level"] == "NOT_VERIFIED"
        assert document["limitations"] == REQUIRED_LIMITATION_PARAGRAPH

    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps({"inputs": {}, "queries": 5}))
    assert cli.main(["predict", str(bundle_path), "--request", str(odd)]) == (
        cli.EXIT_REFUSED
    )
    capsys.readouterr()
    assert cli.parse_request({"inputs": {}, "queries": 5})[1] == [5]

    (tmp_path / "bad.json").write_text("{")
    assert (
        cli.main(["predict", str(bundle_path), "--request", str(tmp_path / "bad.json")])
        == cli.EXIT_ERROR
    )
    (tmp_path / "list.json").write_text("[]")
    assert (
        cli.main(
            ["predict", str(bundle_path), "--request", str(tmp_path / "list.json")]
        )
        == cli.EXIT_REFUSED
    )
    capsys.readouterr()

    tampered = tmp_path / "tampered"
    helpers.write_raw(tampered)
    model = tampered / MODEL_PATH
    model.write_bytes(model.read_bytes().replace(b"synthetic", b"Synthetic"))
    request = _request(tmp_path / "r.json")
    for argv in (
        ["predict", str(tampered), "--request", str(request)],
        ["inspect", str(tampered)],
        ["bundle", "verify", str(tampered)],
    ):
        assert cli.main(argv) == cli.EXIT_ERROR
        document = json.loads(capsys.readouterr().out)
        assert document["verdict"] == "BUNDLE_REFUSED"
        assert document["reason_codes"] == ["BUNDLE_FILE_HASH_MISMATCH"]
        assert document["limitations"] == REQUIRED_LIMITATION_PARAGRAPH


def test_campaign_commands_dispatch_to_a_plugin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, list[str]]] = []
    module = types.ModuleType("fake_campaign_plugin")
    module.COMMANDS = {  # type: ignore[attr-defined]
        name: (lambda arguments, name=name: calls.append((name, arguments)) or 0)
        for name in cli.CAMPAIGN_COMMANDS
    }
    module.NOT_A_TABLE = 3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_campaign_plugin", module)
    plugin = "fake_campaign_plugin:COMMANDS"

    argv_by_command = {
        "family compile": ["family", "compile"],
        "bundle build": ["bundle", "build"],
    }
    for command in cli.CAMPAIGN_COMMANDS:
        argv = argv_by_command.get(command, [command])
        assert cli.main([*argv, "--campaign", plugin, "--", "--x", "1"]) == 0
    assert calls == [(name, ["--x", "1"]) for name in cli.CAMPAIGN_COMMANDS]

    assert cli.main(["train", "--campaign", "fake_campaign_plugin:NOT_A_TABLE"]) == 1
    assert cli.main(["train", "--campaign", "no_such_campaign_xyz"]) == 1
    module.COMMANDS = {}  # type: ignore[attr-defined]
    assert cli.main(["train", "--campaign", plugin]) == 1
    assert "has no 'train' command" in capsys.readouterr().err


def test_entry_point_plugins_and_search_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Entry:
        name = "listed"

        @staticmethod
        def load() -> dict[str, object]:
            return {"train": lambda arguments: 7}

    monkeypatch.setattr(cli, "entry_points", lambda group: [Entry()])
    assert cli.main(["train", "--campaign", "listed"]) == 7

    package = tmp_path / "campaigns" / "toy"
    package.mkdir(parents=True)
    (tmp_path / "campaigns" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text("COMMANDS = {'pilot': lambda arguments: 5}\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [
        m for m in sys.modules if m == "campaigns" or m.startswith("campaigns.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    assert (
        cli.main(["pilot", "--campaign", "toy", "--campaign-path", str(tmp_path)]) == 5
    )


def test_dogbone_plugin_table_and_family_compile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from campaigns.dogbone_v1.cli import COMMANDS

    assert set(COMMANDS) == set(cli.CAMPAIGN_COMMANDS)
    assert COMMANDS["family compile"]([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["parameters"] == ["L_total", "W_grip", "W_gauge", "R_fillet"]
    with pytest.raises(SystemExit):
        COMMANDS["pilot"](["--unexpected"])
    with pytest.raises(SystemExit):
        COMMANDS["verify"](["holdout"])
