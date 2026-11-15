"""Prepared geometry joins the same plan, CLI and resume input binding."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from piginot.compiler import cli, planning, records
from piginot.compiler.config_types import AdapterRef, GeometryResourceRef
from piginot.compiler.planning import PlanError, build_plan
from tests.compiler.plan_fixtures import inputs


def test_additional_geometry_inputs_are_checked_before_publication(tmp_path):
    problem, config, _, descriptor = inputs(tmp_path)
    plan = build_plan(
        problem,
        config,
        tmp_path / "run",
        descriptor=descriptor,
    )
    geometry = tmp_path / "geometry.json"
    geometry.write_bytes(b"original")
    plan = replace(
        plan,
        input_paths=(*plan.input_paths, geometry),
        input_sha256=(
            *plan.input_sha256,
            ("geometry-resource", hashlib.sha256(b"original").hexdigest()),
        ),
    )
    records._verify_inputs(plan)
    geometry.write_bytes(b"changed")
    with pytest.raises(PlanError, match="geometry-resource changed"):
        records.write_plan(plan)
    assert not plan.output_path.exists()


def test_compile_cli_resolves_the_explicit_resource_relative_to_its_config(
    tmp_path, monkeypatch, capsys
):
    reference = AdapterRef(
        "cad-fixed-v1", "a" * 64, GeometryResourceRef("../cad/geometry.json", "b" * 64)
    )
    campaign = SimpleNamespace(adapter=reference)
    descriptor = object()
    calls = []
    adapter = SimpleNamespace(descriptor=descriptor)
    monkeypatch.setattr(cli.CampaignConfig, "load", lambda path: campaign)
    monkeypatch.setattr(
        cli,
        "resolve_configured_adapter",
        lambda ref, path: calls.append((ref, path)) or adapter,
        raising=False,
    )
    monkeypatch.setattr(
        cli, "resolve_adapter", lambda name: pytest.fail("unbound CAD resolver")
    )
    plan = SimpleNamespace(document=lambda: {"status": "PLANNED"})

    def build(spec, config, output, **kwargs):
        assert kwargs["descriptor"] is descriptor
        return plan

    monkeypatch.setattr(cli, "build_plan", build)
    monkeypatch.setattr(cli, "write_plan", lambda value: None)
    config = tmp_path / "campaign/config.json"
    assert (
        cli.compile_command(
            tmp_path / "problem.json",
            config,
            tmp_path / "run",
            plan_only=True,
            resume=False,
        )
        == 0
    )
    assert calls == [(reference, config)]
    assert '"PLANNED"' in capsys.readouterr().out


def test_build_plan_does_not_ignore_geometry_resources_with_an_injected_descriptor(
    tmp_path, monkeypatch
):
    problem, config, _, descriptor = inputs(tmp_path)
    original = planning.CampaignConfig.load(config)
    configured = replace(
        original,
        adapter=AdapterRef(
            "cad-fixed-v1", "a" * 64, GeometryResourceRef("missing.json", "b" * 64)
        ),
    )
    monkeypatch.setattr(
        planning.CampaignConfig, "from_document", lambda value: configured
    )

    def unavailable(reference, config_path):
        assert reference == configured.adapter
        raise ValueError("bound geometry is unavailable")

    monkeypatch.setattr(
        planning, "resolve_configured_adapter", unavailable, raising=False
    )
    with pytest.raises(PlanError, match="bound geometry is unavailable"):
        build_plan(
            problem,
            config,
            tmp_path / "run",
            descriptor=descriptor,
        )


@pytest.mark.parametrize(
    "fault", [None, "digest", "labels", "duplicate", "reserved", "descriptor"]
)
def test_plan_binds_all_provider_inputs_and_rejects_incoherent_bindings(
    tmp_path, monkeypatch, fault
):
    problem, config, _, original_descriptor = inputs(tmp_path)
    original = planning.CampaignConfig.load(config)
    configured = replace(
        original,
        adapter=AdapterRef(
            "cad-fixed-v1", "a" * 64, GeometryResourceRef("geometry.json", "b" * 64)
        ),
    )
    descriptor = replace(
        original_descriptor,
        adapter_id="cad-fixed-v1",
        implementation_sha256="a" * 64,
    )
    paths = []
    hashes = []
    for index in range(5):
        path = tmp_path / f"member-{index}"
        path.write_bytes(str(index).encode())
        paths.append((f"geometry-{index}", path))
        hashes.append(
            (f"geometry-{index}", hashlib.sha256(path.read_bytes()).hexdigest())
        )
    if fault == "digest":
        hashes[0] = (hashes[0][0], "0" * 64)
    elif fault == "labels":
        hashes[0] = ("other", hashes[0][1])
    elif fault == "duplicate":
        paths[1] = (paths[0][0], paths[1][1])
        hashes[1] = (hashes[0][0], hashes[1][1])
    elif fault == "reserved":
        paths[0] = ("problem", paths[0][1])
        hashes[0] = ("problem", hashes[0][1])
    bound = SimpleNamespace(
        descriptor=descriptor,
        geometry_inputs=tuple(paths),
        geometry_input_sha256=tuple(hashes),
    )
    monkeypatch.setattr(
        planning.CampaignConfig, "from_document", lambda value: configured
    )
    monkeypatch.setattr(planning, "resolve_configured_adapter", lambda *args: bound)
    provided = original_descriptor if fault == "descriptor" else descriptor
    if fault:
        with pytest.raises(PlanError):
            build_plan(
                problem,
                config,
                tmp_path / "run",
                descriptor=provided,
            )
    else:
        plan = build_plan(
            problem,
            config,
            tmp_path / "run",
            descriptor=provided,
        )
        assert len(plan.input_paths) == 8
        assert plan.input_sha256[-5:] == tuple(hashes)
        records.write_plan(plan)
