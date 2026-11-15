"""A user's explicit settings become bound CAD inputs without running numerics."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from piginot.cad.configuration import prepare_campaign
from piginot.compiler.config import CampaignConfig
from piginot.compiler.public_source import select_public_cases
from piginot.contracts import (
    CanonicalMesh,
    ProblemSpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.doe import PublishedCaseManifest


def mesh_fixture(*, tags=None) -> CanonicalMesh:
    nodes = ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0))
    values = {
        "nodes": nodes,
        "cells": ((0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2)),
        "cell_type": "triangle",
        "boundary_facets": ((0, 3), (3, 4), (4, 5), (5, 2), (2, 1), (1, 0)),
        "semantic_tags": tags
        or {"bottom": (0,), "right": (1, 2), "top": (3,), "left": (4, 5)},
        "region_ids": (1, 1, 1, 1),
        "unit": Unit.MM,
        "orientation": "counterclockwise",
    }
    return CanonicalMesh(
        mesh_id=hash_spec(
            {"schema": "canonical-triangle-mesh-v1", **values}, "content"
        ),
        **values,
        topology_signature="one-connected-square",
        quality_metrics={"minimum_angle_degrees": 26.5, "maximum_aspect_ratio": 2.0},
        source_provenance={"adapter": "pure-cad-test"},
    )


def write_project(path: Path, *, tags=None, kinds=None) -> Path:
    path.mkdir()
    source = b"ISO-10303-21; pure fixture; END-ISO-10303-21;"
    (path / "source.step").write_bytes(source)
    mesh = mesh_fixture(tags=tags)
    payload = canonical_json_bytes(mesh)
    for level in (1, 2, 4):
        (path / f"mesh-{level}.json").write_bytes(payload)
    document = {
        "schema": "piginot.cad-geometry-v1",
        "source": {"path": "source.step", "sha256": hashlib.sha256(source).hexdigest()},
        "meshes": {
            str(level): {
                "path": f"mesh-{level}.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "mesh_id": mesh.mesh_id,
            }
            for level in (1, 2, 4)
        },
        "boundary_kinds": kinds or {name: "line" for name in mesh.semantic_tags},
        "provenance": {"source_format": "step", "domain": "fixed-polygon"},
    }
    (path / "geometry.json").write_bytes(canonical_json_bytes(document))
    return path


def settings() -> dict:
    return {
        "schema": "piginot.cad-settings-v1",
        "name": "my-cad-model",
        "parameters": [
            {
                "name": "E",
                "role": "material",
                "lo": 760.0,
                "hi": 760.0,
                "unit": "MPa",
                "distribution": "fixed",
            },
            {
                "name": "nu",
                "role": "material",
                "lo": 0.23,
                "hi": 0.23,
                "unit": "1",
                "distribution": "fixed",
            },
            {
                "name": "u",
                "role": "load",
                "lo": 0.01,
                "hi": 0.02,
                "unit": "mm",
                "distribution": "uniform",
            },
        ],
        "material": {
            "model": "compressible_neo_hookean",
            "parameters": {"elastic_modulus": "E", "poisson_ratio": "nu"},
        },
        "boundary_conditions": [
            {
                "name": "clamp",
                "region": "left",
                "kind": "dirichlet",
                "components": ["x", "y"],
                "values": [
                    {"expression": "0", "unit": "mm"},
                    {"expression": "0", "unit": "mm"},
                ],
            },
            {
                "name": "pull",
                "region": "right",
                "kind": "dirichlet",
                "components": ["x"],
                "values": [{"expression": "u", "unit": "mm"}],
            },
            {"name": "free_top", "region": "top", "kind": "traction_free"},
            {"name": "free_bottom", "region": "bottom", "kind": "traction_free"},
        ],
        "qois": [
            {
                "name": "free_peak",
                "field": "P_served",
                "reduction": "max-frobenius",
                "region": "top",
                "exclude_regions": ["left", "right"],
            }
        ],
        "verification": {
            "quantile": 0.9,
            "tolerances": {
                "absolute_peak_error": 0.05,
                "p_relative_l2": 0.05,
                "u_relative_l2": 0.02,
            },
        },
        "displacement_scale": "cad_extent_x",
        "training": {
            "optimizer_steps": 1,
            "case_batch_size": 1,
            "checkpoint_interval": 1,
            "learning_rate": 0.001,
            "final_learning_rate": 0.0001,
            "constant_steps": 0,
            "seed": 2902,
        },
        "sampling": {"fit_cases": 2, "screen_cases": 1},
    }


def save_settings(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def prepared(tmp_path):
    project = write_project(tmp_path / "project")
    setting = save_settings(tmp_path / "settings.json", settings())
    return project, setting, tmp_path / "campaign"


def test_prepare_fixed_geometry_campaign_has_real_ranges_and_no_hidden_work(prepared):
    project, setting, output = prepared
    before = setting.read_bytes()
    record = prepare_campaign(project, setting, output)
    problem = ProblemSpec.from_bytes((output / "problem.json").read_bytes())
    config = CampaignConfig.load(output / "config.json")
    source = PublishedCaseManifest.from_bytes((output / "source.json").read_bytes())
    selected = select_public_cases(problem, config, source)
    assert len(selected["fit"]) == 2 and len(selected["screen"]) == 1
    assert not selected["development"]
    assert not source.derived_axes and not source.stratum_axes
    assert {p.role for p in problem.parameters} == {"load", "material"}
    assert {p.name for p in problem.parameters} == {"E", "nu", "u"}
    assert config.training.optimizer_steps == 1 and config.training.seed == 2902
    assert config.reference.refinement_levels == (1, 2, 4)
    assert config.adapter.id == "cad-fixed-v1"
    assert (
        config.adapter.geometry_resource.sha256
        == hashlib.sha256((project / "geometry.json").read_bytes()).hexdigest()
    )
    assert (
        output / config.adapter.geometry_resource.path
    ).resolve() == project / "geometry.json"
    assert (
        config.execution.training.platform_fingerprint["cpu_capability"] == "NOT_PROBED"
    )
    assert record["profile_status"] == "METADATA_ONLY"
    assert record["claim_level"] == "NOT_VERIFIED"
    assert record["settings_sha256"] == hashlib.sha256(before).hexdigest()
    assert record["development"] == record["holdout"] == "NOT_GENERATED_NOT_REQUESTED"
    assert setting.read_bytes() == before
    assert json.loads((output / "preparation.json").read_bytes()) == record
    for name, digest in record["input_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    pieces = problem.document()["lifting"]["pieces"]
    assert len(pieces["clamp"]) == len(pieces["pull"]) == 1
    assert pieces["clamp"][0]["line"] == {
        "start": ["0.0", "1.0"],
        "end": ["0.0", "0.0"],
    }


def test_sampling_is_deterministic_disjoint_and_covers_material_ranges(
    prepared, tmp_path
):
    project, setting, output = prepared
    document = settings()
    document["parameters"][0] = {
        **document["parameters"][0],
        "hi": 800.0,
        "distribution": "uniform",
    }
    save_settings(setting, document)
    prepare_campaign(project, setting, output)
    prepare_campaign(project, setting, tmp_path / "again")
    assert (output / "source.json").read_bytes() == (
        tmp_path / "again/source.json"
    ).read_bytes()
    source = PublishedCaseManifest.from_bytes((output / "source.json").read_bytes())
    assert len({case.leakage_group_id for case in source.cases}) == 3
    assert (
        len({tuple(case.primitive_coordinates.items()) for case in source.cases}) == 3
    )
    assert all(
        760.0 <= case.primitive_coordinates["E"] <= 800.0 for case in source.cases
    )
    assert all(
        case.role.value == "TRAIN" and case.status.value == "ACTIVE"
        for case in source.cases
    )


def test_constraints_are_evaluated_and_recorded_in_source(prepared):
    project, setting, output = prepared
    document = {
        **settings(),
        "constraints": [
            {
                "name": "allowed_u",
                "expression": "u",
                "lo": 0.013,
                "hi": 0.017,
                "unit": "mm",
            }
        ],
    }
    save_settings(setting, document)
    prepare_campaign(project, setting, output)
    source = PublishedCaseManifest.from_bytes((output / "source.json").read_bytes())
    assert source.derived_axes == ("allowed_u",)
    assert all(
        case.derived_coordinates["allowed_u"] == case.primitive_coordinates["u"]
        for case in source.cases
    )
    assert all(
        0.013 <= case.primitive_coordinates["u"] <= 0.017 for case in source.cases
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "other"},
        {"unknown": True},
        {"sampling": {"fit_cases": 1, "screen_cases": 1}},
        {"sampling": {"fit_cases": 2, "screen_cases": True}},
        {"sampling": {"fit_cases": 2, "screen_cases": 1, "extra": 1}},
        {"training": {"optimizer_steps": 1}},
        {
            "reference": {
                "initialization": "cold",
                "refinement_levels": [1, 3],
                "workers": 2,
            }
        },
        {"model": {"architecture_id": "unsupported"}},
        {"model": {"parameter_names": ["E", "nu"]}},
        {"constants": [{"name": "cad_extent_x", "value": 1.0, "unit": "mm"}]},
    ],
)
def test_invalid_settings_refuse_before_output(prepared, change):
    project, setting, output = prepared
    save_settings(setting, {**settings(), **change})
    with pytest.raises(ValueError):
        prepare_campaign(project, setting, output)
    assert not output.exists()


def test_parameter_roles_and_all_fixed_ranges_are_rejected(prepared):
    project, setting, output = prepared
    document = settings()
    document["parameters"][2]["role"] = "geometry"
    save_settings(setting, document)
    with pytest.raises(ValueError, match="material|load"):
        prepare_campaign(project, setting, output)
    document["parameters"][2] = {
        **document["parameters"][2],
        "role": "load",
        "hi": 0.01,
        "distribution": "fixed",
    }
    save_settings(setting, document)
    with pytest.raises(ValueError, match="varying"):
        prepare_campaign(project, setting, output)


@pytest.mark.parametrize("kind", ["arc", "other"])
def test_curved_essential_boundary_refuses(prepared, kind):
    project, setting, output = prepared
    resource = json.loads((project / "geometry.json").read_bytes())
    resource["boundary_kinds"]["left"] = kind
    (project / "geometry.json").write_bytes(canonical_json_bytes(resource))
    with pytest.raises(ValueError, match="straight|line"):
        prepare_campaign(project, setting, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["missing", "unknown", "neumann"])
def test_boundary_selection_is_explicit_and_supported(prepared, change):
    project, setting, output = prepared
    document = settings()
    if change == "missing":
        document["boundary_conditions"] = document["boundary_conditions"][:-1]
    elif change == "unknown":
        document["boundary_conditions"][0]["region"] = "guessed"
    else:
        document["boundary_conditions"][1]["kind"] = "neumann"
    save_settings(setting, document)
    with pytest.raises(ValueError):
        prepare_campaign(project, setting, output)
    assert not output.exists()


def test_noncollinear_or_disconnected_essential_groups_refuse(tmp_path):
    for label, facets in [("noncollinear", (4, 5, 0)), ("disconnected", (5, 2))]:
        project = write_project(
            tmp_path / label,
            tags={"left": facets, "right": (1,), "top": (3,), "bottom": (0,)},
        )
        setting = save_settings(tmp_path / (label + ".json"), settings())
        with pytest.raises(ValueError, match="line|connected|branch"):
            prepare_campaign(project, setting, tmp_path / (label + "-out"))


def test_existing_overlapping_and_symlinked_inputs_refuse(prepared, tmp_path):
    project, setting, output = prepared
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_campaign(project, setting, output)
    for target in (project / "inside", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            prepare_campaign(project, setting, target)
    link = tmp_path / "linked-settings.json"
    try:
        link.symlink_to(setting)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink|regular"):
        prepare_campaign(project, link, tmp_path / "linked-out")


def test_tampered_geometry_refuses_before_output(prepared):
    project, setting, output = prepared
    with (project / "mesh-2.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        prepare_campaign(project, setting, output)
    assert not output.exists()


def test_execution_profile_is_explicit_and_never_executes_numerics(
    prepared, monkeypatch
):
    from piginot.cad import configuration

    project, setting, output = prepared
    monkeypatch.setattr(configuration.metadata, "version", lambda name: "1.0")
    monkeypatch.setattr(configuration, "_cpu_capability", lambda: "CAD_TEST")
    receipt = prepare_campaign(project, setting, output, execution_profile=True)
    assert receipt["profile_status"] == "ENVIRONMENT_CAPTURED_NOT_EXECUTED"
    config = CampaignConfig.load(output / "config.json")
    assert (
        config.execution.training.platform_fingerprint["cpu_capability"] == "CAD_TEST"
    )


def test_missing_execution_dependencies_and_package_overlap_refuse(
    prepared, monkeypatch
):
    from importlib import metadata

    import piginot
    from piginot.cad import configuration

    project, setting, output = prepared

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(configuration.metadata, "version", missing)
    with pytest.raises(ValueError, match="verify"):
        prepare_campaign(project, setting, output, execution_profile=True)
    assert not output.exists()
    with pytest.raises(ValueError, match="package"):
        prepare_campaign(
            project, setting, Path(piginot.__file__).resolve().parent / "not-created"
        )


def test_metadata_preparation_in_fresh_process_imports_no_numerical_stack(prepared):
    project, setting, output = prepared
    root = Path(__file__).resolve().parents[2]
    program = f"""
import importlib.abc, sys
sys.path.insert(0, {str(root / "src")!r})
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch','numpy','scipy','sympy','skfem','gmsh','ezdxf','safetensors'}}:
            raise AssertionError('Unexpected numerical import: ' + fullname)
sys.meta_path.insert(0, Deny())
from pathlib import Path
from piginot.cad.configuration import prepare_campaign
receipt = prepare_campaign(Path({str(project)!r}), Path({str(setting)!r}), Path({str(output)!r}))
assert receipt['profile_status'] == 'METADATA_ONLY'
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("change", ["batch", "initializer"])
def test_incompatible_training_and_reference_settings_refuse(prepared, change):
    project, setting, output = prepared
    document = settings()
    if change == "batch":
        document["training"]["case_batch_size"] = 3
    else:
        document["reference"] = {"initialization": "neo-hookean-full-load-hgo"}
    save_settings(setting, document)
    with pytest.raises(ValueError, match="batch|HGO"):
        prepare_campaign(project, setting, output)
    assert not output.exists()


def test_cross_drive_resource_reference_uses_explicit_absolute_path(
    prepared, monkeypatch
):
    from piginot.cad import configuration

    project, setting, output = prepared

    def different_drive(*args):
        raise ValueError("different drive")

    monkeypatch.setattr(configuration.os.path, "relpath", different_drive)
    prepare_campaign(project, setting, output)
    config = CampaignConfig.load(output / "config.json")
    assert config.adapter.geometry_resource.path == str(project / "geometry.json")


def test_changed_settings_are_refused_before_publishing(prepared, monkeypatch):
    from piginot.cad import configuration

    project, setting, output = prepared
    receipt = configuration._receipt

    def changed(*args, **kwargs):
        result = receipt(*args, **kwargs)
        setting.write_bytes(b"changed")
        return result

    monkeypatch.setattr(configuration, "_receipt", changed)
    with pytest.raises(ValueError, match="settings changed"):
        prepare_campaign(project, setting, output)
    assert not output.exists()


def test_cpu_capability_is_captured_from_the_backend(monkeypatch):
    from types import SimpleNamespace

    from piginot.cad import configuration

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                cpu=SimpleNamespace(get_cpu_capability=lambda: "TEST_CPU")
            )
        ),
    )
    assert configuration._cpu_capability() == "TEST_CPU"


def test_preparation_cannot_write_into_the_installed_adapter_package(
    prepared, monkeypatch, tmp_path
):
    import piginot_campaigns
    from piginot.cad import project as geometry_project

    project, setting, _ = prepared
    installed = tmp_path / "installed-adapters"
    monkeypatch.setattr(piginot_campaigns, "__file__", str(installed / "__init__.py"))
    output = installed / "new-output"
    with pytest.raises(ValueError, match="package"):
        prepare_campaign(project, setting, output)
    with pytest.raises(ValueError, match="package"):
        geometry_project._output_path(setting, output)
    assert not output.exists()


@pytest.mark.parametrize("invalid_constraint", [False, True])
def test_case_sampling_has_a_finite_budget_for_duplicates_and_math_failures(
    monkeypatch, invalid_constraint
):
    from types import SimpleNamespace

    from piginot.cad import cases

    monkeypatch.setattr(cases, "ATTEMPTS_PER_CASE", 2)

    def fail(*args):
        raise ZeroDivisionError("invalid constraint")

    monkeypatch.setattr(cases, "_evaluate_tree", fail)
    spec = SimpleNamespace(
        name="degenerate",
        parameters=(SimpleNamespace(name="u", lo=1, hi=1),),
        constraints=(SimpleNamespace(name="bad", expression=SimpleNamespace(tree=())),)
        if invalid_constraint
        else (),
        scope=lambda values: values,
    )
    with pytest.raises(ValueError, match="exhausted"):
        cases.generate_cases(spec, 2, 1, 0)
