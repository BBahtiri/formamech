from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from piginot.cad.mesh_policy import FeatureSize, MeshPolicy, PointSize
from piginot.contracts import canonical_json_bytes


@pytest.fixture
def policy() -> MeshPolicy:
    return MeshPolicy(
        global_size_mm=1.5,
        minimum_size_mm=0.1,
        curve_tolerance_mm=0.01,
        curvature_points_per_circle=24,
        feature_sizes=(FeatureSize("edge-000004", 0.3, 0.5, 3.0),),
        point_sizes=(PointSize((6.0, 8.0), 0.2, 0.0, 2.0),),
        minimum_angle_deg=20.0,
        maximum_aspect_ratio=5.0,
        maximum_triangles=50000,
        maximum_p2_dofs=200000,
    )


def test_policy_roundtrip_preserves_explicit_settings_and_content_hash(policy):
    document = policy.document()
    assert document["schema"] == "piginot.cad-mesh-policy-v1"
    assert document["feature_sizes"][0] == {
        "boundary": "edge-000004",
        "size_mm": 0.3,
        "distance_min_mm": 0.5,
        "distance_max_mm": 3.0,
    }
    assert document["point_sizes"][0]["center_mm"] == [6.0, 8.0]
    decoded = MeshPolicy.from_document(json.loads(json.dumps(document)))
    assert decoded == policy
    assert decoded.sha256 == hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    assert replace(policy, curve_tolerance_mm=0.02).sha256 != policy.sha256


def test_policy_and_nested_targets_are_immutable_without_retaining_mutable_inputs(
    policy,
):
    with pytest.raises(FrozenInstanceError):
        policy.global_size_mm = 2.0
    with pytest.raises(FrozenInstanceError):
        policy.feature_sizes[0].size_mm = 0.2
    document = policy.document()
    decoded = MeshPolicy.from_document(document)
    document["feature_sizes"][0]["size_mm"] = 100.0
    document["point_sizes"][0]["center_mm"][0] = 100.0
    assert decoded == policy


@pytest.mark.parametrize(
    "field",
    [
        "global_size_mm",
        "minimum_size_mm",
        "curve_tolerance_mm",
        "minimum_angle_deg",
        "maximum_aspect_ratio",
    ],
)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1", None])
def test_invalid_scalar_settings_fail_before_a_mesher_can_be_loaded(
    policy, field, value
):
    with pytest.raises(ValueError, match=field):
        replace(policy, **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "curvature_points_per_circle",
        "maximum_triangles",
        "maximum_p2_dofs",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 8.0, True, "8", None])
def test_budgets_and_curvature_samples_require_positive_real_integers(
    policy, field, value
):
    with pytest.raises(ValueError, match=field):
        replace(policy, **{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_size_mm": 2.0},
        {"curvature_points_per_circle": 7},
        {"minimum_angle_deg": 61},
        {"feature_sizes": (FeatureSize("edge-000004", 2.0, 0.0, 1.0),)},
        {"feature_sizes": (FeatureSize("edge-000004", 0.05, 0.0, 1.0),)},
        {"point_sizes": (PointSize((0.0, 0.0), 2.0, 0.0, 1.0),)},
        {"point_sizes": (PointSize((0.0, 0.0), 0.05, 0.0, 1.0),)},
    ],
)
def test_inconsistent_geometry_and_size_limits_are_refused(policy, changes):
    with pytest.raises(ValueError):
        replace(policy, **changes)


@pytest.mark.parametrize("target", [FeatureSize, PointSize])
@pytest.mark.parametrize(
    "field,value",
    [
        ("size_mm", 0),
        ("size_mm", True),
        ("size_mm", float("inf")),
        ("distance_min_mm", -1),
        ("distance_min_mm", True),
        ("distance_min_mm", float("nan")),
        ("distance_max_mm", 0),
        ("distance_max_mm", True),
        ("distance_max_mm", float("inf")),
    ],
)
def test_target_size_and_transition_ranges_are_validated(target, field, value):
    args = {"size_mm": 0.2, "distance_min_mm": 0.0, "distance_max_mm": 1.0}
    args[field] = value
    args["boundary" if target is FeatureSize else "center_mm"] = (
        "edge-000001" if target is FeatureSize else (0.0, 0.0)
    )
    with pytest.raises(ValueError):
        target(**args)


def test_transition_must_reach_bulk_beyond_its_inner_radius():
    with pytest.raises(ValueError, match="distance_max_mm"):
        FeatureSize("edge-000001", 0.2, 2.0, 1.0)
    with pytest.raises(ValueError, match="distance_max_mm"):
        PointSize((0.0, 0.0), 0.2, 2.0, 2.0)


@pytest.mark.parametrize("boundary", ["", "left", "edge-1", "edge-000001x", 1, None])
def test_feature_selection_uses_the_imported_boundary_vocabulary(boundary):
    with pytest.raises(ValueError, match="boundary"):
        FeatureSize(boundary, 0.2, 0.0, 1.0)


@pytest.mark.parametrize(
    "center",
    [
        [0.0, 1.0],
        (),
        (0.0,),
        (0.0, 1.0, 2.0),
        "xy",
        None,
        (True, 1.0),
        (0.0, float("nan")),
        (float("inf"), 1.0),
    ],
)
def test_point_targets_are_finite_immutable_two_dimensional_coordinates(center):
    with pytest.raises(ValueError, match="center_mm"):
        PointSize(center, 0.2, 0.0, 1.0)


@pytest.mark.parametrize("field", ["feature_sizes", "point_sizes"])
@pytest.mark.parametrize("value", [[], None, ({},), ("invalid",)])
def test_direct_policy_construction_requires_immutable_typed_targets(
    policy, field, value
):
    with pytest.raises(ValueError, match=field):
        replace(policy, **{field: value})


def test_duplicate_selectors_are_rejected_and_empty_target_sets_are_valid(policy):
    with pytest.raises(ValueError, match="duplicate"):
        replace(policy, feature_sizes=policy.feature_sizes * 2)
    with pytest.raises(ValueError, match="duplicate"):
        replace(policy, point_sizes=policy.point_sizes * 2)
    empty = replace(policy, feature_sizes=(), point_sizes=())
    assert MeshPolicy.from_document(empty.document()) == empty


@pytest.mark.parametrize("location", ["root", "feature", "point"])
@pytest.mark.parametrize("change", ["unknown", "missing", "not-object"])
def test_json_vocabulary_is_strict_at_every_level(policy, location, change):
    document = policy.document()
    selected = document if location == "root" else document[location + "_sizes"][0]
    if change == "unknown":
        selected["typo"] = 1
    elif change == "missing":
        selected.pop(next(iter(selected)))
    elif location == "root":
        document = []
    else:
        document[location + "_sizes"][0] = 1
    with pytest.raises(ValueError):
        MeshPolicy.from_document(document)


def test_json_schema_and_array_types_are_strict(policy):
    document = policy.document()
    document["schema"] = "future-schema"
    with pytest.raises(ValueError, match="schema"):
        MeshPolicy.from_document(document)
    for field in ("feature_sizes", "point_sizes"):
        document = policy.document()
        document[field] = tuple(document[field])
        with pytest.raises(ValueError, match=field):
            MeshPolicy.from_document(document)
    document = policy.document()
    document["point_sizes"][0]["center_mm"] = "bad"
    with pytest.raises(ValueError, match="center_mm"):
        MeshPolicy.from_document(document)


@pytest.mark.parametrize("center", [[], [0.0], [0.0, 1.0, 2.0]])
def test_json_point_targets_reject_missing_or_three_dimensional_coordinates(
    policy, center
):
    document = policy.document()
    document["point_sizes"][0]["center_mm"] = center
    with pytest.raises(ValueError, match="center_mm"):
        MeshPolicy.from_document(document)


def test_extreme_integer_lengths_are_reported_as_invalid_policy_inputs(policy):
    with pytest.raises(ValueError, match="global_size_mm"):
        replace(policy, global_size_mm=10**1000)


def test_loading_uses_strict_json_and_rejects_missing_files_and_links(policy, tmp_path):
    path = tmp_path / "mesh-policy.json"
    with pytest.raises(ValueError, match="regular"):
        MeshPolicy.load(path)
    with pytest.raises(ValueError, match="regular"):
        MeshPolicy.load(tmp_path)
    path.write_bytes(canonical_json_bytes(policy.document()))
    assert MeshPolicy.load(path) == policy
    link = tmp_path / "policy-link.json"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("OS cannot create symlinks")
    with pytest.raises(ValueError, match="regular"):
        MeshPolicy.load(link)
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError):
        MeshPolicy.load(path)


def test_policy_metadata_import_does_not_load_the_numerical_stack():
    root = Path(__file__).resolve().parents[2]
    script = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(root / "src")!r})
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'gmsh', 'ezdxf', 'numpy', 'torch', 'scipy', 'skfem'}}:
            raise AssertionError('Numerical import during metadata: ' + fullname)
sys.meta_path.insert(0, Deny())
from piginot.cad.mesh_policy import FeatureSize, MeshPolicy, PointSize
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
