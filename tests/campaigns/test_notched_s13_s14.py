"""Notched S13 close and S14 pieces that need no trained candidate."""

from __future__ import annotations

from campaigns.dogbone_v1.s14_candidate import S14Case
from campaigns.dogbone_v1.s14_reference import nested_vertex_map
from campaigns.notched_v1 import family, s13_close, s14
from piginot.operators import ENERGY_G11_MINIMUM_PARAMETER_VECTORS
from piginot.verification import REGISTERED_FAMILIES
from tests.platform_support import requires_pinned_initial_weights


def test_the_g11_lifting_population_covers_the_box() -> None:
    vectors = s13_close.lifting_parameter_vectors()
    assert len(vectors) == ENERGY_G11_MINIMUM_PARAMETER_VECTORS == 64
    corners = {tuple(vector[n] for n in family.INPUTS) for vector in vectors[:16]}
    assert len(corners) == 16
    for vector in vectors:
        for name, value in vector.items():
            lower, upper = family.BOUNDS[name]
            assert lower <= value <= upper


def test_the_root_vertices_are_the_notch_profile() -> None:
    mesh = family.compiled_family().reference_mesh
    root = s13_close.root_vertices(mesh)
    assert len(root) == 25
    assert min(mesh.nodes[i][0] for i in root) == 0.0


def test_the_coarse_level_nests_in_the_production_mesh() -> None:
    case = S14Case(
        "probe",
        "TRAIN",
        "probe",
        {"width": 20.0, "notch_radius": 3.0, "notch_depth": 4.0},
        {},
        {"width": 0},
    )
    coarse, fine = s14.level_mesh(case, 0.5), s14.level_mesh(case, 1.0)
    mapped = nested_vertex_map(coarse, fine)
    assert len(mapped) == len(coarse.nodes) == 225 and len(fine.nodes) == 833
    assert len(s13_close.root_vertices(coarse)) == 13


def test_slices_and_families_are_campaign_scoped() -> None:
    assert len(s14.GATED_SLICES) == 17 and len(s14.REPORTED_SLICES) == 9
    families = {f.family_id: f for f in s14.campaign_families()}
    assert len(families) == len(REGISTERED_FAMILIES) == 12
    for family_id in ("F05", "F08", "F09", "F10", "F12"):
        assert families[family_id].artifact.startswith(
            "artifacts/campaigns/notched-v1/"
        )
    assert families["F01"].artifact == "artifacts/gates/S2.json"
    assert set(s14.EVALUATION_FILES) >= {
        "campaigns/notched_v1/s14.py",
        "src/piginot/verification/budget.py",
    }


def test_the_dataset_identity_does_not_follow_the_live_environment() -> None:
    """The S17 request plan and snapshot rebuild on the 0.2.0 line (piginot
    0.2.0.dev0, any Python patch release): the plan binds the S10 profile."""

    import json
    from pathlib import Path

    from campaigns.notched_v1 import s13
    from campaigns.notched_v1.s10 import request_plan

    root = Path(s13.__file__).resolve().parents[2]
    committed = json.loads((root / s13.PROTOCOL_PATH).read_bytes())
    assert request_plan(s13.S10_PROFILE).content_hash == committed["request_plan_hash"]
    protocol_fit = tuple(committed["fit_case_ids"])
    assert (
        s13.geometry_snapshot_hash(protocol_fit, s13.S10_PROFILE)
        == committed["dataset_snapshot_hash"]
    )
    assert s13.S10_PROFILE.library_versions["piginot"] == "0.1.0"


@requires_pinned_initial_weights
def test_the_notched_protocol_rebuilds_on_the_current_line() -> None:
    from campaigns.notched_v1 import s13

    protocol = s13.load_committed_protocol()
    assert len(s13.training_cases(protocol)) == len(protocol.fit_case_ids)
