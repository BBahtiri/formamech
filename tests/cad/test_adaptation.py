"""Bounded preparation uses only declared pilots and retains every attempt."""

from dataclasses import FrozenInstanceError, replace

import pytest

from piginot.cad.adaptation import (
    PilotCase,
    PilotStress,
    adapt_mesh,
    stress_jump_indicator,
)
from piginot.cad.mesh_policy import MeshPolicy, PointSize
from piginot.cad.refinement import refine_fixed_mesh
from piginot.contracts import Unit, canonical_json_bytes
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh


def policy(**changes):
    return replace(
        MeshPolicy(
            global_size_mm=1.0,
            minimum_size_mm=0.1,
            curve_tolerance_mm=0.01,
            curvature_points_per_circle=16,
            feature_sizes=(),
            point_sizes=(),
            minimum_angle_deg=15.0,
            maximum_aspect_ratio=10.0,
            maximum_triangles=1000,
            maximum_p2_dofs=10000,
        ),
        **changes,
    )


def rectangle(current=None, level=1):
    mesh = canonicalize_mesh(
        MeshPayload(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            ((0, 1, 2), (0, 2, 3)),
            CellType.TRIANGLE,
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            {"edge-000001": (0, 1, 2, 3)},
            (1, 1),
            Unit.MM,
            {},
        )
    )
    mesh = refine_fixed_mesh(mesh, level)
    current = policy() if current is None else current
    return replace(
        mesh,
        source_provenance={
            "source_sha256": "1" * 64,
            "mesh_policy_sha256": current.sha256,
        },
    )


def stresses(mesh, jump=False):
    return tuple(
        ((2.0 if jump and i == 0 else 1.0, 0.0), (0.0, 1.0))
        for i in range(len(mesh.cells))
    )


def pilots():
    return (PilotCase("pilot-1", "TRAIN-pilot", {"ux": 0.2}),)


def run(**changes):
    settings = {
        "policy": policy(),
        "pilot_cases": pilots(),
        "screen_case_ids": ("screen-1",),
        "remesh": lambda p: rectangle(p),
        "evaluate": lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh)
        ),
        "max_rounds": 2,
        "max_solves": 2,
        "indicator_target": 0.1,
        "mark_fraction": 0.5,
        "refinement_factor": 2.0,
        "max_markers": 4,
    }
    return adapt_mesh(**{**settings, **changes})


def test_uniform_and_zero_stress_have_no_interior_traction_jump():
    mesh = rectangle()
    assert stress_jump_indicator(mesh, stresses(mesh)) == (0.0, 0.0)
    assert stress_jump_indicator(mesh, (((0.0, 0.0), (0.0, 0.0)),) * 2) == (
        0.0,
        0.0,
    )


def test_jump_indicator_is_local_nonzero_and_scale_invariant():
    mesh = rectangle()
    values = stress_jump_indicator(mesh, stresses(mesh, jump=True))
    assert values[0] == pytest.approx(values[1])
    assert values[0] > 0
    scaled = tuple(
        tuple(tuple(1000 * value for value in row) for row in cell)
        for cell in stresses(mesh, jump=True)
    )
    assert stress_jump_indicator(mesh, scaled) == pytest.approx(values)


def test_large_finite_stress_does_not_overflow_into_false_zero_indicators():
    mesh = rectangle()
    scaled = tuple(
        tuple(tuple(1e300 * value for value in row) for row in cell)
        for cell in stresses(mesh, jump=True)
    )
    assert stress_jump_indicator(mesh, scaled) == pytest.approx(
        stress_jump_indicator(mesh, stresses(mesh, jump=True))
    )


def test_tangential_stress_difference_has_zero_traction_jump():
    # The shared diagonal is parallel to (1, 1): each row of the stress
    # difference is tangential, so its product with the edge normal is zero.
    assert stress_jump_indicator(
        rectangle(), (((1.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (0.0, 0.0)))
    ) == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize(
    "values",
    [
        (),
        (((1, 0), (0, 1)),),
        (((True, 0), (0, 1)),) * 2,
        (((float("nan"), 0), (0, 1)),) * 2,
        (((1, 0, 0), (0, 1, 0)),) * 2,
    ],
)
def test_invalid_cell_stress_is_refused(values):
    with pytest.raises(ValueError, match="stress"):
        stress_jump_indicator(rectangle(), values)


def test_invalid_mesh_identity_is_refused():
    with pytest.raises(ValueError, match="identity"):
        stress_jump_indicator(
            replace(rectangle(), mesh_id="0" * 64), stresses(rectangle())
        )


def test_pilot_case_copies_mutable_inputs_and_qualifies_training_role():
    values = {"ux": 0.2}
    case = PilotCase("pilot-1", "TRAIN-pilot", values)
    before = canonical_json_bytes(case.document())
    values["ux"] = 99
    assert canonical_json_bytes(case.document()) == before
    with pytest.raises(TypeError):
        case.parameters["ux"] = 1
    with pytest.raises(FrozenInstanceError):
        case.partition = "HOLDOUT"
    assert PilotStress("pilot-1", "a" * 64, stresses(rectangle())).sha256


def test_large_pilot_stress_layout_is_copied_into_immutable_values():
    cell = [[1.0, 0.0], [0.0, 1.0]]
    values = [cell] * 75000
    pilot = PilotStress("pilot-1", "a" * 64, values)
    cell[0][0] = 99.0
    values.pop()
    assert len(pilot.cell_stress) == 75000
    assert pilot.cell_stress[0] == pilot.cell_stress[-1] == ((1.0, 0.0), (0.0, 1.0))
    assert isinstance(pilot.cell_stress, tuple)


def test_attempt_document_exposes_source_identity_for_geometry_binding():
    document = run().document()
    assert document["attempts"][0]["source_sha256"] == "1" * 64
    failed = run(remesh=lambda p: None).document()
    assert failed["attempts"][0]["source_sha256"] is None


@pytest.mark.parametrize("partition", ["TRAIN-screen", "DEVELOPMENT", "HOLDOUT"])
def test_non_pilot_partitions_are_refused(partition):
    with pytest.raises(ValueError, match="TRAIN-pilot"):
        PilotCase("pilot-1", partition, {"ux": 0.2})


@pytest.mark.parametrize("parameters", [{"ux": True}, {"ux": float("inf")}, {"": 1}])
def test_invalid_pilot_parameters_are_refused(parameters):
    with pytest.raises(ValueError, match="parameter"):
        PilotCase("pilot-1", "TRAIN-pilot", parameters)


def test_target_met_is_explicitly_only_a_heuristic_result():
    result = run()
    assert result.status == "HEURISTIC_TARGET_MET"
    assert result.solves_used == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].indicators == (0.0, 0.0)
    document = result.document()
    assert document["error_bound"] == "NOT_ESTABLISHED"
    assert document["indicator"] == "relative-cell-average-interior-traction-jump-v1"
    assert document["pilot_cases"][0]["parameters"] == {"ux": 0.2}
    assert document["attempts"][0]["mesh_id"] == rectangle().mesh_id
    assert result.sha256 == run().sha256


def test_unresolved_cells_produce_bounded_point_fields_then_stop_on_target():
    seen = []

    def remesh(current):
        seen.append(current)
        return rectangle(current, level=len(seen))

    def evaluate(mesh, case):
        return PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, jump=len(seen) == 1)
        )

    initial = policy()
    result = run(policy=initial, remesh=remesh, evaluate=evaluate)
    assert result.status == "HEURISTIC_TARGET_MET"
    assert result.solves_used == 2
    assert len(result.attempts) == 2
    assert seen[0] == initial
    assert not initial.point_sizes
    assert len(seen[1].point_sizes) == 1
    marker = seen[1].point_sizes[0]
    assert marker.size_mm < initial.global_size_mm
    assert marker.size_mm >= initial.minimum_size_mm
    assert marker.distance_max_mm > marker.distance_min_mm
    assert result.attempts[0].pilots[0].cell_stress == stresses(rectangle(), jump=True)


def test_all_pilots_are_evaluated_and_worst_cell_indicator_drives_marking():
    cases = (*pilots(), PilotCase("pilot-2", "TRAIN-pilot", {"ux": 0.4}))
    result = run(
        pilot_cases=cases,
        evaluate=lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, jump=case.case_id == "pilot-2")
        ),
        max_rounds=1,
    )
    assert result.status == "ROUND_LIMIT"
    assert result.solves_used == 2
    assert len(result.attempts[0].pilots) == 2
    assert max(result.attempts[0].indicators) > 0.1


def test_screen_overlap_is_refused_before_any_callback():
    with pytest.raises(ValueError, match="overlap"):
        run(screen_case_ids=("pilot-1",), remesh=lambda p: pytest.fail("must not mesh"))


def test_empty_or_duplicate_pilot_populations_are_refused():
    with pytest.raises(ValueError, match="pilot"):
        run(pilot_cases=())
    with pytest.raises(ValueError, match="pilot"):
        run(pilot_cases=pilots() * 2)


@pytest.mark.parametrize(
    "change",
    [
        {"max_rounds": True},
        {"max_solves": 0},
        {"max_markers": 0},
        {"mark_fraction": 0},
        {"mark_fraction": 1.1},
        {"indicator_target": -1},
        {"indicator_target": float("nan")},
        {"refinement_factor": 1},
        {"refinement_factor": True},
    ],
)
def test_invalid_budgets_are_refused_before_callbacks(change):
    with pytest.raises(ValueError):
        run(**change, remesh=lambda p: pytest.fail("must not mesh"))


def test_solve_budget_stops_before_a_round_that_cannot_finish_all_pilots():
    result = run(
        pilot_cases=(*pilots(), PilotCase("pilot-2", "TRAIN-pilot", {"ux": 0.4})),
        max_solves=1,
        remesh=lambda p: pytest.fail("must not mesh"),
    )
    assert result.status == "SOLVE_LIMIT"
    assert result.solves_used == 0
    assert not result.attempts


def test_remaining_solve_budget_stops_before_remeshing():
    calls = []

    def remesh(current):
        calls.append(current)
        return rectangle(current)

    result = run(
        max_solves=1,
        remesh=remesh,
        evaluate=lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, True)
        ),
    )
    assert result.status == "SOLVE_LIMIT"
    assert len(calls) == result.solves_used == 1


@pytest.mark.parametrize(
    "limits",
    [{"maximum_triangles": 31}, {"maximum_p2_dofs": 161}],
)
def test_largest_reference_mesh_limits_stop_before_pilot_solve(limits):
    result = run(
        policy=policy(**limits),
        evaluate=lambda *args: pytest.fail("must not solve"),
    )
    assert result.status == "MESH_LIMIT"
    assert result.solves_used == 0
    assert len(result.attempts) == 1


def test_largest_reference_mesh_limits_allow_exact_boundary():
    # A unit square split into two triangles has 32 triangles and 81 P2
    # scalar DOFs at reference level 4, hence 162 vector displacement DOFs.
    assert (
        run(policy=policy(maximum_triangles=32, maximum_p2_dofs=162)).status
        == "HEURISTIC_TARGET_MET"
    )


def test_callback_failures_are_recorded_once_without_retries():
    def failed(*args):
        raise RuntimeError("visible pilot failure")

    result = run(remesh=failed)
    assert result.status == "REMESH_FAILURE"
    assert result.solves_used == 0
    assert "visible pilot failure" in result.attempts[0].reason
    result = run(evaluate=failed)
    assert result.status == "PILOT_FAILURE"
    assert result.solves_used == 1
    assert "visible pilot failure" in result.attempts[0].reason


def test_invalid_mesher_return_is_retained_as_a_serializable_failure():
    result = run(remesh=lambda p: "not a canonical mesh")
    assert result.status == "REMESH_FAILURE"
    assert result.solves_used == 0
    assert result.attempts[0].mesh is None
    assert result.document()["attempts"][0]["mesh_id"] is None


def test_valid_but_different_cad_source_is_refused_after_first_round():
    count = 0

    def remesh(current):
        nonlocal count
        count += 1
        mesh = rectangle(current, level=count)
        if count == 2:
            return replace(
                mesh,
                source_provenance={**mesh.source_provenance, "source_sha256": "2" * 64},
            )
        return mesh

    result = run(
        remesh=remesh,
        evaluate=lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, True)
        ),
    )
    assert result.status == "REMESH_FAILURE"
    assert result.solves_used == 1
    assert len(result.attempts) == 2
    assert result.attempts[0].pilots
    assert "source" in result.attempts[1].reason


@pytest.mark.parametrize(
    "change",
    [{"policy": None}, {"remesh": None}, {"evaluate": None}],
)
def test_invalid_callback_configuration_is_refused(change):
    with pytest.raises(ValueError, match="MeshPolicy"):
        run(**change)


def test_existing_point_fields_must_fit_the_declared_marker_budget():
    initial = policy(
        point_sizes=(
            PointSize((0.2, 0.2), 0.2, 0.1, 0.5),
            PointSize((0.8, 0.8), 0.2, 0.1, 0.5),
        )
    )
    with pytest.raises(ValueError, match="max_markers"):
        run(policy=initial, max_markers=1)


@pytest.mark.parametrize("wrong", ["case", "mesh", "count", "type"])
def test_pilot_callback_must_return_the_requested_case_mesh_and_layout(wrong):
    def evaluate(mesh, case):
        if wrong == "type":
            return None
        return PilotStress(
            "another" if wrong == "case" else case.case_id,
            "0" * 64 if wrong == "mesh" else mesh.mesh_id,
            stresses(mesh)[:1] if wrong == "count" else stresses(mesh),
        )

    result = run(evaluate=evaluate)
    assert result.status == "PILOT_FAILURE"
    assert result.solves_used == 1


@pytest.mark.parametrize("wrong", ["source", "policy", "tags", "topology"])
def test_remesh_must_preserve_source_binding_topology_and_boundary_vocabulary(wrong):
    count = 0

    def remesh(current):
        nonlocal count
        count += 1
        mesh = rectangle(current, level=count)
        if wrong == "source":
            return replace(
                mesh, source_provenance={**mesh.source_provenance, "source_sha256": "x"}
            )
        if wrong == "policy":
            return replace(
                mesh,
                source_provenance={
                    **mesh.source_provenance,
                    "mesh_policy_sha256": "0" * 64,
                },
            )
        if count == 2:
            if wrong == "tags":
                return replace(
                    mesh,
                    semantic_tags={"another": tuple(range(len(mesh.boundary_facets)))},
                )
            return replace(mesh, topology_signature="different")
        return mesh

    result = run(
        remesh=remesh,
        evaluate=lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, True)
        ),
    )
    assert result.status == "REMESH_FAILURE"
    assert result.solves_used == (0 if wrong in ("source", "policy") else 1)


def test_unchanged_mesh_stops_before_duplicate_pilot_solves():
    result = run(
        evaluate=lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, True)
        )
    )
    assert result.status == "STALLED"
    assert len(result.attempts) == 2
    assert result.solves_used == 1


def test_point_field_budget_is_finite_and_explicit():
    initial = policy(point_sizes=(PointSize((0.25, 0.25), 0.2, 0.1, 0.5),))
    result = run(
        policy=initial,
        max_markers=1,
        evaluate=lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, True)
        ),
    )
    assert result.status == "MARKER_LIMIT"
    assert result.solves_used == 1


def test_existing_marker_can_refine_further_without_consuming_another_slot():
    mesh = rectangle()
    center = tuple(
        sum(mesh.nodes[i][axis] for i in mesh.cells[0]) / 3 for axis in (0, 1)
    )
    initial = policy(point_sizes=(PointSize(center, 0.8, 0.2, 2.0),))
    seen = []

    def remesh(current):
        seen.append(current)
        return rectangle(current, level=len(seen))

    result = run(
        policy=initial,
        max_markers=1,
        remesh=remesh,
        evaluate=lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, jump=len(seen) == 1)
        ),
    )
    assert result.status == "HEURISTIC_TARGET_MET"
    assert len(seen[1].point_sizes) == 1
    assert seen[1].point_sizes[0].size_mm == pytest.approx(0.4)


def test_minimum_size_does_not_create_an_unchanged_policy_loop():
    result = run(
        policy=policy(minimum_size_mm=1.0),
        evaluate=lambda mesh, case: PilotStress(
            case.case_id, mesh.mesh_id, stresses(mesh, True)
        ),
    )
    assert result.status == "MINIMUM_SIZE_REACHED"
    assert result.solves_used == 1
