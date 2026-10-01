from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s10_requests import _load_public_sources
from campaigns.dogbone_v1.s11_contingency_source import (
    VerifiedS10Request,
    _rebuild_request,
)
from campaigns.dogbone_v1.s13c_data import (
    DogboneS13cEnergyObjective,
    DogboneS13cGeometryCase,
    _loaded_end_displacement,
    case_energy,
    conditioning_vector,
    geometry_case,
    materialize_dogbone_s13c_training_data,
)
from campaigns.dogbone_v1.s13c_protocol import (
    build_dogbone_s13c_protocol,
    build_initial_operator,
    energy_config,
    load_s13c_normalization,
)
from campaigns.dogbone_v1.strain_pilot import prescribed_displacement_mm
from piginot.contracts import BoundaryConditionSpec, ExecutionProfile
from piginot.doe import EvidenceRole

_PROFILE = ExecutionProfile(
    python_version="3.11.0",
    library_versions={"torch": "test"},
    blas_threads=1,
    device="cpu",
    dtype="float64",
    rng_algorithm="none",
    seed=0,
    container_digest=None,
)


@pytest.fixture(scope="module")
def requests() -> tuple[VerifiedS10Request, ...]:
    protocol = build_dogbone_s13c_protocol()
    _, manifest = _load_public_sources()
    cases = {case.case_id: case for case in manifest.cases}
    family = compiled_family()
    chosen = (protocol.fit_case_ids[0], protocol.fit_case_ids[-1])
    return tuple(
        VerifiedS10Request(
            cases[case_id], _rebuild_request(cases[case_id], _PROFILE, family)
        )
        for case_id in chosen
    )


def _tagged_nodes(case: DogboneS13cGeometryCase, tag: str) -> torch.Tensor:
    mesh = case.request.mesh
    node_ids = sorted(
        {
            node
            for facet in mesh.semantic_tags[tag]
            for node in mesh.boundary_facets[facet]
        }
    )
    return case.domain.vertices[node_ids]


def test_geometry_case_reduces_a_public_train_case(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    normalization = load_s13c_normalization()
    for verified in requests:
        case = geometry_case(verified, normalization, dtype=torch.float64)
        coordinates = verified.case.primitive_coordinates

        assert verified.case.role is EvidenceRole.TRAIN
        assert case.half_length_mm == coordinates["L_total"] / 2.0
        assert case.half_grip_width_mm == coordinates["W_grip"] / 2.0
        assert case.prescribed_displacement_mm == prescribed_displacement_mm(
            coordinates["L_total"], coordinates["nominal_axial_strain"]
        )
        assert case.material.model == "compressible_neo_hookean"
        assert case.domain.cells.shape == (1792, 3)
        assert case.domain.area > 0.0
        assert case.conditioning.shape == (5,)


def test_lifting_reproduces_every_quarter_model_boundary_condition(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    normalization = load_s13c_normalization()
    model = build_initial_operator().to(torch.float64)
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.5)
    for verified in requests:
        case = geometry_case(verified, normalization, dtype=torch.float64)

        def displacement(
            points: torch.Tensor, case: DogboneS13cGeometryCase = case
        ) -> torch.Tensor:
            return model.displacement(
                query_xy=points.unsqueeze(0),
                parameters=case.parameters(),
                frame=case.frame(),
                lifting=case.lifting(),
            )[0].detach()

        left = displacement(_tagged_nodes(case, "left_symmetry"))
        bottom = displacement(_tagged_nodes(case, "bottom"))
        right = displacement(_tagged_nodes(case, "right_grip"))
        free = displacement(_tagged_nodes(case, "right_arc"))

        assert torch.equal(left[:, 0], torch.zeros_like(left[:, 0]))
        assert torch.equal(bottom[:, 1], torch.zeros_like(bottom[:, 1]))
        assert torch.allclose(
            right[:, 0],
            torch.full_like(right[:, 0], case.prescribed_displacement_mm),
            rtol=0.0,
            atol=1e-12,
        )
        assert float(free[:, 1].abs().max()) > 0.0


def test_conditioning_uses_frozen_train_z_scores() -> None:
    normalization = load_s13c_normalization()
    index = {name: i for i, name in enumerate(normalization.parameter_names)}
    coordinates = {
        "L_total": normalization.mean[index["L_total"]],
        "W_grip": normalization.mean[index["W_grip"]]
        + normalization.standard_deviation[index["W_grip"]],
        "W_gauge": normalization.mean[index["W_gauge"]],
        "R_fillet": normalization.mean[index["R_fillet"]],
        "nominal_axial_strain": normalization.mean[index["nominal_axial_strain"]],
    }

    vector = conditioning_vector(coordinates, normalization)

    assert torch.allclose(
        vector, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="R_fillet"):
        conditioning_vector(
            {key: value for key, value in coordinates.items() if key != "R_fillet"},
            normalization,
        )
    with pytest.raises(TypeError):
        conditioning_vector(coordinates, object())  # type: ignore[arg-type]


def test_loaded_end_rule_rejects_changed_boundary_conditions(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    request = requests[0].request
    conditions = list(request.boundary_conditions)
    conditions[1] = BoundaryConditionSpec(
        "bottom_y_symmetry", "bottom", "symmetry", ("x",)
    )
    with pytest.raises(ValueError, match="quarter-model"):
        _loaded_end_displacement(
            replace(request, boundary_conditions=tuple(conditions))
        )


def test_energy_objective_is_finite_and_train_only(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    normalization = load_s13c_normalization()
    case = geometry_case(requests[0], normalization, dtype=torch.float32)
    objective = DogboneS13cEnergyObjective((case,), energy_config())
    model = build_initial_operator()

    terms = case_energy(
        model,
        case,
        energy_config(),
        generator=torch.Generator().manual_seed(0),
        create_graph=False,
    )
    assert torch.isfinite(terms.total)
    assert terms.inadmissible_points == 0
    assert terms.minimum_determinant > 1.0
    with pytest.raises(PermissionError):
        objective(model, "candidate-unregistered", 1)
    with pytest.raises(TypeError):
        objective(torch.nn.Linear(1, 1), case.case_id, 1)
    with pytest.raises(ValueError):
        DogboneS13cEnergyObjective((), energy_config())
    with pytest.raises(TypeError):
        DogboneS13cEnergyObjective((case,), object())  # type: ignore[arg-type]


def test_materialization_requires_the_verified_source() -> None:
    with pytest.raises(TypeError):
        materialize_dogbone_s13c_training_data(
            object(),  # type: ignore[arg-type]
            build_dogbone_s13c_protocol(),
            load_s13c_normalization(),
        )


def test_geometry_case_rejects_inconsistent_values(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    case = geometry_case(requests[0], load_s13c_normalization(), dtype=torch.float64)
    with pytest.raises(ValueError):
        replace(case, conditioning=torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError):
        replace(case, half_length_mm=-1.0)
    with pytest.raises(ValueError):
        replace(case, case_id="other")
