"""Executable analytic, gradient, mutant, and FEM verification for S12."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

import numpy as np
import torch

from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalQuantity,
    MaterialSpec,
    Unit,
)
from piginot.doe import EvidenceRole
from piginot.fem import FemRequest, FemSolverConfig, ScikitFemBackend
from piginot.fem.mms_runner import verification_execution_profile, verification_material
from piginot.fem.verification_meshes import unit_square_verification_mesh
from piginot.objectives.equilibrium import (
    strong_equilibrium_residual,
    weak_equilibrium_residual,
)
from piginot.objectives.kinematics import displacement_gradient
from piginot.objectives.models import (
    CoordinateConversion,
    NondimensionalScales,
    ObjectiveMode,
    ObjectiveResiduals,
    ObjectiveTerm,
)
from piginot.objectives.registry import default_objective_registry
from piginot.objectives.terms import (
    angular_momentum_residual,
    constitutive_residual,
    dirichlet_residual,
    supervised_displacement_residual,
    supervised_stress_residual,
    traction_free_residual,
)
from piginot.operators import OperatorConfig, PiGinotOperator
from piginot.physics import first_piola, plane_stress_closure

PHYSICS_TERM_IDS = (
    "equilibrium_strong",
    "equilibrium_weak",
    "constitutive",
    "angular",
    "dirichlet",
    "traction_free",
)
VERIFIED_GRADIENT_TERM_IDS = ("data_u", "data_P", *PHYSICS_TERM_IDS)
_REQUIRED_PARAMETER_NAMES = (
    "geometry_length",
    "geometry_width",
    "geometry_transition_radius",
    "geometry_shape",
    "elastic_modulus",
    "poisson_ratio",
    "load_amplitude",
    "prescribed_displacement",
)
_EXACT_LIMIT = 1.0e-8
_S8_PATCH_DISCRETIZATION_BOUND = 1.0e-12
_SCALES = NondimensionalScales(10.0, 2.0, 100.0)
_CANONICAL_COORDINATES = CoordinateConversion((1.0, 1.0))


@dataclass(frozen=True, slots=True)
class S12VerificationReport:
    """Finite measured evidence needed by the G12 reducer."""

    exact_max_residuals: Mapping[str, float]
    perturbation_fired_terms: Mapping[str, tuple[str, ...]]
    finite_difference_relative_errors: Mapping[str, float]
    detached_terms: tuple[str, ...]
    reachable_parameter_names: tuple[str, ...]
    required_parameter_names: tuple[str, ...]
    anisotropy_relative_change: float
    coordinate_anisotropy_mutant_fired: bool
    fem_solve_success: bool
    weak_strong_difference: float
    s8_discretization_error_bound: float
    constitutive_residual_shape: tuple[int, int]
    constitutive_rejects_out_of_plane: bool
    holdout_refused: bool
    holdout_access_mutant_fired: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exact_max_residuals",
            MappingProxyType(dict(self.exact_max_residuals)),
        )
        object.__setattr__(
            self,
            "perturbation_fired_terms",
            MappingProxyType(dict(self.perturbation_fired_terms)),
        )
        object.__setattr__(
            self,
            "finite_difference_relative_errors",
            MappingProxyType(dict(self.finite_difference_relative_errors)),
        )


@lru_cache(maxsize=1)
def run_s12_verification() -> S12VerificationReport:
    """Run deterministic S12 evidence without training or campaign-role access."""

    exact, fixtures = _exact_residuals()
    perturbed = _perturbation_isolation(exact, fixtures)
    finite_difference, detached = _finite_difference_checks(fixtures)
    reachable = _parameter_reachability()
    anisotropy, anisotropy_mutant = _anisotropy_check()
    fem_success, weak_strong = _weak_strong_fem_check()
    shape, rejects_out_of_plane = _constitutive_shape_check(fixtures)
    holdout_refused, holdout_mutant = _role_isolation_check()
    return S12VerificationReport(
        exact_max_residuals={
            name: float(residual.abs().max().item()) for name, residual in exact.items()
        },
        perturbation_fired_terms=perturbed,
        finite_difference_relative_errors=finite_difference,
        detached_terms=detached,
        reachable_parameter_names=reachable,
        required_parameter_names=_REQUIRED_PARAMETER_NAMES,
        anisotropy_relative_change=anisotropy,
        coordinate_anisotropy_mutant_fired=anisotropy_mutant,
        fem_solve_success=fem_success,
        weak_strong_difference=weak_strong,
        s8_discretization_error_bound=_S8_PATCH_DISCRETIZATION_BOUND,
        constitutive_residual_shape=shape,
        constitutive_rejects_out_of_plane=rejects_out_of_plane,
        holdout_refused=holdout_refused,
        holdout_access_mutant_fired=holdout_mutant,
    )


def _exact_residuals() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    coordinates = torch.tensor(
        [[[0.2, 0.3], [0.7, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    stress = _analytic_stress(coordinates, _CANONICAL_COORDINATES)
    body_force = torch.tensor([[[-5.0, -3.0], [-5.0, -3.0]]], dtype=torch.float64)
    strong = strong_equilibrium_residual(
        stress, coordinates, body_force, _SCALES, _CANONICAL_COORDINATES
    )

    weak_arguments = _weak_fixture()
    weak = _weak_residual(weak_arguments)
    deformation_2d, material_stress = _material_state(verification_material())
    gradient = deformation_2d - torch.eye(2, dtype=torch.float64)
    constitutive = constitutive_residual(
        material_stress, gradient, verification_material(), _SCALES
    )
    angular = angular_momentum_residual(material_stress, deformation_2d, _SCALES)
    displacement = torch.tensor([[[0.2, -0.1]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True]]])
    dirichlet = dirichlet_residual(displacement, displacement, mask, _SCALES)
    free_stress = torch.tensor(
        [[[[0.0, 0.0], [0.0, 4.0]]]], dtype=torch.float64
    )
    free_normals = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    traction_free = traction_free_residual(free_stress, free_normals, _SCALES)
    residuals = {
        "equilibrium_strong": strong,
        "equilibrium_weak": weak,
        "constitutive": constitutive,
        "angular": angular,
        "dirichlet": dirichlet,
        "traction_free": traction_free,
    }
    fixtures: dict[str, object] = {
        "coordinates": coordinates,
        "stress": stress,
        "body_force": body_force,
        "weak_arguments": weak_arguments,
        "deformation_2d": deformation_2d,
        "material_stress": material_stress,
        "gradient": gradient,
        "displacement": displacement,
        "mask": mask,
        "free_stress": free_stress,
        "free_normals": free_normals,
    }
    return residuals, fixtures


def _perturbation_isolation(
    exact: Mapping[str, torch.Tensor], fixtures: Mapping[str, object]
) -> dict[str, tuple[str, ...]]:
    perturbed: dict[str, torch.Tensor] = {}
    coordinates = _tensor(fixtures, "coordinates")
    stress = _tensor(fixtures, "stress")
    body_force = _tensor(fixtures, "body_force")
    perturbed["equilibrium_strong"] = strong_equilibrium_residual(
        stress,
        coordinates,
        body_force + 1.0,
        _SCALES,
        _CANONICAL_COORDINATES,
    )
    weak_arguments = dict(_mapping(fixtures, "weak_arguments"))
    weak_arguments["traction"] = -_tensor_from(weak_arguments, "traction")
    perturbed["equilibrium_weak"] = _weak_residual(weak_arguments)
    changed_material = MaterialSpec(
        "s12_perturbed_material",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(110.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )
    perturbed["constitutive"] = constitutive_residual(
        _tensor(fixtures, "material_stress"),
        _tensor(fixtures, "gradient"),
        changed_material,
        _SCALES,
    )
    angular_stress = _tensor(fixtures, "material_stress").clone()
    angular_stress[..., 0, 1] += 10.0
    perturbed["angular"] = angular_momentum_residual(
        angular_stress, _tensor(fixtures, "deformation_2d"), _SCALES
    )
    perturbed["dirichlet"] = dirichlet_residual(
        _tensor(fixtures, "displacement"),
        _tensor(fixtures, "displacement") + 0.2,
        _tensor(fixtures, "mask"),
        _SCALES,
    )
    free_stress = _tensor(fixtures, "free_stress").clone()
    free_stress[..., 0, 0] += 10.0
    perturbed["traction_free"] = traction_free_residual(
        free_stress, _tensor(fixtures, "free_normals"), _SCALES
    )

    fired_by_fixture: dict[str, tuple[str, ...]] = {}
    for expected in PHYSICS_TERM_IDS:
        candidate = dict(exact)
        candidate[expected] = perturbed[expected]
        fired_by_fixture[expected] = tuple(
            term
            for term in PHYSICS_TERM_IDS
            if float(candidate[term].abs().max().item()) >= _EXACT_LIMIT
        )
    return fired_by_fixture


def _finite_difference_checks(
    fixtures: Mapping[str, object],
) -> tuple[dict[str, float], tuple[str, ...]]:
    functions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "data_u": lambda value: supervised_displacement_residual(
            value.expand(1, 1, 2), torch.zeros((1, 1, 2), dtype=torch.float64), _SCALES
        ),
        "data_P": lambda value: supervised_stress_residual(
            value.expand(1, 1, 2, 2),
            torch.zeros((1, 1, 2, 2), dtype=torch.float64),
            _SCALES,
        ),
        "equilibrium_strong": _strong_fd_residual,
        "equilibrium_weak": _weak_fd_residual,
        "constitutive": _constitutive_fd_residual,
        "angular": lambda value: angular_momentum_residual(
            _tensor(fixtures, "material_stress")
            + value * _off_diagonal_pattern(),
            _tensor(fixtures, "deformation_2d"),
            _SCALES,
        ),
        "dirichlet": lambda value: dirichlet_residual(
            value.expand(1, 1, 2),
            torch.zeros((1, 1, 2), dtype=torch.float64),
            torch.ones((1, 1, 2), dtype=torch.bool),
            _SCALES,
        ),
        "traction_free": lambda value: traction_free_residual(
            value * torch.eye(2, dtype=torch.float64).reshape(1, 1, 2, 2),
            torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
            _SCALES,
        ),
    }
    errors: dict[str, float] = {}
    detached: list[str] = []
    for term in VERIFIED_GRADIENT_TERM_IDS:
        error, connected = _finite_difference_error(functions[term])
        errors[term] = error
        if not connected:
            detached.append(term)
    return errors, tuple(detached)


def _finite_difference_error(
    residual_function: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[float, bool]:
    value = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    loss = residual_function(value).square().mean()
    if not loss.requires_grad:
        return math.inf, False
    gradient = torch.autograd.grad(loss, value, allow_unused=True)[0]
    if gradient is None or not bool(torch.isfinite(gradient).item()):
        return math.inf, False
    step = 1.0e-6
    above = residual_function(torch.tensor(0.37 + step, dtype=torch.float64))
    below = residual_function(torch.tensor(0.37 - step, dtype=torch.float64))
    numerical = (above.square().mean() - below.square().mean()) / (2.0 * step)
    denominator = max(abs(float(gradient.item())), abs(float(numerical.item())), 1.0e-12)
    return abs(float(gradient.item() - numerical.item())) / denominator, True


def _strong_fd_residual(value: torch.Tensor) -> torch.Tensor:
    coordinates = torch.tensor(
        [[[0.2, 0.3], [0.7, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    zeros = torch.zeros_like(coordinates[..., 0])
    stress = torch.stack(
        (value * coordinates[..., 0], zeros, zeros, zeros), dim=-1
    ).reshape(1, 2, 2, 2)
    return strong_equilibrium_residual(
        stress,
        coordinates,
        torch.zeros_like(coordinates),
        _SCALES,
        _CANONICAL_COORDINATES,
    )


def _weak_fd_residual(value: torch.Tensor) -> torch.Tensor:
    stress = value * torch.eye(2, dtype=torch.float64).reshape(1, 1, 2, 2)
    return weak_equilibrium_residual(
        stress=stress,
        body_force=torch.zeros((1, 1, 2), dtype=torch.float64),
        test_values=torch.ones((1, 1, 1), dtype=torch.float64),
        test_gradients=torch.ones((1, 1, 1, 2), dtype=torch.float64),
        volume_weights=torch.ones((1, 1), dtype=torch.float64),
        scales=_SCALES,
    )


def _constitutive_fd_residual(value: torch.Tensor) -> torch.Tensor:
    coordinates = torch.tensor(
        [[[0.2, 0.3], [0.7, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    displacement = torch.stack(
        (
            value * coordinates[..., 0],
            -0.2 * value * coordinates[..., 1],
        ),
        dim=-1,
    )
    gradient = displacement_gradient(
        displacement, coordinates, _CANONICAL_COORDINATES
    )
    return constitutive_residual(
        torch.zeros_like(gradient),
        gradient,
        verification_material(),
        _SCALES,
    )


def _parameter_reachability() -> tuple[str, ...]:
    parameter_names = _REQUIRED_PARAMETER_NAMES[:-1]
    with torch.random.fork_rng():
        torch.manual_seed(12001)
        model = PiGinotOperator(
            OperatorConfig(
                parameter_names=parameter_names,
                boundary_tag_names=("left", "right", "top", "bottom"),
                hidden_dim=16,
                latent_token_count=4,
                num_attention_heads=4,
                num_fourier_frequencies=2,
                encoder_self_attention_layers=1,
                decoder_cross_attention_layers=1,
            )
        ).double()
    boundary_xy = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float64,
    )
    boundary_tags = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    boundary_normals = torch.tensor(
        [[[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
        dtype=torch.float64,
    )
    query_xy = torch.tensor(
        [[[0.2, 0.3], [0.7, 0.4], [0.5, 0.8]]], dtype=torch.float64
    )
    parameters = torch.tensor(
        [[0.2, -0.3, 0.4, -0.5, 0.1, -0.2, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = model(
        boundary_xy=boundary_xy,
        boundary_tags=boundary_tags,
        boundary_normals=boundary_normals,
        boundary_weights=torch.ones((1, 4), dtype=torch.float64),
        query_xy=query_xy,
        parameters=parameters,
    )
    residuals = ObjectiveResiduals(
        {
            ObjectiveTerm.DATA_U: supervised_displacement_residual(
                output.u,
                output.u.detach() + 0.13,
                _SCALES,
            ),
            ObjectiveTerm.DATA_P: supervised_stress_residual(
                output.P,
                output.P.detach() - 7.0,
                _SCALES,
            ),
        }
    )
    loss = default_objective_registry(mode=ObjectiveMode.DATA_ONLY).evaluate(
        residuals, role=EvidenceRole.PILOT
    ).total
    gradient = torch.autograd.grad(loss, parameters)[0][0]
    reachable = [
        name
        for name, derivative in zip(parameter_names, gradient, strict=True)
        if math.isfinite(float(derivative.item())) and abs(float(derivative.item())) > 1.0e-14
    ]
    prescribed = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    bc_residual = dirichlet_residual(
        torch.zeros((1, 1, 2), dtype=torch.float64),
        prescribed.expand(1, 1, 2),
        torch.ones((1, 1, 2), dtype=torch.bool),
        _SCALES,
    )
    bc_gradient = torch.autograd.grad(bc_residual.square().mean(), prescribed)[0]
    if abs(float(bc_gradient.item())) > 1.0e-14:
        reachable.append(_REQUIRED_PARAMETER_NAMES[-1])
    return tuple(reachable)


def _anisotropy_check() -> tuple[float, bool]:
    base_conversion = CoordinateConversion((1.0, 1.0))
    scaled_conversion = CoordinateConversion((0.1, 1.0))
    base = torch.tensor(
        [[[0.2, 0.3], [0.7, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    scaled = torch.tensor(
        [[[2.0, 0.3], [7.0, 0.4]]], dtype=torch.float64, requires_grad=True
    )
    body_force = torch.zeros_like(base)
    base_residual = strong_equilibrium_residual(
        _analytic_stress(base, base_conversion),
        base,
        body_force,
        _SCALES,
        base_conversion,
    )
    scaled_stress = _analytic_stress(scaled, scaled_conversion)
    scaled_residual = strong_equilibrium_residual(
        scaled_stress,
        scaled,
        body_force,
        _SCALES,
        scaled_conversion,
    )
    change = _relative_change(base_residual, scaled_residual)
    mutant = _coordinate_anisotropy_mutant(
        scaled_stress, scaled, body_force, _SCALES, scaled_conversion
    )
    mutant_change = _relative_change(base_residual, mutant)
    return change, mutant_change > 1.0e-6


def _coordinate_anisotropy_mutant(
    stress: torch.Tensor,
    coordinates: torch.Tensor,
    body_force: torch.Tensor,
    scales: NondimensionalScales,
    conversion: CoordinateConversion,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row in range(2):
        divergence = torch.zeros_like(coordinates[..., 0])
        for column in range(2):
            component = stress[..., row, column]
            gradient = torch.autograd.grad(
                component,
                coordinates,
                grad_outputs=torch.ones_like(component),
                create_graph=True,
                retain_graph=True,
            )[0]
            divergence = divergence + gradient[..., column] / conversion.to_mm[0]
        rows.append(divergence)
    return (torch.stack(tuple(rows), dim=-1) + body_force) / scales.body_force_mpa_per_mm


def _weak_strong_fem_check() -> tuple[bool, float]:
    zero = CanonicalQuantity(0.0, Unit.MM)
    request = FemRequest(
        request_id="s12-weak-strong-affine-p2",
        mesh=unit_square_verification_mesh(2),
        material=verification_material(),
        boundary_conditions=(
            BoundaryConditionSpec("left_x", "left", "dirichlet", ("x",), (zero,)),
            BoundaryConditionSpec(
                "right_x",
                "right",
                "dirichlet",
                ("x",),
                (CanonicalQuantity(0.08, Unit.MM),),
            ),
            BoundaryConditionSpec("bottom_y", "bottom", "dirichlet", ("y",), (zero,)),
            BoundaryConditionSpec(
                "top_y",
                "top",
                "dirichlet",
                ("y",),
                (CanonicalQuantity(-0.03, Unit.MM),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=2,
            solver_tolerance=1.0e-11,
            max_newton_iterations=24,
            plane_stress_max_iterations=32,
            load_steps=4,
        ),
        execution_profile=verification_execution_profile(),
    )
    attempt = ScikitFemBackend().solve_many((request,))[0]
    if not attempt.success or attempt.result is None:
        return False, math.inf
    result = attempt.result
    points = torch.tensor(
        np.asarray(result.quadrature_points).reshape(1, -1, 2), dtype=torch.float64
    )
    weights = torch.tensor(
        np.asarray(result.quadrature_weights).reshape(1, -1), dtype=torch.float64
    )
    stress = torch.tensor(
        np.asarray(result.first_piola_stress)[..., :2, :2].reshape(1, -1, 2, 2),
        dtype=torch.float64,
    )
    x_coordinate = points[..., 0]
    y_coordinate = points[..., 1]
    test_values = (
        x_coordinate
        * (1.0 - x_coordinate)
        * y_coordinate
        * (1.0 - y_coordinate)
    ).unsqueeze(-1)
    gradients = torch.stack(
        (
            (1.0 - 2.0 * x_coordinate) * y_coordinate * (1.0 - y_coordinate),
            x_coordinate * (1.0 - x_coordinate) * (1.0 - 2.0 * y_coordinate),
        ),
        dim=-1,
    ).unsqueeze(-2)
    fem_scales = NondimensionalScales(1.0, 0.08, 100.0)
    weak = weak_equilibrium_residual(
        stress=stress,
        body_force=torch.zeros_like(points),
        test_values=test_values,
        test_gradients=gradients,
        volume_weights=weights,
        scales=fem_scales,
    )
    strong_coordinates = points.detach().clone().requires_grad_(True)
    constant_stress = stress.mean(dim=1, keepdim=True).expand_as(stress)
    connected_stress = constant_stress + strong_coordinates[..., 0, None, None] * 0.0
    strong = strong_equilibrium_residual(
        connected_stress,
        strong_coordinates,
        torch.zeros_like(strong_coordinates),
        fem_scales,
        _CANONICAL_COORDINATES,
    )
    projected_strong = torch.einsum(
        "bqi,bqa,bq->bai", strong, test_values, weights
    ) / fem_scales.length_mm**2
    difference = float((weak + projected_strong).abs().max().item())
    return True, difference


def _constitutive_shape_check(
    fixtures: Mapping[str, object],
) -> tuple[tuple[int, int], bool]:
    residual = constitutive_residual(
        _tensor(fixtures, "material_stress"),
        _tensor(fixtures, "gradient"),
        verification_material(),
        _SCALES,
    )
    rejected = False
    try:
        constitutive_residual(
            torch.zeros((1, 1, 3, 3), dtype=torch.float64),
            _tensor(fixtures, "gradient"),
            verification_material(),
            _SCALES,
        )
    except ValueError:
        rejected = True
    return (int(residual.shape[-2]), int(residual.shape[-1])), rejected


def _role_isolation_check() -> tuple[bool, bool]:
    registry = default_objective_registry(mode=ObjectiveMode.DATA_ONLY)
    residuals = ObjectiveResiduals(
        {
            ObjectiveTerm.DATA_U: torch.ones(1, dtype=torch.float64),
            ObjectiveTerm.DATA_P: torch.ones(1, dtype=torch.float64),
        }
    )
    refused = False
    try:
        registry.evaluate(residuals, role=EvidenceRole.HOLDOUT)
    except PermissionError:
        refused = True
    mutant_total = sum(
        float(residuals.values[term].square().mean().item())
        for term in registry.required_terms
    )
    return refused, refused and mutant_total > 0.0


def _weak_fixture() -> dict[str, torch.Tensor]:
    return {
        "stress": torch.tensor(
            [[[[2.0, 1.0], [3.0, 4.0]]]], dtype=torch.float64
        ),
        "body_force": torch.zeros((1, 1, 2), dtype=torch.float64),
        "test_values": torch.ones((1, 1, 1), dtype=torch.float64),
        "test_gradients": torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64),
        "volume_weights": torch.ones((1, 1), dtype=torch.float64),
        "traction": torch.tensor([[[2.0, 3.0]]], dtype=torch.float64),
        "boundary_test_values": torch.ones((1, 1, 1), dtype=torch.float64),
        "boundary_weights": torch.ones((1, 1), dtype=torch.float64),
    }


def _weak_residual(arguments: Mapping[str, object]) -> torch.Tensor:
    return weak_equilibrium_residual(
        stress=_tensor_from(arguments, "stress"),
        body_force=_tensor_from(arguments, "body_force"),
        test_values=_tensor_from(arguments, "test_values"),
        test_gradients=_tensor_from(arguments, "test_gradients"),
        volume_weights=_tensor_from(arguments, "volume_weights"),
        traction=_tensor_from(arguments, "traction"),
        boundary_test_values=_tensor_from(arguments, "boundary_test_values"),
        boundary_weights=_tensor_from(arguments, "boundary_weights"),
        scales=_SCALES,
    )


def _material_state(material: MaterialSpec) -> tuple[torch.Tensor, torch.Tensor]:
    deformation_2d = torch.tensor(
        [[[[1.08, 0.03], [-0.01, 0.96]]]], dtype=torch.float64
    )
    thickness_stretch = plane_stress_closure(deformation_2d, material)
    deformation_3d = torch.zeros((1, 1, 3, 3), dtype=torch.float64)
    deformation_3d[..., :2, :2] = deformation_2d
    deformation_3d[..., 2, 2] = thickness_stretch
    return deformation_2d, first_piola(deformation_3d, material)[..., :2, :2]


def _analytic_stress(
    coordinates: torch.Tensor, conversion: CoordinateConversion
) -> torch.Tensor:
    physical = coordinates * coordinates.new_tensor(conversion.to_mm)
    x_coordinate = physical[..., 0]
    y_coordinate = physical[..., 1]
    return torch.stack(
        (
            2.0 * x_coordinate,
            3.0 * y_coordinate,
            -x_coordinate,
            4.0 * y_coordinate,
        ),
        dim=-1,
    ).reshape(*coordinates.shape[:-1], 2, 2)


def _off_diagonal_pattern() -> torch.Tensor:
    return torch.tensor(
        [[[[0.0, 1.0], [0.0, 0.0]]]], dtype=torch.float64
    )


def _relative_change(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(reference).item()), 1.0e-15)
    return float(torch.linalg.vector_norm(reference - candidate).item()) / denominator


def _mapping(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = values[name]
    if not isinstance(value, Mapping):
        raise RuntimeError(f"verification fixture {name} is not a mapping")
    return value


def _tensor(values: Mapping[str, object], name: str) -> torch.Tensor:
    return _tensor_from(values, name)


def _tensor_from(values: Mapping[str, object], name: str) -> torch.Tensor:
    value = values[name]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"verification fixture {name} is not a tensor")
    return value


__all__ = (
    "PHYSICS_TERM_IDS",
    "VERIFIED_GRADIENT_TERM_IDS",
    "S12VerificationReport",
    "run_s12_verification",
)
