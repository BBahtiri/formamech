"""Executable analytic, gradient, mutant, and FEM verification for S12."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

import torch

from piginot.contracts import (
    CanonicalQuantity,
    MaterialSpec,
    Unit,
)
from piginot.doe import EvidenceRole
from piginot.fem.mms_runner import verification_material
from piginot.objectives._weak_strong import run_weak_strong_fem_check
from piginot.objectives.equilibrium import (
    strong_equilibrium_residual,
    weak_equilibrium_residual,
)
from piginot.objectives.kinematics import displacement_gradient
from piginot.objectives.models import (
    CoordinateConversion,
    NondimensionalScales,
    ObjectiveMode,
    ObjectiveRegistry,
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
    "geometry_axial_extent",
    "geometry_outer_width",
    "geometry_inner_width",
    "geometry_transition_radius",
    "elastic_modulus",
    "poisson_ratio",
    "load_amplitude",
    "prescribed_displacement",
)
VERIFIED_REACHABILITY_LOSS_PATHS = MappingProxyType(
    {
        **{name: "physics_objective_total" for name in _REQUIRED_PARAMETER_NAMES[:-1]},
        _REQUIRED_PARAMETER_NAMES[-1]: "dirichlet",
    }
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
    perturbation_recomputed_term_counts: Mapping[str, int]
    finite_difference_relative_errors: Mapping[str, float]
    detached_terms: tuple[str, ...]
    reachable_parameter_names: tuple[str, ...]
    required_parameter_names: tuple[str, ...]
    reachability_loss_paths: Mapping[str, str]
    anisotropy_relative_change: float
    coordinate_anisotropy_mutant_fired: bool
    fem_solve_success: bool
    weak_strong_difference: float
    weak_residual_max: float
    projected_strong_residual_max: float
    reconstructed_stress_variation: float
    weak_strong_fem_request_id: str
    weak_strong_quadrature_point_count: int
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
            "perturbation_recomputed_term_counts",
            MappingProxyType(dict(self.perturbation_recomputed_term_counts)),
        )
        object.__setattr__(
            self,
            "finite_difference_relative_errors",
            MappingProxyType(dict(self.finite_difference_relative_errors)),
        )
        object.__setattr__(
            self,
            "reachability_loss_paths",
            MappingProxyType(dict(self.reachability_loss_paths)),
        )


@lru_cache(maxsize=1)
def run_s12_verification() -> S12VerificationReport:
    """Run deterministic S12 evidence without training or campaign-role access."""

    exact, fixtures = _exact_residuals()
    perturbed, recomputed_counts = _perturbation_isolation(fixtures)
    finite_difference, detached = _finite_difference_checks(fixtures)
    reachable, reachability_paths = _parameter_reachability()
    anisotropy, anisotropy_mutant = _anisotropy_check()
    weak_strong = run_weak_strong_fem_check()
    shape, rejects_out_of_plane = _constitutive_shape_check(fixtures)
    holdout_refused, holdout_mutant = _role_isolation_check()
    return S12VerificationReport(
        exact_max_residuals={
            name: float(residual.abs().max().item()) for name, residual in exact.items()
        },
        perturbation_fired_terms=perturbed,
        perturbation_recomputed_term_counts=recomputed_counts,
        finite_difference_relative_errors=finite_difference,
        detached_terms=detached,
        reachable_parameter_names=reachable,
        required_parameter_names=_REQUIRED_PARAMETER_NAMES,
        reachability_loss_paths=reachability_paths,
        anisotropy_relative_change=anisotropy,
        coordinate_anisotropy_mutant_fired=anisotropy_mutant,
        fem_solve_success=weak_strong.fem_solve_success,
        weak_strong_difference=weak_strong.difference,
        weak_residual_max=weak_strong.weak_residual_max,
        projected_strong_residual_max=weak_strong.projected_strong_residual_max,
        reconstructed_stress_variation=weak_strong.reconstructed_stress_variation,
        weak_strong_fem_request_id=weak_strong.request_id,
        weak_strong_quadrature_point_count=weak_strong.quadrature_point_count,
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
    weak_arguments = _weak_fixture()
    deformation_2d, material_stress = _material_state(verification_material())
    gradient = deformation_2d - torch.eye(2, dtype=torch.float64)
    displacement = torch.tensor([[[0.2, -0.1]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True]]])
    free_stress = torch.tensor([[[[0.0, 0.0], [0.0, 4.0]]]], dtype=torch.float64)
    free_normals = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    fixtures: dict[str, object] = {
        "equilibrium_strong": {
            "coordinates": coordinates,
            "stress": stress,
            "body_force": body_force,
        },
        "equilibrium_weak": weak_arguments,
        "constitutive": {
            "stress": material_stress,
            "gradient": gradient,
            "material": verification_material(),
        },
        "angular": {"stress": material_stress, "deformation": deformation_2d},
        "dirichlet": {
            "predicted": displacement,
            "prescribed": displacement,
            "mask": mask,
        },
        "traction_free": {"stress": free_stress, "normals": free_normals},
    }
    return _evaluate_physics_residuals(fixtures), fixtures


def _perturbation_isolation(
    fixtures: Mapping[str, object],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    changed_material = MaterialSpec(
        "s12_perturbed_material",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(110.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )
    fired_by_fixture: dict[str, tuple[str, ...]] = {}
    recomputed_counts: dict[str, int] = {}
    for expected in PHYSICS_TERM_IDS:
        candidate = {term: dict(_mapping(fixtures, term)) for term in PHYSICS_TERM_IDS}
        channel = candidate[expected]
        if expected == "equilibrium_strong":
            channel["body_force"] = _tensor_from(channel, "body_force") + 1.0
        elif expected == "equilibrium_weak":
            channel["traction"] = -_tensor_from(channel, "traction")
        elif expected == "constitutive":
            channel["material"] = changed_material
        elif expected == "angular":
            channel["stress"] = (
                _tensor_from(channel, "stress") + 10.0 * _off_diagonal_pattern()
            )
        elif expected == "dirichlet":
            channel["prescribed"] = _tensor_from(channel, "prescribed") + 0.2
        else:
            channel["stress"] = _tensor_from(channel, "stress") + torch.tensor(
                [[[[10.0, 0.0], [0.0, 0.0]]]], dtype=torch.float64
            )
        residuals = _evaluate_physics_residuals(candidate)
        recomputed_counts[expected] = len(residuals)
        fired_by_fixture[expected] = tuple(
            term
            for term in PHYSICS_TERM_IDS
            if float(residuals[term].abs().max().item()) >= _EXACT_LIMIT
        )
    return fired_by_fixture, recomputed_counts


def _evaluate_physics_residuals(
    fixtures: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    strong = _mapping(fixtures, "equilibrium_strong")
    constitutive = _mapping(fixtures, "constitutive")
    material = constitutive["material"]
    if not isinstance(material, MaterialSpec):
        raise RuntimeError("constitutive fixture material is invalid")
    angular = _mapping(fixtures, "angular")
    dirichlet = _mapping(fixtures, "dirichlet")
    traction_free = _mapping(fixtures, "traction_free")
    return {
        "equilibrium_strong": strong_equilibrium_residual(
            _tensor_from(strong, "stress"),
            _tensor_from(strong, "coordinates"),
            _tensor_from(strong, "body_force"),
            _SCALES,
            _CANONICAL_COORDINATES,
        ),
        "equilibrium_weak": _weak_residual(_mapping(fixtures, "equilibrium_weak")),
        "constitutive": constitutive_residual(
            _tensor_from(constitutive, "stress"),
            _tensor_from(constitutive, "gradient"),
            material,
            _SCALES,
        ),
        "angular": angular_momentum_residual(
            _tensor_from(angular, "stress"),
            _tensor_from(angular, "deformation"),
            _SCALES,
        ),
        "dirichlet": dirichlet_residual(
            _tensor_from(dirichlet, "predicted"),
            _tensor_from(dirichlet, "prescribed"),
            _tensor_from(dirichlet, "mask"),
            _SCALES,
        ),
        "traction_free": traction_free_residual(
            _tensor_from(traction_free, "stress"),
            _tensor_from(traction_free, "normals"),
            _SCALES,
        ),
    }


def _finite_difference_checks(
    fixtures: Mapping[str, object],
) -> tuple[dict[str, float], tuple[str, ...]]:
    angular_fixture = _mapping(fixtures, "angular")
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
            _tensor_from(angular_fixture, "stress") + value * _off_diagonal_pattern(),
            _tensor_from(angular_fixture, "deformation"),
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
    denominator = max(
        abs(float(gradient.item())), abs(float(numerical.item())), 1.0e-12
    )
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
    gradient = displacement_gradient(displacement, coordinates, _CANONICAL_COORDINATES)
    return constitutive_residual(
        torch.zeros_like(gradient),
        gradient,
        verification_material(),
        _SCALES,
    )


def _parameter_reachability() -> tuple[tuple[str, ...], dict[str, str]]:
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
        [[[0.2, 0.3], [0.7, 0.4], [0.5, 0.8]]],
        dtype=torch.float64,
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
    displacement_gradient_2d = torch.tensor(
        [[[[0.03, 0.01], [-0.02, -0.01]]]], dtype=torch.float64
    ).expand(1, query_xy.shape[1], 2, 2)
    deformation_2d = displacement_gradient_2d + torch.eye(2, dtype=torch.float64)
    point_count = query_xy.shape[1]
    residuals = ObjectiveResiduals(
        {
            ObjectiveTerm.EQUILIBRIUM_WEAK: weak_equilibrium_residual(
                stress=output.P,
                body_force=torch.zeros_like(output.u),
                test_values=torch.ones((1, point_count, 1), dtype=torch.float64),
                test_gradients=torch.tensor(
                    [[[[1.0, -0.5]], [[-0.2, 0.7]], [[0.4, 0.3]]]],
                    dtype=torch.float64,
                ),
                volume_weights=torch.ones((1, point_count), dtype=torch.float64),
                scales=_SCALES,
            ),
            ObjectiveTerm.CONSTITUTIVE: constitutive_residual(
                output.P,
                displacement_gradient_2d,
                verification_material(),
                _SCALES,
            ),
            ObjectiveTerm.ANGULAR: angular_momentum_residual(
                output.P, deformation_2d, _SCALES
            ),
            ObjectiveTerm.DIRICHLET: dirichlet_residual(
                output.u,
                torch.zeros_like(output.u),
                torch.ones_like(output.u, dtype=torch.bool),
                _SCALES,
            ),
            ObjectiveTerm.TRACTION_FREE: traction_free_residual(
                output.P,
                torch.tensor(
                    [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
                    dtype=torch.float64,
                ),
                _SCALES,
            ),
        }
    )
    loss = (
        default_objective_registry(mode=ObjectiveMode.PHYSICS_ONLY)
        .evaluate(residuals, role=EvidenceRole.PILOT)
        .total
    )
    gradient = torch.autograd.grad(loss, parameters)[0][0]
    reachable = [
        name
        for name, derivative in zip(parameter_names, gradient, strict=True)
        if math.isfinite(float(derivative.item()))
        and abs(float(derivative.item())) > 1.0e-14
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
    paths = dict(VERIFIED_REACHABILITY_LOSS_PATHS)
    return tuple(reachable), paths


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
    return (
        torch.stack(tuple(rows), dim=-1) + body_force
    ) / scales.body_force_mpa_per_mm


def _constitutive_shape_check(
    fixtures: Mapping[str, object],
) -> tuple[tuple[int, int], bool]:
    constitutive_fixture = _mapping(fixtures, "constitutive")
    residual = constitutive_residual(
        _tensor_from(constitutive_fixture, "stress"),
        _tensor_from(constitutive_fixture, "gradient"),
        verification_material(),
        _SCALES,
    )
    rejected = False
    try:
        constitutive_residual(
            torch.zeros((1, 1, 3, 3), dtype=torch.float64),
            _tensor_from(constitutive_fixture, "gradient"),
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
    authorized = registry.evaluate(residuals, role=EvidenceRole.PILOT)
    mutant_total = _holdout_access_mutant(
        registry, residuals, role=EvidenceRole.HOLDOUT
    )
    mutant_matches_reducer = torch.equal(mutant_total, authorized.total)
    return refused, refused and mutant_matches_reducer


def _holdout_access_mutant(
    registry: ObjectiveRegistry,
    residuals: ObjectiveResiduals,
    *,
    role: EvidenceRole,
) -> torch.Tensor:
    """Negative control identical to reduction except for the deleted role guard."""

    if type(registry) is not ObjectiveRegistry:
        raise TypeError("mutant registry must be an exact ObjectiveRegistry")
    if type(residuals) is not ObjectiveResiduals:
        raise TypeError("mutant residuals must be exact ObjectiveResiduals")
    if role is not EvidenceRole.HOLDOUT:
        raise ValueError("mutant must exercise the HOLDOUT role")
    return torch.stack(
        tuple(
            residuals.values[term].square().mean() * registry.term_weights[term]
            for term in registry.required_terms
        )
    ).sum()


def _weak_fixture() -> dict[str, torch.Tensor]:
    return {
        "stress": torch.tensor([[[[2.0, 1.0], [3.0, 4.0]]]], dtype=torch.float64),
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
    return torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]], dtype=torch.float64)


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
    "VERIFIED_REACHABILITY_LOSS_PATHS",
    "VERIFIED_GRADIENT_TERM_IDS",
    "S12VerificationReport",
    "run_s12_verification",
)
