"""Hyperelastic materials defined by a strain-energy formula (S25, ADR 0107).

:class:`FormulaMaterial` evaluates an :class:`EnergyFormula` with torch
operations and derives everything else by automatic differentiation: the
first Piola stress ``P = dW/dF``, the tangent ``dP/dF`` and the plane-stress
closure ``P33(F2, F33) = 0`` with its implicit-function derivative. It offers
the five functions of :mod:`piginot.physics`, which dispatches every model
other than the closed-form Neo-Hookean here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch

from piginot.contracts import Dimension, MaterialSpec, Unit
from piginot.contracts.energy_formula import (
    EnergyFormula,
    EnergyFormulaError,
    Tree,
)
from piginot.materials.values import fibre_directions, formula_moduli
from piginot.physics._validation import (
    determinant_is_admissible,
    require_admissible,
    validate_tensor,
)
from piginot.physics.errors import (
    InadmissibleDeformationError,
    InvalidMaterialError,
    NewtonNonConvergenceError,
    PlaneStressClosureError,
)

_FUNCTIONS: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "ramp": torch.relu,
}
"""``ramp`` is ``relu``: its derivative at 0 is 0, so a switched fibre term
carries no load and no stiffness at ``I4 = 1`` (ADR 0108 decision 5)."""
CLOSURE_TOLERANCE = 16.0
"""Newton stops when its ``log F33`` step is below this many machine
epsilons (relative, at least 1); the energy's rounding error is larger than
the closed-form kernel's half epsilon."""

BRACKET_STEPS = 24
"""Steps of the ``log F33`` bracket search: each side starts 0.25 from the
guess, doubles while its residual has the wrong sign and halves back while
the energy is undefined there."""


def _step(width: torch.Tensor, done: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    undefined = ~torch.isfinite(value)
    return torch.where(done, width, torch.where(undefined, 0.5 * width, 2.0 * width))


def torch_value(tree: Tree, variables: Mapping[str, Any]) -> Any:
    """``tree`` as torch operations on ``variables`` (tensors or floats)."""

    kind = tree[0]
    if kind == "number":
        return float(tree[1])
    if kind == "name":
        return variables[cast(str, tree[1])]
    if kind == "unary":
        operand = torch_value(cast(Tree, tree[2]), variables)
        return operand if tree[1] == "+" else -operand
    if kind == "call":
        argument = torch_value(cast(Tree, tree[2]), variables)
        if not isinstance(argument, torch.Tensor):
            argument = torch.tensor(float(argument), dtype=torch.float64)
        return _FUNCTIONS[cast(str, tree[1])](argument)
    if kind == "power":
        return torch_value(cast(Tree, tree[1]), variables) ** float(tree[2])
    left = torch_value(cast(Tree, tree[2]), variables)
    right = torch_value(cast(Tree, tree[3]), variables)
    operator = tree[1]
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    return left / right


def invariants(deformation: torch.Tensor) -> dict[str, torch.Tensor]:
    """``I1``, ``I2``, ``J``, ``I1bar`` and ``I2bar`` of 3D deformations."""

    cauchy_green = deformation.transpose(-2, -1) @ deformation
    first = torch.diagonal(cauchy_green, dim1=-2, dim2=-1).sum(dim=-1)
    squared = (cauchy_green * cauchy_green).sum(dim=(-2, -1))
    second = 0.5 * (first * first - squared)
    determinant = torch.linalg.det(deformation)
    return {
        "I1": first,
        "I2": second,
        "J": determinant,
        "I1bar": determinant ** (-2.0 / 3.0) * first,
        "I2bar": determinant ** (-4.0 / 3.0) * second,
    }


def fibre_invariants(
    deformation: torch.Tensor, directions: Mapping[str, tuple[float, float, float]]
) -> dict[str, torch.Tensor]:
    """``I4<label> = a.C a`` and ``I5<label> = a.C^2 a`` of each family."""

    if not directions:
        return {}
    cauchy_green = deformation.transpose(-2, -1) @ deformation
    result: dict[str, torch.Tensor] = {}
    for label, direction in directions.items():
        a = torch.tensor(direction, dtype=deformation.dtype, device=deformation.device)
        stretched = cauchy_green @ a
        result[f"I4{label}"] = (stretched * a).sum(dim=-1)
        result[f"I5{label}"] = (stretched * stretched).sum(dim=-1)
    return result


def embed(deformation_2d: torch.Tensor, f33: torch.Tensor) -> torch.Tensor:
    """The 3D deformation ``[[F2, 0], [0, F33]]``."""

    zeros_column = torch.zeros(
        (*deformation_2d.shape[:-2], 2, 1),
        dtype=deformation_2d.dtype,
        device=deformation_2d.device,
    )
    upper = torch.cat((deformation_2d, zeros_column), dim=-1)
    bottom = torch.cat((zeros_column.transpose(-2, -1), f33[..., None, None]), dim=-1)
    return torch.cat((upper, bottom), dim=-2)


def material_values(formula: EnergyFormula, material: MaterialSpec) -> dict[str, float]:
    """The formula's parameter values from a material contract."""

    if not isinstance(material, MaterialSpec):
        raise InvalidMaterialError("material must be a MaterialSpec")
    if set(material.properties) != set(formula.parameters):
        raise InvalidMaterialError(
            f"{material.model} needs exactly the properties "
            f"{sorted(formula.parameters)}"
        )
    values: dict[str, float] = {}
    for name, unit in formula.parameters.items():
        quantity = material.properties[name]
        expected = {
            Unit.MPA: Dimension.STRESS,
            Unit.DEG: Dimension.ANGLE,
        }.get(unit, Dimension.DIMENSIONLESS)
        if quantity.unit is not unit or quantity.dimension is not expected:
            raise InvalidMaterialError(f"{name} must use the unit {unit.value}")
        values[name] = float(quantity.value)
    try:
        formula.check_values(values)
    except EnergyFormulaError as exc:
        raise InvalidMaterialError(str(exc)) from exc
    return values


@dataclass(frozen=True, slots=True)
class FormulaMaterial:
    """A strain-energy formula with bound parameter values."""

    formula: EnergyFormula
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        try:
            self.formula.check_values(dict(self.values))
        except EnergyFormulaError as exc:
            raise InvalidMaterialError(str(exc)) from exc

    @property
    def moduli(self) -> tuple[float, float]:
        """Declared small-strain shear and bulk moduli in MPa."""

        return formula_moduli(self.formula, self.values)

    @property
    def directions(self) -> dict[str, tuple[float, float, float]]:
        """Unit direction of each fibre family (S26)."""

        return fibre_directions(self.formula, self.values)

    # ---------------------------------------------------------------- energy

    def _energy(self, deformation: torch.Tensor) -> torch.Tensor:
        variables: dict[str, Any] = dict(self.values)
        variables.update(invariants(deformation))
        variables.update(fibre_invariants(deformation, self.directions))
        energy = torch_value(self.formula.energy.tree, variables)
        if not isinstance(energy, torch.Tensor):
            energy = torch.full(
                deformation.shape[:-2],
                float(energy),
                dtype=deformation.dtype,
                device=deformation.device,
            )
        return energy

    def energy_density(self, deformation: torch.Tensor) -> torch.Tensor:
        """Strain-energy density in MPa of 3D deformations."""

        deformation = validate_tensor(deformation, (3, 3))
        require_admissible(torch.linalg.det(deformation))
        energy = self._energy(deformation)
        _require_finite(energy.detach())
        return energy

    def admissibility(self, deformation: torch.Tensor) -> bool:
        """Whether every deformation has an admitted ``J`` and finite energy."""

        deformation = validate_tensor(deformation, (3, 3))
        determinant = torch.linalg.det(deformation)
        if not bool(determinant_is_admissible(determinant).all().item()):
            return False
        with torch.no_grad():
            energy = self._energy(deformation.detach())
        return bool(torch.isfinite(energy).all().item())

    # ---------------------------------------------------------------- stress

    def first_piola(self, deformation: torch.Tensor) -> torch.Tensor:
        """``P = dW/dF`` in MPa; differentiable when ``deformation`` is."""

        deformation = validate_tensor(deformation, (3, 3))
        require_admissible(torch.linalg.det(deformation))
        if deformation.requires_grad and torch.is_grad_enabled():
            energy = self._energy(deformation)
            _require_finite(energy.detach())
            (stress,) = torch.autograd.grad(
                energy.sum(), deformation, create_graph=True
            )
            return stress
        with torch.enable_grad():
            leaf = deformation.detach().requires_grad_(True)
            energy = self._energy(leaf)
            _require_finite(energy.detach())
            (stress,) = torch.autograd.grad(energy.sum(), leaf)
        return stress.detach()

    def consistent_tangent(self, deformation: torch.Tensor) -> torch.Tensor:
        """``dP_iJ/dF_kL`` with layout ``(..., i, J, k, L)``."""

        deformation = validate_tensor(deformation, (3, 3))
        require_admissible(torch.linalg.det(deformation))
        with torch.enable_grad():
            leaf = deformation.detach().requires_grad_(True)
            (stress,) = torch.autograd.grad(
                self._energy(leaf).sum(), leaf, create_graph=True
            )
            rows = []
            for i in range(3):
                for j in range(3):
                    (row,) = torch.autograd.grad(
                        stress[..., i, j].sum(), leaf, retain_graph=True
                    )
                    rows.append(row)
        tangent = torch.stack(rows, dim=-3)
        return tangent.reshape(*deformation.shape[:-2], 3, 3, 3, 3).detach()

    # ------------------------------------------------------- plane stress

    def thickness_terms(
        self, deformation_2d: torch.Tensor, f33: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``P33``, ``dP33/dF33`` and ``dP33/dF2`` at ``[[F2, 0], [0, F33]]``."""

        with torch.enable_grad():
            leaf = embed(deformation_2d.detach(), f33.detach()).requires_grad_(True)
            (stress,) = torch.autograd.grad(
                self._energy(leaf).sum(), leaf, create_graph=True
            )
            normal = stress[..., 2, 2]
            (row,) = torch.autograd.grad(normal.sum(), leaf)
        return normal.detach(), row[..., 2, 2], row[..., :2, :2]

    def _residual(
        self, deformation_2d: torch.Tensor, log_f33: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``r = F33 P33`` and ``dr/d log F33``; non-finite outside the domain."""

        f33 = torch.exp(log_f33)
        determinant = torch.linalg.det(deformation_2d) * f33
        safe = determinant_is_admissible(determinant)
        normal, slope, _ = self.thickness_terms(
            deformation_2d, torch.where(safe, f33, torch.ones_like(f33))
        )
        residual = torch.where(safe, f33 * normal, torch.full_like(f33, math.nan))
        derivative = torch.where(
            safe, f33 * (normal + f33 * slope), torch.full_like(f33, math.nan)
        )
        return residual, derivative

    def solve_f33(
        self,
        deformation_2d: torch.Tensor,
        initial_f33: float = 1.0,
        max_iterations: int = 32,
    ) -> torch.Tensor:
        """The thickness stretch with ``P33 = 0`` (safeguarded Newton in log F33)."""

        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise PlaneStressClosureError("max_iterations must be an integer")
        if max_iterations <= 0:
            raise NewtonNonConvergenceError(
                "plane-stress Newton solve requires at least one iteration"
            )
        if not (math.isfinite(initial_f33) and initial_f33 > 0.0):
            raise PlaneStressClosureError("initial_f33 must be finite and positive")
        deformation_2d = deformation_2d.detach()
        centre = torch.full(
            deformation_2d.shape[:-2],
            math.log(initial_f33),
            dtype=deformation_2d.dtype,
            device=deformation_2d.device,
        )
        lower, upper = self._bracket(deformation_2d, centre)
        log_f33 = 0.5 * (lower + upper)
        epsilon = torch.finfo(deformation_2d.dtype).eps
        converged = torch.zeros_like(log_f33, dtype=torch.bool)
        for _ in range(max_iterations):
            residual, derivative = self._residual(deformation_2d, log_f33)
            correction = residual / derivative
            tolerance = (
                CLOSURE_TOLERANCE * epsilon * torch.clamp(log_f33.abs(), min=1.0)
            )
            converged = converged | (residual == 0.0) | (correction.abs() <= tolerance)
            if bool(converged.all().item()):
                break
            newton = log_f33 - correction
            inside = torch.isfinite(newton) & (newton > lower) & (newton < upper)
            candidate = torch.where(inside, newton, 0.5 * (lower + upper))
            value, _ = self._residual(deformation_2d, candidate)
            lower = torch.where(value <= 0.0, candidate, lower)
            upper = torch.where(value > 0.0, candidate, upper)
            log_f33 = torch.where(converged, log_f33, candidate)
        residual, derivative = self._residual(deformation_2d, log_f33)
        tolerance = CLOSURE_TOLERANCE * epsilon * torch.clamp(log_f33.abs(), min=1.0)
        done = (residual == 0.0) | ((residual / derivative).abs() <= tolerance)
        if not bool(done.all().item()):
            raise NewtonNonConvergenceError(
                f"plane-stress Newton solve did not converge in {max_iterations} "
                "iterations"
            )
        return torch.exp(log_f33)

    def _bracket(
        self, deformation_2d: torch.Tensor, centre: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``log F33`` values with ``r < 0`` below and ``r > 0`` above the root."""

        below = torch.full_like(centre, 0.25)
        above = torch.full_like(centre, 0.25)
        for _ in range(BRACKET_STEPS):
            low_value, _ = self._residual(deformation_2d, centre - below)
            high_value, _ = self._residual(deformation_2d, centre + above)
            low_ok = low_value < 0.0
            high_ok = high_value > 0.0
            if bool((low_ok & high_ok).all().item()):
                return centre - below, centre + above
            below = _step(below, low_ok, low_value)
            above = _step(above, high_ok, high_value)
        raise PlaneStressClosureError(
            "no plane-stress root bracket inside the energy's domain"
        )

    def plane_stress_closure(
        self,
        deformation_2d: torch.Tensor,
        *,
        initial_f33: float = 1.0,
        max_iterations: int = 32,
    ) -> torch.Tensor:
        """``F33`` with ``P33 = 0``, differentiable by the implicit function
        theorem: ``dF33/dF2 = -(dP33/dF2) / (dP33/dF33)``."""

        deformation_2d = validate_tensor(deformation_2d, (2, 2))
        require_admissible(torch.linalg.det(deformation_2d))
        apply: Callable[..., torch.Tensor] = _FormulaClosure.apply
        return apply(deformation_2d, self, initial_f33, max_iterations)


class _FormulaClosure(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        deformation_2d: torch.Tensor,
        material: FormulaMaterial,
        initial_f33: float,
        max_iterations: int,
    ) -> torch.Tensor:
        f33 = material.solve_f33(deformation_2d, initial_f33, max_iterations)
        _, slope, cross = material.thickness_terms(deformation_2d, f33)
        ctx.save_for_backward(-cross / slope[..., None, None])
        return f33

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (derivative,) = ctx.saved_tensors
        return grad_output[..., None, None] * derivative, None, None, None


def _require_finite(values: torch.Tensor) -> None:
    if not bool(torch.isfinite(values).all().item()):
        raise InadmissibleDeformationError(
            "the strain energy is not defined at this deformation"
        )


__all__ = (
    "BRACKET_STEPS",
    "CLOSURE_TOLERANCE",
    "FormulaMaterial",
    "embed",
    "fibre_invariants",
    "invariants",
    "material_values",
    "torch_value",
)
