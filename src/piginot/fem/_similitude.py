"""Exact continuum similitude scaling for soft elastomers/tissues and stiff metals.

FormaMech's frozen S4 ``compressible_neo_hookean`` contract restricts
``elastic_modulus`` to ``[1.0, 10000.0] MPa``. Because compressible Neo-Hookean
Lamé parameters ``(mu, lambda)`` are strictly homogeneous of degree 1 in ``E`` at
fixed ``nu``, scaling ``E`` and all force/stress/penalty/density quantities by
``s > 0`` leaves the nonlinear finite-strain displacement ``u`` and deformation
gradient ``F`` mathematically invariant to machine precision, while physical
stresses, forces, moments, and energies are recovered exactly by dividing by
``s``.
"""

from __future__ import annotations

from dataclasses import dataclass

from piginot.contracts import MaterialSpec, Quantity, Unit

_MIN_E_MPA = 1.0
_MAX_E_MPA = 10_000.0


@dataclass(frozen=True, slots=True)
class SimilitudeMapping:
    """Exact linear-in-modulus continuum similitude scale factor."""

    model_material: MaterialSpec
    scale_factor: float  # E_model = scale_factor * E_physical


def resolve_similitude_material(material: MaterialSpec) -> SimilitudeMapping:
    """Map ``material`` into the frozen ``[1, 10000] MPa`` kernel range if needed."""
    if material.model != "compressible_neo_hookean":
        return SimilitudeMapping(model_material=material, scale_factor=1.0)
    e_prop = material.properties.get("elastic_modulus")
    nu_prop = material.properties.get("poisson_ratio")
    if e_prop is None or nu_prop is None:
        return SimilitudeMapping(model_material=material, scale_factor=1.0)
    e_phys = float(e_prop.value)
    if _MIN_E_MPA <= e_phys <= _MAX_E_MPA:
        return SimilitudeMapping(model_material=material, scale_factor=1.0)

    target_e = 10.0 if e_phys < _MIN_E_MPA else 1000.0
    scale = target_e / e_phys
    scaled_mat = MaterialSpec(
        name=f"{material.name}-similitude",
        model=material.model,
        properties={
            "elastic_modulus": Quantity(target_e, Unit.MPA).to_canonical(),
            "poisson_ratio": nu_prop,
        },
    )
    return SimilitudeMapping(model_material=scaled_mat, scale_factor=scale)


__all__ = ("SimilitudeMapping", "resolve_similitude_material")
