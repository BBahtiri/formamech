from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from piginot.contracts import (
    BoundaryConditionSpec,
    CampaignPlan,
    CanonicalMesh,
    DerivedExpression,
    DerivedQuantitySpec,
    GeometryFamilySpec,
    LoadSpec,
    MaterialSpec,
    ParameterSpec,
    ProblemFamilySpec,
    QoISpec,
    Quantity,
    ToleranceSpec,
    Unit,
)
from piginot.contracts.errors import (
    BareDimensionalValueError,
    ContractValidationError,
    NonFiniteNumberError,
    UnknownFieldError,
)

FROZEN_CONTRACT_TYPES = (
    ProblemFamilySpec,
    GeometryFamilySpec,
    ParameterSpec,
    DerivedQuantitySpec,
    MaterialSpec,
    BoundaryConditionSpec,
    LoadSpec,
    QoISpec,
    ToleranceSpec,
    CampaignPlan,
    CanonicalMesh,
)


def _length(value: float, unit: Unit = Unit.MM):
    return Quantity(value, unit).to_canonical()


def _dimensionless(value: float):
    return Quantity(value, Unit.ONE).to_canonical()


def _force(value: float):
    return Quantity(value, Unit.N).to_canonical()


def _stress(value: float):
    return Quantity(value, Unit.MPA).to_canonical()


def _parameter_specs() -> tuple[ParameterSpec, ParameterSpec]:
    return (
        ParameterSpec(
            name="width",
            lower_bound=_length(2.0, Unit.CM),
            upper_bound=_length(4.0, Unit.CM),
            description="overall width",
        ),
        ParameterSpec(
            name="cut_depth",
            lower_bound=_length(1.0),
            upper_bound=_length(5.0),
            description="symmetric cut depth",
        ),
    )


def _derived_quantity() -> DerivedQuantitySpec:
    return DerivedQuantitySpec(
        name="net_section_factor",
        expression=DerivedExpression.parse("width / (width - 2 * cut_depth)"),
        lower_bound=_dimensionless(1.07),
        upper_bound=_dimensionless(2.65),
        unit=Unit.ONE,
        description="dimensionless remaining-section ratio",
    )


def _mesh_inputs() -> dict[str, object]:
    return {
        "mesh_id": "mesh-001",
        "nodes": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "cells": [[0, 1, 2]],
        "cell_type": "triangle",
        "boundary_facets": [[0, 1], [1, 2], [2, 0]],
        "semantic_tags": {"fixed_edge": [0], "loaded_edge": [1]},
        "region_ids": [7],
        "unit": Unit.MM,
        "orientation": "counterclockwise",
        "topology_signature": "triangle:3-nodes:3-boundary-facets",
        "quality_metrics": {"minimum_angle_degrees": 45.0},
        "source_provenance": {"provider": "contract-test"},
    }


def _problem_inputs() -> dict[str, object]:
    geometry = GeometryFamilySpec(
        name="symmetric_plate",
        parameters=_parameter_specs(),
        derived_quantities=(_derived_quantity(),),
        topology_signature="fixed-topology-v1",
        semantic_tags=("fixed_edge", "loaded_edge"),
    )
    material = MaterialSpec(
        name="elastic_material",
        model="finite_strain",
        properties={"elastic_modulus": _stress(210_000.0)},
    )
    boundary_condition = BoundaryConditionSpec(
        name="fixed_support",
        boundary_tag="fixed_edge",
        kind="dirichlet",
        components=("x", "y"),
        values=(_length(0.0), _length(0.0)),
    )
    load = LoadSpec(
        name="edge_force",
        kind="traction",
        boundary_tag="loaded_edge",
        components=("x",),
        values=(_force(100.0),),
    )
    qoi = QoISpec(
        name="average_stress",
        field="first_piola",
        reduction="region_average",
        region_tag="loaded_edge",
        unit=Unit.MPA,
    )
    tolerance = ToleranceSpec(
        name="stress_error",
        metric="relative_l2",
        limit=_dimensionless(0.15),
    )
    campaign = CampaignPlan(
        campaign_id="campaign-001",
        sample_count=40,
        seed=314159,
    )
    return {
        "name": "plate_problem",
        "geometry": geometry,
        "material": material,
        "boundary_conditions": [boundary_condition],
        "loads": [load],
        "qois": [qoi],
        "tolerances": [tolerance],
        "campaign": campaign,
        "schema_version": "1.0",
    }


@pytest.mark.parametrize(
    "contract_type",
    FROZEN_CONTRACT_TYPES,
    ids=lambda contract_type: contract_type.__name__,
)
def test_required_contracts_are_frozen_dataclasses(contract_type: type[object]) -> None:
    assert is_dataclass(contract_type)
    assert contract_type.__dataclass_params__.frozen


@pytest.mark.parametrize(
    "contract_type",
    FROZEN_CONTRACT_TYPES,
    ids=lambda contract_type: contract_type.__name__,
)
def test_mutating_any_declared_contract_field_raises(
    contract_type: type[object],
) -> None:
    instance = object.__new__(contract_type)

    for field in fields(contract_type):
        with pytest.raises(FrozenInstanceError):
            setattr(instance, field.name, object())


@pytest.mark.parametrize(
    "contract_type",
    FROZEN_CONTRACT_TYPES,
    ids=lambda contract_type: contract_type.__name__,
)
def test_from_mapping_rejects_unknown_fields_with_typed_error(
    contract_type: type[object],
) -> None:
    with pytest.raises(UnknownFieldError):
        contract_type.from_mapping({"unexpected_field": "not permitted"})


def test_parameter_contract_requires_explicit_canonical_quantities() -> None:
    parameter = ParameterSpec(
        name="width",
        lower_bound=Quantity(0.02, Unit.M).to_canonical(),
        upper_bound=Quantity(4.0, Unit.CM).to_canonical(),
    )

    assert parameter.lower_bound == _length(20.0)
    assert parameter.upper_bound == _length(40.0)

    with pytest.raises(BareDimensionalValueError):
        ParameterSpec(name="width", lower_bound=20.0, upper_bound=_length(40.0))

    with pytest.raises(ContractValidationError):
        ParameterSpec(
            name="width",
            lower_bound=Quantity(20.0, Unit.MM),
            upper_bound=Quantity(40.0, Unit.MM),
        )


def test_parameter_names_must_be_referenceable_by_derived_expressions() -> None:
    with pytest.raises(ContractValidationError, match="identifier"):
        ParameterSpec(
            name="class",
            lower_bound=_length(1.0),
            upper_bound=_length(2.0),
        )


def test_parameter_rejects_invalid_identity_bounds_and_description() -> None:
    with pytest.raises(ContractValidationError):
        ParameterSpec("not a name", _length(1.0), _length(2.0))
    with pytest.raises(ContractValidationError):
        ParameterSpec("width", _length(3.0), _length(2.0))
    with pytest.raises(ContractValidationError):
        ParameterSpec("width", _length(1.0), _dimensionless(2.0))
    with pytest.raises(ContractValidationError):
        ParameterSpec("width", _length(1.0), _length(2.0), description=7)


def test_parameter_from_mapping_accepts_valid_fields_and_wraps_missing_fields() -> None:
    parameter = ParameterSpec.from_mapping(
        {
            "name": "width",
            "lower_bound": _length(1.0),
            "upper_bound": _length(2.0),
            "description": "width range",
        }
    )

    assert parameter.name == "width"
    with pytest.raises(ContractValidationError):
        ParameterSpec.from_mapping({"name": "width"})


def test_parameter_and_derived_contract_fields_are_explicit() -> None:
    parameter_fields = {field.name for field in fields(ParameterSpec)}
    derived_fields = {field.name for field in fields(DerivedQuantitySpec)}
    geometry_fields = {field.name for field in fields(GeometryFamilySpec)}

    assert {"name", "lower_bound", "upper_bound", "description"} <= parameter_fields
    assert {
        "name",
        "expression",
        "lower_bound",
        "upper_bound",
        "unit",
        "description",
    } <= derived_fields
    assert {"name", "parameters", "derived_quantities"} <= geometry_fields


def test_geometry_detaches_caller_collections_and_is_deeply_immutable() -> None:
    parameters = list(_parameter_specs())
    derived_quantities = [_derived_quantity()]
    geometry = GeometryFamilySpec(
        name="symmetric_plate",
        parameters=parameters,
        derived_quantities=derived_quantities,
    )

    parameters.clear()
    derived_quantities.clear()

    assert tuple(spec.name for spec in geometry.parameters) == (
        "width",
        "cut_depth",
    )
    assert tuple(spec.name for spec in geometry.derived_quantities) == (
        "net_section_factor",
    )
    assert isinstance(geometry.parameters, tuple)
    assert isinstance(geometry.derived_quantities, tuple)
    assert hash(geometry)


def test_geometry_rejects_duplicate_names_across_its_declarations() -> None:
    width, cut_depth = _parameter_specs()
    duplicate_width = ParameterSpec(
        name="width",
        lower_bound=_length(50.0),
        upper_bound=_length(60.0),
    )

    with pytest.raises(ContractValidationError):
        GeometryFamilySpec(
            name="invalid-family",
            parameters=(width, cut_depth, duplicate_width),
            derived_quantities=(),
        )


def test_derived_quantity_rejects_invalid_contract_fields() -> None:
    expression = DerivedExpression.parse("width / width")
    valid = {
        "name": "ratio",
        "expression": expression,
        "lower_bound": _dimensionless(0.0),
        "upper_bound": _dimensionless(1.0),
        "unit": Unit.ONE,
    }

    invalid_overrides = (
        {"expression": "width / width"},
        {"unit": Unit.CM},
        {"unit": Unit.N},
        {"lower_bound": _dimensionless(2.0)},
        {"description": 7},
    )
    for override in invalid_overrides:
        with pytest.raises(ContractValidationError):
            DerivedQuantitySpec(**(valid | override))


def test_derived_quantity_validation_rejects_bad_parameter_sequences() -> None:
    derived = _derived_quantity()
    width, _ = _parameter_specs()

    with pytest.raises(ContractValidationError):
        derived.validate_against((object(),))
    with pytest.raises(ContractValidationError):
        derived.validate_against((width, width))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("name", ""),
        ("parameters", (object(),)),
        ("derived_quantities", (object(),)),
        ("semantic_tags", ("",)),
        ("topology_signature", ""),
    ],
)
def test_geometry_rejects_invalid_structural_declarations(
    field_name: str,
    invalid_value: object,
) -> None:
    values: dict[str, object] = {
        "name": "plate_family",
        "parameters": (),
        "derived_quantities": (),
        "semantic_tags": (),
        "topology_signature": None,
    }
    values[field_name] = invalid_value

    with pytest.raises(ContractValidationError):
        GeometryFamilySpec(**values)


def test_material_contract_detaches_properties_and_behaves_as_mapping() -> None:
    properties = {
        "elastic_modulus": _stress(210_000.0),
        "poisson_ratio": _dimensionless(0.3),
    }
    material = MaterialSpec("elastic_material", "finite_strain", properties)
    properties.clear()

    assert len(material.properties) == 2
    assert tuple(material.properties) == ("elastic_modulus", "poisson_ratio")
    assert material.properties["poisson_ratio"] == _dimensionless(0.3)
    with pytest.raises(KeyError):
        material.properties["missing"]
    with pytest.raises(TypeError):
        material.properties["new"] = _dimensionless(1.0)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("name", ""),
        ("model", ""),
        ("properties", (("elastic_modulus", _stress(1.0)),)),
        ("properties", {"": _stress(1.0)}),
        ("properties", {"elastic_modulus": Quantity(1.0, Unit.MPA)}),
    ],
)
def test_material_rejects_invalid_declarations(
    field_name: str,
    invalid_value: object,
) -> None:
    values: dict[str, object] = {
        "name": "elastic_material",
        "model": "finite_strain",
        "properties": {"elastic_modulus": _stress(1.0)},
    }
    values[field_name] = invalid_value

    with pytest.raises(ContractValidationError):
        MaterialSpec(**values)


def test_boundary_and_load_contracts_freeze_component_values() -> None:
    components = ["x", "y"]
    boundary_values = [_length(0.0), _length(0.0)]
    load_values = [_force(10.0), _force(20.0)]

    boundary = BoundaryConditionSpec(
        "fixed_support", "fixed_edge", "dirichlet", components, boundary_values
    )
    load = LoadSpec("edge_force", "traction", "loaded_edge", components, load_values)
    components.clear()
    boundary_values.clear()
    load_values.clear()

    assert boundary.components == ("x", "y")
    assert boundary.values == (_length(0.0), _length(0.0))
    assert load.components == ("x", "y")
    assert load.values == (_force(10.0), _force(20.0))


@pytest.mark.parametrize(
    "invalid_value",
    [
        {"name": "", "boundary_tag": "edge", "kind": "dirichlet"},
        {"name": "bc", "boundary_tag": "", "kind": "dirichlet"},
        {"name": "bc", "boundary_tag": "edge", "kind": ""},
        {
            "name": "bc",
            "boundary_tag": "edge",
            "kind": "dirichlet",
            "components": ("",),
        },
        {
            "name": "bc",
            "boundary_tag": "edge",
            "kind": "dirichlet",
            "components": ("x",),
            "values": (Quantity(0.0, Unit.MM),),
        },
        {
            "name": "bc",
            "boundary_tag": "edge",
            "kind": "dirichlet",
            "components": ("x", "y"),
            "values": (_length(0.0),),
        },
    ],
)
def test_boundary_condition_rejects_invalid_declarations(
    invalid_value: dict[str, object],
) -> None:
    with pytest.raises(ContractValidationError):
        BoundaryConditionSpec(**invalid_value)


@pytest.mark.parametrize(
    "invalid_value",
    [
        {"name": "", "kind": "traction"},
        {"name": "load", "kind": ""},
        {"name": "load", "kind": "traction", "boundary_tag": ""},
        {"name": "load", "kind": "traction", "components": ("",)},
        {
            "name": "load",
            "kind": "traction",
            "components": ("x",),
            "values": (Quantity(1.0, Unit.N),),
        },
        {
            "name": "load",
            "kind": "traction",
            "components": ("x", "y"),
            "values": (_force(1.0),),
        },
    ],
)
def test_load_rejects_invalid_declarations(
    invalid_value: dict[str, object],
) -> None:
    with pytest.raises(ContractValidationError):
        LoadSpec(**invalid_value)


def test_qoi_tolerance_and_campaign_validate_their_public_fields() -> None:
    qoi = QoISpec(
        "average_stress", "first_piola", "region_average", "loaded_edge", Unit.MPA
    )
    tolerance = ToleranceSpec("stress_error", "relative_l2", _dimensionless(0.15))
    campaign = CampaignPlan("campaign-001", 40, 0)

    assert qoi.unit is Unit.MPA
    assert tolerance.limit == _dimensionless(0.15)
    assert campaign.seed == 0


@pytest.mark.parametrize("field_name", ["name", "field", "reduction", "region_tag"])
def test_qoi_rejects_empty_text_fields(field_name: str) -> None:
    values: dict[str, object] = {
        "name": "average_stress",
        "field": "first_piola",
        "reduction": "region_average",
        "region_tag": "loaded_edge",
        "unit": Unit.MPA,
    }
    values[field_name] = ""

    with pytest.raises(ContractValidationError):
        QoISpec(**values)


def test_qoi_rejects_noncanonical_unit() -> None:
    with pytest.raises(ContractValidationError):
        QoISpec("average_stress", "first_piola", "average", "edge", Unit.GPA)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("name", ""),
        ("metric", ""),
        ("limit", Quantity(0.1, Unit.ONE)),
        ("limit", _dimensionless(-0.1)),
    ],
)
def test_tolerance_rejects_invalid_declarations(
    field_name: str,
    invalid_value: object,
) -> None:
    values: dict[str, object] = {
        "name": "stress_error",
        "metric": "relative_l2",
        "limit": _dimensionless(0.15),
    }
    values[field_name] = invalid_value

    with pytest.raises(ContractValidationError):
        ToleranceSpec(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("campaign_id", ""),
        ("sample_count", True),
        ("sample_count", 0),
        ("seed", True),
        ("seed", -1),
    ],
)
def test_campaign_rejects_invalid_declarations(
    field_name: str,
    invalid_value: object,
) -> None:
    values: dict[str, object] = {
        "campaign_id": "campaign-001",
        "sample_count": 40,
        "seed": 0,
    }
    values[field_name] = invalid_value

    with pytest.raises(ContractValidationError):
        CampaignPlan(**values)


def test_problem_family_composes_and_detaches_all_nested_spec_lists() -> None:
    values = _problem_inputs()
    problem = ProblemFamilySpec(**values)

    values["boundary_conditions"].clear()
    values["loads"].clear()
    values["qois"].clear()
    values["tolerances"].clear()

    assert len(problem.boundary_conditions) == 1
    assert len(problem.loads) == 1
    assert len(problem.qois) == 1
    assert len(problem.tolerances) == 1
    assert hash(problem)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("name", ""),
        ("schema_version", ""),
        ("geometry", object()),
        ("material", object()),
        ("campaign", object()),
        ("boundary_conditions", (object(),)),
        ("loads", (object(),)),
        ("qois", (object(),)),
        ("tolerances", (object(),)),
    ],
)
def test_problem_family_rejects_invalid_nested_contracts(
    field_name: str,
    invalid_value: object,
) -> None:
    values = _problem_inputs()
    values[field_name] = invalid_value

    with pytest.raises(ContractValidationError):
        ProblemFamilySpec(**values)


def test_canonical_mesh_is_a_deeply_immutable_structural_contract() -> None:
    source = _mesh_inputs()
    mesh = CanonicalMesh(**source)

    source["nodes"][0][0] = 99.0
    source["cells"][0][0] = 2
    source["boundary_facets"].append([0, 2])
    source["semantic_tags"]["fixed_edge"].append(2)
    source["region_ids"].append(8)
    source["quality_metrics"]["minimum_angle_degrees"] = 1.0
    source["source_provenance"]["provider"] = "mutated"

    assert mesh.nodes == ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    assert mesh.cells == ((0, 1, 2),)
    assert mesh.boundary_facets == ((0, 1), (1, 2), (2, 0))
    assert tuple(mesh.semantic_tags["fixed_edge"]) == (0,)
    assert mesh.region_ids == (7,)
    assert mesh.quality_metrics["minimum_angle_degrees"] == 45.0
    assert mesh.source_provenance["provider"] == "contract-test"
    assert hash(mesh)

    with pytest.raises(TypeError):
        mesh.semantic_tags["new_edge"] = (2,)
    with pytest.raises(TypeError):
        mesh.quality_metrics["aspect_ratio"] = 1.0
    with pytest.raises(TypeError):
        mesh.source_provenance["provider"] = "other"


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("nodes", [[0.0, 0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        ("cells", [[0, 1, 3]]),
        ("boundary_facets", [[0, 1, 2]]),
        ("region_ids", [7, 8]),
        ("unit", Unit.CM),
    ],
)
def test_canonical_mesh_rejects_malformed_structural_payloads(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _mesh_inputs()
    payload[field_name] = invalid_value

    with pytest.raises(ContractValidationError):
        CanonicalMesh(**payload)


def test_canonical_mesh_rejects_non_finite_numbers_with_typed_error() -> None:
    payload = _mesh_inputs()
    payload["nodes"] = [[float("nan"), 0.0], [1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(NonFiniteNumberError):
        CanonicalMesh(**payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("nodes", []),
        ("nodes", [["zero", 0.0], [1.0, 0.0], [0.0, 1.0]]),
        ("cells", [[0, 1]]),
        ("cells", [[0, 1, True]]),
        ("boundary_facets", [[0, 3]]),
        ("boundary_facets", [[0, True]]),
        ("region_ids", [True]),
        ("semantic_tags", {"fixed_edge": [3]}),
        ("semantic_tags", (("fixed_edge", [0]),)),
        ("semantic_tags", {"": [0]}),
        ("quality_metrics", (("minimum_angle_degrees", 45.0),)),
        ("quality_metrics", {"": 45.0}),
        ("source_provenance", (("provider", "contract-test"),)),
        ("source_provenance", {"provider": 7}),
    ],
)
def test_canonical_mesh_rejects_additional_invalid_payloads(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _mesh_inputs()
    payload[field_name] = invalid_value

    with pytest.raises(ContractValidationError):
        CanonicalMesh(**payload)
