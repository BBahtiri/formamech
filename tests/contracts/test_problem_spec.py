"""S24 ProblemSpec contract: reading, identity and every rejection rule."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from piginot.contracts import ProblemSpec, ProblemSpecError
from piginot.contracts.problem_spec import (
    BoundaryCondition,
    Constant,
    Expression,
    Lifting,
    ParameterRange,
    Piece,
    Value,
)
from piginot.contracts.units import Dimension, Unit
from tests.problem_helpers import adf_lifting, edited, plate_document, plate_mesh

Document = dict[str, Any]


def spec(document: Document | None = None) -> ProblemSpec:
    return ProblemSpec.from_document(plate_document() if document is None else document)


def refused(document: Document) -> str:
    with pytest.raises(ProblemSpecError) as error:
        ProblemSpec.from_document(document)
    return error.value.rule


# ------------------------------------------------------------------ reading


@pytest.mark.parametrize("lifting", ["quarter", "adf"])
def test_document_round_trips(lifting: str) -> None:
    document = plate_document(lifting)
    problem = ProblemSpec.from_document(document)
    assert problem.document() == document
    assert ProblemSpec.from_document(problem.document()) == problem
    assert ProblemSpec.from_bytes(json.dumps(document).encode()) == problem


def test_load_reads_a_file(tmp_path: Any) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(plate_document()), encoding="utf-8")
    assert ProblemSpec.load(path) == spec()
    assert ProblemSpec.load(str(path)).problem_hash == spec().problem_hash


def test_values_and_scope_add_the_constants() -> None:
    problem = spec()
    inputs = {"width": 12.0, "height": 5, "strain": 0.07, "modulus": 760.0}
    with pytest.raises(KeyError):
        problem.values(inputs)
    inputs["poisson"] = 0.3
    values = problem.values(inputs)
    assert values["unit_length"] == 1.0 and values["height"] == 5.0
    assert problem.scope({"width": 3.0}) == {"unit_length": 1.0, "width": 3.0}
    assert problem.parameter("strain").role == "load"
    with pytest.raises(KeyError):
        problem.parameter("missing")


def test_expressions_evaluate_left_to_right_without_eval() -> None:
    from piginot.problem.expressions import value

    expression = Expression.parse("0.5 * a * b - -c / d", "e")
    assert expression.names == {"a", "b", "c", "d"}
    assert value(expression, {"a": 3.0, "b": 4.0, "c": 2.0, "d": 4.0}) == 6.5
    assert value(expression, {"a": 0.1, "b": 0.7, "c": 0.0, "d": 1.0}) == (
        (0.5 * 0.1) * 0.7
    )
    assert Expression.parse("+2", "e").constant == 2.0
    assert Expression.parse("-(2 - 6) / 2 * 3 + 1", "e").constant == 7.0
    assert Expression.parse("x", "e").constant is None
    assert Expression.parse("(x)", "e") == Expression.parse(" x ", "e")
    with pytest.raises(ZeroDivisionError):
        value(Expression.parse("a / b", "e"), {"a": 1.0, "b": 0.0})
    with pytest.raises(ZeroDivisionError):
        _ = Expression.parse("1 / 0", "e").constant
    assert value(Expression.parse("a + b", "e"), {"a": 1.0, "b": 2.0}) == 3.0


def test_the_contract_folds_constants_only() -> None:
    from piginot.contracts.problem_spec import _constant_value

    with pytest.raises(ValueError, match="no names"):
        _constant_value(Expression.parse("x", "e").tree)


@pytest.mark.parametrize(
    "source", ["__import__('os')", "a ** 2", "f(a)", "a.b", "[a]", "", 3]
)
def test_unsafe_or_invalid_expressions_are_refused(source: object) -> None:
    with pytest.raises(ProblemSpecError):
        Expression.parse(source, "e")


# ----------------------------------------------------------------- identity


def _encodings(document: Document) -> Iterator[bytes]:
    text = json.dumps(document, indent=2)
    yield text.encode("utf-8")
    yield b"\xef\xbb\xbf" + text.encode("utf-8")
    yield text.encode("utf-16")
    yield text.encode("utf-32")
    reordered = json.dumps(
        dict(reversed(list(document.items()))), separators=(",", ":")
    )
    yield reordered.encode("utf-8")


def test_hash_is_stable_under_formatting() -> None:
    expected = spec().problem_hash
    for payload in _encodings(plate_document()):
        assert ProblemSpec.from_bytes(payload).problem_hash == expected
    spelled = json.dumps(plate_document()).replace('"lo": 760', '"lo": 7.6e2')
    spelled = spelled.replace('"hi": 760', '"hi": 760.0')
    spelled = spelled.replace('"strain * width"', '"(strain)*width"')
    spelled = spelled.replace('"width / height"', '"(width/height)"')
    assert spelled != json.dumps(plate_document())
    assert ProblemSpec.from_bytes(spelled.encode()).problem_hash == expected


def _leaves(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path


def _edits(value: Any) -> list[Any]:
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, (int, float)):
        return [value * 1.25 + 0.5, value * 0.8]
    return [value + " + 0", value + "z"]


@pytest.mark.parametrize("lifting", ["quarter", "adf"])
def test_every_semantic_leaf_edit_changes_the_hash(lifting: str) -> None:
    document = plate_document(lifting)
    expected = ProblemSpec.from_document(document).problem_hash
    accepted = 0
    for path in _leaves(document):
        target: Any = document
        for key in path:
            target = target[key]
        for value in _edits(target):
            try:
                changed = ProblemSpec.from_document(edited(document, path, value))
            except ProblemSpecError:
                continue
            accepted += 1
            assert changed.problem_hash != expected, path
    assert accepted >= 30


# ---------------------------------------------------------------- rejections

Edit = Callable[[Document], Document]


def _set(*path: Any, value: Any) -> Edit:
    return lambda document: edited(document, path, value)


def _drop(*path: Any) -> Edit:
    return lambda document: edited(document, path, ...)


def _append(*path: Any, value: Any) -> Edit:
    def edit(document: Document) -> Document:
        result = copy.deepcopy(document)
        target: Any = result
        for key in path:
            target = target[key]
        target.append(value)
        return result

    return edit


_FREE = {"name": "free", "region": "top", "kind": "traction_free"}
_PULL = plate_document()["boundary_conditions"][2]

REJECTIONS: list[tuple[str, Edit, str]] = [
    ("not an object", lambda d: [], "type"),  # type: ignore[arg-type, return-value]
    ("unknown top key", _set("extra", value=1), "unknown-key"),
    ("unknown nested key", _set("parameters", 0, "mean", value=1), "unknown-key"),
    ("missing key", _drop("fem"), "missing-key"),
    ("missing nested key", _drop("geometry", "frame"), "missing-key"),
    ("schema", _set("schema", value="piginot.problem-spec-v0"), "schema"),
    ("spec name", _set("name", value="Plate V1"), "name"),
    ("blank text", _set("name", value=" plate"), "type"),
    ("physics", _set("physics", value="3d-solid"), "physics"),
    ("no parameters", _set("parameters", value=[]), "missing-key"),
    ("parameters not a list", _set("parameters", value={}), "type"),
    ("parameter identifier", _set("parameters", 0, "name", value="2w"), "name"),
    ("parameter role", _set("parameters", 0, "role", value="boundary"), "role"),
    ("missing unit", _drop("parameters", 0, "unit"), "missing-key"),
    ("unit not text", _set("parameters", 0, "unit", value=1), "unit"),
    ("non-canonical unit", _set("parameters", 3, "unit", value="GPa"), "unit"),
    ("unknown unit", _set("parameters", 0, "unit", value="furlong"), "unit"),
    ("lo above hi", _set("parameters", 0, "lo", value=30), "range"),
    ("lo not a number", _set("parameters", 0, "lo", value="10"), "type"),
    ("boolean number", _set("parameters", 0, "hi", value=True), "type"),
    ("fixed not fixed", _set("parameters", 3, "hi", value=800), "distribution"),
    ("uniform degenerate", _set("parameters", 0, "hi", value=10), "distribution"),
    (
        "distribution",
        _set("parameters", 0, "distribution", value="normal"),
        "distribution",
    ),
    (
        "duplicate parameter",
        _append("parameters", value=plate_document()["parameters"][0]),
        "duplicate",
    ),
    (
        "constant shadows parameter",
        _set("constants", 0, "name", value="width"),
        "duplicate",
    ),
    ("constant unit", _set("constants", 0, "unit", value="cm"), "unit"),
    (
        "constraint unknown name",
        _set("constraints", 0, "expression", value="w / height"),
        "expression",
    ),
    (
        "constraint dimension",
        _set("constraints", 0, "expression", value="width"),
        "expression",
    ),
    (
        "constraint syntax",
        _set("constraints", 0, "expression", value="width ** 2"),
        "expression",
    ),
    ("constraint range", _set("constraints", 0, "lo", value=9), "range"),
    ("mesh id", _set("geometry", "reference_mesh_id", value="A" * 64), "hash"),
    ("no regions", _set("geometry", "regions", value=[]), "region"),
    ("duplicate region", _append("geometry", "regions", value="top"), "duplicate"),
    (
        "frame dimension",
        _set("geometry", "frame", "extent", 0, value="strain"),
        "expression",
    ),
    ("frame pair", _set("geometry", "frame", "origin", value=["0"]), "type"),
    ("material model", _set("material", "model", value="mooney_rivlin"), "material"),
    (
        "material properties",
        _drop("material", "parameters", "poisson_ratio"),
        "material",
    ),
    ("material parameters", _set("material", "parameters", value=[]), "type"),
    (
        "material unknown parameter",
        _set("material", "parameters", "poisson_ratio", value="nu"),
        "material",
    ),
    (
        "material wrong role",
        _set("material", "parameters", "poisson_ratio", value="strain"),
        "material",
    ),
    ("bc region", _set("boundary_conditions", 0, "region", value="hole"), "region"),
    ("bc kind", _set("boundary_conditions", 0, "kind", value="roller"), "kind"),
    (
        "bc component",
        _set("boundary_conditions", 0, "components", value=["z"]),
        "components",
    ),
    (
        "bc repeated component",
        _set("boundary_conditions", 0, "components", value=["x", "x"]),
        "duplicate",
    ),
    ("bc no components", _drop("boundary_conditions", 0, "components"), "components"),
    (
        "symmetry with values",
        _set(
            "boundary_conditions",
            0,
            "values",
            value=[{"expression": "0", "unit": "mm"}],
        ),
        "values",
    ),
    (
        "free with components",
        _append("boundary_conditions", value={**_FREE, "components": ["x"]}),
        "components",
    ),
    (
        "value count",
        _set("boundary_conditions", 2, "components", value=["x", "y"]),
        "values",
    ),
    (
        "value form",
        _set("boundary_conditions", 2, "values", 0, value={"expression": "0"}),
        "missing-key",
    ),
    (
        "value both forms",
        _set(
            "boundary_conditions",
            2,
            "values",
            0,
            value={"expression": "0", "unit": "mm", "nominal_strain": "strain"},
        ),
        "unknown-key",
    ),
    (
        "value unit",
        _set(
            "boundary_conditions",
            2,
            "values",
            0,
            value={"expression": "strain * width", "unit": "MPa"},
        ),
        "unit",
    ),
    (
        "value dimension",
        _set(
            "boundary_conditions",
            2,
            "values",
            0,
            value={"expression": "strain", "unit": "mm"},
        ),
        "expression",
    ),
    (
        "nominal strain unknown",
        _set("boundary_conditions", 2, "values", 0, value={"nominal_strain": "eps"}),
        "expression",
    ),
    (
        "nominal strain dimension",
        _set("boundary_conditions", 2, "values", 0, value={"nominal_strain": "width"}),
        "expression",
    ),
    (
        "dirichlet and neumann",
        _append(
            "boundary_conditions",
            value={
                "name": "push",
                "region": "right",
                "kind": "neumann",
                "components": ["x"],
                "values": [{"expression": "1", "unit": "MPa"}],
            },
        ),
        "dirichlet-neumann",
    ),
    (
        "neumann strain",
        _append(
            "boundary_conditions",
            value={
                "name": "push",
                "region": "top",
                "kind": "neumann",
                "components": ["y"],
                "values": [{"nominal_strain": "strain"}],
            },
        ),
        "values",
    ),
    (
        "neumann unit",
        _append(
            "boundary_conditions",
            value={
                "name": "push",
                "region": "top",
                "kind": "neumann",
                "components": ["y"],
                "values": [{"expression": "1", "unit": "mm"}],
            },
        ),
        "unit",
    ),
    (
        "prescribed twice",
        _append("boundary_conditions", value={**_PULL, "name": "pull2"}),
        "duplicate",
    ),
    (
        "bc name twice",
        _append("boundary_conditions", value={**_FREE, "name": "pull"}),
        "duplicate",
    ),
    (
        "load path",
        _set("boundary_conditions", 2, "load_path", value=[0, 1]),
        "load-path",
    ),
    ("no y constraint", _drop("boundary_conditions", 1), "rigid-modes"),
    ("lifting kind", _set("lifting", "kind", value="harmonic"), "lifting"),
    (
        "lifting scale",
        _set("lifting", "displacement_scale", value="strain"),
        "expression",
    ),
    (
        "quarter with pieces",
        _set("lifting", "pieces", value=adf_lifting()["pieces"]),
        "lifting",
    ),
    (
        "quarter frame",
        _set("geometry", "frame", "origin", value=["1", "0"]),
        "lifting-coverage",
    ),
    (
        "quarter structure",
        _set("boundary_conditions", 1, "components", value=["x"]),
        "rigid-modes",
    ),
    (
        "quarter loaded y",
        _set(
            "boundary_conditions",
            1,
            value={
                "name": "bottom_y",
                "region": "bottom",
                "kind": "dirichlet",
                "components": ["y"],
                "values": [{"expression": "1", "unit": "mm"}],
            },
        ),
        "lifting-coverage",
    ),
    ("verification quantile", _set("verification", "quantile", value=1.0), "range"),
    (
        "verification quantities",
        _drop("verification", "tolerances", "u_relative_l2"),
        "verification",
    ),
    (
        "verification tolerance",
        _set("verification", "tolerances", "u_relative_l2", value=0),
        "range",
    ),
    ("verification tolerances", _set("verification", "tolerances", value=[]), "type"),
    ("split role", _drop("split", "reserve"), "missing-key"),
    ("split count", _set("split", "train", value=0), "range"),
    ("split negative", _set("split", "pilot", value=-1), "range"),
    ("split integer", _set("split", "pilot", value=1.5), "type"),
    ("fem order", _set("fem", "element_order", value=3), "range"),
    ("fem tolerance", _set("fem", "solver_tolerance", value=0), "range"),
    ("qoi name twice", _append("qois", value=plate_document()["qois"][0]), "duplicate"),
    ("qoi region", _set("qois", 0, "region", value="hole"), "region"),
    (
        "qoi exclude region",
        _set("qois", 0, "exclude_regions", value=["hole"]),
        "region",
    ),
    ("qoi field", _set("qois", 0, "field", value="u"), "qoi"),
    ("qoi reduction", _set("qois", 0, "reduction", value="mean"), "qoi"),
    ("json", lambda d: d, "json"),
]


@pytest.mark.parametrize(
    ("label", "edit", "rule"), REJECTIONS, ids=[item[0] for item in REJECTIONS]
)
def test_each_rule_refuses_a_spec(label: str, edit: Edit, rule: str) -> None:
    if rule == "json":
        with pytest.raises(ProblemSpecError) as error:
            ProblemSpec.from_bytes(b'{"a": 1, "a": 2}')
        assert error.value.rule == "json"
        return
    assert refused(edit(plate_document())) == rule


ADF_REJECTIONS: list[tuple[str, Edit, str]] = [
    ("missing condition", _drop("lifting", "pieces", "pull"), "lifting-coverage"),
    (
        "extra condition",
        _set(
            "lifting",
            "pieces",
            "top_free",
            value=[{"line": {"start": ["0", "0"], "end": ["1", "0"]}}],
        ),
        "lifting-coverage",
    ),
    ("no pieces", _set("lifting", "pieces", "pull", value=[]), "lifting"),
    ("pieces not an object", _set("lifting", "pieces", value=[]), "type"),
    ("piece not an object", _set("lifting", "pieces", "pull", 0, value=[]), "type"),
    (
        "piece kind",
        _set("lifting", "pieces", "pull", 0, value={"spline": {}}),
        "lifting",
    ),
    (
        "two kinds",
        _set("lifting", "pieces", "pull", 0, value={"line": {}, "arc": {}}),
        "lifting",
    ),
    (
        "arc without center",
        _set(
            "lifting",
            "pieces",
            "pull",
            0,
            value={"arc": {"start": ["0", "0"], "end": ["1", "0"]}},
        ),
        "missing-key",
    ),
    (
        "line with center",
        _set(
            "lifting",
            "pieces",
            "pull",
            0,
            value={
                "line": {"start": ["0", "0"], "end": ["1", "0"], "center": ["0", "0"]}
            },
        ),
        "unknown-key",
    ),
    (
        "untrimmed arc",
        _set(
            "lifting",
            "pieces",
            "pull",
            0,
            value={
                "arc": {"start": ["0", "0"], "end": ["1", "0"], "center": ["0", "1"]},
                "trimmed": False,
            },
        ),
        "lifting",
    ),
    (
        "trimmed flag",
        _set("lifting", "pieces", "pull", 0, "trimmed", value="yes"),
        "type",
    ),
    (
        "piece dimension",
        _set("lifting", "pieces", "pull", 0, "line", "end", 1, value="strain"),
        "expression",
    ),
]


@pytest.mark.parametrize(
    ("label", "edit", "rule"), ADF_REJECTIONS, ids=[item[0] for item in ADF_REJECTIONS]
)
def test_each_adf_rule_refuses_a_spec(label: str, edit: Edit, rule: str) -> None:
    assert refused(edit(plate_document("adf"))) == rule


def test_neumann_traction_validates() -> None:
    document = _append(
        "boundary_conditions",
        value={
            "name": "push",
            "region": "top",
            "kind": "neumann",
            "components": ["y"],
            "values": [{"expression": "1", "unit": "MPa"}],
        },
    )(plate_document())
    problem = spec(document)
    assert problem.boundary_conditions[-1].kind == "neumann"
    assert not problem.boundary_conditions[-1].is_dirichlet


def test_section_constructors_check_their_own_invariants() -> None:
    with pytest.raises(ProblemSpecError, match="canonical"):
        ParameterRange("a", "geometry", 0.0, 1.0, Unit.CM, "uniform")
    with pytest.raises(ProblemSpecError, match="expression or a nominal"):
        Value()
    zero = Expression.parse("0", "e")
    with pytest.raises(ProblemSpecError, match="needs a unit"):
        Value(expression=zero)
    assert Value(expression=zero, unit=Unit.MM).is_zero
    assert not Value(nominal_strain="e").is_zero
    with pytest.raises(ProblemSpecError, match="reserved"):
        BoundaryCondition("a", "r", "symmetry", ("x",), (), load_path=[1])  # type: ignore[arg-type]
    pair = (zero, zero)
    with pytest.raises(ProblemSpecError, match="center"):
        Piece("line", pair, pair, center=pair)
    with pytest.raises(ProblemSpecError, match="a piece is one of"):
        Piece("spline", pair, pair)
    with pytest.raises(ProblemSpecError, match="true or false"):
        Piece("line", pair, pair, trimmed=1)  # type: ignore[arg-type]
    with pytest.raises(ProblemSpecError, match="at least one piece"):
        Lifting("adf", zero, {"a": ()})
    with pytest.raises(ProblemSpecError, match="canonical"):
        Constant("c", 1.0, Unit.M)
    condition = BoundaryCondition("a", "r", "symmetry", ("x",))
    assert condition.value("x") is None and condition.is_zero("x")
    assert Constant("c", 1.0, Unit.MM).document() == {
        "name": "c",
        "value": 1.0,
        "unit": "mm",
    }


def test_bare_numbers_take_the_expected_dimension() -> None:
    Expression.parse("25", "e").check({}, Dimension.LENGTH, "e")
    with pytest.raises(ProblemSpecError, match="unknown parameter"):
        Expression.parse("2 * w", "e").check({}, Dimension.LENGTH, "e")


# --------------------------------------------------------------- mesh rules


def test_the_reference_mesh_passes_the_mesh_rules() -> None:
    spec().check_mesh(plate_mesh())
    spec(plate_document("adf")).check_mesh(plate_mesh())


def test_another_mesh_is_refused() -> None:
    with pytest.raises(ProblemSpecError) as error:
        spec().check_mesh(plate_mesh(mesh_id="c" * 64))
    assert error.value.rule == "mesh-id"


def test_a_region_absent_from_the_mesh_is_refused() -> None:
    document = _append("geometry", "regions", value="hole")(plate_document())
    with pytest.raises(ProblemSpecError) as error:
        spec(document).check_mesh(plate_mesh())
    assert error.value.rule == "region"


def test_a_free_rotation_is_refused() -> None:
    document = plate_document()
    document["boundary_conditions"] = [
        {
            "name": "bottom_x",
            "region": "bottom",
            "kind": "symmetry",
            "components": ["x"],
        },
        {"name": "left_y", "region": "left", "kind": "symmetry", "components": ["y"]},
    ]
    document["lifting"] = {
        "kind": "adf",
        "displacement_scale": "strain * width",
        "pieces": {
            "bottom_x": [{"line": {"start": ["0", "0"], "end": ["width", "0"]}}],
            "left_y": [{"line": {"start": ["0", "height"], "end": ["0", "0"]}}],
        },
    }
    problem = spec(document)
    with pytest.raises(ProblemSpecError) as error:
        problem.check_mesh(plate_mesh())
    assert error.value.rule == "rigid-modes"
    document["boundary_conditions"][1]["components"] = ["x", "y"]
    spec(document).check_mesh(plate_mesh())


def test_a_nominal_strain_on_a_parallel_region_is_refused() -> None:
    document = plate_document("adf")
    document["boundary_conditions"][2]["region"] = "top"
    document["boundary_conditions"][3]["region"] = "right"
    with pytest.raises(ProblemSpecError) as error:
        spec(document).check_mesh(plate_mesh())
    assert error.value.rule == "nominal-strain-region"
