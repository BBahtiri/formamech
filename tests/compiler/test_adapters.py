"""Installed compiler adapters are resolved by registered identity only."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from types import SimpleNamespace

import pytest

from piginot.compiler import adapters


def descriptor() -> adapters.AdapterDescriptor:
    return adapters.AdapterDescriptor(
        adapter_id="unit-adapter",
        version="1.0",
        implementation_sha256="a" * 64,
        geometries=(adapters.GeometryIdentity("b" * 64, "c" * 64),),
        materials=("compressible_neo_hookean", "formula-isotropic"),
        liftings=("quarter", "adf"),
        architectures=("pi-ginot-energy-parametric-film-v1",),
        reference_initializations=("cold",),
    )


@dataclass(frozen=True)
class Installed:
    descriptor: adapters.AdapterDescriptor


def test_registered_adapter_resolves_without_other_plugins_loading(monkeypatch):
    selected = Installed(descriptor())
    other = SimpleNamespace(name="other", load=lambda: pytest.fail("wrong plugin"))
    entry = SimpleNamespace(name="unit-adapter", load=lambda: selected)

    def entries(*, group):
        assert group == "piginot.compiler.adapters"
        return (other, entry)

    monkeypatch.setattr(adapters, "entry_points", entries)
    assert adapters.resolve_adapter("unit-adapter") is selected
    assert descriptor().document()["geometries"] == [
        {"reference_mesh_id": "b" * 64, "morph_map_sha256": "c" * 64}
    ]


@pytest.mark.parametrize("name", ["module:attribute", "../plugin", "", " plugin"])
def test_arbitrary_import_or_noncanonical_names_refuse_before_discovery(
    monkeypatch, name
):
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: pytest.fail("discovery"))
    with pytest.raises(adapters.AdapterError, match="identity"):
        adapters.resolve_adapter(name)


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_ambiguous_registered_names_refuse(monkeypatch, count):
    entry = SimpleNamespace(name="unit-adapter", load=lambda: pytest.fail("load"))
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: (entry,) * count)
    with pytest.raises(adapters.AdapterError, match="registered"):
        adapters.resolve_adapter("unit-adapter")


def test_unavailable_plugin_is_an_explicit_capability_error(monkeypatch):
    def unavailable():
        raise ImportError("optional extra absent")

    entry = SimpleNamespace(name="unit-adapter", load=unavailable)
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: (entry,))
    with pytest.raises(adapters.AdapterError, match="optional extra absent"):
        adapters.resolve_adapter("unit-adapter")


@pytest.mark.parametrize("value", [object(), SimpleNamespace(descriptor={})])
def test_untyped_plugin_descriptor_is_rejected(monkeypatch, value):
    entry = SimpleNamespace(name="unit-adapter", load=lambda: value)
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: (entry,))
    with pytest.raises(adapters.AdapterError, match="descriptor"):
        adapters.resolve_adapter("unit-adapter")


def test_descriptor_must_agree_with_the_registered_name(monkeypatch):
    entry = SimpleNamespace(name="different", load=lambda: Installed(descriptor()))
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: (entry,))
    with pytest.raises(adapters.AdapterError, match="identity"):
        adapters.resolve_adapter("different")


@pytest.mark.parametrize(
    "change",
    [
        {"adapter_id": "bad:import"},
        {"version": " "},
        {"implementation_sha256": "A" * 64},
        {"geometries": ()},
        {"geometries": (object(),)},
        {"geometries": (adapters.GeometryIdentity("b" * 64, "c" * 64),) * 2},
        {"materials": ["compressible_neo_hookean"]},
        {"materials": ("compressible_neo_hookean",) * 2},
        {"liftings": ("harmonic",)},
        {"architectures": ("pi-ginot-boundary-energy-v1",)},
        {"reference_initializations": ("change-iterations-on-failure",)},
    ],
)
def test_descriptor_contract_rejects_unsupported_or_ambiguous_metadata(change):
    with pytest.raises(adapters.AdapterError):
        replace(descriptor(), **change)


@pytest.mark.parametrize("values", [("bad", "c" * 64), ("b" * 64, "bad")])
def test_geometry_identity_requires_both_exact_digests(values):
    with pytest.raises(adapters.AdapterError):
        adapters.GeometryIdentity(*values)


def test_metadata_is_immutable_and_documents_do_not_mutate_it():
    original = descriptor()
    with pytest.raises(FrozenInstanceError):
        original.version = "2.0"
    document = original.document()
    document["geometries"].clear()
    assert len(original.geometries) == 1
