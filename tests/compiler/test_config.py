from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from piginot.compiler.config import CampaignConfig, CompilerConfigError
from piginot.contracts import canonical_json_bytes
from tests.compiler.config_fixtures import config_document


def _at(document: dict[str, Any], path: str) -> dict[str, Any]:
    current = document
    for name in path.split(".") if path else ():
        current = current[name]
    return current


def test_roundtrip_normalizes_defaults_and_binds_every_input(tmp_path: Path) -> None:
    document = config_document()
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    config = CampaignConfig.load(path)
    assert config == CampaignConfig.from_document(document)
    assert config.source.path == "must-not-read.json"
    assert config.source.fit_case_ids == ("fit-001", "fit-002")
    assert config.model.hidden_dim == 64
    assert config.model.parameter_names == ("length", "nominal_axial_strain")
    assert config.execution.training.dtype == "float32"
    assert (
        config.execution.reference.dtype
        == config.execution.inference_dtype
        == "float64"
    )
    assert config.execution.reference.seed == 2802
    serialized = config.document()
    assert "$type" not in json.dumps(serialized)
    assert CampaignConfig.from_document(serialized) == config
    assert (
        config.config_sha256
        == hashlib.sha256(canonical_json_bytes(serialized)).hexdigest()
    )
    assert (
        CampaignConfig.from_document(
            dict(reversed(list(serialized.items())))
        ).config_sha256
        == config.config_sha256
    )
    changed = copy.deepcopy(serialized)
    changed["reference"]["refinement_levels"] = [1, 2, 4]
    assert CampaignConfig.from_document(changed).config_sha256 != config.config_sha256


def test_contract_copies_and_freezes_nested_values() -> None:
    document = config_document()
    config = CampaignConfig.from_document(document)
    original_hash = config.config_sha256
    document["model"]["parameter_names"].append("later")
    document["execution"]["training"]["library_versions"]["torch"] = "changed"
    assert config.model.parameter_names == ("length", "nominal_axial_strain")
    assert config.config_sha256 == original_hash
    exported = config.document()
    exported["source"]["fit_case_ids"].clear()
    exported["execution"]["training"]["platform_fingerprint"]["os"] = "changed"
    assert config.config_sha256 == original_hash
    with pytest.raises(FrozenInstanceError):
        config.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.training.seed = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.execution.training.library_versions["torch"] = "changed"  # type: ignore[index]


_SECTIONS = (
    "",
    "adapter",
    "source",
    "model",
    "training",
    "objective",
    "reference",
    "execution",
    "execution.training",
    "execution.reference",
    "selection",
)


@pytest.mark.parametrize("section", _SECTIONS)
def test_unknown_keys_are_refused_at_every_object(section: str) -> None:
    document = config_document()
    _at(document, section)["silent_override"] = True
    with pytest.raises(CompilerConfigError) as failure:
        CampaignConfig.from_document(document)
    assert failure.value.rule == "unknown-key"


@pytest.mark.parametrize("section", _SECTIONS)
def test_required_keys_are_refused_at_every_object(section: str) -> None:
    document = config_document()
    target = _at(document, section)
    del target[next(iter(target))]
    with pytest.raises(CompilerConfigError) as failure:
        CampaignConfig.from_document(document)
    assert failure.value.rule == "missing-key"


@pytest.mark.parametrize("value", [None, [], "config", 1, True])
def test_root_requires_an_object(value: object) -> None:
    with pytest.raises(CompilerConfigError):
        CampaignConfig.from_document(value)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("", "schema", "future-schema"),
        ("", "name", " bad"),
        ("", "name", ""),
        ("", "normalization", "all-evidence"),
        ("adapter", "id", "module:callable"),
        ("adapter", "implementation_sha256", "A" * 64),
        ("adapter", "implementation_sha256", "a" * 63),
        ("source", "kind", "private-store"),
        ("source", "path", ""),
        ("source", "sha256", "invalid"),
        ("source", "fit_case_ids", []),
        ("source", "fit_case_ids", "fit-001"),
        ("source", "fit_case_ids", ["same", "same"]),
        ("source", "screen_case_ids", []),
        ("source", "screen_case_ids", ["fit-001"]),
        ("source", "development_case_ids", ["fit-001"]),
        ("source", "development_case_ids", ["development-001"]),
        ("model", "architecture_id", ""),
        ("model", "parameter_names", []),
        ("model", "parameter_names", ["length", "length"]),
        ("model", "parameter_names", [" length"]),
        ("model", "hidden_dim", 0),
        ("model", "hidden_dim", 65),
        ("model", "num_attention_heads", True),
        ("model", "output_init_std", 0),
        ("model", "output_init_std", float("nan")),
        ("model", "output_init_std", 2**2048),
        ("training", "learning_rate", 0),
        ("training", "learning_rate", "0.001"),
        ("training", "learning_rate", float("inf")),
        ("training", "final_learning_rate", 1),
        ("training", "constant_steps", 5),
        ("objective", "barrier_weight", 0),
        ("objective", "barrier_threshold", -1),
        ("objective", "admissibility_floor", 0.05),
        ("reference", "initialization", "automatic-rescue"),
        ("reference", "refinement_levels", []),
        ("reference", "refinement_levels", [2, 4]),
        ("reference", "refinement_levels", [1, 1]),
        ("reference", "refinement_levels", [1, 4, 2]),
        ("reference", "refinement_levels", [1, True]),
        ("reference", "workers", 3),
        ("reference", "workers", True),
        ("selection", "checkpoint_rule", "best-on-development"),
        ("selection", "development", 1),
        ("execution", "inference_dtype", "float32"),
        ("execution.training", "dtype", "float16"),
        ("execution.training", "device", "cuda"),
        ("execution.training", "blas_threads", 2),
        ("execution.training", "container_digest", 1),
        ("execution.training", "library_versions", []),
        ("execution.training", "seed", 2),
        ("execution.reference", "dtype", "float32"),
        ("execution.reference", "python_version", "3.12.0"),
        ("execution.reference", "library_versions", {"torch": "different"}),
        ("execution.reference", "platform_fingerprint", {"os": "Linux"}),
    ],
)
def test_invalid_values_and_conflicting_inputs_refuse(
    section: str, field: str, value: object
) -> None:
    document = config_document()
    _at(document, section)[field] = value
    with pytest.raises(CompilerConfigError):
        CampaignConfig.from_document(document)


@pytest.mark.parametrize(
    "field",
    [
        "optimizer_steps",
        "case_batch_size",
        "checkpoint_interval",
        "seed",
        "constant_steps",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, -1, 2**53])
def test_training_integer_contract(field: str, value: object) -> None:
    document = config_document()
    document["training"][field] = value
    with pytest.raises(CompilerConfigError):
        CampaignConfig.from_document(document)


def test_explicit_development_and_full_profile_roundtrip() -> None:
    document = config_document()
    document["selection"]["development"] = True
    document["source"]["development_case_ids"] = ["development-001"]
    for key in ("training", "reference"):
        document["execution"][key]["container_digest"] = "sha256:" + "c" * 64
    document["execution"]["training"]["dtype"] = "float64"
    document["reference"]["initialization"] = "neo-hookean-full-load-hgo"
    config = CampaignConfig.from_document(document)
    assert config.selection.development is True
    assert config.execution.training.container_digest == "sha256:" + "c" * 64
    assert CampaignConfig.from_document(config.document()) == config
    with pytest.raises(CompilerConfigError):
        replace(config, selection=replace(config.selection, development=False))


def test_direct_values_preserve_exact_nested_contracts() -> None:
    config = CampaignConfig.from_document(config_document())
    with pytest.raises(CompilerConfigError):
        replace(config, adapter=object())
    with pytest.raises(CompilerConfigError):
        replace(config.execution, training=object())
    with pytest.raises(CompilerConfigError):
        replace(
            config.execution,
            training=replace(config.execution.training, platform_fingerprint=None),
        )
    with pytest.raises(CompilerConfigError):
        replace(config.reference, refinement_levels=[])


@pytest.mark.parametrize("versions", [{}, {"torch": " 2.13.0"}])
def test_direct_profiles_preserve_parser_metadata_rules(
    versions: dict[str, str],
) -> None:
    config = CampaignConfig.from_document(config_document())
    training = replace(config.execution.training, library_versions=versions)
    reference = replace(config.execution.reference, library_versions=versions)
    with pytest.raises(CompilerConfigError):
        replace(config.execution, training=training, reference=reference)


@pytest.mark.parametrize(
    "missing",
    ["numpy", "torch", "scipy", "scikit-fem", "sympy", "safetensors", "threadpoolctl"],
)
def test_profiles_bind_every_used_numerical_library(missing):
    document = config_document()
    # Start complete so this isolates each omission even before implementation.
    versions = {
        name: "1.0"
        for name in (
            "numpy",
            "torch",
            "scipy",
            "scikit-fem",
            "sympy",
            "safetensors",
            "threadpoolctl",
        )
    }
    for profile_name in ("training", "reference"):
        document["execution"][profile_name]["library_versions"] = {
            key: value for key, value in versions.items() if key != missing
        }
    with pytest.raises(CompilerConfigError, match="numerical library"):
        CampaignConfig.from_document(document)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema":1,"schema":2}',
        '{"x":NaN}',
        '{"x":9007199254740992}',
        "{",
        '"not an object"',
    ],
)
def test_load_strict_json(payload: str, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(CompilerConfigError):
        CampaignConfig.load(path)


def test_load_does_not_open_source_and_imports_no_numerical_stack(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_document()))
    program = (
        "import sys; from piginot.compiler.config import CampaignConfig; "
        f"c=CampaignConfig.load({str(path)!r}); "
        "assert not any(n == p or n.startswith(p+'.') for n in sys.modules "
        "for p in ('torch','numpy','scipy','skfem','piginot.fem','piginot.problem')); "
        "assert c.source.path == 'must-not-read.json'"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
