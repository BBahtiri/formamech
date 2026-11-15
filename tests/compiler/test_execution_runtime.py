"""Runtime identity guards are checked with synthetic observations."""

from dataclasses import replace

import pytest

from piginot.compiler import execution_runtime
from piginot.contracts import ExecutionProfile
from tests.compiler.config_fixtures import profile


def declared():
    return ExecutionProfile(**profile())


def observed():
    return {
        "python_version": "3.11.15",
        "library_versions": profile()["library_versions"],
        "platform_fingerprint": {
            "os": "Linux",
            "machine": "x86_64",
            "cpu_capability": "AVX2",
        },
        "torch_threads": 1,
        "threadpools": [{"num_threads": 1}],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("python_version", "3.12.0"),
        ("library_versions", {"numpy": "0", "torch": "2.13.0"}),
        (
            "platform_fingerprint",
            {"os": "Linux", "machine": "arm64", "cpu_capability": "AVX2"},
        ),
        ("torch_threads", 2),
        ("threadpools", [{"num_threads": 8}]),
    ],
)
def test_mismatched_runtime_refuses(key, value):
    assert execution_runtime.profile_mismatches(declared(), {**observed(), key: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"device": "cuda"},
        {"blas_threads": 2},
        {"rng_algorithm": "unimplemented"},
        {"container_digest": "sha256:" + "a" * 64},
    ],
)
def test_unattested_or_unsupported_identity_refuses(changes):
    assert execution_runtime.profile_mismatches(
        replace(declared(), **changes), observed()
    )


def test_matching_runtime_passes():
    assert execution_runtime.profile_mismatches(declared(), observed()) == []


def test_runtime_refuses_an_omitted_numerical_library_version():
    incomplete = replace(declared(), library_versions={"torch": "2.13.0"})
    assert any(
        "numerical library" in message
        for message in execution_runtime.profile_mismatches(incomplete, observed())
    )


def test_guard_limits_observes_both_profiles_and_restores(monkeypatch):
    events = []
    state = {"threads": 8}
    monkeypatch.setattr(execution_runtime, "load_blas", lambda: events.append("load"))
    monkeypatch.setattr(
        execution_runtime.torch, "get_num_threads", lambda: state["threads"]
    )
    monkeypatch.setattr(
        execution_runtime.torch, "set_num_threads", lambda n: state.update(threads=n)
    )
    monkeypatch.setattr(
        execution_runtime,
        "threadpool_limits",
        lambda **kwargs: type(
            "Limits",
            (),
            {"restore_original_limits": lambda self: events.append("restore")},
        )(),
    )
    monkeypatch.setattr(
        execution_runtime, "observe_runtime", lambda versions: observed()
    )
    guard = execution_runtime.RuntimeGuard()
    result = guard.probe(declared(), replace(declared(), dtype="float64"))
    assert result["status"] == "PASS"
    assert state["threads"] == 1
    guard.close()
    guard.close()
    assert state["threads"] == 8
    assert events == ["load", "restore"]


def test_missing_declared_library_refuses(monkeypatch):
    monkeypatch.setattr(execution_runtime, "threadpool_info", lambda: [])
    monkeypatch.setattr(
        execution_runtime.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(
            execution_runtime.metadata.PackageNotFoundError(name)
        ),
    )
    result = execution_runtime.observe_runtime(("absent",))
    assert result["library_versions"]["absent"] == "UNAVAILABLE"


def test_runtime_library_wrappers_and_blas_imports(monkeypatch):
    from types import SimpleNamespace

    limiter = object()
    monkeypatch.setattr(
        execution_runtime,
        "import_module",
        lambda name: SimpleNamespace(
            threadpool_info=lambda: [], threadpool_limits=lambda **kwargs: limiter
        ),
    )
    assert execution_runtime.threadpool_info() == []
    assert execution_runtime.threadpool_limits(limits=1) is limiter
    execution_runtime.load_blas()
    monkeypatch.setattr(execution_runtime.metadata, "version", lambda name: "1")
    assert execution_runtime.observe_runtime(("synthetic",))["library_versions"] == {
        "synthetic": "1"
    }


def test_missing_platform_attestation_refuses():
    assert execution_runtime.profile_mismatches(
        replace(declared(), platform_fingerprint=None), observed()
    )
