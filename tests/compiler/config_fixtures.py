"""Pure configuration examples; no source stores or numerical dependencies."""

from __future__ import annotations

from typing import Any


def profile(*, dtype: str = "float32", seed: int = 2801) -> dict[str, Any]:
    return {
        "python_version": "3.11.15",
        "library_versions": {
            "numpy": "2.2.0",
            "torch": "2.13.0",
            "scipy": "1.16.0",
            "scikit-fem": "12.0.0",
            "sympy": "1.14.0",
            "safetensors": "0.8.0",
            "threadpoolctl": "3.7.0",
        },
        "blas_threads": 1,
        "device": "cpu",
        "dtype": dtype,
        "rng_algorithm": "torch-manual-seed",
        "seed": seed,
        "platform_fingerprint": {
            "os": "Linux",
            "machine": "x86_64",
            "cpu_capability": "AVX2",
        },
    }


def config_document() -> dict[str, Any]:
    return {
        "schema": "piginot.campaign-config-v1",
        "name": "tiny-fixture",
        "adapter": {"id": "unit-adapter", "implementation_sha256": "a" * 64},
        "source": {
            "kind": "public-manifest",
            "path": "must-not-read.json",
            "sha256": "b" * 64,
            "fit_case_ids": ["fit-001", "fit-002"],
            "screen_case_ids": ["screen-001"],
            "development_case_ids": [],
        },
        "model": {
            "architecture_id": "pi-ginot-energy-parametric-film-v1",
            "parameter_names": ["length", "nominal_axial_strain"],
        },
        "normalization": "train-fit",
        "training": {
            "optimizer_steps": 4,
            "case_batch_size": 1,
            "checkpoint_interval": 1,
            "learning_rate": 0.001,
            "final_learning_rate": 0.0001,
            "constant_steps": 0,
            "seed": 2801,
        },
        "objective": {
            "points_per_cell": 3,
            "barrier_weight": 1000.0,
            "barrier_threshold": 0.05,
            "admissibility_floor": 1e-6,
        },
        "reference": {
            "initialization": "cold",
            "refinement_levels": [1, 2],
            "workers": 2,
        },
        "execution": {
            "training": profile(),
            "reference": profile(dtype="float64", seed=2802),
            "inference_dtype": "float64",
        },
        "selection": {"checkpoint_rule": "final", "development": False},
    }
