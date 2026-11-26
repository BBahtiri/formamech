"""Gate G-V5-3c tests: License-tiered dataset registry & neutral multi-scale industry benchmarks."""

from __future__ import annotations

from pathlib import Path

import pytest

from piginot.benchmarks import (
    IndustryBenchmarkConfig,
    LicenseTier,
    list_external_datasets,
    prepare_dataset_cache,
    resolve_dataset_spec,
    run_industry_benchmark_suite,
)


def test_external_dataset_license_hygiene_blocks_noncommercial_by_default(
    tmp_path: Path,
) -> None:
    """Tier C non-commercial corpora (CC-BY-NC-4.0 / GrabCAD NC) are blocked unless opted in."""
    clean = list_external_datasets(include_noncommercial=False)
    clean_names = {d["name"] for d in clean}
    assert "deepjeb" in clean_names
    assert "nist-ambench" in clean_names
    assert "nasa-pcoe" in clean_names
    assert "shift-battery" not in clean_names
    assert "simjeb" not in clean_names

    # Explicit refusal on Tier C datasets without opt-in
    with pytest.raises(PermissionError, match="Tier C"):
        resolve_dataset_spec("shift-battery", allow_noncommercial=False)
    with pytest.raises(PermissionError, match="Tier C"):
        resolve_dataset_spec("simjeb", allow_noncommercial=False)

    # Explicit opt-in resolves Tier C metadata
    spec_sb = resolve_dataset_spec("shift-battery", allow_noncommercial=True)
    assert spec_sb.license_tier is LicenseTier.TIER_C_NONCOMMERCIAL_OPT_IN

    # Zero-network pure-MIT synthetic fallback writes & verifies SHA-256 cache manifest
    cache1 = prepare_dataset_cache(
        "deepjeb", tmp_path / "cache", synthetic_sample_count=5
    )
    assert cache1.source_mode == "mit_synthetic_fallback"
    assert len(cache1.sample_files) == 5
    cache2 = prepare_dataset_cache(
        "deepjeb", tmp_path / "cache", synthetic_sample_count=5
    )
    assert cache1.manifest_sha256 == cache2.manifest_sha256

    # Tampering with a cached file triggers fail-closed SHA-256 verification error
    cache1.sample_files[0].write_bytes(b"corrupted-bytes")
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        prepare_dataset_cache("deepjeb", tmp_path / "cache")


def test_neutral_industry_benchmark_suite_multi_scale_and_bc_shift(
    tmp_path: Path,
) -> None:
    """Gate G-V5-3c: Multi-scale (>=3 scales) + BC-shift track emits JSON, MD & deterministic hash."""
    cfg = IndustryBenchmarkConfig(
        dataset_name="deepjeb",
        scales=(3, 5, 7),
        epochs_per_scale=6,
        hidden_dim=16,
        seed=42,
    )
    rep1 = run_industry_benchmark_suite(cfg, work_dir=tmp_path / "run1")
    rep2 = run_industry_benchmark_suite(cfg, work_dir=tmp_path / "run2")

    assert len(rep1.scale_results) == 3
    for pt in rep1.scale_results:
        assert pt.worst_case_disp_mae_mm >= pt.p95_disp_mae_mm >= 0.0
        assert pt.worst_case_disp_mae_mm >= pt.mean_disp_mae_mm
        assert pt.break_even_n_star is not None and pt.break_even_n_star > 0.0

    # Held-out BC-shift stress track detects shifted load layouts
    assert rep1.bc_shift_result.ood_auroc >= 0.99
    assert rep1.bc_shift_result.shifted_recall_at_threshold == 1.0

    # Deterministic report_hash across independent runs
    assert rep1.report_hash == rep2.report_hash
    assert (tmp_path / "run1" / "industry_benchmark.json").is_file()
    md = (tmp_path / "run1" / "industry_benchmark.md").read_text(encoding="utf-8")
    assert "Worst-Case MAE (mm)" in md
    assert "BC-Layout OOD AUROC" in md
