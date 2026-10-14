"""The S19 fast goldens compared at their recorded thread count (S26)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from campaigns import fast_goldens, g19_gate

_REPOSITORY = Path(__file__).resolve().parents[2]
_AVX512 = "AVX512|x86_64|3.11.15|Linux|2.13.0+cu130"


def test_every_golden_profile_has_a_thread_count() -> None:
    import json

    goldens = json.loads((_REPOSITORY / g19_gate.GOLDEN_PATH).read_text())
    assert set(goldens["profiles"]) == set(fast_goldens.GOLDEN_THREADS)


def test_the_count_is_pinned_while_serving_and_restored() -> None:
    seen: list[int] = []

    def digest(root: Path, name: str, request_file: str) -> str:
        seen.append(torch.get_num_threads())
        return "0" * 64

    before = torch.get_num_threads()
    result = fast_goldens.compare(_REPOSITORY, _AVX512, digest=digest)
    assert result is not None
    assert result["threads"] == 1 and seen == [1, 1, 1]
    assert not any(result["documents"].values())
    assert torch.get_num_threads() == before


def test_an_unknown_profile_is_not_compared() -> None:
    assert fast_goldens.compare(_REPOSITORY, "SSE|arm|3.12|Darwin|x") is None


def test_the_thread_count_is_restored_after_an_error() -> None:
    before = torch.get_num_threads()
    with pytest.raises(RuntimeError), fast_goldens.torch_threads(1):
        raise RuntimeError("boom")
    assert torch.get_num_threads() == before


def test_the_avx512_goldens_match_at_one_thread() -> None:
    if g19_gate.execution_profile()["cpu_capability"] != "AVX512":
        pytest.skip("the AVX512 goldens are reproduced on AVX512 only")
    result = fast_goldens.compare(_REPOSITORY, _AVX512)
    assert result is not None and all(result["documents"].values()), result


def test_this_profiles_goldens_match_at_their_thread_count() -> None:
    key = g19_gate.profile_key(g19_gate.execution_profile())
    if key not in fast_goldens.GOLDEN_THREADS:
        pytest.skip("no S19 golden was recorded for this execution profile")
    result = fast_goldens.compare(_REPOSITORY)
    assert result is not None and all(result["documents"].values()), result
