"""Fast-serving documents against the S19 goldens, at the recorded thread count.

The S19 goldens (``artifacts/fork/S19-BUNDLE-OUTPUTS.json``) hash
``serve(mode="fast")`` documents per execution profile, but the profile key
does not include the torch thread count, and the documents depend on it
(measured 14 Oct 2026: on the AVX512 profile they match with one thread and
differ with two). G22-G25 compared them with whatever thread count the shell
gave. This module pins the count each golden was recorded with, so a gate
from G26 on compares like with like whatever ``OMP_NUM_THREADS`` says:

- AVX512 (cloud): 1 thread, reproduced in the cloud on 14 Oct 2026;
- AVX2 (owner machine): the default of its 8 cores, with which G22-G25 matched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from campaigns import g19_gate

GOLDEN_THREADS: Final = {
    "AVX2|x86_64|3.11.15|Linux|2.13.0+cu130": 8,
    "AVX512|x86_64|3.11.15|Linux|2.13.0+cu130": 1,
}


@contextmanager
def torch_threads(count: int) -> Iterator[None]:
    """Run the block with ``count`` intra-op threads, then restore."""

    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def served_digest(root: Path, name: str, request_file: str) -> str:
    """SHA-256 of the fast document of one shipped example (the S19 rule)."""

    from piginot.bundle import load_bundle
    from piginot.cli import parse_request
    from piginot.serving import serve

    bundle = load_bundle(root / "bundles" / name)
    document = json.loads(
        (root / "bundles" / "requests" / request_file).read_text("utf-8")
    )
    inputs, queries = parse_request(document)
    served = serve(bundle, inputs, queries, mode="fast").document()
    return hashlib.sha256(
        json.dumps(served, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def compare(
    root: Path,
    key: str | None = None,
    *,
    digest: Callable[[Path, str, str], str] = served_digest,
) -> dict[str, Any] | None:
    """Each example's match with its golden at the recorded thread count.

    ``None`` when this profile has no golden or no recorded thread count.
    """

    goldens = json.loads((root / g19_gate.GOLDEN_PATH).read_text("utf-8"))["profiles"]
    key = g19_gate.profile_key(g19_gate.execution_profile()) if key is None else key
    if key not in goldens or key not in GOLDEN_THREADS:
        return None
    threads = GOLDEN_THREADS[key]
    with torch_threads(threads):
        documents = {
            f"{name}/{request_file}": digest(root, name, request_file)
            == goldens[key][f"{name}/{request_file}"]
            for name, request_file in g19_gate.EXAMPLES
        }
    return {"profile": key, "threads": threads, "documents": documents}


__all__ = ("GOLDEN_THREADS", "compare", "served_digest", "torch_threads")
