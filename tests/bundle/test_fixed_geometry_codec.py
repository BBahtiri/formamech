from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import rfc8785

from piginot.bundle.specs import SpecDecodeError, decode_morph_map
from piginot.contracts import canonical_json_bytes
from piginot.morphing import FixedGeometryMap


def test_fixed_geometry_codec_round_trips_exact_recipe_bytes() -> None:
    mapping = FixedGeometryMap()
    payload = canonical_json_bytes(mapping)
    decoded = decode_morph_map(payload)
    assert type(decoded) is FixedGeometryMap
    assert decoded == mapping
    assert canonical_json_bytes(decoded) == payload
    assert decoded.bind({}).map_point((-1.5, 3.0)) == (-1.5, 3.0)


@pytest.mark.parametrize(
    "document",
    (
        {"$type": "piginot.morphing.FixedGeometryMap", "scale": 2.0},
        {"$type": "piginot.morphing.FixedGeometryMap", "parameter_names": []},
        {"$type": "piginot.morphing.FixedGeometryMap", "kind": "other"},
        {"$type": "piginot.morphing.BoundFixedGeometryDeformation"},
        {"$type": "piginot.morphing.fixed.FixedGeometryMap"},
        {"$type": "external.FixedGeometryMap"},
        {},
    ),
)
def test_fixed_codec_refuses_extra_fields_and_unregistered_types(document) -> None:
    with pytest.raises(SpecDecodeError):
        decode_morph_map(rfc8785.dumps(document))


def test_fixed_codec_refuses_noncanonical_bytes() -> None:
    payload = canonical_json_bytes(FixedGeometryMap())
    with pytest.raises(SpecDecodeError, match="round-trip"):
        decode_morph_map(payload + b"\n")


def test_fixed_recipe_loads_in_a_fresh_source_process_without_fem() -> None:
    root = Path(__file__).resolve().parents[2]
    code = """
import json, sys
from piginot.bundle.specs import decode_morph_map
mapping = decode_morph_map(b'{"$type":"piginot.morphing.FixedGeometryMap"}')
bound = mapping.bind({})
assert bound.inverse_point((2.0, 3.0)) == (2.0, 3.0)
assert not any(name == 'skfem' or name.startswith(('skfem.', 'piginot.fem', 'campaigns.')) for name in sys.modules)
print(json.dumps({'point': bound.map_point((2.0, 3.0)), 'jacobian': bound.jacobian((2.0, 3.0))}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root.parent,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "point": [2.0, 3.0],
        "jacobian": [[1.0, 0.0], [0.0, 1.0]],
    }
