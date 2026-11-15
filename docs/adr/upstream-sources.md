# Upstream sources and licences

- Status: Accepted with the release ADR [0032](https://github.com/BBahtiri/verified-neural-operators/blob/7f7127820e706a71543941e2d917ca84fc298aa9/docs/adr/0032-release-0.1.0.md)
- Date: 2026-10-06
- Gate: `G18.licenses`
- Project licence: MIT (`LICENSE`, "Copyright (c) 2026 PI-GINOT contributors")

This record lists every component adapted from other work, with its source,
commit, digest and licence compatibility. STEPS asks for "every adapted
component from GNOT, GINO or PointNet++". The shipped code draws on none of
these three directly. It descends from GINOT through the author's own
PI-GINOT repositories, so the record follows that lineage.

Digests are SHA-256 of the file with CRLF normalised to LF. Git blob ids are
given where the upstream checkout was available.

## 1. What a release distributes

The distribution description and contents scan below were recorded for the
0.1.0 release. The current development wheel additionally includes
`piginot_campaigns` and its packaged geometry resources. The development
sdist includes that package and `hatch_build.py` as well as the existing
source and metadata files. Neither build target includes `experiments/` or
local experiment outputs.

The wheel contains `src/piginot` and its metadata. The sdist contains
`src/piginot`, `LICENSE`, `README.md`, `CITATION.cff`, `CHANGELOG.md`,
`pyproject.toml` and `uv.lock` (`[tool.hatch.build.targets.sdist]`). Neither
contains `experiments/`. G18 rebuilds both and checks their file lists.

A line scan found **no copied line** in `src/`. It compared every
whitespace-normalised source line of 40 characters or more against two
upstream trees: `PI-GINOT-DogBone@9f67941` (vendored tree) and
`PI-GINOT@12c9c1c` (`models/`, `config.py`). The shipped operators are
re-implementations of the upstream designs, written in the contract style of
this package.

### 1.1 Served operator: `pi-ginot-energy-parametric-film-v1`

| Item | Value |
| --- | --- |
| Shipped file | `src/piginot/operators/energy_operator.py` (`ENERGY_ARCHITECTURE_ID`) |
| Adapted from | the parameter-conditioned decoder of the energy-form PI-GINOT study |
| Source repository | `https://github.com/BBahtiri/PI-GINOT` |
| Commit | `12c9c1c7a13eaabcc2a289a886c34858d0330bad` |
| Source files | `models/physics_decoder.py` (SHA-256 `7401ad96d45d94b2bd8f8a4522878f317c09d75d8b326bebc630e85b3c0b7c7b`, blob `e38edc4`), `models/pi_ginot.py` (SHA-256 `57487c2547958fc47e0b0dea519e1c427cc336ab1ea98bcdc7431738701a6207`, blob `bbd4c62`) |
| Upstream licence | MIT, "Copyright (c) 2025 Betim Bahtiri" (`LICENSE` SHA-256 `64fd2f3673386742e9374f2fceb125fbc1a4b80cb687e7a8206e2ebfd1338eeb`) |
| How adapted | design re-implemented: parameter MLP to latent tokens, NeRF query features, pooled-token bias, pre-norm residual cross-attention blocks with FiLM, linear correction head; exact Dirichlet lifting and constitutive stress added here (ADR 0018) |
| Compatibility | compatible: MIT into MIT, and no upstream text is copied |

### 1.2 S11 operator: `pi-ginot-s11-boundary-cross-attention-v1` and `pi-ginot-s11c-multiscale-local-stress-v1`

| Item | Value |
| --- | --- |
| Shipped file | `src/piginot/operators/architecture.py` (not served by any bundle; ADR 0028) |
| Adapted from | the branch/trunk operator of the DogBone PI-GINOT |
| Source repository | `https://github.com/BBahtiri/PI-GINOT-DogBone` |
| Commit | `9f6794141beca9d3e66433465c3b07039a7e41d2` |
| Source files | `models/pi_ginot.py` (SHA-256 `b31033fa294c061ef16df421ab4e007b6e04f8edeeded933a8dfa8873cc660c5`, blob `a3e1348`), `models/geometry_encoder.py` (`fe7466e1eb10f2adce81b78da8a4ccec6b4aae72d3adb8976d3bd900716c824b`, blob `ac19623`), `models/physics_decoder.py` (`305c784733c6aea6a28d65f3ddf31ba46e39454b10cc09cb705bece16df0e557`, blob `b152cb2`) |
| Upstream licence | MIT, "Copyright (c) 2025 Betim Bahtiri" (same `LICENSE` digest) |
| How adapted | contract-hardened re-implementation (ADR 0012). Weighted set attention replaces the upstream point-sampling path, so no farthest-point sampling, ball query or PointNet++ set abstraction is shipped |
| Compatibility | compatible: MIT into MIT, and no upstream text is copied |

### 1.3 Published ideas used, no code adapted

| Idea | Reference | Where |
| --- | --- | --- |
| GINOT architecture (boundary encoder, query decoder with cross-attention) | Q. Liu, V. Zhong, H. Meidani, D. Abueidda, S. Koric, P. Geubelle, *Geometry-Informed Neural Operator Transformer*, arXiv:2504.19452; code at `https://github.com/QibangLiu/GINOT` | the design lineage of both operators, through the PI-GINOT repositories |
| NeRF positional encoding | B. Mildenhall et al., *NeRF*, ECCV 2020 | `_NerfEncoding`, `FourierFeatures` |
| Pre-norm residual cross-attention | A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 | `_ResidualCrossAttentionBlock` |
| Feature-wise linear modulation | E. Perez et al., *FiLM*, AAAI 2018 | `_FiLM` |
| GNOT | Z. Hao et al., *GNOT*, ICML 2023 | not used: no code or component adapted |
| GINO | Z. Li et al., *Geometry-Informed Neural Operator*, NeurIPS 2023 | not used: no code or component adapted |
| PointNet++ | C. R. Qi et al., *PointNet++*, NeurIPS 2017 | not shipped; see §2 for the spike |

The name PI-GINOT does not claim a faithful reproduction of GNOT, GINO or
GINOT (README, "Name and architecture").

## 2. What only the repository holds: the S1 spike

`experiments/spike/` is quarantined: no build target includes it and `src/`
cannot import it (`tests/test_spike_isolation.py`). It runs the exact
PI-GINOT-DogBone implementation for the S1 spike.

| Files under `experiments/spike/upstream_pi_ginot/` | Origin | Licence | Status |
| --- | --- | --- | --- |
| `LICENSE`, `config.py`, `models/__init__.py`, `models/geometry_encoder.py`, `models/physics_decoder.py`, `models/pi_ginot.py` | `BBahtiri/PI-GINOT-DogBone@9f67941` (digests in `experiments/spike/model.py`, `UPSTREAM_SOURCE_SHA256`) | MIT, Betim Bahtiri | redistributed; compatible |
| `models/modules/__init__.py`, `point_encoding.py`, `point_position_embedding.py`, `transformer.py`, `pointnet2_utils.py` | GINOT (Qibang Liu), copied into `PI-GINOT-DogBone@9f67941`. `pointnet2_utils.py` modifies `openai/shap-e` (MIT, 2023) and `yanx27/Pointnet_Pointnet2_pytorch` (MIT, 2019 benny). Two files carry "Copyright © Qibang Liu 2025. All Rights Reserved." | GINOT has no licence file (checked 2026-10-06) | **not redistributed** from 0.1.0 on: removed from the tree and ignored by Git; pinned in `experiments/spike/upstream_pi_ginot/THIRD_PARTY.json` and fetched with `python -m experiments.spike.fetch_upstream` |

`fetch_upstream` writes nothing unless every file matches its pin. With the
files fetched, the spike's own digest test passes as before.

The owner had them removed from the whole Git history as well (2026-10-06,
`git filter-repo --invert-paths`). No commit reaches the five files or their
blobs. Every commit after `6f5f0cb` was rewritten, and no other file changed.
`artifacts/release/HISTORY-REWRITE.json` maps the original commit ids, which
evidence written before the rewrite still names, to the rewritten ones
(`campaigns/history.py`).

## 3. Runtime dependencies

Dependencies are installed from PyPI under their own licences and are never
vendored (`pyproject.toml`, `uv.lock`). The base install needs only `rfc8785`
(Apache-2.0). The `runtime` extra, used to load a bundle and predict, adds
NumPy (BSD-3-Clause), PyTorch (BSD-3-Clause) and safetensors (Apache-2.0).
`verify` adds numerical execution dependencies including threadpoolctl,
scikit-fem, SciPy and SymPy. The `fem`, `physics`, `ml` and `geometry` extras
also declare optional execution capabilities; `dev` contains development tools.

The `cad` extra adds the external Gmsh and ezdxf Python packages.
[Gmsh's licence](https://gmsh.info/#Licensing) is GPL version 2 or later with
its stated external-library linking exception.
[ezdxf is MIT-licensed](https://ezdxf.readthedocs.io/en/stable/introduction.html#license).
These dependencies are installed separately and are not vendored in this
project's wheel. Consult their own licence notices when distributing them.
