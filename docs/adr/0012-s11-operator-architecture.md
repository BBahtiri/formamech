# ADR 0012: S11 PI-GINOT operator architecture

- Status: Accepted
- Date: 2026-09-07
- Step: S11
- Gate: `artifacts/gates/S11.json`

## Context

S11 fixes the neural-operator architecture before any production training. The
starting point is the working PI-GINOT design in
[`BBahtiri/PI-GINOT-DogBone`](https://github.com/BBahtiri/PI-GINOT-DogBone) at
commit `9f6794141beca9d3e66433465c3b07039a7e41d2`, specifically
`models/pi_ginot.py`, `models/geometry_encoder.py` and
`models/physics_decoder.py`. Its load-bearing pattern is an encode-once boundary
geometry representation followed by positional query decoding through
cross-attention and FiLM conditioning.

ADR 0001 also remains binding. The S1 spike measured stress error in the 5%--15%
contingency band and found systematic non-conservative peak-stress
under-prediction. S11 must therefore preserve the implemented PI-GINOT structure
while making predicted stress authoritative and leaving a mandatory architecture
iteration between S11 and S12. Architecture checks cannot be presented as accuracy
evidence.

## Decision

The fixed architecture identity is
`pi-ginot-s11-boundary-cross-attention-v1`. A tagged boundary point cloud supplies
coordinates, semantic-tag features and outward normals. Measure-weighted set
cross-attention maps that unordered cloud to learned latent tokens, which are then
refined by self-attention. The encoded geometry can be reused for arbitrary query
batches.

The decoder applies Fourier positional features to physical query coordinates,
cross-attends to the geometry tokens and applies parameter-vector FiLM in every
decoder block. DogBone conditioning names all seven current geometry, material and
load quantities: `L_total`, `W_grip`, `W_gauge`, `R_fillet`, `E`, `nu` and
`nominal_axial_strain`.

Two independent output heads produce displacement `u` and in-plane first-Piola
stress `P`. `P_served = P_predicted` is the only served-stress path; inference does
not replace it with a constitutive recomputation or a post-hoc nonlinear scaling.
The independent head makes constitutive consistency, angular momentum and the
ADR 0001 traction-free requirement explicit S12 objective responsibilities.

Component-wise Dirichlet treatment defaults to soft enforcement with declared
residual tolerance `1e-6`. Supplying Dirichlet data in soft mode validates its
shape and tensor context but does not silently lift the prediction. Hard
approximate-distance lifting is constructible only with evidence that names the
full parameter-range hash, verification hash, at least two increasing boundary
refinement levels and a maximum boundary residual within the configured tolerance.
No such production lifting evidence is claimed by S11.

Weights cross the persistence boundary only as `.safetensors`. Before tensor
materialization, the loader checks the bounded UTF-8 JSON header, rejects duplicate
keys and non-finite JSON constants, requires the architecture ID and module digest,
and enforces an exact key, shape and dtype allowlist. It also checks every tensor's
byte length, buffer bounds, non-overlap, gap-free packing and complete data-buffer
coverage. The materialized mapping is checked again. Pickle formats and renamed
pickle payloads are never deserialized. This is a safe-format and schema boundary,
not a signature or authenticity claim about learned values.

The module digest binds architecture semantics, lifting evidence and the complete
state-tensor name/shape/dtype schema, but deliberately excludes learned tensor
values. It therefore stays fixed across training while still changing when the
module contract changes.

S11 contains no trainer, optimizer state, production checkpoint, HTTP/UI surface
or access to the legacy `experiments/` workstream.

## Consequences

The implementation is a contract-hardened adaptation of the existing PI-GINOT,
not an unrelated surrogate. Weighted set attention is used in place of the
upstream point-sampling path so boundary-order and finite resampling invariants are
directly measurable. Boundary masks, including padded samples, are part of the
contract and sparse or cross-device indexing masks fail before backend operations.

G11 is structural evidence from an untrained deterministic model. Its DogBone
probe uses the ratified reference mesh and one normalized parameter vector. The
100 boundary permutations, 320-versus-480 resampling comparison and query batching
checks establish the finite invariants named by G11; they do not prove robustness
over the full geometry/material/load envelope and do not establish displacement,
stress or peak-stress accuracy.

Passing G11 does not begin S12. The next activity is the mandatory ADR 0001
architecture iteration between S11 and S12, targeted at the known peak-stress
bias. S12 must then make traction-free residuals on `P_served` first-class alongside
the constitutive, equilibrium and angular terms.

## Verification

Implementation revision `8b3db083fa2fb8f6a861305ae89ecc3f7c0a0a23` produced
canonical gate artifact content hash
`8f5396827f2dea671c50ad6c1a8a1b147ba4b9f51a44c74b845131d394c68a24`.
The exact focused command recorded in that artifact passed 92 tests at 89.90%
branch-aware operator-package coverage. G11 reduced all nine checks to PASS:

- 100 random boundary permutations changed the output by at most
  `7.743864392418148e-16`;
- 320-versus-480 boundary resampling changed it by
  `0.0087578953803021`, below 2%;
- node-order and query-batch changes were `0` and
  `3.6105796056741125e-16`;
- all seven declared parameters reached both output paths;
- the boundary-mask control was invariant and the `IGNORE_BOUNDARY_MASK` mutant
  fired;
- pickle was refused before deserialization;
- both output shapes and the authoritative predicted-stress identity were verified;
  and
- scoped coverage exceeded the required 80%.

The gate-artifact test re-executes the deterministic evidence, reproduces the
generator payload byte-for-byte, checks canonical JSON and binds the expected
content hash. The final repository-wide regression passed 1,704 tests with 2
skipped. S11-scoped Ruff and mypy checks passed.
