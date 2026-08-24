# ADR 0002: S8 FEM verification thresholds

- Status: Proposed, pending human sign-off
- Date: 2026-08-24
- Step: S8
- Gate: G8 (FAIL until this ADR is explicitly accepted and exact-head CI passes)
- Evidence target: `artifacts/gates/S8.json`

## Context

S8 verifies the S7 primary FEM solver using symbolic manufactured solutions,
mesh-refinement studies, physical invariants, error-estimator calibration, and seeded
defects. `STEPS-v1.0.md` requires the calibrated nondimensional thresholds to be
recorded here before any required threshold can be treated as `PASS`.

The current implementation uses four SymPy-derived nonlinear plane-stress MMS
families: pure Dirichlet, mixed Dirichlet/traction, pure traction with body force,
and strongly varying thickness stretch. P1 and P2 studies run on four refinement
levels. The reported gate order is the finest-pair asymptotic order. The all-level
log-log fit is retained as diagnostic evidence, but it is not the threshold statistic.
The strongly varying thickness-stretch case uses 16 continuation steps because fewer
steps cross an inadmissible intermediate state; the other cases use 8.

Pure-traction MMS cases use global KKT constraints for zero mean x translation, zero
mean y translation, and zero mean rotation. These constraints remove rigid modes
without adding physical point anchors that would pollute the convergence study.

## Decision

The proposed complete and finite G8 threshold table is below. Thresholds fixed by
`STEPS-v1.0.md` are not widened from the observed data; their calibration factor is
therefore not applicable. The two estimator interval factors are the only
data-derived safety factors.

| Check | Fixed population and statistic | Threshold | Calibration, safety factor, and rationale |
| --- | --- | --- | --- |
| `G8.order-p1` | Four MMS cases, structured P1, levels 3-4: maximum absolute deviation of finest-pair displacement L2 order from 2.0 | `<= 0.15` | No data-derived factor; fixed by STEPS. Finest pair measures the asymptotic regime; all four levels remain reported. |
| `G8.order-p2` | Four MMS cases, structured P2, levels 3-4: maximum absolute deviation of finest-pair displacement L2 order from 3.0 | `<= 0.15` | No data-derived factor; fixed by STEPS, with the same asymptotic-order policy. |
| `G8.order-stress` | All four cases at P1/P2, levels 3-4: maximum absolute deviation of first-Piola L2 order from element order | `<= 0.2` | No data-derived factor; fixed by STEPS. Full 3-by-3 first-Piola error is integrated. |
| `G8.patch` | Structured and deterministically perturbed meshes at P1/P2: maximum DOF error divided by maximum exact nonzero affine displacement | `<= 1e-12` | No data-derived factor; fixed by STEPS. Normalization makes the gate statistic nondimensional. |
| `G8.objectivity` | One declared finite-deformation state and a 37-degree proper rotation: maximum relative energy-invariance and stress-covariance error | `<= 1e-10` | No data-derived factor; fixed by STEPS and scaled symmetrically by compared invariant norms. |
| `G8.angular` | The declared finite-deformation state: `norm(skew(sigma)) / norm(sigma)` | `< 1e-10` | No data-derived factor; fixed by STEPS. |
| `G8.energy` | One diagonal displacement path, 16-point Gauss integration plus final state (17 solves): relative work/energy difference | `< 1e-8` | No data-derived factor; fixed by STEPS. Path integration is independent of the final-state `external_work` field. |
| `G8.reaction` | Level-2 P2 square under 0.2 MPa right nominal traction: relative resultant imbalance | `< 1e-9` | No data-derived factor; fixed by STEPS. |
| `G8.small-strain` | Level-2 P1/P2 square at 0.1 percent axial strain: maximum relative displacement disagreement with plane-stress linear elasticity | `< 0.005` | No data-derived factor; fixed by STEPS. |
| `G8.estimator-brackets` | Minimum across two 16-outcome validation populations, each spanning 8 MMS case/order families on structured and unstructured finest meshes, of the true-error interval bracketing rate | `>= 0.95` (`16/16`) | Richardson factor `1.97`; recovery factor `2.61`. Each is 1.25 times the maximum calibration symmetric ratio, rounded upward to the next `0.01`. Every validation outcome is excluded from factor selection. |
| `G8.estimator-conservative` | Same two 16-outcome validation populations: minimum rate whose calibrated upper bound is at least true error | `>= 0.90` (`15/16`) | Same separately calibrated factors; the validation population never selects them. |
| `G8.seeded-defects` | Six fixed injections: body-force sign, boundary normal, tag swap, closure wrong root, inconsistent tangent, anisotropic scaling | `6 of 6` | Exact count fixed by STEPS; each control and mutant is separated by a declared real-check threshold. |
| `G8.coverage` | `tests/fem` over `src/piginot/fem` | `>= 85 percent` | No data-derived factor; fixed by STEPS. Exact-head CI is authoritative. |

The tag-swap mutant preserves each component-wise prescribed value while exchanging
the left and right boundary identities. It is detected by the normalized affine-patch
metric. Merely exchanging the MMS boundary-type partition is not a valid mutant:
exact spatial providers re-evaluate exact data on the new boundary, so the exact
solution still satisfies that different problem.

Richardson estimator calibration uses the first two structured coarse/fine pairs
across all eight MMS studies. Its validation population contains the structured and
deterministically perturbed unstructured finest pair from every case/order study.
Recovered-stress estimator calibration uses structured levels 1 through 3 across
all eight studies. Its validation population contains structured and unstructured
level 4 from every case/order study. The safety factor is derived only from
calibration data as the maximum symmetric estimate/truth ratio times 1.25, rounded
upward to the next hundredth. The resulting factors are `1.97` for Richardson (16
structured calibration pairs, 16 validation pairs) and `2.61` for recovery (24
structured calibration levels, 16 validation levels). The validation population is
never used to select or widen either factor.

"Excluded from factor selection" does not mean statistically independent. The
structured Richardson validation pair reuses level 3 as its coarse field after that
field participates in a calibration pair; its level-4 truth and its pair estimate do
not select the factor. The 16 outcomes are a deterministic, paired finite population
over 8 MMS case/order families and 2 mesh families, not 16 independent random cases.
The reported rates are therefore exact coverage of this declared population, not an
estimate of a sampling probability.

Sixteen outcomes are the smallest complete balanced structured/unstructured
population that preserves all four cases and both element orders while making the
two STEPS rates operationally distinct: `>= 0.95` requires `16/16`, whereas
`>= 0.90` requires `15/16`. The gate evidence records both family-specific counts,
population sizes, safety factors, and the exact `8 + 8` mesh-family split; it does
not retain rates alone. Construction rejects evidence that does not match this fixed
population.

## Current Measurements

Current local Windows measurements at the S8 worktree are:

| Check | Measured |
| --- | --- |
| `G8.order-p1` | max deviation `0.1072257167822865` |
| `G8.order-p2` | max deviation `0.0540915711204602` |
| `G8.order-stress` | max deviation `0.0240311279477990` |
| `G8.patch` | `3.2722362831057246e-13` (normalized) |
| `G8.objectivity` | `1.1578028973454352e-14` |
| `G8.angular` | `5.434071583190579e-17` |
| `G8.energy` | `1.9048071015750106e-13` |
| `G8.reaction` | `1.7170000876942464e-11` |
| `G8.small-strain` | `0.00015219470443983519` |
| `G8.estimator-brackets` | `1.0` (Richardson `16/16`; recovery `16/16`) |
| `G8.estimator-conservative` | `1.0` (Richardson `16/16`; recovery `16/16`) |
| `G8.seeded-defects` | `6 of 6` |
| `G8.coverage` | `91.41 percent` local branch coverage on Windows/Python 3.13.11 (`NOT_RUN` as a gate status pending sign-off and exact-head CI) |

The complete displacement L2 finest-pair orders are:

| Case | P1 | P2 |
| --- | ---: | ---: |
| `mixed_dirichlet_traction` | `1.9708111180223764` | `3.0077013113865716` |
| `pure_dirichlet` | `1.9552003307727461` | `3.05409157112046` |
| `pure_traction_body_force` | `1.8927742832177135` | `2.9657727062299877` |
| `strongly_varying_f33` | `2.000222304550899` | `3.000768258874081` |

The four-level log-log displacement fits are non-gating diagnostics. They make the
choice of an asymptotic finest-pair statistic visible rather than silently replacing
the requested four-level study:

| Case | P1 four-level fit | P2 four-level fit |
| --- | ---: | ---: |
| `mixed_dirichlet_traction` | `1.8279845379119573` | `3.042909167647907` |
| `pure_dirichlet` | `1.785876370376533` | `3.124608984711192` |
| `pure_traction_body_force` | `1.6761719664946202` | `2.9073110823957107` |
| `strongly_varying_f33` | `2.0004893998536755` | `3.0037308826269586` |

In particular, the pure-traction P1 four-level fit would not meet the `0.15`
tolerance, while its finest-pair order does. Human sign-off on this ADR explicitly
accepts the finest-pair asymptotic statistic; it must not be inferred merely from the
passing local number. Every all-level displacement L2/H1 and stress fit is also
required to remain finite and positive as a non-gating sanity guard.

The complete first-Piola L2 finest-pair orders are:

| Case | P1 | P2 |
| --- | ---: | ---: |
| `mixed_dirichlet_traction` | `0.9873776446924758` | `1.987287374940322` |
| `pure_dirichlet` | `0.980407259486106` | `1.980657357167235` |
| `pure_traction_body_force` | `0.975968872052201` | `1.986326118984617` |
| `strongly_varying_f33` | `1.0002060515321298` | `1.999401048069107` |

## Consequences

- No G8 threshold is `PASS` until this ADR is explicitly accepted.
- Local single-platform measurements are not sufficient for G8. Exact branch-head CI
  must pass Linux and Windows on Python 3.11 and 3.12.
- The S8 independent-solver protocol is present, but the current cross-discretization
  evidence remains a discretization-error check, not a second concrete solver.
- S8 establishes FEM verification inside the manufactured and physical-check
  population above. It is not experimental validation and does not certify arbitrary
  geometries or loads.
- Any later threshold loosening or change of population requires a new ADR before a
  failing result can be reclassified.

## Sign-off requested

Explicit acceptance of this ADR means accepting all of the following together:

1. the finite 13-check table and its fixed populations;
2. finest-pair levels 3-4 as the order gate statistic, with all four-level fits kept
   as visible diagnostics;
3. Richardson and recovery safety factors `1.97` and `2.61`, selected only from their
   stated structured calibration populations, with the two deterministic 16-outcome
   validation populations evaluated without retuning; and
4. the stated limitations on solver independence and population scope.

Until that acceptance is recorded, every threshold-dependent G8 check remains
`NOT_RUN` and G8 remains `FAIL`, even when its local measurement is numerically inside
the proposed bound.

## Verification

The current evidence commands are:

- `python -m pytest tests/fem/test_mms_convergence.py -q`
- `python -m pytest tests/fem/test_mms_estimators.py -q`
- `python -m pytest tests/fem/test_s8_physical_runner.py -q`
- `python -m pytest tests/fem/test_seeded_defects.py -q`
- `python -m pytest tests/fem/test_s8_gate.py -q`
- `python -m pytest tests/fem --cov=src/piginot/fem --cov-branch --cov-fail-under=85`
- full CI matrix: Linux/Windows times Python 3.11/3.12 at the exact branch head
