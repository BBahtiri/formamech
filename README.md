# PI-GINOT

> **Read the [limitations](docs/limitations.md) first.** PI-GINOT verifies
> neural-operator predictions against FEM solutions of one plane-stress model,
> inside each bundle's declared envelope; it does not validate them against
> experiments, and it claims nothing outside that envelope.

## Quick start

Predict the stress field of a notched plate with a shipped bundle (Python 3.11–3.13):

```text
git clone --branch v0.1.0 https://github.com/BBahtiri/verified-neural-operators.git
cd verified-neural-operators
python -m pip install -e ".[runtime]"
piginot predict bundles/notched-v1-cp-freeze --request bundles/requests/notched-in-domain.json --output prediction.json
```

The [user guide](docs/user-guide.md) covers three ways to use the software:

- predict with the two shipped [bundles](bundles/README.md);
- train a small model end to end in about two minutes;
- build a verified model for your own geometry.

It also explains the claim levels and refusals.

## About

PI-GINOT is an early-stage open-source project for verified, problem-specific
physics-informed neural operators in 2D finite-strain elasticity. Release 0.1.0
completes the STEPS-v1.0 plan (S0–S18). It ships contracts and a content store,
a plane-stress physics kernel and FEM solver with their verification, geometry
morphing, preregistered datasets, the energy-form operator, deterministic
training, verification with claim levels, hash-closed bundles with a command
line, a mutation suite, examples and benchmarks. Two families have bundles: the
DogBone (`HOLDOUT_VERIFIED`) and a centre-notched plate (`DEVELOPMENT_VERIFIED`).
The history below records how each step was reached.

The S11 PI-GINOT operator architecture and the
[S12 nondimensional objective registry](docs/adr/0013-s12-objective-registry.md)
came first. The mandatory pre-S12 architecture screen trained four diagnostic
candidates under a fixed TRAIN-only protocol and formally fell back to the S11
baseline; G12 passed.

S13 training is complete. The six registered S13 hybrid fine-tunes of the
S11 operator and the S13b peak pilot failed the frozen TRAIN-screen guardrails
(ADR 0015); one-case diagnostics traced the failure to inaccurate displacement
gradients (ADR 0016). S13c replaces the operator with the energy-form PI-GINOT:
exact Dirichlet lifting, a FiLM cross-attention decoder conditioned on the shape
and load parameters, a potential-energy objective, and stress derived from the
predicted displacement gradient
([ADR 0018](docs/adr/0018-s13c-energy-form-operator.md), accepted 2026-10-02;
[plan and outcome](docs/campaigns/s13c-enhanced-model-plan.md)). Its registered
TRAIN-only checkpoint passes every frozen S13 guardrail: median absolute
root-peak error 0.83% against the incumbent's 8.09%, P-field relative L2 0.91%
against 19.20%, and no folded interior point. The amended G11 gate
(`artifacts/gates/S11-ENERGY.json`) and G13 (`artifacts/gates/S13.json`) pass,
and `CP-FREEZE` freezes that checkpoint for S14. The joint-hybrid contrast of
ADR 0017 is parked unexecuted. S14 then measured the frozen candidate under a signed protocol
([plan and results](docs/campaigns/s14-verification-plan.md), ADRs
[0019](docs/adr/0019-s14-verification-decisions.md),
[0020](docs/adr/0020-s14-protocol-methods.md) and
[0021](docs/adr/0021-s14-development-budget-acceptance.md)). On the 64 HOLDOUT
cases the error budget passes every slice (p90 including FEM terms: stress
field 2.8%, root peak 4.0%, displacement 0.29%); on DEVELOPMENT two peak
slices exceeded 5% by at most 0.054 points and were accepted by the decision
owner. Inference is about 1,000x faster than the FEM solver on one thread.
G14 passes (`artifacts/gates/S14.json`): every required check passes,
including the reducer over the twelve verification families that
[ADR 0022](docs/adr/0022-replace-implementation-v1.0-oss.md) defines in place
of the unavailable `IMPLEMENTATION-v1.0-OSS.md`, and the claim level is
`HOLDOUT_VERIFIED`. S15 serves the candidate as a hash-closed bundle
([guide](docs/bundle.md), [ADR 0023](docs/adr/0023-s15-bundle-and-cli.md),
[plan and results](docs/campaigns/s15-bundle-plan.md)): it refuses every
request outside its envelope, reproduces the S14 vertex fields bit for bit, and
loads and predicts from a clean install with runtime dependencies only. G15
passes (`artifacts/gates/S15.json`). S16 checks the gates with the twenty
mutants of ADR 0022, each applied as a source edit of the production function
that holds the defect ([ADR 0024](docs/adr/0024-s16-mutation-suite.md),
[plan and results](docs/campaigns/s16-mutation-plan.md)): every mutant fires
its primary gate, the fired-gate signatures are the committed baseline, and a
release job reports any lost detection. G16 passes (`artifacts/gates/S16.json`).
S17 adds three [examples](docs/examples.md), committed
[benchmarks](docs/benchmarks.md) that reproduce from pinned seeds, a refusal
suite over both families (every fixture refused with the expected reason
code) and the documentation site
([ADR 0025](docs/adr/0025-s17-examples-and-second-family.md),
[plan and results](docs/campaigns/s17-plan.md)). The second family, a
centre-notched plate, ran the whole evidence chain through the same framework
calls but **misses the S14 tolerances**: DEVELOPMENT all-cases p90 stress
field 5.24%, peak 5.34%, accepted by the decision owner
([ADR 0030](docs/adr/0030-notched-development-budget-acceptance.md)); HOLDOUT
6.01% and 6.40% against 5%. Its bundle therefore serves
`DEVELOPMENT_VERIFIED`, not `HOLDOUT_VERIFIED`. G17 passes with
`G17.non-dogbone` failing as computed and accepted by the decision owner
([ADR 0031](docs/adr/0031-g17-non-dogbone-acceptance.md)).
Release 0.1.0 (S18) is recorded in the [changelog](CHANGELOG.md) and
[ADR 0032](docs/adr/0032-release-0.1.0.md): the upstream sources and licences
are [recorded](docs/adr/upstream-sources.md), and the decision owner signed off
the limitations page.
The project does not certify a trained
neural operator. Read the
[project limitations](docs/limitations.md) before using its numerical results.

## Name and architecture

The repository is named `verified-neural-operators`. The project, the Python
package and the command keep the name PI-GINOT (`piginot`).

The name PI-GINOT does not claim a faithful reproduction of GNOT or GINO, nor
of GINOT. It names this project's own physics-informed operator, which
re-implements the design of the author's PI-GINOT repositories in the
contract style of this package. Both published bundles serve exactly one
architecture:

```text
architecture_id: pi-ginot-energy-parametric-film-v1
```

This is the energy-form operator in `src/piginot/operators/energy_operator.py`
([ADR 0018](docs/adr/0018-s13c-energy-form-operator.md),
[ADR 0028](docs/adr/0028-single-architecture-in-v1.md)). Its sources,
commits, digests and licences are recorded in
[upstream sources](docs/adr/upstream-sources.md).

## Requirements

- Python 3.11, 3.12, or 3.13
- Git

## Development setup

```text
python -m venv .venv
python -m pip install -e ".[dev,geometry,physics,fem,ml]"
```

Activate the virtual environment using the command appropriate for your shell before
installing.

## Checks

```text
ruff check .
ruff format --check .
mypy --strict src/
pytest
pytest tests/physics --cov=src/piginot/physics --cov-fail-under=95
pytest tests/geometry --cov=src/piginot/geometry --cov-fail-under=85
pytest tests/morphing --cov=src/piginot/morphing --cov-fail-under=85
pytest tests/fem --cov=src/piginot/fem --cov-fail-under=85
pytest tests/doe tests/test_s9_anti_scope.py --cov=src/piginot/doe --cov-fail-under=90
pytest tests/operators --cov=src/piginot/operators --cov-fail-under=80
pytest tests/objectives --cov=src/piginot/objectives --cov-fail-under=85
pytest tests/training tests/test_s13_anti_scope.py --cov=src/piginot/training --cov-fail-under=80
pytest tests/verification tests/campaigns/test_dogbone_s14_envelope.py tests/test_s14_anti_scope.py --cov=src/piginot/verification --cov-fail-under=85
pytest tests/mutation tests/test_s16_anti_scope.py --cov=src/piginot/mutation --cov-fail-under=85
python -m campaigns.dogbone_v1.s16_mutation run --report mutation-report.json --compare artifacts/mutation/S16-SIGNATURES.json
pytest tests/bundle tests/store/test_cli.py tests/test_version.py tests/test_s15_anti_scope.py tests/test_s5_anti_scope.py tests/campaigns/test_dogbone_s15_bundle.py --cov=src/piginot/bundle --cov=piginot.cli --cov-branch --cov-fail-under=85
pip-audit
```

## Command line

```text
piginot --version
piginot store prune --root PATH
piginot store prune --root PATH --dry-run
piginot bundle verify BUNDLE
piginot inspect BUNDLE
piginot predict BUNDLE --request request.json [--no-confidence]
piginot <campaign-command> --campaign NAME [-- arguments]
```

The prune command opens an existing marked store and removes only PRUNABLE bytes
while retaining their identities and recipe lineage. `bundle verify`, `inspect`
and `predict` serve a bundle ([guide](docs/bundle.md)); install
`piginot[runtime]` to serve without the FEM stack. The campaign commands
(`family compile`, `pilot`, `campaign`, `simulate`, `train`, `verify`,
`holdout-reveal`, `bundle build`) run a campaign plugin, for example
`python -m piginot.cli train --campaign dogbone_v1 -- ...` from the repository
root.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

PI-GINOT is licensed under the [MIT License](LICENSE). Every component adapted
from other work, and its licence, is recorded in
[upstream sources](docs/adr/upstream-sources.md).
