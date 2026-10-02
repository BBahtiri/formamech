# PI-GINOT

PI-GINOT is an early-stage open-source project for verified, problem-specific
physics-informed neural operators in 2D finite-strain elasticity. The repository
currently implements the S2 contract layer through the verified S10 DogBone dataset,
S11 PI-GINOT operator architecture, and
[S12 nondimensional objective registry](docs/adr/0013-s12-objective-registry.md).
The mandatory pre-S12 architecture screen trained four diagnostic candidates under
a fixed TRAIN-only protocol and formally fell back to the S11 baseline. S12 is now
complete and G12 is PASS: 89 focused tests pass with 92.99% branch coverage, while
the full suite records 1,928 passed and 2 skipped.

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
ADR 0017 is parked unexecuted. The project does not yet certify or serve a
trained neural operator. Read the
[project limitations](docs/limitations.md) before using its numerical results.

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
pip-audit
```

## Command line

```text
piginot --version
piginot store prune --root PATH
piginot store prune --root PATH --dry-run
```

The prune command opens an existing marked store and removes only PRUNABLE bytes
while retaining their identities and recipe lineage.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

PI-GINOT is licensed under the [MIT License](LICENSE).
