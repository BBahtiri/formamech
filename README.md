# PI-GINOT

PI-GINOT is an early-stage open-source project for verified, problem-specific
physics-informed neural operators in 2D finite-strain elasticity. The repository
currently implements the S2 contract layer, S3 local content store, S4 constitutive
physics kernel, S5 tagged 2D triangle geometry ingestion, S6 reference-mesh morphing,
S7 finite-element solves, S8 independent FEM verification, and S9 preregistered DOE,
evidence-role, leakage, replacement, and holdout-commitment contracts. It does not
yet generate verified training datasets, train, certify, or serve neural operators.
Read the [project limitations](docs/limitations.md) before using or evaluating future
results.

## Requirements

- Python 3.11, 3.12, or 3.13
- Git

## Development setup

```text
python -m venv .venv
python -m pip install -e ".[dev,geometry,physics,fem]"
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
