# PI-GINOT

PI-GINOT is an early-stage open-source project for verified, problem-specific
physics-informed neural operators in 2D finite-strain elasticity. The repository
currently implements the S2 contract layer through the verified S10 DogBone dataset,
S11 PI-GINOT operator architecture, and
[S12 nondimensional objective registry](docs/adr/0013-s12-objective-registry.md).
The mandatory pre-S12 architecture screen trained four diagnostic candidates under
a fixed TRAIN-only protocol and formally fell back to the S11 baseline. S12 is now
complete and G12 is PASS: 89 focused tests pass with 92.99% branch coverage, while
the full suite records 1,928 passed and 2 skipped. S13 deterministic production
training is next. The project does not yet certify or serve a trained neural
operator. Read the
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
