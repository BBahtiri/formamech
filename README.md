# PI-GINOT

PI-GINOT is an early-stage open-source project for verified, problem-specific
physics-informed neural operators in 2D finite-strain elasticity. The repository
currently implements the S2 contract layer and S3 local content store; it does not
yet generate, train, verify, or serve neural operators. Read the
[project limitations](docs/limitations.md) before using or evaluating future
results.

## Requirements

- Python 3.11, 3.12, or 3.13
- Git

## Development setup

```text
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the virtual environment using the command appropriate for your shell before
installing.

## Checks

```text
ruff check .
ruff format --check .
mypy --strict src/
pytest
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
