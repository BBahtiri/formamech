# Contributing to PI-GINOT

Thank you for helping build PI-GINOT.

## Development workflow

1. Use Python 3.11, 3.12, or 3.13 in a virtual environment.
2. Install the development dependencies with
   `python -m pip install -e ".[dev]"`.
3. Write a failing test before implementation.
4. Make the smallest change that passes, then improve without widening scope.
5. Run the complete local checks documented in the README.

Changes must preserve the geometry-agnostic source boundary and the isolation between
packaged code and disposable experiments. Do not weaken a verification threshold to
make a check pass, and do not describe numerical verification as experimental
validation.

Keep pull requests focused. Explain the behavior being changed, the tests run, and
any required check that could not run.
