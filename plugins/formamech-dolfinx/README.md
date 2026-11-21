# `formamech-dolfinx` (Optional LGPL-3.0 External Plugin)

This external plugin integrates **FEniCSx / DOLFINx** (`dolfinx`, `ufl`, `petsc4py`) into
FormaMech via the `formamech.fem.backends` entry point without introducing LGPL-3.0
dependencies into FormaMech's MIT-licensed core package or breaking native Windows installs.

## Licensing & Platform Isolation

- **Core `formamech`**: MIT licensed, pure Python/PyTorch/SciPy/scikit-fem, runs natively on
  Windows, macOS, and Linux.
- **`formamech-dolfinx`**: LGPL-3.0-or-later, Linux / Docker / conda-forge only (where
  `fenics-dolfinx` and `petsc4py` are distributed).

## Installation (Linux / Docker / conda-forge)

```bash
conda create -n formamech-dolfinx -c conda-forge fenics-dolfinx petsc4py python=3.12
conda activate formamech-dolfinx
pip install -e ../../ -e .
```

Once installed, `formamech.fem.available_backends()` automatically discovers `"dolfinx"`, and
`formamech predict <bundle> --mode exact --backend dolfinx` delegates to `DolfinxBackend`.

## Third Medium Contact (TMC) Contract

`ThirdMediumContactConfig` specifies the regularized hyperelastic void-medium formulation
(`k_third ~= 1e-5`, `k_reg ~= 1e-3`, HuHu regularization) for self-contact and multi-body
3D finite-strain benchmarks inside DOLFINx/UFL.
