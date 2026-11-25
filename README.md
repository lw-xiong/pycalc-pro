# PyCalc Pro

[![CI](https://img.shields.io/github/actions/workflow/status/lw-xiong/pycalc-pro/ci.yml?style=for-the-badge)](https://github.com/lw-xiong/pycalc-pro/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/v2.1.0-2b2b2b?style=for-the-badge)](https://github.com/lw-xiong/pycalc-pro/releases)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-2b2b2b?style=for-the-badge&logo=python)](https://python.org)

PyCalc Pro is a high-performance mathematical computing engine for Python. It provides
arithmetic, transcendental, physics, unit-conversion, and sequence operations with a safe
AST expression evaluator, batch processing, memory pooling, and optional Numba/GPU
acceleration — designed to be simple to import and easy to integrate into AI workflows.

## What changed in this release

The previous version was published with broken packaging (`core`/`interface` as separate
top-level packages, missing `utils`), a phantom C++ shared library that crashed on import,
and GPU/C++ API calls that didn't match the shipped accelerator. This release:

- Repackages everything under one importable `pycalc_pro` package.
- Replaces the unshipped C++ bridge with working NumPy fallbacks (no crash, no phantom `.so`).
- Aligns the GPU/C++ call sites with the actual accelerator API.
- Wires the full math and physics surface through the `PyCalcAI` interface.
- Fixes the Fibonacci sequence generator (was appending the wrong term).
- Adds a pytest suite (70 tests) and CI on every push/PR to `main`.

## Install

```bash
# From source (recommended)
git clone https://github.com/lw-xiong/pycalc-pro.git
cd pycalc-pro
pip install -e .
pycalc

# Or straight from GitHub
pip install git+https://github.com/lw-xiong/pycalc-pro.git
```

Optional accelerators (the engine falls back to plain NumPy/Python without them):

```bash
pip install "pycalc-pro[numba]"   # Numba JIT for numerical kernels
pip install "pycalc-pro[gpu]"     # CuPy GPU acceleration (CUDA required)
```

`pyproject.toml` declares `numpy`, `scipy`, and `pint` as runtime dependencies;
`numba` and `cupy` are optional.

## Quick start

```python
from pycalc_pro import PyCalcAI

c = PyCalcAI()

# Safe expression evaluation
print(c.compute("2 + 3 * 4"))            # 14

# Mathematics
print(c.power(2, 10))                    # 1024.0
print(c.sqrt(16))                        # 4.0
print(c.exp(1))                          # 2.718...
print(c.factorial(5))                    # 120
print(c.logarithm(100, 10))              # 2.0
print(c.sin(90), c.cos(60), c.tan(45))   # 1.0 0.5 1.0
print(c.mod(10, 3), c.abs(-7))           # 1.0 7.0
print(c.nth_root(27, 3))                 # 3.0

# Physics
print(c.kinetic_energy(10, 5))           # 125.0
print(c.relativistic_gamma(149896229))   # ~1.1547 (at 0.5 c)
print(c.time_dilation(1, 149896229))     # > 1
print(c.schwarzschild_radius(5.9722e24)) # ~0.00887 m (Earth)

# Unit conversions (requires pint)
print(c.convert_units(100, "kg", "lb"))  # ~220.46

# Sequences
print(c.fibonacci(8))                    # [0, 1, 1, 2, 3, 5, 8, 13]
print(c.prime_sequence(6))               # [2, 3, 5, 7, 11, 13]

# Batch
print(c.batch_power([2, 3, 4], [2, 3, 2]))  # [4.0, 27.0, 16.0]
```

## CLI

```bash
pycalc compute "2 + 3 * 4"
pycalc math power 2 10
pycalc physics kinetic_energy 10 20
pycalc units 100 kg lb
pycalc sequence fibonacci 8
```

## Backends

| Capability | Default | Optional accelerator |
|---|---|---|
| Scalar math | NumPy/Python | Numba JIT |
| Batch math | NumPy | Numba / CuPy GPU |
| Physics | NumPy/Python | — |
| Units | Pint | — |

When Numba/CuPy/C++ aren't installed, every operation still works through the NumPy
fallback — no hard crash, no phantom shared library.

## Tests & CI

```bash
python -m pytest tests/
```

70 tests cover math, physics, units, sequences, the safe evaluator, and the calculator
engine. CI runs the suite on Python 3.8 / 3.10 / 3.12 / 3.13 on every push and pull
request to `main`.

## License

BSD 3-Clause. See `LICENSE`.
