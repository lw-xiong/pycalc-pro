# PyCalc Pro v2.0.1

A high-performance **mathematical computing engine** written in Python — designed for **AI systems and scientific computing** with advanced mathematics, physics calculations, and sequence operations featuring multi-backend acceleration and memory optimization.

---

## Version Notes
v2.0.1 — November 2025

* **Performance Upgrade**: Optimizations with C++ extensions and GPU acceleration
* **Multi-Backend Support**: C++, Numba JIT, and pure Python fallbacks
* **Memory Optimization**: Advanced memory pooling for reduced allocations
* **Expanded Modules**: Sequences, unit operations, and physics calculations
* **License Change**: Updated to BSD 3-Clause license
* **Code Quality**: Fixed Numba signatures and consistent error handling

---

## Features

### Core Mathematical Operations
* **Basic Arithmetic**: Optimized addition, subtraction, multiplication, division
* **Advanced Functions**: Power, roots, logarithms, exponentials
* **Trigonometric Functions**: Sine, cosine, tangent with degree support
* **Special Functions**: Factorial, modulus, absolute value
* **Batch Processing**: Parallel execution for large datasets

### Physics Engine
* **Classical Mechanics**: Kinetic energy, potential energy, centripetal force
* **Relativity**: Time dilation, length contraction, relativistic gamma
* **Quantum Mechanics**: de Broglie wavelength, Schwarzschild radius
* **Thermodynamics**: Ideal gas law calculations
* **Projectile Motion**: Range calculations with custom gravity

### Sequence & Number Theory
* **Mathematical Sequences**: Fibonacci, arithmetic, geometric progressions
* **Prime Operations**: Prime checking, factorization, prime generation
* **Combinatorics**: Permutations, combinations, factorial operations
* **Series Calculations**: Summation, product sequences

### Unit Operations
* **Unit Conversion**: Comprehensive unit system support
* **Dimensional Analysis**: Automatic unit validation and conversion
* **Physical Constants**: Extensive library of scientific constants

### Performance Optimizations
* **C++ Extensions**: Critical operations accelerated with C++
* **GPU Acceleration**: CUDA support for large batch operations
* **Memory Pooling**: Reduced allocation overhead
* **Numba JIT**: Just-in-time compilation for numerical functions
* **Smart Backend Selection**: Automatic optimization based on operation size
* **Error Handling**: Comprehensive validation and error codes

---

## Architecture
```bash
pycalc-pro/
├── core/ # Core mathematical engines
│ ├── math_ops.py # Phase 3 optimized math operations
│ ├── calculator.py # Main calculator engine
│ ├── physics_ops.py # Physics calculations
│ ├── sequences.py # Sequence and number theory
│ ├── unit_ops.py # Unit operations and conversions
│ ├── cpp_bridge.py # C++ extensions interface
│ ├── cpp_extensions.cpp # C++ acceleration code
│ ├── gpu_accelerator.py # GPU acceleration
│ └── performance.py # Performance monitoring
├── interface/ # User interfaces
│ ├── cli.py # Command-line interface
│ └── ai_interface.py # Interface for AI systems
├── utils/ # Support utilities
│ ├── cache.py # Intelligent caching system
│ ├── memory_pool.py # Memory optimization
│ ├── constants.py # Physical and mathematical constants
│ └── evaluator.py # Expression evaluation
├── examples/ # Usage examples
│ └── ai_usage.py # AI system integration examples
└── dist/ # Distribution files
```

---

## Installation & Usage

### Option 1: Install from PyPI (Recommended)
```bash
pip install pycalc-pro
pycalc
```

### Option 2: Install from PyPI for GPU (Recommended)
```bash
pip install pycalc-pro[gpu]
pycalc
```

### Option 3: Install from GitHub
```bash
pip install git+https://github.com/lw-xiong/pycalc-pro.git
pycalc
```

### Option 4: Clone and Run
```bash
git clone https://github.com/lw-xiong/pycalc-pro.git
cd pycalc-pro
pip install -e .
pycalc
```

---

## AI System Integration
```bash
from pycalc_pro.core.math_ops import MathOperations
from pycalc_pro.core.physics_ops import PhysicsOperations

# High-performance math for AI systems
math_engine = MathOperations()
result = math_engine.power(2, 10)  # 1024

# Physics calculations for scientific AI
physics_engine = PhysicsOperations()
ke = physics_engine.kinetic_energy(10, 5)  # 125 J
```

---

## Performance Features
* Memory Optimization: Reduced allocations via memory pooling
* Multi-Backend: Automatic selection of C++ → Numba → Python backends
* Batch Processing: Parallel execution for large-scale computations
* Error Safety: Comprehensive error codes and validation
* GPU Support: Optional CUDA acceleration for large datasets

---

## Author
Li Wen Xiong (李文雄)
GitHub: @lw-xiong

---

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

---

PyCalc Pro v2.0.1 — Optimized for Scientific Computing
