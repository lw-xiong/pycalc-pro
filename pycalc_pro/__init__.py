"""
PyCalc Pro - A high-performance mathematical computing engine for Python.

Advanced mathematics, physics calculations, unit conversions, and sequence
operations with multi-backend acceleration (Numba JIT, optional C++ and GPU),
memory pooling, and safe expression evaluation.
"""

__version__ = "2.1.0"
__author__ = "Li Wen Xiong (李文雄)"
__email__ = "lwx.src@gmail.com"

from .interface.ai_interface import PyCalcAI

__all__ = ["PyCalcAI"]
