"""
C++ bridge for PyCalc Pro.

The accelerated C++ shared library (``libphysics_complete.*``) is not shipped
in this release, so the bridge degrades to pure NumPy fallbacks that return the
same ``(result, error_code)`` contract the callers expect. Every method below
is safe to call without a compiled library; callers wrap calls in ``try`` and
fall back to their own Python implementations, so a missing library never
breaks import or runtime.
"""
import math
import ctypes
import os
import sys

import numpy as np

# Error codes shared with the Python implementations.
OK = 0
ERR_COMPLEX = 1
ERR_LIGHTSPEED = 2
ERR_NAN = 3
ERR_OVERFLOW = 4
ERR_NEGATIVE = 5
ERR_UNDERFLOW = 6
ERR_ZERO_DIV = 7
ERR_INVALID_BASE = 8
ERR_TOO_LARGE = 9


class CPPBridge:
    """NumPy-backed stand-in for the (unshipped) C++ shared library."""

    def __init__(self):
        self._lib = self._load_library()

    # -- loading ---------------------------------------------------------

    def _load_library(self):
        """Try to load the optional compiled C++ shared library."""
        libname = "libphysics_complete.so"
        if sys.platform.startswith("win"):
            libname = "libphysics_complete.dll"
        elif sys.platform.startswith("darwin"):
            libname = "libphysics_complete.dylib"
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), libname)
            lib = ctypes.CDLL(path)
            return lib
        except Exception:
            return None

    # -- scalar math (result, error_code) --------------------------------

    def safe_power(self, base, exponent):
        try:
            if base < 0 and float(exponent) != int(float(exponent)):
                return float("nan"), ERR_COMPLEX
            result = base ** exponent
            if math.isnan(result):
                return float("nan"), ERR_NAN
            if math.isinf(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def safe_sqrt(self, x):
        try:
            if x < 0:
                return float("nan"), ERR_COMPLEX
            result = math.sqrt(x)
            if math.isinf(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def safe_logarithm(self, x, base):
        try:
            if x <= 0:
                return float("nan"), ERR_NEGATIVE
            if base <= 0 or base == 1:
                return float("nan"), ERR_INVALID_BASE
            result = math.log(x, base)
            if math.isnan(result) or math.isinf(result):
                return float("nan"), ERR_NAN
            return float(result), OK
        except Exception:
            return float("nan"), ERR_NAN

    def safe_exp(self, x):
        try:
            result = math.exp(x)
            if math.isinf(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("inf"), ERR_OVERFLOW

    def safe_nth_root(self, a, n):
        try:
            if n == 0:
                return float("nan"), ERR_COMPLEX
            if a < 0:
                if int(n) % 2 == 0:
                    return float("nan"), ERR_COMPLEX
                result = -((-a) ** (1.0 / n))
            else:
                result = a ** (1.0 / n)
            if math.isnan(result) or math.isinf(result):
                return float("nan"), ERR_NAN
            return float(result), OK
        except Exception:
            return float("nan"), ERR_NAN

    def safe_factorial(self, n):
        try:
            n = int(n)
            if n < 0:
                return -1, ERR_NEGATIVE
            if n > 10000:
                return -1, ERR_TOO_LARGE
            result = 1
            for i in range(2, n + 1):
                result *= i
                if result < 0:
                    return -1, ERR_OVERFLOW
            return result, OK
        except Exception:
            return -1, ERR_OVERFLOW

    def fast_inv_sqrt(self, x):
        try:
            if x <= 0:
                return float("nan"), ERR_NEGATIVE
            y = 1.0 / math.sqrt(x)
            return float(y), OK
        except Exception:
            return float("nan"), ERR_NEGATIVE

    def safe_sin(self, x):
        try:
            return float(math.sin(math.radians(x % 360.0))), OK
        except Exception:
            return float("nan"), ERR_NAN

    def safe_cos(self, x):
        try:
            return float(math.cos(math.radians(x % 360.0))), OK
        except Exception:
            return float("nan"), ERR_NAN

    def safe_tan(self, x):
        try:
            radians = math.radians(x % 360.0)
            if abs(math.cos(radians)) < 1e-10:
                return float("nan"), ERR_COMPLEX
            result = math.tan(radians)
            if abs(result) > 1e10:
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_NAN

    def safe_mod(self, a, b):
        try:
            if b == 0:
                return float("nan"), ERR_ZERO_DIV
            result = a % b
            if math.isnan(result) or math.isinf(result):
                return float("nan"), ERR_NAN
            return float(result), OK
        except Exception:
            return float("nan"), ERR_NAN

    def safe_abs(self, x):
        try:
            return float(abs(float(x))), OK
        except Exception:
            return float("nan"), ERR_NAN

    # -- batch math (results, error_codes) -------------------------------

    def batch_power(self, bases, exponents):
        bases = np.asarray(bases, dtype=np.float64)
        exponents = np.asarray(exponents, dtype=np.float64)
        results = np.empty_like(bases)
        error_codes = np.zeros_like(bases, dtype=np.int32)
        with np.errstate(all="ignore"):
            complex_mask = (bases < 0) & (exponents != np.floor(exponents))
            results = np.power(bases, exponents)
            nan_mask = np.isnan(results)
            inf_mask = np.isinf(results)
            error_codes[complex_mask] = ERR_COMPLEX
            error_codes[nan_mask & ~complex_mask] = ERR_NAN
            error_codes[inf_mask] = ERR_OVERFLOW
        return results, error_codes

    def batch_sqrt(self, numbers):
        numbers = np.asarray(numbers, dtype=np.float64)
        results = np.empty_like(numbers)
        error_codes = np.zeros_like(numbers, dtype=np.int32)
        with np.errstate(all="ignore"):
            neg_mask = numbers < 0
            results = np.sqrt(np.where(neg_mask, 0, numbers))
            inf_mask = np.isinf(results)
            error_codes[neg_mask] = ERR_COMPLEX
            error_codes[inf_mask] = ERR_OVERFLOW
        return results, error_codes

    def batch_logarithm(self, numbers, base):
        numbers = np.asarray(numbers, dtype=np.float64)
        results = np.empty_like(numbers)
        error_codes = np.zeros_like(numbers, dtype=np.int32)
        with np.errstate(all="ignore"):
            neg_mask = numbers <= 0
            results = np.log(np.where(neg_mask, 1, numbers)) / np.log(base)
            nan_mask = np.isnan(results) | np.isinf(results)
            error_codes[neg_mask] = ERR_NEGATIVE
            error_codes[nan_mask & ~neg_mask] = ERR_NAN
        return results, error_codes

    def batch_exp(self, numbers):
        numbers = np.asarray(numbers, dtype=np.float64)
        results = np.empty_like(numbers)
        error_codes = np.zeros_like(numbers, dtype=np.int32)
        with np.errstate(all="ignore"):
            results = np.exp(numbers)
            inf_mask = np.isinf(results)
            error_codes[inf_mask] = ERR_OVERFLOW
        return results, error_codes

    # -- physics scalar (result, error_code) -----------------------------

    def kinetic_energy(self, mass, velocity):
        try:
            if mass < 0 or velocity < 0:
                return float("nan"), ERR_NEGATIVE
            result = 0.5 * mass * velocity ** 2
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def relativistic_gamma(self, velocity, c):
        try:
            if velocity < 0:
                return float("nan"), ERR_NEGATIVE
            if velocity >= c:
                return float("inf"), ERR_LIGHTSPEED
            v_c = velocity / c
            result = 1.0 / math.sqrt(1.0 - v_c ** 2)
            return float(result), OK
        except Exception:
            return float("nan"), ERR_NAN

    def potential_energy(self, mass, height, gravity):
        try:
            if mass < 0 or height < 0 or gravity <= 0:
                return float("nan"), ERR_NEGATIVE
            result = mass * gravity * height
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def schwarzschild_radius(self, mass, G, c):
        try:
            if mass < 0:
                return float("nan"), ERR_NEGATIVE
            result = 2.0 * G * mass / (c ** 2)
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def de_broglie_wavelength(self, momentum, h):
        try:
            if momentum <= 0:
                return float("nan"), ERR_NEGATIVE
            result = h / momentum
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def centripetal_force(self, mass, velocity, radius):
        try:
            if mass < 0 or velocity < 0 or radius <= 0:
                return float("nan"), ERR_NEGATIVE
            result = mass * velocity ** 2 / radius
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def time_dilation(self, proper_time, velocity, c):
        try:
            if proper_time < 0:
                return float("nan"), ERR_NEGATIVE
            gamma = 1.0 / math.sqrt(1.0 - (velocity / c) ** 2)
            result = proper_time * gamma
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def length_contraction(self, proper_length, velocity, c):
        try:
            if proper_length < 0:
                return float("nan"), ERR_NEGATIVE
            gamma = 1.0 / math.sqrt(1.0 - (velocity / c) ** 2)
            result = proper_length / gamma
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def projectile_range(self, velocity, angle, gravity):
        try:
            if velocity < 0:
                return float("nan"), ERR_NEGATIVE
            if gravity <= 0:
                return float("nan"), ERR_NEGATIVE
            result = (velocity ** 2) * math.sin(2.0 * math.radians(angle)) / gravity
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    def ideal_gas_law(self, pressure, volume, temperature, moles, R, result_index, _):
        # result_index: 0=pressure,1=volume,2=temperature,3=moles to solve for
        try:
            if R <= 0:
                return float("nan"), ERR_INVALID_BASE
            if result_index == 0:
                result = moles * R * temperature / volume
            elif result_index == 1:
                result = moles * R * temperature / pressure
            elif result_index == 2:
                result = pressure * volume / (moles * R)
            elif result_index == 3:
                result = pressure * volume / (R * temperature)
            else:
                return float("nan"), ERR_INVALID_BASE
            if math.isinf(result) or math.isnan(result):
                return float("inf"), ERR_OVERFLOW
            return float(result), OK
        except Exception:
            return float("nan"), ERR_OVERFLOW

    # -- batch physics (results, error_codes) ----------------------------

    def batch_kinetic_energy(self, masses, velocities):
        masses = np.asarray(masses, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        results = np.empty_like(masses)
        error_codes = np.zeros_like(masses, dtype=np.int32)
        with np.errstate(all="ignore"):
            invalid = (masses < 0) | (velocities < 0)
            results = 0.5 * masses * np.square(velocities)
            inf_mask = np.isinf(results)
            error_codes[invalid] = ERR_NEGATIVE
            error_codes[inf_mask & ~invalid] = ERR_OVERFLOW
        return results, error_codes

    def batch_relativistic_gamma(self, velocities, c):
        velocities = np.asarray(velocities, dtype=np.float64)
        results = np.empty_like(velocities)
        error_codes = np.zeros_like(velocities, dtype=np.int32)
        with np.errstate(all="ignore"):
            neg = velocities < 0
            light = velocities >= c
            v_c = np.where(light, 0.0, velocities) / c
            results = 1.0 / np.sqrt(1.0 - np.square(v_c))
            results[light] = np.inf
            error_codes[neg] = ERR_NEGATIVE
            error_codes[light] = ERR_LIGHTSPEED
        return results, error_codes

    def batch_potential_energy(self, masses, heights, gravity):
        masses = np.asarray(masses, dtype=np.float64)
        heights = np.asarray(heights, dtype=np.float64)
        results = np.empty_like(masses)
        error_codes = np.zeros_like(masses, dtype=np.int32)
        with np.errstate(all="ignore"):
            invalid = (masses < 0) | (heights < 0) | (gravity <= 0)
            results = masses * heights * gravity
            inf_mask = np.isinf(results)
            error_codes[invalid] = ERR_NEGATIVE
            error_codes[inf_mask & ~invalid] = ERR_OVERFLOW
        return results, error_codes

    # -- calculator vector ops (ctypes-style out buffer) -----------------

    def vector_add(self, a, b, out, size):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        np.add(a, b, out=out)
        return out

    def vector_multiply(self, a, b, out, size):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        np.multiply(a, b, out=out)
        return out

    def vector_subtract(self, a, b, out, size):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        np.subtract(a, b, out=out)
        return out

    def vector_divide(self, a, b, out, size):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        with np.errstate(all="ignore"):
            np.divide(a, b, out=out, where=b != 0)
            out[b == 0] = np.nan
        return out

    def batch_arithmetic(self, arrays, operation, n_arrays, array_size):
        arrays = [np.asarray(arr, dtype=np.float64).flatten() for arr in arrays]
        max_len = max(arr.size for arr in arrays)
        stacked = np.zeros((len(arrays), max_len), dtype=np.float64)
        for i, arr in enumerate(arrays):
            stacked[i, : arr.size] = arr
        with np.errstate(all="ignore"):
            if operation == 0:  # ADD
                return np.sum(stacked, axis=1)
            if operation == 1:  # MULTIPLY
                return np.prod(stacked, axis=1)
            if operation == 2:  # SUBTRACT
                return stacked[:, 0] - np.sum(stacked[:, 1:], axis=1)
            if operation == 3:  # DIVIDE
                denom = np.prod(stacked[:, 1:], axis=1)
                return np.where(denom == 0, np.nan, stacked[:, 0] / denom)
        raise ValueError(f"Unsupported operation: {operation}")


# Global bridge instance (always importable, never crashes).
global_cpp_bridge = CPPBridge()
