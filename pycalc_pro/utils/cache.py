"""
Optimized caching system for mathematical operations.
"""
from collections import OrderedDict
from typing import Optional


class MathCache:
    """Efficient caching for mathematical operations."""

    def __init__(self, max_size: int = 1000):
        self.factorial_cache = {}
        self.power_cache = OrderedDict()
        self.sqrt_cache = OrderedDict()
        self.max_size = max_size

        # Precompute common values
        self._precompute_common()

    def _precompute_common(self):
        """Precompute common mathematical values."""
        # Factorials
        for i in range(0, 21):
            self.factorial_cache[i] = self._compute_factorial(i)

    def _compute_factorial(self, n: int) -> int:
        """Compute factorial."""
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    # -- factorial -------------------------------------------------------

    def get_factorial(self, n: int) -> Optional[int]:
        """Get factorial from cache."""
        return self.factorial_cache.get(n)

    def set_factorial(self, n: int, value: int) -> None:
        """Store factorial in cache."""
        self.factorial_cache[n] = value

    def invalidate_factorial(self, n: int) -> None:
        """Drop a cached factorial entry."""
        self.factorial_cache.pop(n, None)

    # -- power -----------------------------------------------------------

    def get_power(self, base: float, exponent: float):
        """Get a cached power result."""
        return self.power_cache.get((base, exponent))

    def set_power(self, base: float, exponent: float, value) -> None:
        """Store a power result in the cache (LRU-evicted at max_size)."""
        key = (base, exponent)
        if key in self.power_cache:
            self.power_cache.move_to_end(key)
        self.power_cache[key] = value
        if len(self.power_cache) > self.max_size:
            self.power_cache.popitem(last=False)

    def invalidate_power(self, base: float, exponent: float) -> None:
        """Drop a cached power entry."""
        self.power_cache.pop((base, exponent), None)

    # -- sqrt ------------------------------------------------------------

    def get_sqrt(self, x: float):
        """Get a cached sqrt result."""
        return self.sqrt_cache.get(x)

    def set_sqrt(self, x: float, value) -> None:
        """Store a sqrt result in the cache (LRU-evicted at max_size)."""
        if x in self.sqrt_cache:
            self.sqrt_cache.move_to_end(x)
        self.sqrt_cache[x] = value
        if len(self.sqrt_cache) > self.max_size:
            self.sqrt_cache.popitem(last=False)

    def invalidate_sqrt(self, x: float) -> None:
        """Drop a cached sqrt entry."""
        self.sqrt_cache.pop(x, None)

    # -- stats -----------------------------------------------------------

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'factorial_cache_size': len(self.factorial_cache),
            'power_cache_size': len(self.power_cache),
            'sqrt_cache_size': len(self.sqrt_cache),
        }
