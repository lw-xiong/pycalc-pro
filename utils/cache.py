"""
Optimized caching system for mathematical operations
"""
from collections import OrderedDict
from typing import Optional

class MathCache:
    """Efficient caching for mathematical operations"""
    
    def __init__(self, max_size: int = 1000):
        self.factorial_cache = {}
        self.power_cache = OrderedDict()
        self.sqrt_cache = OrderedDict()
        self.max_size = max_size
        
        # Precompute common values
        self._precompute_common()
    
    def _precompute_common(self):
        """Precompute common mathematical values"""
        # Factorials
        for i in range(0, 21):
            self.factorial_cache[i] = self._compute_factorial(i)
    
    def _compute_factorial(self, n: int) -> int:
        """Compute factorial"""
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    
    def get_factorial(self, n: int) -> Optional[int]:
        """Get factorial from cache"""
        return self.factorial_cache.get(n)
    
    def set_factorial(self, n: int, value: int) -> None:
        """Store factorial in cache"""
        self.factorial_cache[n] = value
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'factorial_cache_size': len(self.factorial_cache),
            'power_cache_size': len(self.power_cache),
            'sqrt_cache_size': len(self.sqrt_cache)
        }