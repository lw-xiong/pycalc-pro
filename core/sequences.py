"""
Mathematical sequences - optimized for AI
"""
import math, threading
from typing import List, Optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..utils.cache import MathCache

class SequenceOperations:
    """Mathematical sequences for AI systems"""
    
    def __init__(self, cache: MathCache):
        self.cache = cache
        self.engine = None  # Will be set by AI interface
        self._fibonacci_cache = [0, 1]
        self._prime_cache = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        self._lock = threading.RLock()
        # Precompute Fibonacci cache
        self._precompute_fibonacci()
    
    def _precompute_fibonacci(self):
        """Precompute Fibonacci numbers at startup"""
        with self._lock:
            if len(self._fibonacci_cache) < 100:
                a, b = self._fibonacci_cache[-2], self._fibonacci_cache[-1]
                for _ in range(len(self._fibonacci_cache), 100):
                    a, b = b, a + b
                    self._fibonacci_cache.append(a)
    
    def fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence with caching - RETURNS LIST"""
        if not isinstance(n, int) or n < 0:
            return []
        
        if n == 0:
            return []
        
        with self._lock:
            if n <= len(self._fibonacci_cache):
                return self._fibonacci_cache[:n]
            else:
                # Extend the cache if needed
                a, b = self._fibonacci_cache[-2], self._fibonacci_cache[-1]
                result = self._fibonacci_cache.copy()
                for _ in range(len(self._fibonacci_cache), n):
                    a, b = b, a + b
                    result.append(a)
                self._fibonacci_cache = result
                return result  # Returns list
    
    def arithmetic_sequence(self, first: float, difference: float, n: int) -> List[float]:
        """Generate arithmetic sequence using vectorized formula - RETURNS LIST"""
        if not isinstance(n, int) or n <= 0:
            return []
        
        try:
            if HAS_NUMPY:
                # Vectorized implementation: a_n = a1 + (n-1)*d
                n_arr = np.arange(n, dtype=np.float64)
                sequence = first + difference * n_arr
                return sequence.tolist()
            else:
                # Fallback to list comprehension (still faster than for loop)
                return [first + i * difference for i in range(n)]
        except (ValueError, TypeError, OverflowError) as e:
            return []
    
    def geometric_sequence(self, first: float, ratio: float, n: int) -> List[float]:
        """Generate geometric sequence using vectorized formula - RETURNS LIST"""
        if not isinstance(n, int) or n <= 0:
            return []
        
        if ratio == 0 and first == 0:
            return [0.0] * n
        
        try:
            if HAS_NUMPY:
                # Vectorized implementation: a_n = a1 * r^(n-1)
                n_arr = np.arange(n, dtype=np.float64)
                sequence = first * (ratio ** n_arr)
                
                # Handle potential overflow
                sequence = np.where(np.isinf(sequence) | np.isnan(sequence), float('inf'), sequence)
                return sequence.tolist()
            else:
                # Fallback to list comprehension with overflow checking
                result = []
                current = first
                for i in range(n):
                    result.append(current)
                    if i < n - 1:  # Don't multiply after last element
                        current *= ratio
                        if abs(current) > 1e308:  # Check for overflow
                            current = float('inf')
                return result
        except (ValueError, TypeError, OverflowError) as e:
            return []
    
    def prime_sequence(self, n: int) -> List[int]:
        """Generate first n prime numbers using optimized sieve - RETURNS LIST"""
        if not isinstance(n, int) or n <= 0:
            return []
        
        if n == 0:
            return []
        
        with self._lock:
            # Check if we already have enough primes cached
            if n <= len(self._prime_cache):
                return self._prime_cache[:n]
            
            # Use optimized sieve for larger sequences
            if n > 1000:
                new_primes = self._sieve_primes(n)
                self._prime_cache = new_primes
                return new_primes
            
            # Extend cache using trial division
            primes = self._prime_cache.copy()
            num = self._prime_cache[-1] + 2
            
            while len(primes) < n:
                if self.is_prime_optimized(num, primes):
                    primes.append(num)
                num += 2
            
            self._prime_cache = primes
            return primes
    
    def _sieve_primes(self, n: int) -> List[int]:
        """Sieve of Eratosthenes for larger prime sequences - RETURNS LIST"""
        if n == 0:
            return []
        if n == 1:
            return [2]
        
        # Estimate upper bound using prime number theorem with safety margin
        if n < 6:
            limit = 12
        else:
            limit = int(n * (math.log(n) + math.log(math.log(n)))) + 1000
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        primes = []
        
        for p in range(2, limit + 1):
            if sieve[p]:
                primes.append(p)
                if len(primes) >= n:
                    break
                # Start from p*p, with step p
                start = p * p
                if start > limit:
                    continue
                sieve[start:limit + 1:p] = [False] * len(range(start, limit + 1, p))
        
        return primes[:n]
    
    def is_prime(self, n: int) -> bool:
        """Check if a number is prime"""
        if not isinstance(n, int) or n < 2:
            return False
        
        # Check small primes first
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        
        # Use cached primes for faster checking
        with self._lock:
            # Check against cached primes first
            sqrt_n = int(math.isqrt(n))
            for p in self._prime_cache:
                if p > sqrt_n:
                    break
                if n % p == 0:
                    return False
            
            # If we have enough cached primes, we're done
            if self._prime_cache[-1] >= sqrt_n:
                return True
        
        # Otherwise use optimized trial division
        return self.is_prime_optimized(n)
    
    def is_prime_optimized(self, n: int, known_primes: Optional[List[int]] = None) -> bool:
        """Optimized prime check with known primes"""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        
        # Check divisibility up to sqrt(n)
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # NEW OPTIONAL OPTIMIZED METHODS
    def batch_is_prime(self, numbers: List[int]) -> List[bool]:
        """Batch prime checking - optional optimization"""
        if not numbers:
            return []
        
        return [self.is_prime(n) for n in numbers]
    
    def fibonacci_number(self, n: int) -> int:
        """Get the nth Fibonacci number directly"""
        if not isinstance(n, int) or n < 0:
            return 0
        
        if n < len(self._fibonacci_cache):
            return self._fibonacci_cache[n]
        
        # Compute directly using Binet's formula approximation for large n
        if n > 1000:
            phi = (1 + math.sqrt(5)) / 2
            return int(round(phi ** n / math.sqrt(5)))
        
        # Otherwise compute sequentially
        sequence = self.fibonacci(n + 1)
        return sequence[-1] if sequence else 0
    
    # NEW: Optional NumPy-only methods that return arrays for maximum performance
    def arithmetic_sequence_array(self, first: float, difference: float, n: int) -> np.ndarray:
        """Generate arithmetic sequence as NumPy array (NumPy only)"""
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for array output")
        if not isinstance(n, int) or n <= 0:
            return np.array([], dtype=np.float64)
        
        n_arr = np.arange(n, dtype=np.float64)
        return first + difference * n_arr
    
    def geometric_sequence_array(self, first: float, ratio: float, n: int) -> np.ndarray:
        """Generate geometric sequence as NumPy array (NumPy only)"""
        if not HAS_NUMPY:
            raise RuntimeError("NumPy is required for array output")
        if not isinstance(n, int) or n <= 0:
            return np.array([], dtype=np.float64)
        
        n_arr = np.arange(n, dtype=np.float64)
        result = first * (ratio ** n_arr)
        return np.where(np.isinf(result) | np.isnan(result), np.inf, result)
    
    def get_sequence_info(self) -> dict:
        """Get information about available sequences"""
        return {
            "fibonacci_cache_size": len(self._fibonacci_cache),
            "prime_cache_size": len(self._prime_cache),
            "max_precomputed_fibonacci": self._fibonacci_cache[-1] if self._fibonacci_cache else 0,
            "max_precomputed_prime": self._prime_cache[-1] if self._prime_cache else 0,
            "numpy_available": HAS_NUMPY
        }