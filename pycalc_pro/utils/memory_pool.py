"""
Memory pooling system for optimized array operations
"""
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque

class MemoryPool:
    """
    Optimized memory pooling for array operations
    Reduces allocation overhead by reusing arrays
    """
    
    def __init__(self, max_pool_size: int = 100, preallocate_mb: int = 50):
        self.max_pool_size = max_pool_size
        self.preallocate_mb = preallocate_mb
        self._pools = defaultdict(deque)  # shape_dtype -> deque of arrays
        self._lock = threading.RLock()
        self._allocation_stats = defaultdict(int)
        self._hits = 0
        self._misses = 0
        
        # Preallocate common array sizes
        self._preallocate_common_arrays()
    
    def _preallocate_common_arrays(self):
        """Preallocate arrays for common operations"""
        common_shapes = [
            (10,), (100,), (1000,), (10000,),  # 1D arrays
            (10, 10), (100, 10), (10, 100),     # 2D arrays  
            (1000, 10), (100, 100)              # Medium 2D
        ]
        
        dtypes = [np.float64, np.int64]
        
        with self._lock:
            total_memory = 0
            for shape in common_shapes:
                for dtype in dtypes:
                    # Calculate memory usage
                    array_memory = np.prod(shape) * np.dtype(dtype).itemsize
                    
                    # Only preallocate if it doesn't exceed our memory budget
                    if total_memory + array_memory <= self.preallocate_mb * 1024 * 1024:
                        try:
                            arr = np.empty(shape, dtype=dtype)
                            key = (shape, dtype)
                            self._pools[key].append(arr)
                            total_memory += array_memory
                            self._allocation_stats[key] += 1
                        except MemoryError:
                            break  # Stop if we run out of memory
    
    def get_array(self, shape: Tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
        """Get an array from pool or allocate new one"""
        key = (shape, dtype)
        
        with self._lock:
            if key in self._pools and self._pools[key]:
                self._hits += 1
                return self._pools[key].popleft()
            
            self._misses += 1
            self._allocation_stats[key] += 1
            return np.empty(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray) -> None:
        """Return an array to the pool for reuse"""
        if array is None:
            return
            
        key = (array.shape, array.dtype)
        
        with self._lock:
            if len(self._pools[key]) < self.max_pool_size:
                # Reset array to zeros (optional, for safety)
                # array.fill(0)  # Uncomment if you want clean arrays
                self._pools[key].append(array)
    
    def get_array_like(self, template: np.ndarray) -> np.ndarray:
        """Get array with same shape and dtype as template"""
        return self.get_array(template.shape, template.dtype)
    
    def batch_get_arrays(self, shapes: List[Tuple], dtypes: List[type]) -> List[np.ndarray]:
        """Get multiple arrays at once"""
        return [self.get_array(shape, dtype) for shape, dtype in zip(shapes, dtypes)]
    
    def batch_return_arrays(self, arrays: List[np.ndarray]) -> None:
        """Return multiple arrays to pool"""
        for arr in arrays:
            self.return_array(arr)
    
    def clear_pool(self, key: Optional[Tuple] = None):
        """Clear specific pool or all pools"""
        with self._lock:
            if key:
                if key in self._pools:
                    self._pools[key].clear()
            else:
                self._pools.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self._lock:
            total_arrays = sum(len(pool) for pool in self._pools.values())
            total_preallocated = sum(self._allocation_stats.values())
            total_operations = self._hits + self._misses
            
            return {
                "total_pooled_arrays": total_arrays,
                "pool_hits": self._hits,
                "pool_misses": self._misses,
                "hit_ratio": self._hits / total_operations if total_operations > 0 else 0,
                "preallocated_arrays": total_preallocated,
                "pool_sizes": {str(k): len(v) for k, v in self._pools.items()},
                "allocation_stats": dict(self._allocation_stats)
            }
    
    def optimize_pool(self):
        """Optimize pool by removing least used arrays"""
        with self._lock:
            # Simple strategy: keep at most half of max_pool_size for each key
            for key in list(self._pools.keys()):
                pool = self._pools[key]
                if len(pool) > self.max_pool_size // 2:
                    # Remove excess arrays (oldest first)
                    excess = len(pool) - (self.max_pool_size // 2)
                    for _ in range(excess):
                        pool.popleft()

global_memory_pool = MemoryPool()