"""
Performance monitoring and optimization
"""
import time
from functools import wraps
from typing import Dict, List, Any
import threading
from collections import defaultdict, deque
import statistics

class PerformanceMonitor:
    """Track and optimize performance with advanced metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.operation_times = defaultdict(lambda: deque(maxlen=max_history))
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage = deque(maxlen=100)
        self._lock = threading.RLock()
        self.start_time = time.time()
        
        # Performance thresholds
        self.slow_threshold = 0.1  # 100ms
        self.very_slow_threshold = 1.0  # 1 second
    
    def time_it(self, func):
        """Decorator to time function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            
            func_name = func.__name__
            execution_time = end - start
            
            with self._lock:
                self.operation_times[func_name].append(execution_time)
            
            return result
        return wrapper
    
    def record_cache_hit(self):
        """Record a cache hit"""
        with self._lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        with self._lock:
            self.cache_misses += 1
    
    def record_memory_usage(self, usage_mb: float):
        """Record memory usage in MB"""
        with self._lock:
            self.memory_usage.append((time.time(), usage_mb))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self._lock:
            stats = {}
            total_calls = 0
            total_time = 0.0
            
            for func_name, times in self.operation_times.items():
                if times:
                    times_list = list(times)
                    stats[func_name] = {
                        'calls': len(times_list),
                        'avg_time': statistics.mean(times_list),
                        'min_time': min(times_list),
                        'max_time': max(times_list),
                        'total_time': sum(times_list),
                        'std_dev': statistics.stdev(times_list) if len(times_list) > 1 else 0.0,
                        'slow_calls': len([t for t in times_list if t > self.slow_threshold]),
                        'very_slow_calls': len([t for t in times_list if t > self.very_slow_threshold])
                    }
                    total_calls += len(times_list)
                    total_time += sum(times_list)
            
            # Cache statistics
            total_cache_operations = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / total_cache_operations * 100) if total_cache_operations > 0 else 0
            
            # Memory statistics
            memory_stats = {}
            if self.memory_usage:
                memory_values = [usage for _, usage in self.memory_usage]
                memory_stats = {
                    'current_mb': memory_values[-1] if memory_values else 0,
                    'avg_mb': statistics.mean(memory_values),
                    'max_mb': max(memory_values),
                    'min_mb': min(memory_values)
                }
            
            return {
                'operations': stats,
                'summary': {
                    'total_calls': total_calls,
                    'total_time_seconds': total_time,
                    'avg_call_time': total_time / total_calls if total_calls > 0 else 0,
                    'uptime_seconds': time.time() - self.start_time
                },
                'cache': {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'hit_rate_percent': cache_hit_rate,
                    'total_operations': total_cache_operations
                },
                'memory': memory_stats,
                'thresholds': {
                    'slow_threshold_seconds': self.slow_threshold,
                    'very_slow_threshold_seconds': self.very_slow_threshold
                }
            }
    
    def get_slow_operations(self) -> List[Dict[str, Any]]:
        """Get list of operations that exceed performance thresholds"""
        stats = self.get_performance_stats()
        slow_ops = []
        
        for func_name, op_stats in stats['operations'].items():
            if op_stats['slow_calls'] > 0:
                slow_ops.append({
                    'function': func_name,
                    'avg_time': op_stats['avg_time'],
                    'max_time': op_stats['max_time'],
                    'slow_calls': op_stats['slow_calls'],
                    'very_slow_calls': op_stats['very_slow_calls'],
                    'total_calls': op_stats['calls']
                })
        
        return sorted(slow_ops, key=lambda x: x['avg_time'], reverse=True)
    
    def optimize_cache_size(self, current_hit_rate: float, target_hit_rate: float = 80.0) -> Dict[str, Any]:
        """Dynamically adjust cache sizes based on usage patterns"""
        stats = self.get_performance_stats()
        
        recommendation = {
            'current_hit_rate': current_hit_rate,
            'target_hit_rate': target_hit_rate,
            'adjustment_needed': current_hit_rate < target_hit_rate,
            'recommended_action': 'maintain' if current_hit_rate >= target_hit_rate else 'increase',
            'reason': ''
        }
        
        if current_hit_rate < target_hit_rate:
            recommendation['reason'] = f'Cache hit rate {current_hit_rate:.1f}% below target {target_hit_rate}%'
        else:
            recommendation['reason'] = f'Cache hit rate {current_hit_rate:.1f}% meets or exceeds target'
        
        return recommendation
    
    def reset_stats(self):
        """Reset all performance statistics"""
        with self._lock:
            self.operation_times.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.memory_usage.clear()
            self.start_time = time.time()
    
    def get_performance_report(self) -> str:
        """Generate a human-readable performance report"""
        stats = self.get_performance_stats()
        slow_ops = self.get_slow_operations()
        
        report = []
        report.append("=== PERFORMANCE REPORT ===")
        report.append(f"Uptime: {stats['summary']['uptime_seconds']:.1f}s")
        report.append(f"Total Operations: {stats['summary']['total_calls']}")
        report.append(f"Total Time: {stats['summary']['total_time_seconds']:.3f}s")
        report.append(f"Average Call Time: {stats['summary']['avg_call_time']*1000:.2f}ms")
        report.append(f"Cache Hit Rate: {stats['cache']['hit_rate_percent']:.1f}%")
        
        if stats['memory']:
            report.append(f"Memory Usage: {stats['memory']['current_mb']:.1f}MB (avg: {stats['memory']['avg_mb']:.1f}MB)")
        
        if slow_ops:
            report.append("\nSLOW OPERATIONS:")
            for op in slow_ops[:5]:  # Top 5 slowest
                report.append(f"  {op['function']}: {op['avg_time']*1000:.2f}ms avg, {op['slow_calls']} slow calls")
        
        return "\n".join(report)

# Global performance monitor
perf_monitor = PerformanceMonitor()