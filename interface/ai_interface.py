"""
Clean AI interface for PyCalc Pro
"""
from typing import Union, List, Dict, Any
import threading
from ..core.calculator import CalculatorEngine
from ..core.math_ops import MathOperations
from ..core.physics_ops import PhysicsOperations
from ..core.unit_ops import UnitOperations
from ..core.sequences import SequenceOperations
from ..utils.evaluator import SafeEvaluator
from ..core.performance import PerformanceMonitor as perf_monitor

class PyCalcAI:
    """
    Main AI interface for PyCalc Pro
    Fast, reliable, and easy to integrate
    """
    
    def __init__(self):
        self.engine = CalculatorEngine()
        self.math = MathOperations(self.engine.cache)
        self.physics = PhysicsOperations()
        self.units = UnitOperations()
        self.sequences = SequenceOperations(self.engine.cache)
        self.evaluator = SafeEvaluator(self.engine)
        self._lock = threading.RLock()
        
        # Connect operations to engine for state management
        self._connect_operations()
    
    def _connect_operations(self):
        """Connect all operations to the main engine for state management"""
        self.math.engine = self.engine
        self.physics.engine = self.engine
        self.units.engine = self.engine
        self.sequences.engine = self.engine
    
    # === CORE AI INTERFACE ===
    @perf_monitor.time_it
    def compute(self, expression: str) -> Union[float, str]:
        """
        Compute any mathematical expression
        Perfect for AI natural language processing
        """
        try:
            result = self.evaluator.safe_eval(expression)
            self.engine.set_last_result(result)
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    @perf_monitor.time_it
    def batch_compute(self, expressions: List[str]) -> List[Union[float, str]]:
        """
        Compute multiple expressions efficiently
        Ideal for AI processing multiple calculations
        """
        results = []
        for expr in expressions:
            results.append(self.compute(expr))
        return results
    
    # === MATHEMATICAL OPERATIONS ===
    @perf_monitor.time_it
    def power(self, base: float, exponent: float) -> Union[float, str]:
        result = self.math.power(base, exponent)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def sqrt(self, x: float) -> Union[float, str]:
        result = self.math.sqrt(x)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def factorial(self, n: int) -> Union[int, str]:
        result = self.math.factorial(n)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def logarithm(self, x: float, base: float = 10) -> Union[float, str]:
        result = self.math.logarithm(x, base)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def sin(self, degrees: float) -> Union[float, str]:
        result = self.math.sin(degrees)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def cos(self, degrees: float) -> Union[float, str]:
        result = self.math.cos(degrees)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def tan(self, degrees: float) -> Union[float, str]:
        result = self.math.tan(degrees)
        self.engine.set_last_result(result)
        return result
    
    # === PHYSICS OPERATIONS ===
    @perf_monitor.time_it
    def kinetic_energy(self, mass: float, velocity: float) -> Union[float, str]:
        result = self.physics.kinetic_energy(mass, velocity)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def potential_energy(self, mass: float, height: float, gravity: float = 9.80665) -> Union[float, str]:
        result = self.physics.potential_energy(mass, height, gravity)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def time_dilation(self, proper_time: float, velocity: float) -> Union[float, str]:
        result = self.physics.time_dilation(proper_time, velocity)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def projectile_range(self, initial_velocity: float, angle_degrees: float, gravity: float = 9.80665) -> Union[float, str]:
        result = self.physics.projectile_range(initial_velocity, angle_degrees, gravity)
        self.engine.set_last_result(result)
        return result
    
    # === UNIT CONVERSIONS ===
    @perf_monitor.time_it
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> Union[float, str]:
        result = self.units.convert(value, from_unit, to_unit)
        self.engine.set_last_result(result)
        return result
    
    # === SEQUENCES ===
    @perf_monitor.time_it
    def fibonacci(self, n: int) -> List[int]:
        result = self.sequences.fibonacci(n)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def prime_sequence(self, n: int) -> List[int]:
        result = self.sequences.prime_sequence(n)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def arithmetic_sequence(self, first: float, difference: float, n: int) -> List[float]:
        result = self.sequences.arithmetic_sequence(first, difference, n)
        self.engine.set_last_result(result)
        return result
    
    @perf_monitor.time_it
    def geometric_sequence(self, first: float, ratio: float, n: int) -> List[float]:
        result = self.sequences.geometric_sequence(first, ratio, n)
        self.engine.set_last_result(result)
        return result
    
    # === MEMORY OPERATIONS ===
    def store_memory(self, value: float) -> None:
        self.engine.store_memory(value)
    
    def recall_memory(self) -> float:
        return self.engine.recall_memory()
    
    def clear_memory(self) -> None:
        self.engine.clear_memory()
    
    # === BATCH OPERATIONS ===
    @perf_monitor.time_it
    def batch_power(self, bases: List[float], exponents: List[float]) -> List[Union[float, str]]:
        """Batch power operations"""
        return self.math.batch_power(bases, exponents)
    
    @perf_monitor.time_it
    def batch_sqrt(self, numbers: List[float]) -> List[Union[float, str]]:
        """Batch square root operations"""
        return self.math.batch_sqrt(numbers)
    
    @perf_monitor.time_it
    def batch_kinetic_energy(self, masses: List[float], velocities: List[float]) -> List[Union[float, str]]:
        """Batch kinetic energy calculations"""
        return self.physics.batch_kinetic_energy(masses, velocities)
    
    # === CONTEXT AND STATE ===
    def get_context(self) -> Dict[str, Any]:
        """Get current context for AI reasoning"""
        with self._lock:
            return {
                'last_result': self.engine.last_result,
                'memory': self.engine.memory,
                'cache_stats': self.engine.cache.get_stats(),
                'performance_stats': perf_monitor.get_performance_stats(),
                'available_operations': self.get_available_operations(),
                'system_info': self.get_system_info()
            }
    
    def get_available_operations(self) -> List[str]:
        """Get list of available operations for AI planning"""
        return [
            'compute(expression)',
            'power(base, exponent)',
            'sqrt(x)',
            'factorial(n)',
            'logarithm(x, base)',
            'sin(degrees)', 'cos(degrees)', 'tan(degrees)',
            'kinetic_energy(mass, velocity)',
            'potential_energy(mass, height, gravity)',
            'time_dilation(proper_time, velocity)',
            'projectile_range(velocity, angle, gravity)',
            'convert_units(value, from_unit, to_unit)',
            'fibonacci(n)',
            'prime_sequence(n)',
            'arithmetic_sequence(first, difference, n)',
            'geometric_sequence(first, ratio, n)',
            'store_memory(value)',
            'recall_memory()',
            'batch_power(bases, exponents)',
            'batch_sqrt(numbers)',
            'batch_kinetic_energy(masses, velocities)'
        ]
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities"""
        return {
            "math_optimization": self.math.get_optimization_status(),
            "sequence_info": self.sequences.get_sequence_info(),
            "unit_cache_info": self.units.get_cache_info(),
            "evaluator_cache_info": self.evaluator.get_cache_info(),
            "performance_report": perf_monitor.get_performance_report()
        }
    
    def reset_state(self):
        """Reset calculator state"""
        with self._lock:
            self.engine = CalculatorEngine()
            self.math = MathOperations(self.engine.cache)
            self.evaluator = SafeEvaluator(self.engine)
            self._connect_operations()
            perf_monitor.reset_stats()
    
    def get_help(self) -> Dict[str, Any]:
        """Get help information for AI integration"""
        return {
            "description": "PyCalc Pro - High-performance math/physics engine for AI",
            "version": "1.0.0",
            "features": [
                "Mathematical operations with Numba optimization",
                "Physics calculations with relativistic effects",
                "Unit conversions with Pint integration",
                "Mathematical sequences generation",
                "Safe expression evaluation",
                "Batch processing for AI workloads",
                "Performance monitoring and caching"
            ],
            "usage_examples": {
                "basic_math": "compute('2 + 3 * 4')",
                "physics": "kinetic_energy(10, 20)",
                "units": "convert_units(100, 'kg', 'lb')",
                "sequences": "fibonacci(10)",
                "batch": "batch_power([2, 3, 4], [2, 3, 2])"
            }
        }