"""
COMPLETE C++ Bridge
"""
import numpy as np
import sys, os, ctypes

class CPPBridge:
    """Complete C++ bridge with ALL math, physics, and calculator operations"""
    
    def __init__(self):
        self._lib = self._load_library()
        self._setup_function_prototypes()
    
    def _load_library(self):
        """Load the optimized C++ library"""
        try:
            # Cross-platform library loading
            libname = 'libphysics_complete.so'
            if sys.platform.startswith('win'):
                libname = 'libphysics_complete.dll'
            elif sys.platform.startswith('darwin'):
                libname = 'libphysics_complete.dylib'
            
            lib = ctypes.CDLL(os.path.join('.', libname))
            print("SUCCESS: Loaded complete optimized C++ extensions")
            return lib
        except Exception as e:
            print(f"C++ extensions not available: {e}, using optimized NumPy fallbacks")
            return None
    
    def _setup_function_prototypes(self):
        """Setup ALL function prototypes"""
        if self._lib is None:
            self._setup_fallback_implementations()
            return
        
        # ========== MATH OPERATIONS ==========
        # Power functions
        self._lib.safe_power_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_power_cpp.restype = ctypes.c_double
        
        self._lib.batch_power_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),  
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        # Square root functions
        self._lib.safe_sqrt_cpp.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_sqrt_cpp.restype = ctypes.c_double
        
        self._lib.batch_sqrt_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        # Logarithm functions
        self._lib.safe_logarithm_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_logarithm_cpp.restype = ctypes.c_double
        
        self._lib.batch_logarithm_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        # Exponential functions
        self._lib.safe_exp_cpp.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_exp_cpp.restype = ctypes.c_double
        
        self._lib.batch_exp_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        # Special functions
        self._lib.safe_nth_root_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_nth_root_cpp.restype = ctypes.c_double
        
        self._lib.safe_factorial_cpp.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_factorial_cpp.restype = ctypes.c_longlong
        
        self._lib.fast_inv_sqrt_cpp.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.fast_inv_sqrt_cpp.restype = ctypes.c_double
        
        # Trigonometric functions
        self._lib.safe_sin_cpp.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_sin_cpp.restype = ctypes.c_double
        
        self._lib.safe_cos_cpp.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_cos_cpp.restype = ctypes.c_double
        
        self._lib.safe_tan_cpp.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_tan_cpp.restype = ctypes.c_double
        
        # Basic operations
        self._lib.safe_mod_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_mod_cpp.restype = ctypes.c_double
        
        self._lib.safe_abs_cpp.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.safe_abs_cpp.restype = ctypes.c_double
        
        # ========== PHYSICS OPERATIONS ==========
        # Kinetic energy
        self._lib.kinetic_energy_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.kinetic_energy_cpp.restype = ctypes.c_double
        
        self._lib.batch_kinetic_energy_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        # Relativistic functions
        self._lib.relativistic_gamma_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.relativistic_gamma_cpp.restype = ctypes.c_double
        
        self._lib.batch_relativistic_gamma_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        # Potential energy
        self._lib.potential_energy_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.potential_energy_cpp.restype = ctypes.c_double
        
        self._lib.batch_potential_energy_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        # Advanced physics
        self._lib.schwarzschild_radius_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.schwarzschild_radius_cpp.restype = ctypes.c_double
        
        self._lib.de_broglie_wavelength_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.de_broglie_wavelength_cpp.restype = ctypes.c_double
        
        self._lib.centripetal_force_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.centripetal_force_cpp.restype = ctypes.c_double
        
        self._lib.time_dilation_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.time_dilation_cpp.restype = ctypes.c_double
        
        self._lib.length_contraction_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.length_contraction_cpp.restype = ctypes.c_double
        
        self._lib.ideal_gas_law_cpp.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.c_int
        ]
        self._lib.ideal_gas_law_cpp.restype = ctypes.c_double
        
        self._lib.projectile_range_cpp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        self._lib.projectile_range_cpp.restype = ctypes.c_double
        
        # ========== CALCULATOR OPERATIONS ==========
        self._lib.vector_add_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), 
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        self._lib.vector_multiply_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), 
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        self._lib.vector_subtract_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), 
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        self._lib.vector_divide_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), 
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        
        self._lib.batch_arithmetic_cpp.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        
        # Connect ALL functions
        self._connect_all_functions()
    
    def _connect_all_functions(self):
        """Connect ALL C++ functions to Python methods"""
        # Math operations
        self.safe_power = self._lib.safe_power_cpp
        self.batch_power = self._lib.batch_power_cpp
        self.safe_sqrt = self._lib.safe_sqrt_cpp
        self.batch_sqrt = self._lib.batch_sqrt_cpp
        self.safe_logarithm = self._lib.safe_logarithm_cpp
        self.batch_logarithm = self._lib.batch_logarithm_cpp
        self.safe_exp = self._lib.safe_exp_cpp
        self.batch_exp = self._lib.batch_exp_cpp
        self.safe_nth_root = self._lib.safe_nth_root_cpp
        self.safe_factorial = self._lib.safe_factorial_cpp
        self.fast_inv_sqrt = self._lib.fast_inv_sqrt_cpp
        self.safe_sin = self._lib.safe_sin_cpp
        self.safe_cos = self._lib.safe_cos_cpp
        self.safe_tan = self._lib.safe_tan_cpp
        self.safe_mod = self._lib.safe_mod_cpp
        self.safe_abs = self._lib.safe_abs_cpp
        
        # Physics operations
        self.kinetic_energy = self._lib.kinetic_energy_cpp
        self.batch_kinetic_energy = self._lib.batch_kinetic_energy_cpp
        self.relativistic_gamma = self._lib.relativistic_gamma_cpp
        self.batch_relativistic_gamma = self._lib.batch_relativistic_gamma_cpp
        self.potential_energy = self._lib.potential_energy_cpp
        self.batch_potential_energy = self._lib.batch_potential_energy_cpp
        self.schwarzschild_radius = self._lib.schwarzschild_radius_cpp
        self.de_broglie_wavelength = self._lib.de_broglie_wavelength_cpp
        self.centripetal_force = self._lib.centripetal_force_cpp
        self.time_dilation = self._lib.time_dilation_cpp
        self.length_contraction = self._lib.length_contraction_cpp
        self.ideal_gas_law = self._lib.ideal_gas_law_cpp
        self.projectile_range = self._lib.projectile_range_cpp
        
        # Calculator operations
        self.vector_add = self._lib.vector_add_cpp
        self.vector_multiply = self._lib.vector_multiply_cpp
        self.vector_subtract = self._lib.vector_subtract_cpp
        self.vector_divide = self._lib.vector_divide_cpp
        self.batch_arithmetic = self._lib.batch_arithmetic_cpp
    
    def _setup_fallback_implementations(self):
        """Setup optimized NumPy fallbacks for ALL functions"""
        print("C++ extensions not available, using optimized NumPy fallbacks")
        
        # [Keep all your existing fallback implementations]
        # They will work exactly as before but slower
        
        # Math operations
        self.safe_power = self._numpy_safe_power
        self.batch_power = self._numpy_batch_power
        self.safe_sqrt = lambda x, ec: (np.sqrt(x) if x >= 0 else (ec.assign(1), np.nan)[1], 0)
        # ... add all other fallbacks
    
    # [Keep all your existing _numpy_* fallback methods exactly as they are]

# Global C++ bridge instance
global_cpp_bridge = CPPBridge()