"""
Mathematical and physical constants
Updated with latest CODATA 2018 values
"""

# Physical constants (CODATA 2018 values)
PHYSICS_CONSTANTS = {
    'c': 299792458,                          # Speed of light in vacuum (m/s)
    'G': 6.67430e-11,                        # Gravitational constant (m³/kg/s²)
    'h': 6.62607015e-34,                     # Planck constant (J·s)
    'hbar': 1.054571817e-34,                 # Reduced Planck constant (J·s)
    'k': 1.380649e-23,                       # Boltzmann constant (J/K)
    'g': 9.80665,                            # Standard gravity (m/s²)
    'e': 1.602176634e-19,                    # Elementary charge (C)
    'me': 9.1093837015e-31,                  # Electron mass (kg)
    'mp': 1.67262192369e-27,                 # Proton mass (kg)
    'mn': 1.67492749804e-27,                 # Neutron mass (kg)
    'R': 8.314462618,                        # Gas constant (J/mol·K)
    'Na': 6.02214076e23,                     # Avogadro constant (1/mol)
    'sigma': 5.670374419e-8,                 # Stefan-Boltzmann constant (W/m²·K⁴)
    'epsilon0': 8.8541878128e-12,            # Vacuum permittivity (F/m)
    'mu0': 1.25663706212e-6,                 # Vacuum permeability (N/A²)
    'R_inf': 10973731.568160,                # Rydberg constant (1/m)
    'alpha': 7.2973525693e-3,                # Fine-structure constant
    'phi': 1.618033988749895,                # Golden ratio
}

# Mathematical constants
MATH_CONSTANTS = {
    'pi': 3.14159265358979323846,            # Pi
    'e': 2.71828182845904523536,             # Euler's number
    'gamma': 0.57721566490153286060,         # Euler-Mascheroni constant
    'phi': 1.61803398874989484820,           # Golden ratio
    'sqrt2': 1.41421356237309504880,         # Square root of 2
    'sqrt3': 1.73205080756887729352,         # Square root of 3
    'ln2': 0.69314718055994530941,           # Natural log of 2
    'ln10': 2.30258509299404568402,          # Natural log of 10
}

# Astronomical constants
ASTRONOMICAL_CONSTANTS = {
    'AU': 1.49597870700e11,                  # Astronomical Unit (m)
    'ly': 9.4607304725808e15,                # Light-year (m)
    'pc': 3.0856775814913673e16,             # Parsec (m)
    'M_sun': 1.98847e30,                     # Solar mass (kg)
    'R_sun': 6.957e8,                        # Solar radius (m)
    'L_sun': 3.828e26,                       # Solar luminosity (W)
}

# Conversion factors
CONVERSION_FACTORS = {
    'deg_to_rad': 0.017453292519943295,      # Degrees to radians
    'rad_to_deg': 57.29577951308232,         # Radians to degrees
    'eV_to_J': 1.602176634e-19,              # Electronvolt to joules
    'J_to_eV': 6.241509074460763e18,         # Joules to electronvolts
    'amu_to_kg': 1.66053906660e-27,          # Atomic mass unit to kg
    'kg_to_amu': 6.02214076e26,              # kg to atomic mass units
}

# Unit systems
UNIT_SYSTEMS = {
    'SI': {
        'length': 'meter',
        'mass': 'kilogram', 
        'time': 'second',
        'current': 'ampere',
        'temperature': 'kelvin',
        'amount': 'mole',
        'luminosity': 'candela'
    },
    'CGS': {
        'length': 'centimeter',
        'mass': 'gram',
        'time': 'second'
    },
    'IMPERIAL': {
        'length': 'foot',
        'mass': 'pound',
        'time': 'second'
    }
}

# Common physical values
PHYSICAL_VALUES = {
    'earth_mass': 5.9722e24,                 # Earth mass (kg)
    'earth_radius': 6.371e6,                 # Earth radius (m)
    'moon_mass': 7.342e22,                   # Moon mass (kg)
    'moon_radius': 1.7371e6,                 # Moon radius (m)
    'sun_mass': 1.9885e30,                   # Sun mass (kg)
    'sun_radius': 6.957e8,                   # Sun radius (m)
    'atmospheric_pressure': 101325,          # Standard atmospheric pressure (Pa)
    'water_density': 997,                    # Water density at 25°C (kg/m³)
    'air_density': 1.225,                    # Air density at sea level (kg/m³)
}

# Get all constants combined
def get_all_constants():
    """Get all constants in a single dictionary"""
    all_constants = {}
    all_constants.update(PHYSICS_CONSTANTS)
    all_constants.update(MATH_CONSTANTS)
    all_constants.update(ASTRONOMICAL_CONSTANTS)
    all_constants.update(CONVERSION_FACTORS)
    all_constants.update(PHYSICAL_VALUES)
    return all_constants

# Constant categories for easy access
CONSTANT_CATEGORIES = {
    'physics': PHYSICS_CONSTANTS,
    'math': MATH_CONSTANTS,
    'astronomy': ASTRONOMICAL_CONSTANTS,
    'conversions': CONVERSION_FACTORS,
    'values': PHYSICAL_VALUES
}