"""
Unit conversion operations using Pint
"""
from typing import Union, Dict, List, Any
try:
    import pint
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity
    HAS_PINT = True
except ImportError:
    HAS_PINT = False

class UnitOperations:
    """Unit conversions for AI systems"""
    
    def __init__(self):
        self.ureg = ureg if HAS_PINT else None
        self.engine = None
        self._conversion_cache = {}
        
    def convert(self, value: float, from_unit: str, to_unit: str) -> Union[float, str]:
        """Convert between units using Pint with caching"""
        if not HAS_PINT:
            return "Error: Pint not available for unit conversions"
        
        # Create cache key
        cache_key = (value, from_unit, to_unit)
        if cache_key in self._conversion_cache:
            return self._conversion_cache[cache_key]
        
        try:
            # Validate inputs
            if not isinstance(value, (int, float)):
                return "Error: Value must be numeric"
            
            if not isinstance(from_unit, str) or not isinstance(to_unit, str):
                return "Error: Units must be strings"
            
            # Handle temperature conversions specially (they have offsets)
            from_lower = from_unit.lower()
            to_lower = to_unit.lower()
            
            if from_lower in ['celsius', 'c', 'degc'] and to_lower in ['fahrenheit', 'f', 'degf']:
                result = (value * 9/5) + 32
            elif from_lower in ['fahrenheit', 'f', 'degf'] and to_lower in ['celsius', 'c', 'degc']:
                result = (value - 32) * 5/9
            elif from_lower in ['celsius', 'c', 'degc'] and to_lower in ['kelvin', 'k']:
                result = value + 273.15
            elif from_lower in ['kelvin', 'k'] and to_lower in ['celsius', 'c', 'degc']:
                result = value - 273.15
            elif from_lower in ['fahrenheit', 'f', 'degf'] and to_lower in ['kelvin', 'k']:
                result = (value - 32) * 5/9 + 273.15
            elif from_lower in ['kelvin', 'k'] and to_lower in ['fahrenheit', 'f', 'degf']:
                result = (value - 273.15) * 9/5 + 32
            else:
                # Use Pint for other conversions
                quantity = Q_(value, from_unit)
                converted = quantity.to(to_unit)
                result = converted.magnitude
            
            # Cache the result
            if len(self._conversion_cache) < 1000:  # Limit cache size
                self._conversion_cache[cache_key] = result
            
            return result
            
        except pint.errors.UndefinedUnitError:
            return f"Error: Unknown unit '{from_unit}' or '{to_unit}'"
        except pint.errors.DimensionalityError:
            return f"Error: Cannot convert from '{from_unit}' to '{to_unit}' - incompatible units"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def batch_convert(self, values: List[float], from_unit: str, to_unit: str) -> List[Union[float, str]]:
        """Convert multiple values between units"""
        if not HAS_PINT:
            return ["Error: Pint not available for unit conversions"] * len(values)
        
        results = []
        for value in values:
            results.append(self.convert(value, from_unit, to_unit))
        return results
    
    def get_unit_conversions(self) -> Dict[str, Dict[str, tuple]]:
        """Common unit conversions with categories"""
        return {
            "length": {
                "meters to feet": ("m", "ft"),
                "feet to meters": ("ft", "m"),
                "miles to kilometers": ("mile", "km"),
                "kilometers to miles": ("km", "mile"),
                "inches to centimeters": ("inch", "cm"),
                "centimeters to inches": ("cm", "inch"),
                "yards to meters": ("yard", "m"),
                "meters to yards": ("m", "yard"),
            },
            "mass": {
                "kilograms to pounds": ("kg", "lb"),
                "pounds to kilograms": ("lb", "kg"),
                "grams to ounces": ("g", "oz"),
                "ounces to grams": ("oz", "g"),
                "tons to kilograms": ("ton", "kg"),
                "kilograms to tons": ("kg", "ton"),
            },
            "temperature": {
                "celsius to fahrenheit": ("degC", "degF"),
                "fahrenheit to celsius": ("degF", "degC"),
                "celsius to kelvin": ("degC", "kelvin"),
                "kelvin to celsius": ("kelvin", "degC"),
                "fahrenheit to kelvin": ("degF", "kelvin"),
                "kelvin to fahrenheit": ("kelvin", "degF"),
            },
            "energy": {
                "joules to calories": ("joule", "calorie"),
                "calories to joules": ("calorie", "joule"),
                "electronvolts to joules": ("eV", "joule"),
                "joules to electronvolts": ("joule", "eV"),
                "kilowatt-hours to joules": ("kilowatt_hour", "joule"),
            },
            "pressure": {
                "pascals to atmospheres": ("pascal", "atmosphere"),
                "atmospheres to pascals": ("atmosphere", "pascal"),
                "pascals to psi": ("pascal", "psi"),
                "psi to pascals": ("psi", "pascal"),
            },
            "time": {
                "seconds to minutes": ("second", "minute"),
                "minutes to seconds": ("minute", "second"),
                "hours to seconds": ("hour", "second"),
                "days to hours": ("day", "hour"),
            },
            "volume": {
                "liters to gallons": ("liter", "gallon"),
                "gallons to liters": ("gallon", "liter"),
                "milliliters to liters": ("milliliter", "liter"),
                "liters to milliliters": ("liter", "milliliter"),
            }
        }
    
    def get_available_units(self) -> Dict[str, List[str]]:
        """Get available units by category"""
        if not HAS_PINT:
            return {"error": ["Pint not available"]}
        
        try:
            # Common units by category
            return {
                "length": ["meter", "foot", "mile", "kilometer", "inch", "centimeter", "yard", "millimeter"],
                "mass": ["kilogram", "pound", "gram", "ounce", "ton", "milligram"],
                "temperature": ["celsius", "fahrenheit", "kelvin"],
                "energy": ["joule", "calorie", "electronvolt", "kilowatt_hour", "btu"],
                "pressure": ["pascal", "atmosphere", "psi", "bar", "torr"],
                "time": ["second", "minute", "hour", "day", "week", "month", "year"],
                "volume": ["liter", "gallon", "quart", "pint", "milliliter", "cubic_meter"],
                "speed": ["meter/second", "mile/hour", "kilometer/hour", "knot"],
                "area": ["meter^2", "foot^2", "inch^2", "acre", "hectare"]
            }
        except Exception:
            return {"error": ["Unable to load unit information"]}
    
    def validate_unit(self, unit_str: str) -> bool:
        """Validate if a unit string is recognized"""
        if not HAS_PINT:
            return False
        
        try:
            # Try to create a quantity with the unit
            _ = Q_(1, unit_str)
            return True
        except:
            return False
    
    def get_conversion_factors(self) -> Dict[str, float]:
        """Get common conversion factors"""
        return {
            "meters_to_feet": 3.28084,
            "feet_to_meters": 0.3048,
            "miles_to_kilometers": 1.60934,
            "kilometers_to_miles": 0.621371,
            "kilograms_to_pounds": 2.20462,
            "pounds_to_kilograms": 0.453592,
            "inches_to_centimeters": 2.54,
            "centimeters_to_inches": 0.393701,
            "gallons_to_liters": 3.78541,
            "liters_to_gallons": 0.264172,
        }
    
    def clear_cache(self):
        """Clear the conversion cache"""
        self._conversion_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            "cache_size": len(self._conversion_cache),
            "pint_available": HAS_PINT,
            "cached_conversions_count": len(self._conversion_cache),
            "cache_limit": 1000
        }
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance and capability information"""
        return {
            "pint_available": HAS_PINT,
            "cache_enabled": True,
            "cache_size": len(self._conversion_cache),
            "supported_categories": list(self.get_unit_conversions().keys()),
            "temperature_optimized": True
        }