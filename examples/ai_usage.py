"""
Example of how AI would use PyCalc Pro
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pycalc_pro.interface.ai_interface import PyCalcAI

def demonstrate_ai_usage():
    """Show how AI systems would use the calculator"""
    
    # Initialize calculator (AI does this once)
    calc = PyCalcAI()
    
    # Example 1: Mathematical reasoning
    print("=== MATHEMATICAL REASONING ===")
    result1 = calc.compute("2^8 + factorial(5) - sqrt(144)")
    print(f"Complex calculation: {result1}")
    
    # Example 2: Physics problem solving
    print("\n=== PHYSICS CALCULATIONS ===")
    ke = calc.kinetic_energy(10, 20)  # 10kg at 20m/s
    print(f"Kinetic Energy: {ke} J")
    
    # Example 3: Unit conversions for real-world data
    print("\n=== UNIT CONVERSIONS ===")
    pounds = calc.convert_units(70, "kg", "lb")
    print(f"70 kg = {pounds} lb")
    
    # Example 4: Batch processing (common in AI)
    print("\n=== BATCH PROCESSING ===")
    calculations = [
        "2 + 3 * 4",
        "sin(45) + cos(45)",
        "log(100, 10)"
    ]
    results = calc.batch_compute(calculations)
    for expr, result in zip(calculations, results):
        print(f"{expr} = {result}")
    
    # Example 5: Memory and context (important for AI state)
    print("\n=== CONTEXT AWARENESS ===")
    context = calc.get_context()
    print(f"Last result: {context['last_result']}")
    print(f"Memory: {context['memory']}")
    print(f"Available operations: {len(context['available_operations'])}")

if __name__ == "__main__":
    demonstrate_ai_usage()