"""
Optional CLI interface for PyCalc Pro
"""
import cmd
import sys
from typing import List, Optional
from ..interface.ai_interface import PyCalcAI

class CLI(cmd.Cmd):
    """Simple CLI for testing and manual use"""
    
    prompt = 'pycalc> '
    
    def __init__(self):
        super().__init__()
        self.calc = PyCalcAI()
        self.intro = self._get_welcome_message()
    
    def _get_welcome_message(self) -> str:
        """Get welcome message"""
        return """
╔═══════════════════════════════════════╗
║          PyCalc Pro CLI               ║
║    High-Performance Math Engine       ║
╚═══════════════════════════════════════╝

Type 'help' for available commands or 'exit' to quit.
Type 'compute 2 + 3' to calculate expressions.
"""
    
    def do_compute(self, arg):
        """Compute a mathematical expression: compute 2 + 3 * 4"""
        if not arg:
            print("Error: Please provide an expression")
            return
        
        result = self.calc.compute(arg)
        print(f"Result: {result}")
    
    def do_math(self, arg):
        """Math operations: math power 2 3 | math sqrt 16 | math factorial 5"""
        parts = arg.split()
        if len(parts) < 2:
            print("Usage: math <function> <args>")
            print("Functions: power, sqrt, factorial, log, sin, cos, tan")
            return
        
        func_name = parts[0]
        args = parts[1:]
        
        try:
            if func_name == 'power':
                if len(args) != 2:
                    print("Usage: math power <base> <exponent>")
                    return
                result = self.calc.power(float(args[0]), float(args[1]))
            elif func_name == 'sqrt':
                if len(args) != 1:
                    print("Usage: math sqrt <number>")
                    return
                result = self.calc.sqrt(float(args[0]))
            elif func_name == 'factorial':
                if len(args) != 1:
                    print("Usage: math factorial <number>")
                    return
                result = self.calc.factorial(int(args[0]))
            elif func_name == 'log':
                if len(args) not in [1, 2]:
                    print("Usage: math log <number> [base]")
                    return
                base = float(args[1]) if len(args) == 2 else 10
                result = self.calc.logarithm(float(args[0]), base)
            elif func_name in ['sin', 'cos', 'tan']:
                if len(args) != 1:
                    print(f"Usage: math {func_name} <degrees>")
                    return
                if func_name == 'sin':
                    result = self.calc.sin(float(args[0]))
                elif func_name == 'cos':
                    result = self.calc.cos(float(args[0]))
                elif func_name == 'tan':
                    result = self.calc.tan(float(args[0]))
            else:
                print(f"Unknown math function: {func_name}")
                print("Available: power, sqrt, factorial, log, sin, cos, tan")
                return
            
            print(f"Result: {result}")
            
        except ValueError as e:
            print(f"Error: Invalid input - {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    def do_physics(self, arg):
        """Physics calculations: physics kinetic_energy 10 20"""
        parts = arg.split()
        if len(parts) < 3:
            print("Usage: physics <function> <args>")
            print("Functions: kinetic_energy, potential_energy, time_dilation, projectile_range")
            return
        
        func_name = parts[0]
        args = list(map(float, parts[1:]))
        
        try:
            if func_name == 'kinetic_energy':
                if len(args) != 2:
                    print("Usage: physics kinetic_energy <mass> <velocity>")
                    return
                result = self.calc.kinetic_energy(args[0], args[1])
            elif func_name == 'potential_energy':
                if len(args) not in [2, 3]:
                    print("Usage: physics potential_energy <mass> <height> [gravity]")
                    return
                gravity = args[2] if len(args) == 3 else 9.80665
                result = self.calc.potential_energy(args[0], args[1], gravity)
            elif func_name == 'time_dilation':
                if len(args) != 2:
                    print("Usage: physics time_dilation <proper_time> <velocity>")
                    return
                result = self.calc.time_dilation(args[0], args[1])
            elif func_name == 'projectile_range':
                if len(args) not in [2, 3]:
                    print("Usage: physics projectile_range <velocity> <angle> [gravity]")
                    return
                gravity = args[2] if len(args) == 3 else 9.80665
                result = self.calc.projectile_range(args[0], args[1], gravity)
            else:
                print(f"Unknown physics function: {func_name}")
                print("Available: kinetic_energy, potential_energy, time_dilation, projectile_range")
                return
            
            print(f"Result: {result}")
            
        except ValueError as e:
            print(f"Error: Invalid input - {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    def do_units(self, arg):
        """Convert units: units 100 kg lb"""
        parts = arg.split()
        if len(parts) != 3:
            print("Usage: units <value> <from_unit> <to_unit>")
            print("Example: units 100 kg lb")
            return
        
        try:
            value, from_unit, to_unit = float(parts[0]), parts[1], parts[2]
            result = self.calc.convert_units(value, from_unit, to_unit)
            print(f"Result: {result}")
        except ValueError:
            print("Error: Value must be a number")
        except Exception as e:
            print(f"Error: {e}")
    
    def do_sequence(self, arg):
        """Generate sequences: sequence fibonacci 10 | sequence primes 20"""
        parts = arg.split()
        if len(parts) < 2:
            print("Usage: sequence <type> <count>")
            print("Types: fibonacci, primes, arithmetic, geometric")
            return
        
        seq_type = parts[0]
        try:
            count = int(parts[1])
            
            if seq_type == 'fibonacci':
                result = self.calc.fibonacci(count)
            elif seq_type == 'primes':
                result = self.calc.prime_sequence(count)
            elif seq_type == 'arithmetic':
                if len(parts) != 4:
                    print("Usage: sequence arithmetic <count> <first> <difference>")
                    return
                first, diff = float(parts[2]), float(parts[3])
                result = self.calc.arithmetic_sequence(first, diff, count)
            elif seq_type == 'geometric':
                if len(parts) != 4:
                    print("Usage: sequence geometric <count> <first> <ratio>")
                    return
                first, ratio = float(parts[2]), float(parts[3])
                result = self.calc.geometric_sequence(first, ratio, count)
            else:
                print(f"Unknown sequence type: {seq_type}")
                print("Available: fibonacci, primes, arithmetic, geometric")
                return
            
            print(f"Result: {result}")
            
        except ValueError:
            print("Error: Count must be an integer")
        except Exception as e:
            print(f"Error: {e}")
    
    def do_memory(self, arg):
        """Memory operations: memory store 42 | memory recall | memory clear"""
        parts = arg.split()
        if not parts:
            print("Usage: memory <store|recall|clear> [value]")
            return
        
        operation = parts[0]
        
        if operation == 'store':
            if len(parts) != 2:
                print("Usage: memory store <value>")
                return
            try:
                value = float(parts[1])
                self.calc.store_memory(value)
                print(f"Stored {value} in memory")
            except ValueError:
                print("Error: Value must be a number")
        
        elif operation == 'recall':
            value = self.calc.recall_memory()
            print(f"Memory: {value}")
        
        elif operation == 'clear':
            self.calc.clear_memory()
            print("Memory cleared")
        
        else:
            print(f"Unknown memory operation: {operation}")
            print("Available: store, recall, clear")
    
    def do_batch(self, arg):
        """Batch operations: batch power '2,3,4' '2,3,2'"""
        parts = arg.split()
        if len(parts) < 3:
            print("Usage: batch <operation> <list1> <list2>")
            print("Operations: power, sqrt, kinetic_energy")
            return
        
        operation = parts[0]
        
        try:
            # Parse comma-separated lists
            list1 = [float(x) for x in parts[1].split(',')]
            
            if operation == 'sqrt':
                result = self.calc.batch_sqrt(list1)
            elif operation in ['power', 'kinetic_energy']:
                if len(parts) != 3:
                    print(f"Usage: batch {operation} <list1> <list2>")
                    return
                list2 = [float(x) for x in parts[2].split(',')]
                
                if operation == 'power':
                    result = self.calc.batch_power(list1, list2)
                elif operation == 'kinetic_energy':
                    result = self.calc.batch_kinetic_energy(list1, list2)
            else:
                print(f"Unknown batch operation: {operation}")
                print("Available: power, sqrt, kinetic_energy")
                return
            
            print(f"Results: {result}")
            
        except ValueError:
            print("Error: All values must be numbers")
        except Exception as e:
            print(f"Error: {e}")
    
    def do_status(self, arg):
        """Show system status and performance"""
        context = self.calc.get_context()
        print("\n=== SYSTEM STATUS ===")
        print(f"Last Result: {context['last_result']}")
        print(f"Memory: {context['memory']}")
        print(f"Cache Stats: {context['cache_stats']}")
        print(f"Performance: {context['performance_stats']['summary']}")
    
    def do_help(self, arg):
        """Show help information"""
        if not arg:
            print("\nAvailable Commands:")
            print("  compute <expr>    - Evaluate mathematical expression")
            print("  math <func> <args> - Mathematical functions")
            print("  physics <func> <args> - Physics calculations") 
            print("  units <val> <from> <to> - Unit conversions")
            print("  sequence <type> <n> - Generate sequences")
            print("  memory <op> [val] - Memory operations")
            print("  batch <op> <list1> [list2] - Batch operations")
            print("  status - System status")
            print("  help [command] - Show help")
            print("  exit - Exit calculator")
            print("\nType 'help <command>' for detailed help.")
        else:
            super().do_help(arg)
    
    def do_exit(self, arg):
        """Exit the calculator"""
        print("Thank you for using PyCalc Pro. Goodbye!")
        return True
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def default(self, line):
        """Handle unknown commands"""
        print(f"Unknown command: {line}")
        print("Type 'help' for available commands.")

def main():
    """Main entry point for CLI"""
    try:
        CLI().cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting CLI: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()