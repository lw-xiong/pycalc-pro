# Advanced Calculator CLI
# by Li Wen Xiong
# A feature-rich math and sequence calculator with a clean menu-driven interface.

import math, sys, time, functools, array
from colorama import init, Fore, Style
from functools import lru_cache

# Increase max digits for large integers
sys.set_int_max_str_digits(1000000)

# Optional NumPy - use if available for massive speed boosts
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import Numba for JIT compilation
try:
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

init(autoreset=True)

# =========================
# EXTREME OPTIMIZATIONS
# =========================
# Precompute everything possible at module load
SQRT_CACHE = {}
FACTORIAL_CACHE = {0: 1, 1: 1, 2: 2, 3: 6, 4: 24, 5: 120}
PRIME_CACHE = {2: True, 3: True, 5: True, 7: True}
FIBONACCI_PRECOMPUTED = array.array('Q', [0, 1])  # Unsigned long long

# Precompute first 90 Fibonacci numbers at startup
a, b = 0, 1
for _ in range(88):  # Already have 2, need 88 more to total 90
    a, b = b, a + b
    FIBONACCI_PRECOMPUTED.append(a)

# Common trigonometric values in degrees (0-360)
TRIG_CACHE = {}
for angle in range(0, 361, 15):  # Every 15 degrees
    rad = math.radians(angle)
    TRIG_CACHE[angle] = (math.sin(rad), math.cos(rad), math.tan(rad))

# =========================
# JIT COMPILED FUNCTIONS
# =========================
if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def numba_is_prime(n):
        if n <= 1: return False
        if n <= 3: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    @njit(fastmath=True, cache=True)
    def numba_fibonacci(n):
        a, b = 0, 1
        result = array.array('Q', [0]) * n
        for i in range(n):
            result[i] = a
            a, b = b, a + b
        return result
else:
    # Fallback to pure Python
    def numba_is_prime(n): return is_prime(n)
    def numba_fibonacci(n): return list(fibonacci_gen(n))

# =========================
# BASIC OPERATIONS
# =========================
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): 
    return "Error: Division by zero" if b == 0 else a * (1.0 / b)  # Faster than /

# ==============================================
# ADVANCED MATH FUNCTIONS (MAXIMUM OPTIMIZATION)
# ==============================================
def power(a, b): 
    # Special cases for common exponents
    if b == 2: return a * a
    if b == 0.5: return math.sqrt(a)
    if b == 1: return a
    if b == 0: return 1
    return a ** b

@lru_cache(maxsize=2000)
def sqrt(a):
    if a in SQRT_CACHE:
        return SQRT_CACHE[a]
    result = math.sqrt(a)
    SQRT_CACHE[a] = result
    return result

@lru_cache(maxsize=1000)
def factorial(a):
    if a < 0: return "Error: Factorial of negative number"
    if a != int(a): return "Error: Factorial requires integer"
    a_int = int(a)
    if a_int in FACTORIAL_CACHE:
        return FACTORIAL_CACHE[a_int]
    if a_int > 10000: return "Error: Number too large for factorial"
    
    # Compute and cache progressively
    result = FACTORIAL_CACHE.get(a_int - 1, 1)
    for i in range(max(FACTORIAL_CACHE.keys()) + 1 if FACTORIAL_CACHE else 1, a_int + 1):
        result *= i
        FACTORIAL_CACHE[i] = result
    return result

def logarithm(a, base=10):
    if a <= 0: return "Error: Logarithm of non-positive number"
    if base <= 0 or base == 1: return "Error: Invalid base"
    return math.log(a, base)

def sine(a):
    # Use cache for common angles, otherwise compute directly
    normalized = a % 360
    if normalized in TRIG_CACHE:
        return TRIG_CACHE[normalized][0]
    return math.sin(math.radians(a))

def cosine(a):
    normalized = a % 360
    if normalized in TRIG_CACHE:
        return TRIG_CACHE[normalized][1]
    return math.cos(math.radians(a))

def tangent(a):
    normalized = a % 360
    if normalized in TRIG_CACHE:
        return TRIG_CACHE[normalized][2]
    angle_rad = math.radians(a)
    if abs(math.cos(angle_rad)) < 1e-10:
        return "Error: Tangent undefined at this angle"
    return math.tan(angle_rad)

def exponential(a): return math.exp(a)
def absolute(a): return -a if a < 0 else a  # Faster than abs()
def modulus(a, b): return a % b

# ================================================
# SEQUENCES & NUMBER THEORY (EXTREME OPTIMIZATION)
# ================================================
MAX_FIB_TERMS = 90
TERMS_PER_ROW = 5
MAX_GEOM_VALUE = 1e100

def fibonacci_gen(n):
    """Ultra-fast Fibonacci using precomputed values or JIT"""
    if n <= len(FIBONACCI_PRECOMPUTED):
        return FIBONACCI_PRECOMPUTED[:n]
    elif HAS_NUMBA:
        return numba_fibonacci(n)
    else:
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b

def arithmetic_sequence_gen(a1, d, n):
    """Vectorized if possible, otherwise optimized iteration"""
    if HAS_NUMPY:
        arr = np.arange(n, dtype=np.float64) * d + a1
        return arr
    else:
        current = a1
        for _ in range(n):
            yield current
            current += d

def geometric_sequence_gen(a1, r, n):
    """Optimized with early termination"""
    val = a1
    count = 0
    while count < n and abs(val) <= MAX_GEOM_VALUE:
        yield val
        val *= r
        count += 1

def is_prime(n):
    """Ultra-fast prime checking with multiple optimization layers"""
    if n in PRIME_CACHE:
        return PRIME_CACHE[n]
    if n < 2: 
        PRIME_CACHE[n] = False
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        PRIME_CACHE[n] = True
        return True
    if n % 2 == 0 or n % 3 == 0:
        PRIME_CACHE[n] = False
        return False
    
    # Use JIT compiled function if available
    if HAS_NUMBA:
        result = numba_is_prime(n)
    else:
        # Optimized manual check
        if n % 5 == 0 or n % 7 == 0 or n % 11 == 0 or n % 13 == 0:
            PRIME_CACHE[n] = False
            return False
        i = 17
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                PRIME_CACHE[n] = False
                return False
            i += 6
    
    PRIME_CACHE[n] = True
    return True

def prime_sequence(n):
    """Hybrid approach: sieve for large n, trial for small n"""
    if n <= 0: 
        return
    
    print(Fore.YELLOW + f"Generating {n} primes...")
    
    if n <= 1000:
        # Trial division with caching (fast for small n)
        count, num = 0, 2
        while count < n:
            if is_prime(num):
                yield num
                count += 1
                if count % max(1, n//20) == 0 or count < 10:
                    progress = count * 30 // n
                    bar = "[" + "#"*progress + "-"*(30-progress) + f"] {count}/{n}"
                    print(Fore.GREEN + "\r" + bar, end="", flush=True)
            num += 1
        print()
    else:
        # Segmented sieve for large n (memory efficient)
        yield from segmented_sieve(n)

def segmented_sieve(n):
    """Memory-efficient segmented sieve for very large prime sequences"""
    if n == 0:
        return
    
    # Estimate upper bound using prime number theorem
    if n < 6:
        limit = [2, 3, 5, 7, 11, 13][n-1] if n > 0 else 0
    else:
        from math import log
        limit = int(n * (log(n) + log(log(n)))) + 1000
    
    # Simple sieve for small range, segmented for large range
    if limit <= 10**7:
        yield from simple_sieve_with_progress(n, limit)
    else:
        yield from true_segmented_sieve(n, limit)

def simple_sieve_with_progress(n, limit):
    """Simple sieve with progress reporting"""
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    
    # Optimized sieve marking
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = bytearray(b'\x00') * ((limit - i*i) // i + 1)
    
    # Yield primes with progress
    count = 0
    for p in range(2, limit + 1):
        if sieve[p]:
            yield p
            count += 1
            if count % max(1, n//20) == 0:
                progress = count * 30 // n
                bar = "[" + "#"*progress + "-"*(30-progress) + f"] {count}/{n}"
                print(Fore.GREEN + "\r" + bar, end="", flush=True)
            if count >= n:
                break
    print()

def true_segmented_sieve(n, limit):
    """True segmented sieve for very large ranges"""
    count = 0
    segment_size = min(10**6, limit)
    
    # Generate small primes for segment sieving
    small_primes = list(simple_sieve_with_progress(min(10000, n), int(limit**0.5) + 1000))
    
    low = 0
    while count < n and low <= limit:
        high = min(low + segment_size, limit)
        sieve = bytearray(b'\x01') * (high - low + 1)
        
        for p in small_primes:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                sieve[j - low] = 0
        
        # Yield primes from current segment
        for i in range(max(2, low), high + 1):
            if i - low < len(sieve) and sieve[i - low]:
                yield i
                count += 1
                if count % max(1, n//20) == 0:
                    progress = count * 30 // n
                    bar = "[" + "#"*progress + "-"*(30-progress) + f"] {count}/{n}"
                    print(Fore.GREEN + "\r" + bar, end="", flush=True)
                if count >= n:
                    break
        if count >= n:
            break
        low += segment_size
    print()

# =========================
# SAFE PRINTING (OPTIMIZED)
# =========================
def print_sequence(seq, name="Sequence", max_terms=100):
    """Ultra-fast sequence printing with batch processing"""
    if hasattr(seq, '__array__') or hasattr(seq, '__len__'):
        # Handle numpy arrays and lists efficiently
        try:
            seq_list = list(seq) if not hasattr(seq, '__array__') else seq.tolist()
        except (TypeError, MemoryError):
            seq_list = []
            for i, item in enumerate(seq):
                if i >= max_terms:
                    print(Fore.RED + f"{name} too long to display ({i+1}+ terms). Please request <= {max_terms}.")
                    return
                seq_list.append(item)
    else:
        # Handle generators
        seq_list = []
        try:
            for i, item in enumerate(seq):
                if i >= max_terms:
                    print(Fore.RED + f"{name} too long to display ({i+1}+ terms). Please request <= {max_terms}.")
                    return
                seq_list.append(item)
        except TypeError:
            print(Fore.RED + f"Cannot display {name}")
            return
    
    total = len(seq_list)
    if total > max_terms:
        print(Fore.RED + f"{name} too long to display ({total} terms). Please request <= {max_terms}.")
        return
        
    print(Fore.CYAN + f"{name} (length {total}):")
    
    # Batch string building for maximum speed
    lines = []
    current_line = []
    for i, val in enumerate(seq_list, 1):
        current_line.append(f"{val:<20}")
        if i % TERMS_PER_ROW == 0:
            lines.append(''.join(current_line))
            current_line = []
    
    if current_line:
        lines.append(''.join(current_line))
    
    # Single print call for all lines
    if lines:
        print('\n'.join(lines))

# =========================
# INPUT HELPER (OPTIMIZED)
# =========================
def get_number(prompt, allow_float=True, max_val=None):
    """Fast input with minimal validation overhead"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                print(Fore.RED + "Empty input! Enter a number.")
                continue
                
            # Fast path for common cases
            if user_input.isdigit() and not allow_float:
                val = int(user_input)
            else:
                val = float(user_input) if allow_float else int(user_input)
                
            if max_val is not None and val > max_val:
                print(Fore.RED + f"Value too large! Max allowed: {max_val}")
                continue
            return val
        except ValueError:
            print(Fore.RED + "Invalid input! Enter a number.")
        except KeyboardInterrupt:
            print("\n" + Fore.YELLOW + "Input cancelled.")
            raise

# ============================
# MENU SYSTEM ( MAXIMUM SPEED)
# ============================
def print_header(title):
    width = 60
    print(Fore.CYAN + "╔" + "═"*(width-2) + "╗")
    print(Fore.YELLOW + f"║{title.center(width-2)}║")
    print(Fore.CYAN + "╚" + "═"*(width-2) + "╝")

def basic_menu():
    ops = {"1": add, "2": subtract, "3": multiply, "4": divide}
    shortcuts = {"b":"5","basic operations":"5","a":"1","add":"1",
                 "s":"2","subtract":"2","m":"3","multiply":"3","d":"4","divide":"4"}
    while True:
        print_header("Basic Operations")
        print("1. Add (a)\n2. Subtract (s)\n3. Multiply (m)\n4. Divide (d)\n5. Back (b)")
        choice = input(Fore.GREEN + "Choose an option: ").lower()
        if choice in ["5","b","back"]: break
        choice = shortcuts.get(choice, choice)
        if choice in ops:
            a = get_number("Enter first number: ")
            b = get_number("Enter second number: ")
            print(Fore.MAGENTA + f"Result: {ops[choice](a,b)}")
        else:
            print(Fore.RED + "Invalid choice!")

def advanced_menu():
    while True:
        print_header("Advanced Math")
        left_options = ["1. Power (p)", "2. Square Root (s)", "3. Factorial (f)", "4. Logarithm (l)", "5. Sine (i)", "6. Cosine (c)"]
        right_options = ["7. Tangent (t)", "8. Exponential (e)", "9. Absolute (a)", "10. Modulus (m)", "11. Back (b)"]
        for i in range(max(len(left_options), len(right_options))):
            left = left_options[i] if i < len(left_options) else ""
            right = right_options[i] if i < len(right_options) else ""
            print(f"{left:<25}{right}")
        choice = input(Fore.GREEN + "Choose an option: ").lower()
        if choice in ["11","b","back"]: break
        try:
            if choice in ["1","p","power"]:
                a = get_number("Base: "); b = get_number("Exponent: "); print(Fore.MAGENTA + f"Result: {power(a,b)}")
            elif choice in ["2","s","square root"]:
                a = get_number("Number: "); print(Fore.MAGENTA + f"Result: {sqrt(a)}")
            elif choice in ["3","f","factorial"]:
                a = get_number("Number: ", allow_float=False); print(Fore.MAGENTA + f"Result: {factorial(a)}")
            elif choice in ["4","l","logarithm"]:
                a = get_number("Number: "); base = get_number("Base (0=10): "); base = 10 if base==0 else base
                print(Fore.MAGENTA + f"Result: {logarithm(a, base)}")
            elif choice in ["5","i","sine"]:
                a = get_number("Angle in degrees: "); print(Fore.MAGENTA + f"Result: {sine(a)}")
            elif choice in ["6","c","cosine"]:
                a = get_number("Angle in degrees: "); print(Fore.MAGENTA + f"Result: {cosine(a)}")
            elif choice in ["7","t","tangent"]:
                a = get_number("Angle in degrees: "); print(Fore.MAGENTA + f"Result: {tangent(a)}")
            elif choice in ["8","e","exponential"]:
                a = get_number("Number: "); print(Fore.MAGENTA + f"Result: {exponential(a)}")
            elif choice in ["9","a","absolute"]:
                a = get_number("Number: "); print(Fore.MAGENTA + f"Result: {absolute(a)}")
            elif choice in ["10","m","modulus"]:
                a = get_number("Number 1: "); b = get_number("Number 2: "); print(Fore.MAGENTA + f"Result: {modulus(a,b)}")
        except Exception as e:
            print(Fore.RED + f"Error: {e}")

def sequences_menu():
    while True:
        print_header("Sequences & Number Theory")
        print("1. Fibonacci (f)\n2. Arithmetic Sequence (a)\n3. Geometric Sequence (g)\n4. Prime Numbers (p)\n5. Back (b)")
        choice = input(Fore.GREEN + "Choose an option: ").lower()
        if choice in ["5","b","back"]: break
        elif choice in ["1","f","fibonacci"]:
            while True:
                n = int(get_number(f"Enter number of terms (max {MAX_FIB_TERMS}): ", allow_float=False))
                if n > MAX_FIB_TERMS:
                    print(Fore.RED + f"Number too large! Max {MAX_FIB_TERMS}.")
                    continue
                print_sequence(fibonacci_gen(n), "Fibonacci Sequence", max_terms=MAX_FIB_TERMS)
                break
        elif choice in ["2","a","arithmetic sequence"]:
            a1 = get_number("First term: "); d = get_number("Difference: "); n = int(get_number("Number of terms: ", allow_float=False))
            print_sequence(arithmetic_sequence_gen(a1, d, n), "Arithmetic Sequence")
        elif choice in ["3","g","geometric sequence"]:
            a1 = get_number("First term: "); r = get_number("Ratio: "); n = int(get_number("Number of terms: ", allow_float=False))
            print_sequence(geometric_sequence_gen(a1, r, n), "Geometric Sequence")
        elif choice in ["4","p","prime numbers"]:
            n = int(get_number("Number of primes: ", allow_float=False))
            print_sequence(prime_sequence(n), "Prime Numbers", max_terms=100)
        else:
            print(Fore.RED + "Invalid choice!")

# =============
# MAIN PROGRAM
# =============
def main():
    while True:
        print_header("Welcome to PyCalc Pro v1.1.0")
        print("1. Basic Operations (b)\n2. Advanced Math (a)\n3. Sequences & Number Theory (s)\n4. Exit (e)")
        choice = input(Fore.GREEN + "Choose a category: ").lower()
        if choice in ["1","b","basic operations"]: basic_menu()
        elif choice in ["2","a","advanced math"]: advanced_menu()
        elif choice in ["3","s","sequences & number theory"]: sequences_menu()
        elif choice in ["4","e","exit"]:
            print(Fore.CYAN + "Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid choice!")

if __name__ == "__main__":
    main()


