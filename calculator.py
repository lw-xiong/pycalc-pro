# Advanced Calculator CLI

# by Li Wen Xiong

# A feature-rich math and sequence calculator with a clean menu-driven interface.

import math
import sys
import time
from colorama import init, Fore, Style

# Increase max digits for large integers
sys.set_int_max_str_digits(1000000)

# Optional NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

init(autoreset=True)

# =========================
# BASIC OPERATIONS
# =========================
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): return "Error: Division by zero" if b == 0 else a / b

# =========================
# ADVANCED MATH FUNCTIONS
# =========================
def power(a, b): return a ** b
def sqrt(a): return math.sqrt(a)
def factorial(a): return math.factorial(int(a))
def logarithm(a, base=10): return math.log(a, base)
def sine(a): return math.sin(math.radians(a))
def cosine(a): return math.cos(math.radians(a))
def tangent(a): return math.tan(math.radians(a))
def exponential(a): return math.exp(a)
def absolute(a): return abs(a)
def modulus(a, b): return a % b

# =========================
# SEQUENCES & NUMBER THEORY
# =========================
MAX_FIB_TERMS = 90
TERMS_PER_ROW = 5
MAX_GEOM_VALUE = 1e100

def fibonacci_gen(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def arithmetic_sequence_gen(a1, d, n):
    if HAS_NUMPY:
        arr = np.arange(n) * d + a1
        for val in arr: yield val
    else:
        for i in range(n): yield a1 + i * d

def geometric_sequence_gen(a1, r, n):
    val = a1
    for _ in range(n):
        if abs(val) > MAX_GEOM_VALUE:
            break
        yield val
        val *= r

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def prime_sequence(n):
    """Generator with CLI progress bar for long primes"""
    if n <= 0: return
    count, num = 0, 2
    print(Fore.YELLOW + f"Generating {n} primes...")
    while count < n:
        if is_prime(num):
            yield num
            count += 1
            if count % max(1, n//50) == 0 or count < 10:
                progress = int(count/n*30)
                bar = "[" + "#"*progress + "-"*(30-progress) + f"] {count}/{n}"
                print(Fore.GREEN + "\r" + bar, end="")
        num += 1
    print()  # newline after done

# =========================
# SAFE PRINTING
# =========================
def print_sequence(seq, name="Sequence", max_terms=100):
    seq_list = list(seq)
    total = len(seq_list)
    if total > max_terms:
        print(Fore.RED + f"{name} too long to display ({total} terms). Please request <= {max_terms}.")
        return
    print(Fore.CYAN + f"{name} (length {total}):")
    for i, val in enumerate(seq_list, 1):
        print(f"{str(val):<20}", end="")
        if i % TERMS_PER_ROW == 0:
            print()
    if total % TERMS_PER_ROW != 0:
        print()

# =========================
# INPUT HELPER
# =========================
def get_number(prompt, allow_float=True, max_val=None):
    while True:
        try:
            val = float(input(prompt)) if allow_float else int(input(prompt))
            if max_val and val > max_val:
                print(Fore.RED + f"Value too large! Max allowed: {max_val}")
                continue
            return val
        except ValueError:
            print(Fore.RED + "Invalid input! Enter a number.")

# =========================
# MENU SYSTEM
# =========================
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

# =========================
# MAIN PROGRAM
# =========================
def main():
    while True:
        print_header("Welcome to PyCalc Pro v2.0")
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




