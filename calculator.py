# Advanced Calculator CLI

# by Li Wen Xiong

# A feature-rich math and sequence calculator with a clean menu-driven interface.

import math
from colorama import init, Fore
from tabulate import tabulate

init(autoreset=True)

# =========================
# BASIC OPERATIONS
# =========================
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b):
    if b == 0: return "Error: Division by zero"
    return a / b

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
def fibonacci(n):
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

def arithmetic_sequence(a1, d, n):
    return [a1 + i * d for i in range(n)]

def geometric_sequence(a1, r, n):
    return [a1 * (r ** i) for i in range(n)]

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def prime_sequence(n):
    if n == 0: return []
    seq = [2]
    num = 3
    while len(seq) < n:
        if is_prime(num):
            seq.append(num)
        num += 2
    return seq

# =========================
# MENU SYSTEM
# =========================
def print_header(title):
    print(Fore.CYAN + "=" * 40)
    print(Fore.YELLOW + title.center(40))
    print(Fore.CYAN + "=" * 40)

def basic_menu():
    while True:
        print_header("Basic Operations")
        print("1. Add\n2. Subtract\n3. Multiply\n4. Divide\n5. Back")
        choice = input(Fore.GREEN + "Choose an option: ")

        if choice == "1":
            a, b = map(float, input("Enter two numbers: ").split())
            print(Fore.MAGENTA + f"Result: {add(a,b)}")
        elif choice == "2":
            a, b = map(float, input("Enter two numbers: ").split())
            print(Fore.MAGENTA + f"Result: {subtract(a,b)}")
        elif choice == "3":
            a, b = map(float, input("Enter two numbers: ").split())
            print(Fore.MAGENTA + f"Result: {multiply(a,b)}")
        elif choice == "4":
            a, b = map(float, input("Enter two numbers: ").split())
            print(Fore.MAGENTA + f"Result: {divide(a,b)}")
        elif choice == "5":
            break
        else:
            print(Fore.RED + "Invalid choice!")

def advanced_menu():
    while True:
        print_header("Advanced Math")
        print("1. Power\n2. Square Root\n3. Factorial\n4. Logarithm\n5. Sine\n6. Cosine\n7. Tangent\n8. Exponential\n9. Absolute\n10. Modulus\n11. Back")
        choice = input(Fore.GREEN + "Choose an option: ")

        if choice == "1":
            a, b = map(float, input("Enter base and exponent: ").split())
            print(Fore.MAGENTA + f"Result: {power(a,b)}")
        elif choice == "2":
            a = float(input("Enter a number: "))
            print(Fore.MAGENTA + f"Result: {sqrt(a)}")
        elif choice == "3":
            a = float(input("Enter a number: "))
            print(Fore.MAGENTA + f"Result: {factorial(a)}")
        elif choice == "4":
            a = float(input("Enter a number: "))
            base = input("Enter base (default 10): ")
            base = float(base) if base else 10
            print(Fore.MAGENTA + f"Result: {logarithm(a,base)}")
        elif choice == "5":
            a = float(input("Enter angle (in degrees): "))
            print(Fore.MAGENTA + f"Result: {sine(a)}")
        elif choice == "6":
            a = float(input("Enter angle (in degrees): "))
            print(Fore.MAGENTA + f"Result: {cosine(a)}")
        elif choice == "7":
            a = float(input("Enter angle (in degrees): "))
            print(Fore.MAGENTA + f"Result: {tangent(a)}")
        elif choice == "8":
            a = float(input("Enter a number: "))
            print(Fore.MAGENTA + f"Result: {exponential(a)}")
        elif choice == "9":
            a = float(input("Enter a number: "))
            print(Fore.MAGENTA + f"Result: {absolute(a)}")
        elif choice == "10":
            a, b = map(float, input("Enter two numbers: ").split())
            print(Fore.MAGENTA + f"Result: {modulus(a,b)}")
        elif choice == "11":
            break
        else:
            print(Fore.RED + "Invalid choice!")

def sequences_menu():
    while True:
        print_header("Sequences & Number Theory")
        print("1. Fibonacci\n2. Arithmetic Sequence\n3. Geometric Sequence\n4. Prime Numbers\n5. Back")
        choice = input(Fore.GREEN + "Choose an option: ")

        if choice == "1":
            n = int(input("Enter number of terms: "))
            seq = fibonacci(n)
            print(Fore.MAGENTA + tabulate([[x] for x in seq], headers=["Fibonacci Sequence"]))
        elif choice == "2":
            a1, d, n = map(float, input("Enter a1, difference, and n: ").split())
            seq = arithmetic_sequence(a1, d, int(n))
            print(Fore.MAGENTA + tabulate([[x] for x in seq], headers=["Arithmetic Sequence"]))
        elif choice == "3":
            a1, r, n = map(float, input("Enter a1, ratio, and n: ").split())
            seq = geometric_sequence(a1, r, int(n))
            print(Fore.MAGENTA + tabulate([[x] for x in seq], headers=["Geometric Sequence"]))
        elif choice == "4":
            n = int(input("Enter number of primes: "))
            seq = prime_sequence(n)
            print(Fore.MAGENTA + tabulate([[x] for x in seq], headers=["Prime Numbers"]))
        elif choice == "5":
            break
        else:
            print(Fore.RED + "Invalid choice!")

# =========================
# MAIN PROGRAM
# =========================
def main():
    while True:
        print_header("Welcome to Advanced Calculator")
        print("1. Basic Operations\n2. Advanced Math\n3. Sequences & Number Theory\n4. Exit")
        choice = input(Fore.GREEN + "Choose a category: ")

        if choice == "1": basic_menu()
        elif choice == "2": advanced_menu()
        elif choice == "3": sequences_menu()
        elif choice == "4":
            print(Fore.CYAN + "Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid choice! Please try again.")

if __name__ == "__main__":

    main()


