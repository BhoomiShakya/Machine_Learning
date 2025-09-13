'''
    Real World Example: Multiprocessing for CPU-bound Tasks
    Scenario:Factorial Calculation
    Factorial calculations, especially for large numbers,
    involvee significant computational work.
    Multiprocessing can be used to distribute the workload across multiple CPU cores, improving performance.
'''
import multiprocessing
import math
import time

def compute_factorial(number):
    print(f"Compute factorial of {number}")
    result = math.factorial(number)
    print(f"Factorial of {number} is computed")
    return result

if __name__ == "__main__":
    numbers = [900, 800, 700]
    start_time = time.time()

    with multiprocessing.Pool() as pool:
        results = pool.map(compute_factorial, numbers)

    end_time = time.time()

    print(f"Results: [factorials computed]")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
