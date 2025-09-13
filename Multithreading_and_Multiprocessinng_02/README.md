# Multithreading and Multiprocessing Examples

This repository contains comprehensive examples demonstrating the use of multithreading and multiprocessing in Python. These concepts are essential for improving performance in different types of applications, whether they are I/O-bound or CPU-bound.

## 📚 Overview

The examples in this repository cover:
- **Basic multithreading** with manual thread creation
- **Basic multiprocessing** with manual process creation
- **Advanced multithreading** using ThreadPoolExecutor
- **Advanced multiprocessing** using ProcessPoolExecutor
- **Real-world use cases** for both threading and multiprocessing

## 🚀 Getting Started

### Prerequisites
- Python 3.6+
- Required packages (for web scraping example):
  ```bash
  pip install requests beautifulsoup4
  ```

## 📁 File Structure

### Basic Examples

#### 1. `multithreading.py` - Basic Multithreading
**Purpose**: Demonstrates fundamental multithreading concepts using manual thread creation.

**Key Features**:
- Creates two threads that run concurrently
- One thread prints numbers (0-4)
- Another thread prints letters (a-e)
- Measures execution time to show performance improvement
- Uses `threading.Thread()` for manual thread creation
- Demonstrates `start()` and `join()` methods

**Use Case**: I/O-bound tasks that can run concurrently (like printing with delays)

#### 2. `multiprocessing_demo.py` - Basic Multiprocessing
**Purpose**: Shows basic multiprocessing concepts using manual process creation.

**Key Features**:
- Creates two processes that run in parallel
- One process calculates squares of numbers
- Another process calculates cubes of numbers
- Measures execution time
- Uses `multiprocessing.Process()` for manual process creation
- Includes the `if __name__ == "__main__":` guard (essential for multiprocessing)

**Use Case**: CPU-bound tasks that benefit from true parallelism

### Advanced Examples

#### 3. `advance_multi_threading.py` - ThreadPoolExecutor
**Purpose**: Demonstrates advanced multithreading using ThreadPoolExecutor.

**Key Features**:
- Uses `concurrent.futures.ThreadPoolExecutor`
- Processes a list of numbers with a thread pool
- Limits the number of concurrent threads (`max_workers=3`)
- More efficient than manual thread creation
- Automatic thread management and cleanup

**Use Case**: When you need to process multiple items concurrently with controlled concurrency

#### 4. `advance_multi_processing.py` - ProcessPoolExecutor
**Purpose**: Shows advanced multiprocessing using ProcessPoolExecutor.

**Key Features**:
- Uses `concurrent.futures.ProcessPoolExecutor`
- Processes a list of numbers with a process pool
- Limits the number of concurrent processes (`max_workers=3`)
- More efficient than manual process creation
- Automatic process management and cleanup

**Use Case**: CPU-intensive tasks that benefit from multiple CPU cores

### Real-World Examples

#### 5. `factorial_multi_processing.py` - CPU-Bound Task Example
**Purpose**: Real-world example of multiprocessing for CPU-intensive calculations.

**Key Features**:
- Calculates factorials of large numbers (900, 800, 700)
- Uses `multiprocessing.Pool()` for parallel computation
- Demonstrates significant performance improvement for CPU-bound tasks
- Shows how to distribute computational workload across CPU cores

**Use Case**: Mathematical computations, data processing, scientific calculations

#### 6. `usecase_multi_threading.py` - I/O-Bound Task Example
**Purpose**: Real-world example of multithreading for I/O-bound tasks (web scraping).

**Key Features**:
- Fetches content from multiple URLs concurrently
- Uses `requests` and `BeautifulSoup` for web scraping
- Demonstrates significant performance improvement for I/O-bound tasks
- Shows how to handle multiple network requests efficiently

**Use Case**: Web scraping, API calls, file I/O operations, database queries

## 🔍 Key Concepts Explained

### Multithreading vs Multiprocessing

| Aspect | Multithreading | Multiprocessing |
|--------|----------------|-----------------|
| **Memory** | Shared memory space | Separate memory spaces |
| **CPU Usage** | Single CPU core (concurrent) | Multiple CPU cores (parallel) |
| **Best For** | I/O-bound tasks | CPU-bound tasks |
| **Overhead** | Low | Higher |
| **Communication** | Easy (shared memory) | Requires inter-process communication |

### When to Use Each

**Use Multithreading for:**
- I/O-bound operations (network requests, file operations)
- Tasks that spend time waiting
- User interface responsiveness
- Web scraping and API calls

**Use Multiprocessing for:**
- CPU-intensive calculations
- Mathematical computations
- Data processing
- Tasks that benefit from true parallelism

## 🛠️ Running the Examples

1. **Basic Examples**:
   ```bash
   python multithreading.py
   python multiprocessing_demo.py
   ```

2. **Advanced Examples**:
   ```bash
   python advance_multi_threading.py
   python advance_multi_processing.py
   ```

3. **Real-World Examples**:
   ```bash
   python factorial_multi_processing.py
   python usecase_multi_threading.py
   ```

## 📊 Performance Comparison

The examples include timing measurements to demonstrate performance improvements:

- **Sequential vs Concurrent**: Basic examples show how concurrent execution reduces total time
- **Thread Pool vs Manual**: Advanced examples show more efficient resource management
- **Real-world Impact**: Factorial and web scraping examples show practical performance gains

## 🔧 Best Practices

1. **Always use `if __name__ == "__main__":`** for multiprocessing code
2. **Choose the right tool**: Threading for I/O-bound, multiprocessing for CPU-bound
3. **Use thread/process pools** for better resource management
4. **Handle exceptions** properly in concurrent code
5. **Avoid shared state** when possible to prevent race conditions
6. **Measure performance** to ensure your optimization is effective

## 🚨 Common Pitfalls

- **GIL (Global Interpreter Lock)**: Python's GIL limits true parallelism in threading
- **Race Conditions**: Shared state can cause unpredictable behavior
- **Resource Overhead**: Too many threads/processes can hurt performance
- **Memory Usage**: Multiprocessing uses more memory than threading

## 📖 Learning Path

1. Start with `multithreading.py` to understand basic concepts
2. Move to `multiprocessing_demo.py` to see the difference
3. Explore advanced examples to learn better practices
4. Study real-world examples to see practical applications

## 🤝 Contributing

Feel free to add more examples or improve existing ones. Consider adding:
- Examples with error handling
- More complex real-world scenarios
- Performance benchmarking tools
- Examples with different Python versions

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy Coding! 🐍✨**

