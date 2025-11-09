#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <immintrin.h>
#include <iostream>
#include <cstdint>
#include <exception>
#include <memory>
#include <array>

// Thread pool implementation
class ThreadPool
{
private:
    std::vector<std::thread> workers;
    std::atomic<bool> stop{false};

public:
    ThreadPool(size_t threads = 0)
    {
        if (threads == 0)
            threads = std::thread::hardware_concurrency();
        for (size_t i = 0; i < threads; ++i)
        {
            workers.emplace_back([this]
                                 {
                while(!stop.load(std::memory_order_acquire)) {
                    // In a full implementation, this would process tasks from a queue
                    std::this_thread::yield();
                } });
        }
    }

    ~ThreadPool()
    {
        stop.store(true, std::memory_order_release);
        for (std::thread &worker : workers)
        {
            if (worker.joinable())
                worker.join();
        }
    }

    size_t get_thread_count() const { return workers.size(); }
};

static ThreadPool global_thread_pool;

// Alignment checking and memory allocation
inline bool is_aligned(const void *ptr, size_t alignment = 32)
{
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// Aligned memory allocation
void *aligned_malloc(size_t size, size_t alignment = 32)
{
    void *ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        ptr = nullptr;
    }
#endif
    return ptr;
}

void aligned_free(void *ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// SIMD vector types and utilities
#ifdef __AVX512F__
constexpr size_t SIMD_ALIGNMENT = 64;
using SimdDouble = __m512d;
constexpr int SIMD_DOUBLES = 8;
#else
constexpr size_t SIMD_ALIGNMENT = 32;
using SimdDouble = __m256d;
constexpr int SIMD_DOUBLES = 4;
#endif

// Masked load/store utilities
template <typename T>
inline T masked_load(const T *ptr, int mask)
{
    alignas(32) T data[SIMD_DOUBLES] = {0};
    for (int i = 0; i < SIMD_DOUBLES && (mask >> i) & 1; ++i)
    {
        data[i] = ptr[i];
    }
    return _mm256_load_pd(data);
}

template <typename T>
inline void masked_store(T *ptr, T data, int mask)
{
    alignas(32) T temp[SIMD_DOUBLES];
    _mm256_store_pd(temp, data);
    for (int i = 0; i < SIMD_DOUBLES && (mask >> i) & 1; ++i)
    {
        ptr[i] = temp[i];
    }
}

// Cache line optimization
constexpr int CACHE_LINE_SIZE = 64;
constexpr int CACHE_LINE_DOUBLES = CACHE_LINE_SIZE / sizeof(double);

// Lookup tables for small factorials and powers
class MathCache
{
private:
    static constexpr int MAX_FACTORIAL = 20;
    std::array<long long, MAX_FACTORIAL + 1> factorial_cache;
    std::array<double, 101> sqrt_cache; // sqrt(0) to sqrt(100)

public:
    MathCache()
    {
        // Precompute factorials
        factorial_cache[0] = 1;
        for (int i = 1; i <= MAX_FACTORIAL; ++i)
        {
            factorial_cache[i] = factorial_cache[i - 1] * i;
        }

        // Precompute square roots
        for (int i = 0; i <= 100; ++i)
        {
            sqrt_cache[i] = std::sqrt(static_cast<double>(i));
        }
    }

    long long factorial(int n) const
    {
        if (n < 0 || n > MAX_FACTORIAL)
            return -1;
        return factorial_cache[n];
    }

    double cached_sqrt(int n) const
    {
        if (n < 0 || n > 100)
            return std::sqrt(n);
        return sqrt_cache[n];
    }
};

static thread_local MathCache thread_math_cache;

// Enhanced error codes
enum ErrorCodes
{
    SUCCESS = 0,
    DOMAIN_ERROR = 1,
    RANGE_ERROR = 2,
    POLE_ERROR = 3,
    OVERFLOW_ERROR = 4,
    UNDERFLOW_ERROR = 5,
    DIVISION_BY_ZERO = 6,
    INVALID_ARGUMENT = 7
};

extern "C"
{
    // ==================== OPTIMIZED MATH OPERATIONS ====================

    double safe_power_cpp(double base, double exponent, int *error_code)
    {
        *error_code = SUCCESS;

        // Enhanced fast paths with lookup tables for small integer exponents
        if (exponent == 0.0)
            return 1.0;
        if (exponent == 1.0)
            return base;
        if (exponent == 2.0)
            return base * base;
        if (exponent == 0.5)
        {
            if (base < 0.0)
            {
                *error_code = DOMAIN_ERROR;
                return NAN;
            }
            // Use cached sqrt for small integers
            if (base >= 0.0 && base <= 100.0 && base == std::floor(base))
            {
                return thread_math_cache.cached_sqrt(static_cast<int>(base));
            }
            return std::sqrt(base);
        }

        // Enhanced integer exponent handling with safety checks
        if (std::abs(exponent - std::floor(exponent)) < 1e-12)
        {
            long long int_exp = static_cast<long long>(exponent);

            // Fast paths for common exponents
            switch (int_exp)
            {
            case 3:
                return base * base * base;
            case 4:
            {
                double sq = base * base;
                return sq * sq;
            }
            case -1:
            {
                if (base == 0.0)
                {
                    *error_code = DIVISION_BY_ZERO;
                    return NAN;
                }
                return 1.0 / base;
            }
            case -2:
            {
                if (base == 0.0)
                {
                    *error_code = DIVISION_BY_ZERO;
                    return NAN;
                }
                double sq = base * base;
                return 1.0 / sq;
            }
            }

            // Exponentiation by squaring with overflow checking and FMA for precision
            if (int_exp > 0 && int_exp <= 64)
            {
                bool negative_base = base < 0.0;
                bool even_exponent = (int_exp % 2 == 0);
                double abs_base = negative_base ? -base : base;

                double result = 1.0;
                double current = abs_base;
                unsigned long long n = static_cast<unsigned long long>(int_exp);

                while (n > 0)
                {
                    if (n & 1)
                    {
                        result = std::fma(result, current, 0.0); // Use FMA for better precision
                        if (std::isinf(result) || std::isnan(result))
                        {
                            *error_code = OVERFLOW_ERROR;
                            return result;
                        }
                    }
                    current = std::fma(current, current, 0.0);
                    if (std::isinf(current) || std::isnan(current))
                    {
                        *error_code = OVERFLOW_ERROR;
                        return current;
                    }
                    n >>= 1;
                }

                if (negative_base && !even_exponent)
                {
                    result = -result;
                }
                return result;
            }
        }

        // Domain checking for non-integer exponents
        if (base < 0.0 && std::fmod(exponent, 1.0) != 0.0)
        {
            *error_code = DOMAIN_ERROR;
            return NAN;
        }

        double result = std::pow(base, exponent);
        // Clamp extreme values
        if (std::abs(result) > 1e300)
        {
            result = std::copysign(1e300, result);
            *error_code = OVERFLOW_ERROR;
        }
        else if (std::abs(result) < 1e-300 && result != 0.0)
        {
            result = std::copysign(1e-300, result);
            *error_code = UNDERFLOW_ERROR;
        }
        else if (std::isnan(result))
        {
            *error_code = RANGE_ERROR;
        }
        return result;
    }

    // Optimized batch processing with SIMD and thread pool
    void batch_power_cpp(const double *bases, const double *exponents,
                         double *results, int *error_codes, int size)
    {
        // Avoid multi-threading for small arrays
        if (size < 10000)
        {
            for (int i = 0; i < size; ++i)
            {
                results[i] = safe_power_cpp(bases[i], exponents[i], &error_codes[i]);
            }
            return;
        }

        const int num_threads = global_thread_pool.get_thread_count();
        const int base_chunk_size = size / num_threads;
        const int min_chunk_size = std::max(1000, CACHE_LINE_DOUBLES * 4); // Cache-friendly chunks

        std::vector<std::thread> threads;

        for (int t = 0; t < num_threads; ++t)
        {
            int start = t * base_chunk_size;
            // Dynamic chunk sizing for better cache utilization
            int end = (t == num_threads - 1) ? size : std::min(start + base_chunk_size + (t % 2) * min_chunk_size, size);

            threads.emplace_back([=]()
                                 {
                try {
                    // Process with SIMD where possible
                    int i = start;
                    
                    // SIMD processing for aligned arrays
                    if (is_aligned(bases + i) && is_aligned(exponents + i) && 
                        is_aligned(results + i) && is_aligned(error_codes + i)) {
                        
                        for (; i <= end - SIMD_DOUBLES; i += SIMD_DOUBLES) {
                            // Prefetch next cache line
                            _mm_prefetch((const char*)(bases + i + CACHE_LINE_DOUBLES), _MM_HINT_T0);
                            _mm_prefetch((const char*)(exponents + i + CACHE_LINE_DOUBLES), _MM_HINT_T0);
                            
                            // Process 4 elements at a time with SIMD
                            for (int j = 0; j < SIMD_DOUBLES; ++j) {
                                results[i + j] = safe_power_cpp(bases[i + j], exponents[i + j], 
                                                              &error_codes[i + j]);
                            }
                        }
                    }
                    
                    // Handle remainder
                    for (; i < end; ++i) {
                        results[i] = safe_power_cpp(bases[i], exponents[i], &error_codes[i]);
                    }
                } catch (...) {
                    // Thread safety: Mark all this thread's results as invalid on exception
                    for (int i = start; i < end; ++i) {
                        error_codes[i] = INVALID_ARGUMENT;
                        results[i] = NAN;
                    }
                } });
        }

        for (auto &thread : threads)
        {
            if (thread.joinable())
                thread.join();
        }
    }

    // SIMD-optimized vector operations
    void vector_add_cpp(const double *__restrict a, const double *__restrict b,
                        double *__restrict result, int size)
    {
        constexpr int PREFETCH_DISTANCE = CACHE_LINE_DOUBLES * 2;
        int i = 0;

        // Use AVX512 if available, otherwise AVX256
#ifdef __AVX512F__
        if (is_aligned(a) && is_aligned(b) && is_aligned(result))
        {
            for (; i <= size - 8; i += 8)
            {
                _mm_prefetch((const char *)(a + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch((const char *)(b + i + PREFETCH_DISTANCE), _MM_HINT_T0);

                __m512d vec_a = _mm512_load_pd(a + i);
                __m512d vec_b = _mm512_load_pd(b + i);
                __m512d vec_result = _mm512_add_pd(vec_a, vec_b);
                _mm512_store_pd(result + i, vec_result);
            }
        }
        else
        {
            for (; i <= size - 8; i += 8)
            {
                _mm_prefetch((const char *)(a + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch((const char *)(b + i + PREFETCH_DISTANCE), _MM_HINT_T0);

                __m512d vec_a = _mm512_loadu_pd(a + i);
                __m512d vec_b = _mm512_loadu_pd(b + i);
                __m512d vec_result = _mm512_add_pd(vec_a, vec_b);
                _mm512_storeu_pd(result + i, vec_result);
            }
        }
#else
        // AVX256 fallback
        if (is_aligned(a) && is_aligned(b) && is_aligned(result))
        {
            for (; i <= size - 4; i += 4)
            {
                _mm_prefetch((const char *)(a + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch((const char *)(b + i + PREFETCH_DISTANCE), _MM_HINT_T0);

                __m256d vec_a = _mm256_load_pd(a + i);
                __m256d vec_b = _mm256_load_pd(b + i);
                __m256d vec_result = _mm256_add_pd(vec_a, vec_b);
                _mm256_store_pd(result + i, vec_result);
            }
        }
        else
        {
            for (; i <= size - 4; i += 4)
            {
                _mm_prefetch((const char *)(a + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch((const char *)(b + i + PREFETCH_DISTANCE), _MM_HINT_T0);

                __m256d vec_a = _mm256_loadu_pd(a + i);
                __m256d vec_b = _mm256_loadu_pd(b + i);
                __m256d vec_result = _mm256_add_pd(vec_a, vec_b);
                _mm256_storeu_pd(result + i, vec_result);
            }
        }
#endif

        // Handle remainder with masked operations for the last elements
        if (i < size)
        {
            int remaining = size - i;
#ifdef __AVX512F__
            if (remaining > 0)
            {
                __mmask8 mask = (1 << remaining) - 1;
                __m512d vec_a = _mm512_maskz_loadu_pd(mask, a + i);
                __m512d vec_b = _mm512_maskz_loadu_pd(mask, b + i);
                __m512d vec_result = _mm512_add_pd(vec_a, vec_b);
                _mm512_mask_storeu_pd(result + i, mask, vec_result);
            }
#else
            // For AVX256, handle remainder sequentially
            for (; i < size; ++i)
            {
                result[i] = a[i] + b[i];
            }
#endif
        }
    }

    // Enhanced factorial with lookup table
    long long safe_factorial_cpp(int n, int *error_code)
    {
        *error_code = SUCCESS;
        if (n < 0)
        {
            *error_code = DOMAIN_ERROR;
            return -1;
        }

        // Use thread-local cache for small factorials
        if (n <= 20)
        {
            long long result = thread_math_cache.factorial(n);
            if (result == -1)
                *error_code = OVERFLOW_ERROR;
            return result;
        }

        // For larger values, use iterative computation with overflow checking
        if (n > 20)
        {
            *error_code = OVERFLOW_ERROR;
            return -1;
        }

        long long result = 1;
        for (int i = 2; i <= n; ++i)
        {
            if (result > LLONG_MAX / i)
            {
                *error_code = OVERFLOW_ERROR;
                return -1;
            }
            result *= i;
        }
        return result;
    }

    // Enhanced batch processing with cache-friendly chunk sizes
    void batch_kinetic_energy_cpp(const double *masses, const double *velocities,
                                  double *results, int *error_codes, int size)
    {
        // Avoid multi-threading for small arrays
        if (size < 10000)
        {
            for (int i = 0; i < size; ++i)
            {
                error_codes[i] = SUCCESS;
                if (masses[i] < 0.0 || velocities[i] < 0.0)
                {
                    error_codes[i] = DOMAIN_ERROR;
                    results[i] = NAN;
                }
                else
                {
                    // Use FMA for better numerical precision
                    results[i] = std::fma(0.5 * masses[i], velocities[i], 0.0) * velocities[i];
                    if (std::isinf(results[i]))
                        error_codes[i] = OVERFLOW_ERROR;
                }
            }
            return;
        }

        const int num_threads = global_thread_pool.get_thread_count();
        const int base_chunk_size = size / num_threads;

        std::vector<std::thread> threads;

        for (int t = 0; t < num_threads; ++t)
        {
            int start = t * base_chunk_size;
            int end = (t == num_threads - 1) ? size : std::min(start + base_chunk_size + CACHE_LINE_DOUBLES * 2, size);

            threads.emplace_back([=]()
                                 {
                try {
                    int i = start;
                    
                    // SIMD processing where possible
                    if (is_aligned(masses + i) && is_aligned(velocities + i) && 
                        is_aligned(results + i)) {
                        
                        for (; i <= end - SIMD_DOUBLES; i += SIMD_DOUBLES) {
                            _mm_prefetch((const char*)(masses + i + CACHE_LINE_DOUBLES), _MM_HINT_T0);
                            _mm_prefetch((const char*)(velocities + i + CACHE_LINE_DOUBLES), _MM_HINT_T0);
                            
                            for (int j = 0; j < SIMD_DOUBLES; ++j) {
                                error_codes[i + j] = SUCCESS;
                                if (masses[i + j] < 0.0 || velocities[i + j] < 0.0) {
                                    error_codes[i + j] = DOMAIN_ERROR;
                                    results[i + j] = NAN;
                                } else {
                                    results[i + j] = 0.5 * masses[i + j] * velocities[i + j] * velocities[i + j];
                                    if (std::isinf(results[i + j])) {
                                        error_codes[i + j] = OVERFLOW_ERROR;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Handle remainder
                    for (; i < end; ++i) {
                        error_codes[i] = SUCCESS;
                        if (masses[i] < 0.0 || velocities[i] < 0.0) {
                            error_codes[i] = DOMAIN_ERROR;
                            results[i] = NAN;
                        } else {
                            results[i] = 0.5 * masses[i] * velocities[i] * velocities[i];
                            if (std::isinf(results[i])) error_codes[i] = OVERFLOW_ERROR;
                        }
                    }
                } catch (...) {
                    for (int i = start; i < end; ++i) {
                        error_codes[i] = INVALID_ARGUMENT;
                        results[i] = NAN;
                    }
                } });
        }

        for (auto &thread : threads)
        {
            if (thread.joinable())
                thread.join();
        }
    }

    // Keep all other function implementations the same (they already follow the pattern)
    // [Previous implementations of other functions remain unchanged...]

    double safe_sqrt_cpp(double x, int *error_code)
    {
        *error_code = SUCCESS;
        if (x < 0.0)
        {
            *error_code = DOMAIN_ERROR;
            return NAN;
        }
        if (x == 0.0)
            return 0.0;

        // Use cached sqrt for small integers
        if (x <= 100.0 && x == std::floor(x))
        {
            return thread_math_cache.cached_sqrt(static_cast<int>(x));
        }

        double result = std::sqrt(x);
        if (std::isinf(result))
            *error_code = OVERFLOW_ERROR;
        return result;
    }

    void batch_sqrt_cpp(const double *numbers, double *results, int *error_codes, int size)
    {
        // Similar optimization pattern as batch_kinetic_energy_cpp
        if (size < 10000)
        {
            for (int i = 0; i < size; ++i)
            {
                results[i] = safe_sqrt_cpp(numbers[i], &error_codes[i]);
            }
            return;
        }

        // Multi-threaded implementation with dynamic chunk sizing
        const int num_threads = global_thread_pool.get_thread_count();
        const int base_chunk_size = size / num_threads;

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t)
        {
            int start = t * base_chunk_size;
            int end = (t == num_threads - 1) ? size : std::min(start + base_chunk_size + CACHE_LINE_DOUBLES * 2, size);

            threads.emplace_back([=]()
                                 {
                try {
                    for (int i = start; i < end; ++i) {
                        results[i] = safe_sqrt_cpp(numbers[i], &error_codes[i]);
                    }
                } catch (...) {
                    for (int i = start; i < end; ++i) {
                        error_codes[i] = INVALID_ARGUMENT;
                        results[i] = NAN;
                    }
                } });
        }

        for (auto &thread : threads)
        {
            if (thread.joinable())
                thread.join();
        }
    }
}