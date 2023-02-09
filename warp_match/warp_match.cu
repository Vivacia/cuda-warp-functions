#include <cstdio>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

/**
 * __match_any_sync(unsigned mask, T value); broadcast-and-compare of a value across
 * threads in a warp after synchronizing threads named in mask. Returns mask of threads that
 * have same value of value in mask
 *
 * __match_all_sync(unsigned mask, T value, int *pred); Returns mask if all threads in mask
 * have the same value for value; otherwise 0 is returned. Predicate pred is set to true if
 * all threads in mask have the same value of value; otherwise the predicate is set to false.
 */

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARP MATCH FUNCTION IMPLEMENTATIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Maintain an array of predicates since each thread has access to a predicate register.
 * We simulate this with the array instead of actual registers.
*/

__device__ int match_any_sync_shared_var_arr[32];
__device__ bool match_any_sync_updated[32] = {0};

__device__ int match_any_sync(unsigned mask, unsigned value) {
    int tid = threadIdx.x;
    int ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        match_any_sync_shared_var_arr[tid] = value;
        match_any_sync_updated[tid] = true;
    }

    for (int i = 0; i < 32; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !match_any_sync_updated[i]);
        if (value == match_any_sync_shared_var_arr[tid])
            ret |= (0x1 << i);
    }

    return ret;
}

__device__ int match_all_sync_shared_var_arr[32];
__device__ bool match_all_sync_updated[32] = {0};

__device__ int match_all_sync(unsigned mask, unsigned value, int *pred) {
    int tid = threadIdx.x;
    int ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        match_all_sync_shared_var_arr[tid] = value;
        match_all_sync_updated[tid] = true;
    }

    for (int i = 0; i < 32; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !match_all_sync_updated[i]);
        if (value == match_all_sync_shared_var_arr[tid]) {
            ret |= (0x1 << i);
        } else {
            *pred = false;
            return 0;
        }
    }

    *pred = true;
    return ret;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARP MATCH FUNCTION BENCHMARKS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

// For benchmarking
const int NUM_ROUNDS = 10;

// MATCH_ALL_SYNC /////////////////////////////////////////////////////////////////////////////////////////

__device__ int val = 0;

__global__ void test_match_all_sync_default_true(int arg) {
    int pred[1] = {0};
    unsigned ret = __match_all_sync(0xffffffff, arg, pred);
    if (ret != 0xffffffff || *pred != true)
        printf("Thread %d failed.\n", threadIdx.x);
}

__global__ void test_match_all_sync_custom_true(int arg) {
    int pred[1] = {0};
    int ret = match_all_sync(0xffffffff, arg, pred);
    if (ret != 0xffffffff || *pred != true)
        printf("Thread %d failed.\n", threadIdx.x);
}

__global__ void test_match_all_sync_default_false() {
    int pred[1] = {0};
    atomicAdd(&val, 1);
    int ret = __match_all_sync(0xffffffff, val, pred);
    if (ret != 0 || *pred != false)
        printf("Thread %d failed.\n", threadIdx.x);
}

__global__ void test_match_all_sync_custom_false() {
    int pred[1] = {0};
    atomicAdd(&val, 1);
    unsigned ret = match_all_sync(0xffffffff, val, pred);
    if (ret != 0 || *pred != false)
        printf("Thread %d failed.\n", threadIdx.x);
}

void benchmark_match_all_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;
    srand(time(0));

    for (int i = 0; i < NUM_ROUNDS; i++) {
        int arg = rand() % 1000;

        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_match_all_sync_default_true<<< 1, 32 >>>(arg);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_match_all_sync_custom_true<<< 1, 32 >>>(arg);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;

        gettimeofday(&start_default, 0);
        test_match_all_sync_default_false<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        gettimeofday(&start_custom, 0);
        test_match_all_sync_custom_false<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __match_all_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom match_all_sync() = %f ms\n\n", avg_custom);
}

// MATCH_ANY_SYNC /////////////////////////////////////////////////////////////////////////////////////////

__global__ void test_match_any_sync_default_simple() {
    unsigned ret = __match_any_sync(0xffffffff, 0);
    printf("Default thread %d final value = %d\n", threadIdx.x, ret);
}

__global__ void test_match_any_sync_custom_simple() {
    int ret = match_any_sync(0xffffffff, 0);
    printf("Custom thread %d final value = %d\n", threadIdx.x, ret);
}

__global__ void test_match_any_sync_default_alternate() {
    int tid = threadIdx.x;
    unsigned ret = __match_any_sync(0xffffffff, tid % 3);
    printf("Default thread %d final value = %d\n", tid, ret);
}

__global__ void test_match_any_sync_custom_alternate() {
    int tid = threadIdx.x;
    int ret = match_any_sync(0xffffffff, tid % 3);
    printf("Custom thread %d final value = %d\n", tid, ret);
}

__global__ void test_match_any_sync_default_unique() {
    atomicAdd(&val, 1);
    unsigned ret = __match_any_sync(0xffffffff, val);
    printf("Default thread %d final value = %d\n", threadIdx.x, ret);
}

__global__ void test_match_any_sync_custom_unique() {
    atomicAdd(&val, 1);
    int ret = match_any_sync(0xffffffff, val);
    printf("Custom thread %d final value = %d\n", threadIdx.x, ret);
}

void benchmark_match_any_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;
    srand(time(0));

    for (int i = 0; i < NUM_ROUNDS; i++) {
        int arg = rand() % 1000;

        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_match_any_sync_default_simple<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_match_any_sync_custom_simple<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;

        gettimeofday(&start_default, 0);
        test_match_any_sync_default_alternate<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        gettimeofday(&start_custom, 0);
        test_match_any_sync_custom_alternate<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __match_any_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom match_any_sync() = %f ms\n\n", avg_custom);
}

int main() {
    printf("CUDA Warp Match Benchmarks\n\n");
    benchmark_match_all_sync();
    benchmark_match_any_sync();
    return 0;
}