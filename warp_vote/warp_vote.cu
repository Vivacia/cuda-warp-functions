#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

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
 * __all_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask and
 * return non-zero if and only if predicate evaluates to non-zero for all of them.
 * 
 * __any_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask and
 * return non-zero if and only if predicate evaluates to non-zero for any of them.
 *
 * __ballot_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask
 * and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for
 * the Nth thread of the warp and the Nth thread is active.
 *
 * __activemask(): Return a mask with Nth bit set for all threads currently active in calling warp.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARP VOTE FUNCTION IMPLEMENTATIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Maintain an array of predicates since each thread has access to a predicate register.
 * We simulate this with the array instead of actual registers.
*/

// every thread goes through the function. use it to populate array. use that array for computation

__device__ int all_sync_shared_var_arr[32];
__device__ bool all_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ int all_sync(unsigned mask, int predicate) {
    int tid = threadIdx.x;
    int ret = 1;

    if (((0x1 << tid) & mask) != 0) {
        all_sync_shared_var_arr[tid] = predicate;
        all_sync_updated[tid] = true;
    }

    for (int i = 0; i < warpSize; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !all_sync_updated[i]);
        ret &= all_sync_shared_var_arr[i];
    }

    return ret;
}

__device__ int any_sync_shared_var_arr[32];
__device__ bool any_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ int any_sync(unsigned mask, int predicate) {
    int tid = threadIdx.x;
    int ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        any_sync_shared_var_arr[tid] = predicate;
        any_sync_updated[tid] = true;
    }

    for (int i = 0; i < warpSize; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !any_sync_updated[i]);
        ret |= any_sync_shared_var_arr[i];
        if (ret)
            break;
    }

    return ret;
}

__device__ int ballot_sync_shared_var_arr[32];
__device__ bool ballot_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ unsigned ballot_sync(unsigned mask, int predicate) {
    int tid = threadIdx.x;
    int ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        ballot_sync_shared_var_arr[tid] = predicate;
        ballot_sync_updated[tid] = true;
    }

    for (int i = 0; i < warpSize; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !ballot_sync_updated[i]);
        if (ballot_sync_shared_var_arr[i])
            ret |= (0x1 << i);
    }

    return ret;
}

// https://stackoverflow.com/questions/11217117/equivalent-of-usleep-in-cuda-kernel
__device__ unsigned _activemask = 0;
__device__ clock_t *global_now;

__device__ unsigned activemask() {
    int tid = threadIdx.x;
    atomicAnd(&_activemask, tid << 0x1);

    // need to wait until other active threads modify the active mask
    clock_t start = clock();
    clock_t now;
    for (;;) {
        now = clock();
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= 10000) {
            break;
        }
    }
    // Stored "now" in global memory here to prevent the
    // compiler from optimizing away the entire loop.
    *global_now = now;

    return _activemask;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARP VOTE FUNCTION BENCHMARKS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

// For benchmarking
const int NUM_ROUNDS = 1000;

// VOTE ALL_SYNC /////////////////////////////////////////////////////////////////////////////////////////

__global__ void test_all_sync_default_pass() {
    unsigned ret = __all_sync(0xffffffff, 1);
    if (ret != true) {
        printf("Default thread %d failed.\n", threadIdx.x);
    }
}

__global__ void test_all_sync_custom_pass() {
    unsigned ret = all_sync(0xffffffff, 1);
    if (ret != true) {
        printf("Custom thread %d failed.\n", threadIdx.x);
    }
}

__global__ void test_all_sync_default_fail() {
    int tid = threadIdx.x;
    unsigned ret = all_sync(0xffffffff, tid % 2);
    if (ret != false) {
        printf("Default thread %d failed.\n", tid);
    }
}

__global__ void test_all_sync_custom_fail() {
    int tid = threadIdx.x;
    unsigned ret = all_sync(0xffffffff, tid % 2);
    if (ret != false) {
        printf("Custom thread %d failed.\n", tid);
    }
}

void benchmark_all_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;

    for (int i = 0; i < NUM_ROUNDS; i++) {
        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_all_sync_default_pass<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_all_sync_custom_pass<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;

        gettimeofday(&start_default, 0);
        test_all_sync_default_pass<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        gettimeofday(&start_custom, 0);
        test_all_sync_custom_pass<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __all_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom all_sync() = %f ms\n\n", avg_custom);
}


// VOTE ANY_SYNC /////////////////////////////////////////////////////////////////////////////////////////

__global__ void test_any_sync_default_two() {
    int tid = threadIdx.x;
    unsigned ret = __any_sync(0xffffffff, tid % 2);
    if (ret == 0) {
        printf("Default thread %d failed.\n", tid);
    }
}

__global__ void test_any_sync_custom_two() {
    int tid = threadIdx.x;
    unsigned ret = __any_sync(0xffffffff, tid % 2);
    if (ret == 0) {
        printf("Custom thread %d failed.\n", tid);
    }
}

__global__ void test_any_sync_default_four() {
    int tid = threadIdx.x;
    unsigned ret = any_sync(0xffffffff, tid % 4);
    if (ret == 0) {
        printf("Default thread %d failed.\n", tid);
    }
}

__global__ void test_any_sync_custom_four() {
    int tid = threadIdx.x;
    unsigned ret = any_sync(0xffffffff, tid % 4);
    if (ret == 0) {
        printf("Custom thread %d failed.\n", tid);
    }
}

void benchmark_any_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;

    for (int i = 0; i < NUM_ROUNDS; i++) {
        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_any_sync_default_two<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_any_sync_custom_two<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;

        gettimeofday(&start_default, 0);
        test_any_sync_default_four<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        gettimeofday(&start_custom, 0);
        test_any_sync_custom_four<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __any_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom any_sync() = %f ms\n\n", avg_custom);
}


// VOTE BALLOT_SYNC /////////////////////////////////////////////////////////////////////////////////////////

__global__ void test_ballot_sync_default_two() {
    int tid = threadIdx.x;
    unsigned ret = __ballot_sync(0xffffffff, tid % 2);
    if (tid % 2 == 1 && (ret & (0x1 << tid)) == 0) {
        printf("Default thread %d failed.\n", tid);
    }
}

__global__ void test_ballot_sync_custom_two() {
    int tid = threadIdx.x;
    unsigned ret = __ballot_sync(0xffffffff, tid % 2);
    if (tid % 2 == 1 && (ret & (0x1 << tid)) == 0) {
        printf("Custom thread %d failed.\n", tid);
    }
}

__global__ void test_ballot_sync_default_four() {
    int tid = threadIdx.x;
    unsigned ret = __ballot_sync(0xffffffff, tid % 4);
    if (tid % 4 == 1 && (ret & (0x1 << tid)) == 0) {
        printf("Default thread %d failed.\n", tid);
    }
}

__global__ void test_ballot_sync_custom_four() {
    int tid = threadIdx.x;
    unsigned ret = ballot_sync(0xffffffff, tid % 4);
    if (tid % 4 == 1 && (ret & (0x1 << tid)) == 0) {
        printf("Custom thread %d failed.\n", tid);
    }
}

void benchmark_ballot_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;

    for (int i = 0; i < NUM_ROUNDS; i++) {
        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_ballot_sync_default_two<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_ballot_sync_custom_two<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;

        gettimeofday(&start_default, 0);
        test_ballot_sync_default_four<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        gettimeofday(&start_custom, 0);
        test_ballot_sync_custom_four<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __ballot_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom ballot_sync() = %f ms\n\n", avg_custom);
}

// VOTE ACTIVEMASK /////////////////////////////////////////////////////////////////////////////////////////

__global__ void test_activemask_default_two() {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        unsigned ret = __activemask();
        if (ret != 0b10101010101010101010101010101010) {
            printf("Default thread %d failed.\n", tid);
        }
    }
}

__global__ void test_activemask_custom_two() {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        unsigned ret = activemask();
        if (ret != 0b10101010101010101010101010101010) {
            printf("Custom thread %d failed.\n", tid);
        }
    }
}

__global__ void test_activemask_default_four() {
    int tid = threadIdx.x;
    if (tid % 4 == 0) {
        unsigned ret = __activemask();
        if (ret != 0b10001000100010001000100010001000) {
            printf("Default thread %d failed.\n", tid);
        }
    }
}

__global__ void test_activemask_custom_four() {
    int tid = threadIdx.x;
    if (tid % 4 == 0) {
        unsigned ret = activemask();
        if (ret != 0b10001000100010001000100010001000) {
            printf("Custom thread %d failed.\n", tid);
        }
    }
}

void benchmark_activemask() {
    double total_default_time = 0;
    double total_custom_time = 0;

    for (int i = 0; i < NUM_ROUNDS; i++) {
        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_ballot_sync_default_two<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_ballot_sync_custom_two<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;

        gettimeofday(&start_default, 0);
        test_ballot_sync_default_four<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        gettimeofday(&start_custom, 0);
        test_ballot_sync_custom_four<<< 1, 32 >>>();
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __activemask() = %f ms\n", avg_default);
    printf("  Average time to run custom activemask() = %f ms\n\n", avg_custom);
}

int main() {
    printf("CUDA Warp Vote Benchmarks\n\n");
    benchmark_all_sync();
    benchmark_any_sync();
    benchmark_ballot_sync();
    benchmark_activemask();
    return 0;
}