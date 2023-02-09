#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>

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
 * __reduce_add_sync, __reduce_min_sync, __reduce_max_sync,
 * __reduce_and_sync, __reduce_or_sync, __reduce_xor_sync,
 * f(unsigned mask, unsigned/int value)
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARP REDUCE FUNCTION IMPLEMENTATIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ unsigned reduce_add_sync_shared_var_arr[32];
__device__ bool reduce_add_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ unsigned reduce_add_sync(unsigned mask, unsigned value) {
    int tid = threadIdx.x;
    unsigned ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        reduce_add_sync_shared_var_arr[tid] = value;
        reduce_add_sync_updated[tid] = true;
    }

    __syncthreads();

    for (int i = 0; i < warpSize; i++) {
        if (reduce_add_sync_updated[i]) {
            ret += reduce_add_sync_shared_var_arr[i];
            reduce_add_sync_updated[i] = false; // reset
        }
    }

    return ret;
}

__device__ unsigned reduce_min_sync_shared_var_arr[32];
__device__ bool reduce_min_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ unsigned reduce_min_sync(unsigned mask, unsigned value) {
    int tid = threadIdx.x;
    unsigned ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        reduce_min_sync_shared_var_arr[tid] = value;
        reduce_min_sync_updated[tid] = true;
    }

    __syncthreads();

    for (int i = 0; i < warpSize; i++) {
        if (reduce_min_sync_updated[i]
            && ret > reduce_min_sync_shared_var_arr[i]) {
            ret = reduce_min_sync_shared_var_arr[i];
            reduce_min_sync_updated[i] = false; // reset
        }
    }

    return ret;
}

__device__ unsigned reduce_max_sync_shared_var_arr[32];
__device__ bool reduce_max_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ unsigned reduce_max_sync(unsigned mask, unsigned value) {
    int tid = threadIdx.x;
    unsigned ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        reduce_max_sync_shared_var_arr[tid] = value;
        reduce_max_sync_updated[tid] = true;
    }

    __syncthreads();

    for (int i = 0; i < warpSize; i++) {
        if (reduce_max_sync_updated[i]
            && ret < reduce_max_sync_shared_var_arr[i]) {
            ret = reduce_max_sync_shared_var_arr[i];
            reduce_max_sync_updated[i] = false; // reset
        }
    }

    return ret;
}

__device__ unsigned reduce_and_sync_shared_var_arr[32];
__device__ bool reduce_and_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ unsigned reduce_and_sync(unsigned mask, unsigned value) {
    int tid = threadIdx.x;
    unsigned ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        reduce_and_sync_shared_var_arr[tid] = value;
        reduce_and_sync_updated[tid] = true;
    }

    __syncthreads();

    for (int i = 0; i < warpSize; i++) {
        if (reduce_and_sync_updated[i]) {
            ret &= reduce_and_sync_shared_var_arr[i];
            reduce_and_sync_updated[i] = false; // reset
        }
    }

    return ret;
}

__device__ unsigned reduce_or_sync_shared_var_arr[32];
__device__ bool reduce_or_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ unsigned reduce_or_sync(unsigned mask, unsigned value) {
    int tid = threadIdx.x;
    unsigned ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        reduce_or_sync_shared_var_arr[tid] = value;
        reduce_or_sync_updated[tid] = true;
    }

    __syncthreads();

    for (int i = 0; i < warpSize; i++) {
        if (reduce_or_sync_updated[i]) {
            ret |= reduce_or_sync_shared_var_arr[i];
            reduce_or_sync_updated[i] = false; // reset
        }
    }

    return ret;
}

__device__ unsigned reduce_xor_sync_shared_var_arr[32];
__device__ bool reduce_xor_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

__device__ unsigned reduce_xor_sync(unsigned mask, unsigned value) {
    int tid = threadIdx.x;
    unsigned ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        reduce_xor_sync_shared_var_arr[tid] = value;
        reduce_xor_sync_updated[tid] = true;
    }

    __syncthreads();

    for (int i = 0; i < warpSize; i++) {
        if (reduce_xor_sync_updated[i]){
            ret ^= reduce_xor_sync_shared_var_arr[i];
            reduce_xor_sync_updated[i] = false; // reset
        }
    }

    return ret;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARP REDUCE FUNCTION BENCHMARKS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

// For benchmarking
const int NUM_ROUNDS = 10;

// REDUCE_ADD_SYNC /////////////////////////////////////////////////////////////////////////////////////////
__global__ void test_reduce_add_sync_default(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = __reduce_add_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Default thread %d failed.\n", threadIdx.x);
}

__global__ void test_reduce_add_sync_custom(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = reduce_add_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Custom thread %d failed.\n", threadIdx.x);
}

void benchmark_reduce_add_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;
    unsigned test_arr[32];
    unsigned *_test_arr[32];
    unsigned res, *_res;
    srand(time(0));

    for (int i = 0; i < NUM_ROUNDS; i++) {
        res = 0;
        for (int ii = 0; ii < 32; ii++) {
            test_arr[ii] = rand() % 100;
            res += test_arr[ii];
        }

        checkCudaErrors(cudaMalloc((void**) &_test_arr,
        32 * sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_test_arr, &test_arr,
            32 * sizeof(size_t), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**) &_res,
        sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_res, &res,
            sizeof(size_t), cudaMemcpyHostToDevice));

        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_reduce_add_sync_default<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_reduce_add_sync_custom<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __reduce_add_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom reduce_add_sync() = %f ms\n\n", avg_custom);
}


// REDUCE_MIN_SYNC /////////////////////////////////////////////////////////////////////////////////////////
__global__ void test_reduce_min_sync_default(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = __reduce_min_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Default thread %d failed.\n", threadIdx.x);
}

__global__ void test_reduce_min_sync_custom(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = reduce_min_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Custom thread %d failed.\n", threadIdx.x);
}

void benchmark_reduce_min_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;
    unsigned test_arr[32];
    unsigned *_test_arr[32];
    unsigned res, *_res;
    srand(time(0));

    for (int i = 0; i < NUM_ROUNDS; i++) {
        test_arr[0] = rand() % 1000;
        res = test_arr[0] ;
        for (int ii = 1; ii < 32; ii++) {
            test_arr[ii] = rand() % 1000;
            if (test_arr[ii] < res) {
                res = test_arr[ii];
            }
        }

        checkCudaErrors(cudaMalloc((void**) &_test_arr,
        32 * sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_test_arr, &test_arr,
            32 * sizeof(size_t), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**) &_res,
        sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_res, &res,
            sizeof(size_t), cudaMemcpyHostToDevice));

        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_reduce_min_sync_default<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_reduce_min_sync_custom<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __reduce_min_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom reduce_min_sync() = %f ms\n\n", avg_custom);
}


// REDUCE_MAX_SYNC /////////////////////////////////////////////////////////////////////////////////////////
__global__ void test_reduce_max_sync_default(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = __reduce_max_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Default thread %d failed.\n", threadIdx.x);
}

__global__ void test_reduce_max_sync_custom(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = reduce_max_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Custom thread %d failed.\n", threadIdx.x);
}

void benchmark_reduce_max_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;
    unsigned test_arr[32];
    unsigned *_test_arr[32];
    unsigned res, *_res;
    srand(time(0));

    for (int i = 0; i < NUM_ROUNDS; i++) {
        test_arr[0] = rand() % 1000;
        res = test_arr[0] ;
        for (int ii = 1; ii < 32; ii++) {
            test_arr[ii] = rand() % 1000;
            if (test_arr[ii] > res) {
                res = test_arr[ii];
            }
        }

        checkCudaErrors(cudaMalloc((void**) &_test_arr,
        32 * sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_test_arr, &test_arr,
            32 * sizeof(size_t), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**) &_res,
        sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_res, &res,
            sizeof(size_t), cudaMemcpyHostToDevice));

        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_reduce_max_sync_default<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_reduce_max_sync_custom<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __reduce_max_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom reduce_max_sync() = %f ms\n\n", avg_custom);
}


// REDUCE_AND_SYNC /////////////////////////////////////////////////////////////////////////////////////////
__global__ void test_reduce_and_sync_default(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = __reduce_and_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Default thread %d failed.\n", threadIdx.x);
}

__global__ void test_reduce_and_sync_custom(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = reduce_and_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Custom thread %d failed.\n", threadIdx.x);
}

void benchmark_reduce_and_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;
    unsigned test_arr[32];
    unsigned *_test_arr[32];
    unsigned res, *_res;
    srand(time(0));

    for (int i = 0; i < NUM_ROUNDS; i++) {
        res = ~0x0;
        for (int ii = 0; ii < 32; ii++) {
            test_arr[ii] = rand() % 1000;
            res &= test_arr[ii];
        }

        checkCudaErrors(cudaMalloc((void**) &_test_arr,
        32 * sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_test_arr, &test_arr,
            32 * sizeof(size_t), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**) &_res,
        sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_res, &res,
            sizeof(size_t), cudaMemcpyHostToDevice));

        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_reduce_and_sync_default<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_reduce_and_sync_custom<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __reduce_and_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom reduce_and_sync() = %f ms\n\n", avg_custom);
}


// REDUCE_OR_SYNC //////////////////////////////////////////////////////////////////////////////////////////
__global__ void test_reduce_or_sync_default(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = __reduce_or_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Default thread %d failed.\n", threadIdx.x);
}

__global__ void test_reduce_or_sync_custom(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = reduce_or_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Custom thread %d failed.\n", threadIdx.x);
}

void benchmark_reduce_or_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;
    unsigned test_arr[32];
    unsigned *_test_arr[32];
    unsigned res, *_res;
    srand(time(0));

    for (int i = 0; i < NUM_ROUNDS; i++) {
        res = 0;
        for (int ii = 0; ii < 32; ii++) {
            test_arr[ii] = rand() % 1000;
            res |= test_arr[ii];
        }

        checkCudaErrors(cudaMalloc((void**) &_test_arr,
        32 * sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_test_arr, &test_arr,
            32 * sizeof(size_t), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**) &_res,
        sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_res, &res,
            sizeof(size_t), cudaMemcpyHostToDevice));

        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_reduce_or_sync_default<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_reduce_or_sync_custom<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __reduce_or_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom reduce_or_sync() = %f ms\n\n", avg_custom);
}


// REDUCE_XOR_SYNC /////////////////////////////////////////////////////////////////////////////////////////
__global__ void test_reduce_xor_sync_default(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = __reduce_xor_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Default thread %d failed.\n", threadIdx.x);
}

__global__ void test_reduce_xor_sync_custom(unsigned** test_arr, unsigned* res) {
    int tid = threadIdx.x;
    unsigned ret = reduce_xor_sync(0xffffffff, (*test_arr)[tid]);
    if (ret != *res)
        printf("Custom thread %d failed.\n", threadIdx.x);
}

void benchmark_reduce_xor_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;
    unsigned test_arr[32];
    unsigned *_test_arr[32];
    unsigned res, *_res;
    srand(time(0));

    for (int i = 0; i < NUM_ROUNDS; i++) {
        test_arr[0] = rand() % 1000;
        res = test_arr[0];
        for (int ii = 1; ii < 32; ii++) {
            test_arr[ii] = rand() % 1000;
            res ^= test_arr[ii];
        }

        checkCudaErrors(cudaMalloc((void**) &_test_arr,
        32 * sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_test_arr, &test_arr,
            32 * sizeof(size_t), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void**) &_res,
        sizeof(unsigned)));
        checkCudaErrors(cudaMemcpy(_res, &res,
            sizeof(size_t), cudaMemcpyHostToDevice));

        struct timeval start_default, end_default;
        gettimeofday(&start_default, 0);
        test_reduce_xor_sync_default<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_default, 0);
        double duration_default = (1000000.0*(end_default.tv_sec-start_default.tv_sec)
            + end_default.tv_usec-start_default.tv_usec)/1000.0;
        total_default_time += duration_default;

        struct timeval start_custom, end_custom;
        gettimeofday(&start_custom, 0);
        test_reduce_xor_sync_custom<<< 1, 32 >>>(_test_arr, _res);
        checkCudaErrors(cudaDeviceSynchronize());
        gettimeofday(&end_custom, 0);
        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run CUDA __reduce_xor_sync() = %f ms\n", avg_default);
    printf("  Average time to run custom reduce_xor_sync() = %f ms\n\n", avg_custom);
}



int main() {
    printf("CUDA Warp Reduce Benchmarks\n\n");
    benchmark_reduce_add_sync();
    benchmark_reduce_min_sync();
    benchmark_reduce_max_sync();
    benchmark_reduce_and_sync();
    benchmark_reduce_or_sync();
    benchmark_reduce_xor_sync();
    return 0;
}