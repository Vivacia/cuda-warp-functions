#include <cstdio>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <sys/time.h>
#include <level_zero/ze_api.h>

#include "./ze_utils.h"


/**
 * All of the threads exchange data simultaneously.
 * Mask used to determine which threads participate, usually -1.
 * Var is the data it is operating on/temporary store?
 * For xor, lane_mask ^ current_lane_ID/lane_index determines which lane to switch with.
 * Two threads switch their data based on the above. In shuffle up and down, delta = shift.
 * Indexing starts at 0 regardless of width (usually warp size).
 * var = variable to get the value from, which is different for each thread.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARP SHUFFLE FUNCTION IMPLEMENTATIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
 * returns the value of var held by the thread whose ID is given by srcLane.
 */
__shared unsigned shfl_up_sync_shared_var_arr[32];
__shared bool shfl_up_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

unsigned shfl_up_sync(unsigned mask, unsigned var, unsigned int delta, int width=32) {
    int tid = ze_get_local_id(0);

    shfl_up_sync_shared_var_arr[tid] = var;
    shfl_up_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        int sub_id = floor((float) tid / (float) width); // subsection
        int sub_tid = tid % width;
        int srcLane = sub_tid - delta;
        if (srcLane >= 0) {
            while(!shfl_up_sync_updated[srcLane + (width * sub_id)]);
            var = shfl_up_sync_shared_var_arr[srcLane + (width * sub_id)];
            shfl_up_sync_updated[srcLane + (width * sub_id)] = false; // reset
        }
    }
    return var;
}

__shared unsigned shfl_down_sync_shared_var_arr[32];
__shared bool shfl_down_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

unsigned shfl_down_sync(unsigned mask, unsigned var, unsigned int delta, int width=32) {
    int tid = ze_get_local_id(0);

    shfl_down_sync_shared_var_arr[tid] = var;
    shfl_down_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        int sub_id = floor((float) tid / (float) width); // subsection
        int sub_tid = tid % width;
        int srcLane = sub_tid + delta;
        if (srcLane < width) {
            while(!shfl_down_sync_updated[srcLane + (width * sub_id)]);
            var = shfl_down_sync_shared_var_arr[srcLane + (width * sub_id)];
            shfl_down_sync_updated[srcLane + (width * sub_id)] = false; // reset
        }
    }
    return var;
}

__shared unsigned shfl_sync_shared_var_arr[32];
__shared bool shfl_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

unsigned shfl_sync(unsigned mask, unsigned var, int srcLane, int width=32) {
    int tid = ze_get_local_id(0);

    shfl_sync_shared_var_arr[tid] = var;
    shfl_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        if (srcLane >= 0) {
            while(!shfl_sync_updated[srcLane]);
            var = shfl_sync_shared_var_arr[srcLane];
            shfl_sync_updated[srcLane] = false;
        }
    }
    return var;
}

__shared unsigned shfl_xor_sync_shared_var_arr[32];
__shared bool shfl_xor_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

unsigned shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width=32) {
    int tid = ze_get_local_id(0);

    shfl_xor_sync_shared_var_arr[tid] = var;
    shfl_xor_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        int sub_id = floor((float) tid / (float) width);
        int sub_tid = tid % width;
        int srcLane = sub_tid ^ laneMask;
        int src_sub_id = floor((float) srcLane / (float) width);
        if (src_sub_id <= sub_id) {
            while(!shfl_xor_sync_updated[srcLane + (width * src_sub_id)]);
            var = shfl_xor_sync_shared_var_arr[srcLane + (width * src_sub_id)];
            shfl_xor_sync_updated[srcLane + (width * src_sub_id)] = false; // reset
        }
    }
    return var;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// WARP SHUFFLE FUNCTION BENCHMARKS
///////////////////////////////////////////////////////////////////////////////////////////////////////////

// For benchmarking
const int NUM_ROUNDS = 1000;

// WARP SHFL_SYNC /////////////////////////////////////////////////////////////////////////////////////////
// from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#broadcast-of-a-single-value-across-a-warp
// Broadcast of a single value across a warp

__kernel void test_shfl_sync_custom(int arg) {
    int laneId = ze_get_local_id(0) & 0x1f;
    int value;
    if (laneId == 0)        // Note unused variable for
        value = arg;        // all threads except lane 0
    // Synchronize all threads in warp, and get "value" from lane 0
    value = shfl_sync(0xffffffff, value, 0);
    if (value != arg)
        printf("Thread %d failed.\n", ze_get_local_id(0));
}

void benchmark_shfl_sync() {
    double total_default_time = 0;
    double total_custom_time = 0;

    for (int i = 0; i < NUM_ROUNDS; i++) {
        struct timeval start_custom, end_custom;
        
        ze_command_list_handle_t cmdList;
        zeCommandListCreate(device, &cmdList, &commandListDesc);
        ze_event_handle_t event;
        zeCommandListAppendSignalEvent(cmdList, event);
        ze_kernel_handle_t kernelFunction;
        zeKernelCreate(module, "myKernelFunction", &kernelFunction);
        // Set kernel arguments
        int arg1 = 123;
        float arg2 = 3.14f;
        zeKernelSetArgumentValue(kernelFunction, 0, sizeof(arg1), &arg1);
        zeKernelSetArgumentValue(kernelFunction, 1, sizeof(arg2), &arg2);
        ze_group_count_t dispatchDimensions = { 32, 1, 1 };  // 32 work items in the x dimension
        zeCommandListAppendLaunchKernel(cmdList, kernelFunction, &dispatchDimensions, nullptr, 0, nullptr);
        zeCommandListClose(cmdList);
        gettimeofday(&start_custom, 0);
        zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, nullptr);
        zeEventHostSynchronize(event, UINT64_MAX);
        gettimeofday(&end_custom, 0);
        zeEventDestroy(event);
        zeCommandListDestroy(cmdList);

        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_default = total_default_time / NUM_ROUNDS;
    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run custom shfl_sync() = %f ms\n\n", avg_custom);
}


// WARP SHFL_UP_SYNC //////////////////////////////////////////////////////////////////////////////////
// from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#inclusive-plus-scan-across-sub-partitions-of-8-threads
// Inclusive plus-scan across sub-partitions of 8 threads

__shared unsigned custom_results_shfl_up[32];

__kernel void test_shfl_up_sync_custom() {
    int laneId = ze_get_local_id(0) & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    unsigned value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        unsigned n = shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }

    // printf("Custom thread %d final value = %d\n", ze_get_local_id(0), value);
}

void benchmark_shfl_up_sync() {
    double total_custom_time = 0;

    for (int i = 0; i < NUM_ROUNDS; i++) {
        struct timeval start_custom, end_custom;
        
        ze_command_list_handle_t cmdList;
        zeCommandListCreate(device, &cmdList, &commandListDesc);
        ze_event_handle_t event;
        zeCommandListAppendSignalEvent(cmdList, event);
        ze_kernel_handle_t kernelFunction;
        zeKernelCreate(module, "myKernelFunction", &kernelFunction);
        // Set kernel arguments
        int arg1 = 123;
        float arg2 = 3.14f;
        zeKernelSetArgumentValue(kernelFunction, 0, sizeof(arg1), &arg1);
        zeKernelSetArgumentValue(kernelFunction, 1, sizeof(arg2), &arg2);
        ze_group_count_t dispatchDimensions = { 32, 1, 1 };  // 32 work items in the x dimension
        zeCommandListAppendLaunchKernel(cmdList, kernelFunction, &dispatchDimensions, nullptr, 0, nullptr);
        zeCommandListClose(cmdList);
        gettimeofday(&start_custom, 0);
        zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, nullptr);
        zeEventHostSynchronize(event, UINT64_MAX);
        gettimeofday(&end_custom, 0);
        zeEventDestroy(event);
        zeCommandListDestroy(cmdList);

        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run custom shfl_up_sync() = %f ms\n\n", avg_custom);
}


// WARP SHFL_DOWN_SYNC ///////////////////////////////////////////////////////////////////////////////
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
// Warp reduce sum

__shared unsigned custom_results_shfl_down[32];

__kernel void test_shfl_down_sync_custom() {
    int laneId = ze_get_local_id(0) & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    unsigned value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        unsigned n = shfl_down_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) <= i)
            value += n;
    }
}

__shared unsigned test_array[32];

void benchmark_shfl_down_sync() {
    double total_custom_time = 0;

    for (int i = 0; i < NUM_ROUNDS; i++) {
        struct timeval start_custom, end_custom;

        ze_command_list_handle_t cmdList;
        zeCommandListCreate(device, &cmdList, &commandListDesc);
        ze_event_handle_t event;
        zeCommandListAppendSignalEvent(cmdList, event);
        ze_kernel_handle_t kernelFunction;
        zeKernelCreate(module, "myKernelFunction", &kernelFunction);
        // Set kernel arguments
        int arg1 = 123;
        float arg2 = 3.14f;
        zeKernelSetArgumentValue(kernelFunction, 0, sizeof(arg1), &arg1);
        zeKernelSetArgumentValue(kernelFunction, 1, sizeof(arg2), &arg2);
        ze_group_count_t dispatchDimensions = { 32, 1, 1 };  // 32 work items in the x dimension
        zeCommandListAppendLaunchKernel(cmdList, kernelFunction, &dispatchDimensions, nullptr, 0, nullptr);
        zeCommandListClose(cmdList);
        gettimeofday(&start_custom, 0);
        zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, nullptr);
        zeEventHostSynchronize(event, UINT64_MAX);
        gettimeofday(&end_custom, 0);
        zeEventDestroy(event);
        zeCommandListDestroy(cmdList);

        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run custom shfl_down_sync() = %f ms\n\n", avg_custom);
}


// WARP SHFL_XOR_SYNC //////////////////////////////////////////////////////////////////////////////
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#reduction-across-a-warp

__shared unsigned custom_results_shfl_xor[32];


// TODO: turn into spv code and load into level zero
__kernel void test_shfl_xor_sync_custom() {
    int laneId = ze_get_local_id(0) & 0x1f;
    // Seed starting value as inverse lane ID
    unsigned value = 31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += shfl_xor_sync(0xffffffff, value, i, 32);
    // printf("Custom thread %d final value = %d\n", ze_get_local_id(0), value);
}

void benchmark_shfl_xor_sync() {
    double total_custom_time = 0;

    for (int i = 0; i < NUM_ROUNDS; i++) {
        struct timeval start_custom, end_custom;
        
        ze_command_list_handle_t cmdList;
        zeCommandListCreate(device, &cmdList, &commandListDesc);
        ze_event_handle_t event;
        zeCommandListAppendSignalEvent(cmdList, event);
        ze_kernel_handle_t kernelFunction;
        zeKernelCreate(module, "myKernelFunction", &kernelFunction);
        // Set kernel arguments
        int arg1 = 123;
        float arg2 = 3.14f;
        zeKernelSetArgumentValue(kernelFunction, 0, sizeof(arg1), &arg1);
        zeKernelSetArgumentValue(kernelFunction, 1, sizeof(arg2), &arg2);
        ze_group_count_t dispatchDimensions = { 32, 1, 1 };  // 32 work items in the x dimension
        zeCommandListAppendLaunchKernel(cmdList, kernelFunction, &dispatchDimensions, nullptr, 0, nullptr);
        zeCommandListClose(cmdList);

        gettimeofday(&start_custom, 0);
        zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, nullptr);
        zeEventHostSynchronize(event, UINT64_MAX);
        gettimeofday(&end_custom, 0);

        zeEventDestroy(event);
        zeCommandListDestroy(cmdList);

        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run custom shfl_xor_sync() = %f ms\n\n", avg_custom);
}
