__kernel void shfl_down_sync(unsigned mask, unsigned var, unsigned int delta, int width,
    __local unsigned shfl_down_sync_shared_var_arr[], __local unsigned shfl_down_sync_updated[], __local unsigned* n) {
    int tid = get_global_id(0);

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
    *n = var;
}

__kernel void test_shfl_down_sync_custom(__local unsigned shfl_down_sync_shared_var_arr[], __local unsigned shfl_down_sync_updated[]) {
    int laneId = get_global_id(0) & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    unsigned value = 31 - laneId;
    __local unsigned n;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        shfl_down_sync(0xffffffff, value, i, 8, shfl_down_sync_shared_var_arr, shfl_down_sync_updated, &n);
        if ((laneId & 7) <= i)
            value += n;
    }

    // printf("Custom thread %d final value = %d\n", threadIdx.x, value);
}