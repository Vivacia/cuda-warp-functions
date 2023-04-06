__kernel void shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width,
    __local unsigned shfl_xor_sync_shared_var_arr[], __local unsigned shfl_xor_sync_updated[], __local unsigned* n) {
    int tid = get_global_id(0);

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
    *n =  var;
}

__kernel void test_shfl_xor_sync_custom(__local unsigned shfl_xor_sync_shared_var_arr[32], __local unsigned shfl_xor_sync_updated[32]) {
    int laneId = get_global_id(0) & 0x1f;
    // Seed starting value as inverse lane ID
    __local unsigned value;
    unsigned val = 31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2) {
        shfl_xor_sync(0xffffffff, val, i, 32, shfl_xor_sync_shared_var_arr, shfl_xor_sync_updated, &value);
        val += value;
    }
    // printf("Custom thread %d final value = %d\n", threadIdx.x, value);
}