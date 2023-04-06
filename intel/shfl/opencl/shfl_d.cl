__kernel void shfl_sync(unsigned mask, unsigned var, int srcLane, int width,
    __local unsigned shfl_sync_shared_var_arr[], __local unsigned shfl_sync_updated[], __local int* n) {
    int tid = get_global_id(0);

    shfl_sync_shared_var_arr[tid] = var;
    shfl_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        if (srcLane >= 0) {
            while(!shfl_sync_updated[srcLane]);
            var = shfl_sync_shared_var_arr[srcLane];
            shfl_sync_updated[srcLane] = false;
        }
    }
    *n = var;
}

__kernel void test_shfl_sync_custom(__local unsigned shfl_sync_shared_var_arr[], __local unsigned shfl_sync_updated[]) {
    int laneId = get_global_id(0) & 0x1f;
    __local int value;
    if (laneId == 0)        // Note unused variable for
        value = 10;        // all threads except lane 0
    // Synchronize all threads in warp, and get "value" from lane 0
    shfl_sync(0xffffffff, value, 0, 32, shfl_sync_shared_var_arr, shfl_sync_updated,
    &value);
    if (value != 10)
        printf("Thread %zu failed.\n", get_global_id(0));
}