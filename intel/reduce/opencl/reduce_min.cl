__kernel void reduce_min_sync(unsigned mask, unsigned value,
    __local unsigned* reduce_min_sync_shared_var_arr,
    __local unsigned* reduce_min_sync_updated,
    __local unsigned* n) {
    int tid = get_global_id(0); // threadIdx.x;
    unsigned ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        reduce_min_sync_shared_var_arr[tid] = value;
        reduce_min_sync_updated[tid] = true;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 32; i++) {
        if (reduce_min_sync_updated[i]
            && ret > reduce_min_sync_shared_var_arr[i]) {
            ret = reduce_min_sync_shared_var_arr[i];
            reduce_min_sync_updated[i] = false; // reset
        }
    }

    *n = ret;
}

__kernel void test_reduce_min_sync_custom(unsigned** test_arr, unsigned* res,
    __local unsigned* reduce_min_sync_shared_var_arr, __local unsigned* reduce_min_sync_updated) {
    int tid = get_global_id(0); // threadIdx.x;
    __local unsigned n;
    unsigned ret = reduce_min_sync(0xffffffff, (*test_arr)[tid],
        reduce_min_sync_shared_var_arr, reduce_min_sync_updated, &n);
    if (ret != *res)
        printf("Custom thread %d failed.\n", tid);
}