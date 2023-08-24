__kernel void all_sync(unsigned mask, int predicate, __local int all_sync_shared_var_arr[],
    __local int all_sync_updated[], __local int* val) {
    int tid = get_global_id(0);
    int ret = 1;

    if (((0x1 << tid) & mask) != 0) {
        all_sync_shared_var_arr[tid] = predicate;
        all_sync_updated[tid] = true;
    }

    for (int i = 0; i < 32; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !all_sync_updated[i]);
        ret &= all_sync_shared_var_arr[i];
    }

    *val = ret;
}

__kernel void test_all_sync_custom_pass(__local int all_sync_shared_var_arr[],
    __local int all_sync_updated[]) {
    __local int ret;
    all_sync(0xffffffff, 1, all_sync_shared_var_arr, all_sync_updated, &ret);
    if (ret != true) {
        printf("Custom thread %zu failed.\n", get_global_id(0));
    }
}

__kernel void test_all_sync_custom_fail(__local int all_sync_shared_var_arr[],
    __local int all_sync_updated[]) {
    int tid = get_global_id(0);
    __local int ret;
    all_sync(0xffffffff, tid % 2, all_sync_shared_var_arr, all_sync_updated, &ret);
    if (ret != false) {
        printf("Custom thread %d failed.\n", tid);
    }
}