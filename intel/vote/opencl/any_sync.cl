__kernel void any_sync(unsigned mask, int predicate, __local int any_sync_shared_var_arr[],
    __local int any_sync_updated[], __local unsigned* val) {
    int tid = get_global_id(0);
    int ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        any_sync_shared_var_arr[tid] = predicate;
        any_sync_updated[tid] = true;
    }

    for (int i = 0; i < 32; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !any_sync_updated[i]);
        ret |= any_sync_shared_var_arr[i];
        if (ret)
            break;
    }

    *val = ret;
}

__kernel void test_any_sync_custom_two(__local int any_sync_shared_var_arr[],
    __local int any_sync_updated[]) {
    int tid = get_global_id(0);
    __local unsigned ret;
    any_sync(0xffffffff, tid % 2, any_sync_shared_var_arr, any_sync_updated, &ret);
    if (ret == 0) {
        printf("Custom thread %d failed.\n", tid);
    }
}

__kernel void test_any_sync_custom_four(__local int any_sync_shared_var_arr[],
    __local int any_sync_updated[]) {
    int tid = get_global_id(0);
    __local unsigned ret;
    any_sync(0xffffffff, tid % 4, any_sync_shared_var_arr, any_sync_updated, &ret);
    if (ret == 0) {
        printf("Custom thread %d failed.\n", tid);
    }
}