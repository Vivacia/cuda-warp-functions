__kernel void match_any_sync(unsigned mask, __local int *value, __local int *val
    __local int match_any_sync_shared_var_arr[], __local int match_any_sync_updated[],
    __local int *val) {
    int tid = get_global_id(0);
    int ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        match_any_sync_shared_var_arr[tid] = *value;
        match_any_sync_updated[tid] = true;
    }

    for (int i = 0; i < 32; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !match_any_sync_updated[i]);
        if (*value == match_any_sync_shared_var_arr[tid])
            ret |= (0x1 << i);
    }

    *val = ret;
}

__kernel void test_match_any_sync_custom_simple(__local int match_any_sync_shared_var_arr[],
    __local int match_any_sync_updated[]) {
    __local int ret;
    match_any_sync(0xffffffff, 0, match_any_sync_shared_var_arr, match_any_sync_updated, &ret);
    printf("Custom thread %zu final value = %d\n", get_global_id(0), ret);
}

__kernel void test_match_any_sync_custom_alternate(__local int match_any_sync_shared_var_arr[],
    __local int match_any_sync_updated[]) {
    int tid = get_global_id(0);
    __local int ret;
    match_any_sync(0xffffffff, tid % 3, match_any_sync_shared_var_arr, match_any_sync_updated, &ret);
    printf("Custom thread %zu final value = %d\n", tid, ret);
}

__kernel void test_match_any_sync_custom_unique(__local int val, __local int match_any_sync_shared_var_arr[],
    __local int match_any_sync_updated[]) {
    atomic_store(&val, 1+val);
    __local int ret;
    match_any_sync(0xffffffff, val, match_any_sync_shared_var_arr, match_any_sync_updated, &ret);
    printf("Custom thread %zu final value = %d\n", get_global_id(0), ret);
}