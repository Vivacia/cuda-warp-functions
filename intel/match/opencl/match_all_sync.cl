__kernel void match_all_sync(unsigned mask, __local int* value, __local int *pred,
    __local int match_all_sync_shared_var_arr[], __local int match_all_sync_updated[],
    __local int *val) {
    int tid = get_global_id(0);
    int ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        match_all_sync_shared_var_arr[tid] = *value;
        match_all_sync_updated[tid] = true;
    }

    for (int i = 0; i < 32; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !match_all_sync_updated[i]);
        if (*value == match_all_sync_shared_var_arr[tid]) {
            ret |= (0x1 << i);
        } else {
            *pred = false;
            *val = 0;
            return;
        }
    }

    *pred = true;
    *val = ret;
}

__kernel void test_match_all_sync_custom_true(__local int *v, __local int match_all_sync_shared_var_arr[],
    __local int match_all_sync_updated[]) {
    __local int *pred;
    *pred = 0;
    __local int ret;
    match_all_sync(0xffffffff, v, pred, match_all_sync_shared_var_arr, match_all_sync_updated, &ret);
    if (ret != 0xffffffff || *pred != true)
        printf("Thread %zu failed.\n", get_global_id(0));
}

// __kernel void test_match_all_sync_custom_false(__local int val, __local int match_all_sync_shared_var_arr[],
//     __local int match_all_sync_updated[]) {
//     int pred[1] = {0};
//     atomic_store(&val, 1+val);
//     __local int ret;
//     match_all_sync(0xffffffff, val, pred, match_all_sync_shared_var_arr, match_all_sync_updated, &ret);
//     if (ret != 0 || *pred != false)
//         printf("Thread %zu failed.\n", get_global_id(0));
// }