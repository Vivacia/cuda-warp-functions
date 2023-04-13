__kernel unsigned ballot_sync(unsigned mask, int predicate, __local int ballot_sync_shared_var_arr[],
    __local int ballot_sync_updated[], __local unsigned* val) {
    int tid = get_global_id(0);
    int ret = 0;

    if (((0x1 << tid) & mask) != 0) {
        ballot_sync_shared_var_arr[tid] = predicate;
        ballot_sync_updated[tid] = true;
    }

    for (int i = 0; i < warpSize; i++) {
        if (((0x1 << tid) & mask) == 0)
            continue;
        while (((0x1 << tid) & mask) != 0
            && !ballot_sync_updated[i]);
        if (ballot_sync_shared_var_arr[i])
            ret |= (0x1 << i);
    }

    *val = ret;
}

__kernel void test_ballot_sync_custom_two(__local int ballot_sync_shared_var_arr[],
    __local int ballot_sync_updated[]) {
    int tid = get_global_id(0);
    __local unsigned ret;
    __ballot_sync(0xffffffff, tid % 2, ballot_sync_shared_var_arr,
    ballot_sync_updated, &ret);
    if (tid % 2 == 1 && (ret & (0x1 << tid)) == 0) {
        printf("Custom thread %d failed.\n", tid);
    }
}

__kernel void test_ballot_sync_custom_four(__local int ballot_sync_shared_var_arr[],
    __local int ballot_sync_updated[]) {
    int tid = get_global_id(0);
    __local unsigned ret;
    ballot_sync(0xffffffff, tid % 4, ballot_sync_shared_var_arr,
    ballot_sync_updated, &ret);
    if (tid % 4 == 1 && (ret & (0x1 << tid)) == 0) {
        printf("Custom thread %d failed.\n", tid);
    }
}