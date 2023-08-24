__kernel void activemask(__local clock_t* global_now, __local unsigned* ret) {
    int tid = get_global_id(0);
    atomic_and(&_activemask, tid << 0x1);

    // need to wait until other active threads modify the active mask
    clock_t start = clock();
    clock_t now;
    for (;;) {
        now = clock();
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= 10000) {
            break;
        }
    }
    // Stored "now" in global memory here to prevent the
    // compiler from optimizing away the entire loop.
    *global_now = now;

    *ret = _activemask;
}

__kernel void test_activemask_custom_two(__local clock_t global_now) {
    int tid = get_global_id(0);
    __local unsigned ret;
    if (tid % 2 == 0) {
        activemask(&global_now, &ret);
        if (ret != 0b10101010101010101010101010101010) {
            printf("Custom thread %d failed.\n", tid);
        }
    }
}

__kernel void test_activemask_custom_four(__local clock_t global_now) {
    int tid = get_global_id(0);
    __local unsigned ret;
    if (tid % 4 == 0) {
        activemask(&global_now, &ret);
        if (ret != 0b10001000100010001000100010001000) {
            printf("Custom thread %d failed.\n", tid);
        }
    }
}