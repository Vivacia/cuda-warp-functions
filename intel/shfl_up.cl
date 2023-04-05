/*
 * T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
 * returns the value of var held by the thread whose ID is given by srcLane.
 */
unsigned shfl_up_sync_shared_var_arr[32];
bool shfl_up_sync_updated[32] = {0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0};

unsigned shfl_up_sync(unsigned mask, unsigned var, unsigned int delta, int width) {
    int tid = get_global_id(0); // threadIdx.x;

    shfl_up_sync_shared_var_arr[tid] = var;
    shfl_up_sync_updated[tid] = true;

    if (((0x1 << tid) & mask) != 0) {
        int sub_id = floor((float) tid / (float) width); // subsection
        int sub_tid = tid % width;
        int srcLane = sub_tid - delta;
        if (srcLane >= 0) {
            while(!shfl_up_sync_updated[srcLane + (width * sub_id)]);
            var = shfl_up_sync_shared_var_arr[srcLane + (width * sub_id)];
            shfl_up_sync_updated[srcLane + (width * sub_id)] = false; // reset
        }
    }
    return var;
}

void test_shfl_up_sync_custom() {
    int laneId = /*threadIdx.x*/ get_global_id(0) & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    unsigned value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        unsigned n = shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }
}