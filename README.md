## cuda-warp-functions
### warp shuffle functions
- unsigned shfl_up_sync(unsigned mask, unsigned var, unsigned int delta, int width=warpSize)
- unsigned shfl_down_sync(unsigned mask, unsigned var, unsigned int delta, int width=warpSize)
- unsigned shfl_sync(unsigned mask, unsigned var, int srcLane, int width=warpSize)
- unsigned shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width=warpSize)
### warp vote functions
- all_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask and return non-zero if and only if predicate evaluates to non-zero for all of them.
- any_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask and return non-zero if and only if predicate evaluates to non-zero for any of them.
- ballot_sync(unsigned mask, predicate): Evaluate predicate for all non-exited threads in mask and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active.
- activemask(): Return a mask with Nth bit set for all threads currently active in calling warp.
### warp match functions
- T match_any_sync(unsigned mask, T value): Broadcast-and-compare of a value across threads in a warp after synchronizing threads named in mask. Returns mask of threads that have same value of value in mask.
- T match_all_sync(unsigned mask, T value, int *pred): Returns mask if all threads in mask have the same value for value; otherwise 0 is returned. Predicate pred is set to true if all threads in mask have the same value of value; otherwise the predicate is set to false.
### warp reduce functions
- unsigned reduce_add_sync(unsigned mask, unsigned value)
- unsigned reduce_min_sync(unsigned mask, unsigned value)
- unsigned reduce_max_sync(unsigned mask, unsigned value)
- unsigned reduce_and_sync(unsigned mask, unsigned value)
- unsigned reduce_or_sync(unsigned mask, unsigned value)
- unsigned reduce_xor_sync(unsigned mask, unsigned value)


