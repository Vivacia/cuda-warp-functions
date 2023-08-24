## cuda-warp-functions
This folder should be placed in the same directory as the folder containing level-zero code. https://github.com/oneapi-src/level-zero

run `make clean; make` in a sub-directory to build.
run `./<sub-directory> groupSize numGroup` in a sub-directory to run the program/tests.
For example, `./shfl 32 1`.

`shfl`, `reduce`, `match`, and `vote` are the subdirectories to run tests in. `include` contains common header files for the aforementioned subdirectories.
Every subdirectory contains the following:
- a warp .c file: this has the logic that runs on Level Zero GPUs to run the tests - it is the host file.
- `opencl` directory: this contains all the kernel files. All of the tests and warp function implementations are written in OpenCL.
- OpenCL files should automatically compile based on the Makefile. This can be modified to add more tests.

NOTE - activemask() in Level Zero currently does not work - there are two versions of the OpenCL snippet. The issue here is that the OpenCL kernel needs to wait for all threads to execute the activemask() function but there is no way to determine whether a thread has finished executing the function, or even entered it. With activemask(), not all threads exit the function normally. The original implementation depended on a timer that determined the active threads after 10000 cycles from the first thread entering activemask(). However, in OpenCL, there is no way to count the number of cycles that have elapsed. Another method could be to perform busy work to wait for the threads to get through the function. But this does not guarantee that activemask() will provide the correct results.

atomic_store() is not available on the JLSE device, test_match_any_sync_custom_unique() and test_match_all_sync_custom_false() do not work on JLSE.

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


