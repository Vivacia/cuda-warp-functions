CC=gcc
CFLAGS=-g -O2 -lze_loader -Wall
BINARIES=reduce reduce_add_sync.ar reduce_min_sync.ar reduce_max_sync.ar reduce_and_sync.ar reduce_or_sync.ar reduce_xor_sync.ar
all: $(BINARIES)

run: $(addprefix run_, $(BINARIES))

clean:
        rm -f -- $(BINARIES) reduce

reduce_add_sync.ar: opencl/reduce_add_sync.cl
        ocloc compile -file $^ -device \*

reduce_min_sync.ar: opencl/reduce_min_sync.cl
        ocloc compile -file $^ -device \*

reduce_max_sync.ar: opencl/reduce_max_sync.cl
        ocloc compile -file $^ -device \*

reduce_and_sync.ar: opencl/reduce_and_sync.cl
        ocloc compile -file $^ -device \*

reduce_or_sync.ar: opencl/reduce_or_sync.cl
        ocloc compile -file $^ -device \*

reduce_xor_sync.ar: opencl/reduce_xor_sync.cl
        ocloc compile -file $^ -device \*

reduce: warp_shuffle.c ze_utils.c
                $(CC) $^ $(CFLAGS) -o $@

run_reduce: reduce
                ./$< 10 1