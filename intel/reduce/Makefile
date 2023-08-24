CC=gcc
CFLAGS=-g -O2 -lze_loader -Wall
BINARIES=reduce reduce_add.ar reduce_min.ar reduce_max.ar reduce_and.ar reduce_or.ar reduce_xor.ar
all: $(BINARIES)

run: $(addprefix run_, $(BINARIES))

clean:
        rm -f -- $(BINARIES) reduce

reduce_add.ar: opencl/reduce_add.cl
        ocloc compile -file $^ -device \*

reduce_min.ar: opencl/reduce_min.cl
        ocloc compile -file $^ -device \*

reduce_max.ar: opencl/reduce_max.cl
        ocloc compile -file $^ -device \*

reduce_and.ar: opencl/reduce_and.cl
        ocloc compile -file $^ -device \*

reduce_or.ar: opencl/reduce_or.cl
        ocloc compile -file $^ -device \*

reduce_xor.ar: opencl/reduce_xor.cl
        ocloc compile -file $^ -device \*

reduce: warp_reduce.c ze_utils.c
                $(CC) $^ $(CFLAGS) -o $@

run_reduce: reduce
                ./$< 10 1