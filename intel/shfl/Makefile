CC=gcc
CFLAGS=-g -O2 -lze_loader -Wall
BINARIES=shflup shfl_up.ar shfl_down.ar shfl_d.ar shfl_xor.ar
all: $(BINARIES)

run: $(addprefix run_, $(BINARIES))

clean:
        rm -f -- $(BINARIES) shflup

run_shfl_up: shflup
            ./$<

shfl_up.ar: opencl/shfl_up.cl
        ocloc compile -file $^ -device \*

shfl_down.ar: opencl/shfl_down.cl
        ocloc compile -file $^ -device \*

shfl_xor.ar: opencl/shfl_xor.cl
        ocloc compile -file $^ -device \*

shfl_d.ar: opencl/shfl_d.cl
        ocloc compile -file $^ -device \*

shflup: warp_shuffle.c ze_utils.c
                $(CC) $^ $(CFLAGS) -o $@

run_shflup: shflup
                ./$< 10 1