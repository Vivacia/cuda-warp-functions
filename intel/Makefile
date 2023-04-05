CC=gcc
CFLAGS=-g -O2 -lze_loader -Wall
BINARIES=shflup
all: $(BINARIES)

run: $(addprefix run_, $(BINARIES))

clean:
	rm -f -- $(BINARIES) shflup

run_shfl_up: shflup
	    ./$<

# shfl_up.ar: shfl_up.cl
# 	ocloc compile -file $^ -device \*

shflup: warp_shuffle.c ze_utils.c
	        $(CC) $^ $(CFLAGS) -o $@

run_shflup: shflup
		./$< 10 1