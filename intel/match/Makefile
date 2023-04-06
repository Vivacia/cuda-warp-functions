CC=gcc
CFLAGS=-g -O2 -lze_loader -Wall
BINARIES=match match_any_sync.ar match_all_sync.ar
all: $(BINARIES)

run: $(addprefix run_, $(BINARIES))

clean:
        rm -f -- $(BINARIES) match

match_any_sync.ar: opencl/match_any_sync.cl
        ocloc compile -file $^ -device \*

match_all_sync.ar: opencl/match_all_sync.cl
        ocloc compile -file $^ -device \*

match: warp_match.c ze_utils.c
                $(CC) $^ $(CFLAGS) -o $@

run_match: match
                ./$< 10 1