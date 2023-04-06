CC=gcc
CFLAGS=-g -O2 -lze_loader -Wall
BINARIES=vote all_sync.ar any_sync.ar ballot_sync.ar activemask.ar
all: $(BINARIES)

run: $(addprefix run_, $(BINARIES))

clean:
        rm -f -- $(BINARIES) vote

all_sync.ar: opencl/all_sync.cl
        ocloc compile -file $^ -device \*

any_sync.ar: opencl/any_sync.cl
        ocloc compile -file $^ -device \*

ballot_sync.ar: opencl/ballot_sync.cl
        ocloc compile -file $^ -device \*

activemask.ar: opencl/activemask.cl
        ocloc compile -file $^ -device \*

vote: warp_vote.c ze_utils.c
                $(CC) $^ $(CFLAGS) -o $@

run_vote: vote
                ./$< 10 1