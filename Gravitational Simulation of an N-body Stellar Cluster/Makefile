########################################
# Makefile for PX457 2025 Assignment 5 #
########################################

# C compiler and flags. This will need
# changing for MPI/CUDA/OpenMP etc.
CC=mpicc
CFLAGS=-O3 -fopenmp -DTIMING

# Command to use for linking and executable
LD=$(CC)
EXE=gravity
LDFLAGS=-lm -fopenmp

OBJECTS=mt19937ar.o data.o gravity.o

# Default build target
$(EXE) : $(OBJECTS)
        $(LD) $(CFLAGS) -o $(EXE) $(OBJECTS) $(LDFLAGS)

# Purge build files and executable
clean :
        rm -f *.o $(EXE)

# Rules for building object files
%.o: %.c
        $(CC) $(CFLAGS) -c -o $@ $<