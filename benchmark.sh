#!/bin/bash
set -e

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
make clean && make all

for size in $(seq 64 64 4096); do
    ./gemm $1 $size
done
