#!/bin/bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
./build.sh

for size in $(seq 64 64 4096); do
    ./gemm $1 $size
done