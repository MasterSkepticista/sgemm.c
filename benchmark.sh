#!/bin/bash
set -e
make clean
make -j"$(nproc)"

for size in $(seq 64 64 4096); do
    OMP_NUM_THREADS=1 ./gemm $1 $size
done
