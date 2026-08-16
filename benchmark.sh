#!/bin/bash
set -e
make clean
make -j"$(nproc)"

# Sweep through favorable and unfavorable sizes.
# Favorable: multiples of lcm(6, 16, 8, 48) = 48
# Unfavorable: cache associtivity conflicts in powers of 2 and/or multiples of 3.
sizes=(
    48   64   96   128  144  192  240  256
    288  336  384  432  480  512  528  624
    720  768  816  912  1008 1024 1056 1104
    1296 1488 1536 1584 1824 2016 2048 2064
    2160 2592 3024 3072 3120 3648 4080 4096
)

# Loop through each item
for size in "${sizes[@]}"; do
    OMP_NUM_THREADS=1 ./gemm $1 $size
done
