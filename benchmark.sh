#!/bin/bash
./build.sh

for size in $(seq 64 64 4096); do
    ./gemm $1 $size
done