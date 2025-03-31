clang-18 -O3 -march=native gemm.c -o ./gemm
for size in $(seq 6 12); do
    ./gemm $((2**size))
done