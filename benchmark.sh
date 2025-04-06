clang -O2 -march=native -fopenmp gemm.c -o ./gemm
for size in $(seq 480 32 2048); do
    ./gemm $size
done