clang -O2 -march=native gemm.c -o ./gemm
for size in $(seq 480 32 4096); do
    ./gemm $size
done