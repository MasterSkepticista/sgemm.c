clang -O2 \
    -march=native \
    -mprefer-vector-width=512 \
    -I venv/include -lmkl_rt \
    -L $(pwd)/venv/lib \
    gemm.c -o ./gemm
for size in $(seq 256 32 4096); do
    ./gemm 0 $size
done