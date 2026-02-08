export LD_LIBRARY_PATH=$(pwd)/venv/lib:$LD_LIBRARY_PATH
DEBUG_FLAG=""
if [ "$DEBUG" = "1" ]; then
    DEBUG_FLAG="-DDEBUG"
fi
clang -O2 \
    $DEBUG_FLAG \
    -march=native \
    -I venv/include -lmkl_rt -lm \
    -L $(pwd)/venv/lib \
    gemm.c -o ./gemm
for size in $(seq 256 32 4096); do
    ./gemm $1 $size
done