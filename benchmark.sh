DEBUG_FLAG=""
if [ "$DEBUG" = "1" ]; then
    DEBUG_FLAG="-DDEBUG"
fi

clang -O2 \
    $DEBUG_FLAG \
    -march=native \
    -I venv/include -l:libmkl_rt.so.3 -lm \
    -L $(pwd)/venv/lib -Wl,-rpath,$(pwd)/venv/lib \
    gemm.c variants/*.c -o ./gemm

for size in $(seq 32 32 4096); do
    ./gemm $1 $size
done