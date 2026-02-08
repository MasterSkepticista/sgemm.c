DEBUG_FLAG=""
if [ "$DEBUG" = "1" ]; then
    DEBUG_FLAG="-DDEBUG"
fi

clang -O2 \
    $DEBUG_FLAG \
    -march=native \
    -I venv/include -l:libmkl_rt.so.2 -lm \
    -L $(pwd)/venv/lib -Wl,-rpath,$(pwd)/venv/lib \
    gemm.c -o ./gemm

for size in $(seq 256 32 4096); do
    ./gemm $1 $size
done