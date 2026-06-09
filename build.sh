DEBUG_FLAG=""
if [ "$DEBUG" = "1" ]; then
    DEBUG_FLAG="-DDEBUG"
fi

clang -O2 \
    $DEBUG_FLAG \
    -march=native \
    -lopenblas -lm \
    gemm.c variants/*.c -o ./gemm