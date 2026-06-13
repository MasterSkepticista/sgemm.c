DEBUG_FLAG=""
if [ "$DEBUG" = "1" ]; then
    DEBUG_FLAG="-DDEBUG"
fi

gcc -O3 -funroll-loops \
    $DEBUG_FLAG \
    -march=native \
    gemm.c variants/*.c -lopenblas -lm -o ./gemm