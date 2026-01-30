## Optimizing SGEMM in C

A single C file attempt to beat Intel MKL for the single precision GEMM operation.

### Prerequisites

* Install Intel MKL headers and libraries in a python venv activated in the root of this project. We use this to compare roofline GFLOP/s and verify correctness.

  ```bash
  pip install mkl mkl-devel mkl-include
  ```

* Make MKL paths visible to compiler.
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/venv/lib
  ```

* Compile and run. Kernel number `0` refers to MKL reference sgemm implementation. This should give the peak GFLOP/s on your machine.
  ```bash
  clang -DDEBUG -O2 \
    -march=native \
    -mprefer-vector-width=512 \
    -I venv/include -lmkl_rt \
    -L $(pwd)/venv/lib \
    gemm.c -o ./gemm && ./gemm 0 1920
  ```
