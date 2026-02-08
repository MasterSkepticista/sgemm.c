## Optimizing SGEMM in C

A single C file attempt to beat Intel MKL for the single precision GEMM operation.

### Prerequisites

* Install Intel MKL headers and libraries in a python venv activated in the root of this project. We use this to compare roofline GFLOP/s and verify correctness.

  ```bash
  pip install mkl mkl-include
  ```

* Compile.
  ```bash
  clang -O2 \
    DEBUG=1 \
    -march=native \
    -I venv/include -l:libmkl_rt.so.2 -lm \
    -L $(pwd)/venv/lib -Wl,-rpath,$(pwd)/venv/lib \
    gemm.c -o ./gemm
  ```
  Run. Kernel number `0` refers to MKL reference sgemm implementation. This should give the peak GFLOP/s on your machine.
  ```bash
  ./gemm 0 1920
  ```
