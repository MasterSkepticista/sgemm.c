## Optimizing SGEMM in C

An attempt to beat Intel-MKL/openBLAS for the single precision GEMM operation.

### Prerequisites

* Install OpenBLAS
  ```bash
  sudo apt install libopenblas-dev
  ```

* Compile and run.
  ```bash
  DEBUG=1 make -j4 && ./gemm <kernel_num> <size>
  ```
  Kernel number `0` is the reference sgemm implementation (openBLAS). This should give the peak GFLOP/s on your machine.

  Kernel `5` is compiled with AVX-512F instructions. Running it on a CPU without AVX-512F will terminate with an illegal-instruction signal; kernels `0`–`4` do not require AVX-512F.

* Sweep benchmark.
  ```bash
  DEBUG=1 ./benchmark.sh <kernel_num>
  ```

### Kernels
* `0`: Reference OpenBLAS implementation.
* `1`: Simple triple-for-loop with reordering.
* `2`: Same as kernel `1`, but with cache-blocking for consistent performance across all sizes.
* `3`: Reformulation of Matrix-Multiplication as a tiled outer product. More FLOP/s per byte moved.
* `4`: Same as kernel `3`, but with cache-blocking. This kernel should come close to AVX-256 performance limit on your CPU. Calculate the peak manually. E.g. 2.5GHz * 32 FLOPs/cycle = 80GFLOP/s
* `5`: Similar design to kernel `4`, but uses 512-bit wide AVX intrinsics, and requires a different tuning of constants. This kernel should come close to AVX-512 performance limit on your CPU (assume OpenBLAS to be the standard). In contrast, AVX-512 can do 64 FLOPs/cycle.
