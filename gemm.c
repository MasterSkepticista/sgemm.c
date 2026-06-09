/**
 * Optimizing SGEMM in C using AVX (Row-major layout).
 * Build with benchmark.sh, then run: ./gemm 0 1024
 */

#include <immintrin.h>
#include <stdlib.h>

#include "common.h"
#include "variants/variants.h"

#define MEM_ALIGN 64

void launch_kernel(int kernel_num, float* C, float* A, float* B, int M, int N, int K) {
  switch (kernel_num) {
    case 0:
      gemm_mkl(C, A, B, M, N, K);
      break;
    case 1:
      gemm_loop_reorder(C, A, B, M, N, K);
      break;
    case 2:
      gemm_cache_blocked(C, A, B, M, N, K);
      break;
    case 3:
      gemm_outer_product(C, A, B, M, N, K);
      break;
    case 4:
      gemm_outer_product_cache_blocking(C, A, B, M, N, K);
      break;
    case 5:
      gemm_outer_product_cache_blocking_512(C, A, B, M, N, K);
      break;
    default:
      printf("Invalid kernel number `%d`\n", kernel_num);
      exit(EXIT_FAILURE);
  }
}

static double benchmark_kernel(int kernel_num,
                               float* C,
                               float* A,
                               float* B,
                               int M,
                               int N,
                               int K,
                               int repeats) {
  launch_kernel(kernel_num, C, A, B, M, N, K); // Warmup
  double total_time = 0.0;
  for (int i = 0; i < repeats; i++) {
    double start = tick();
    launch_kernel(kernel_num, C, A, B, M, N, K);
    double stop = tick();
    total_time += stop - start;
  }

  double gflops = (2.0 * M * N * K) * 1e-9;
  return gflops / (total_time / repeats);
}

int main(int argc, char** argv) {
  int kernel_num, M, N, K;
  if (argc > 4) {
    kernel_num = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    N = atoi(argv[4]);
  } else if (argc > 2) {
    kernel_num = atoi(argv[1]);
    M = N = K = atoi(argv[2]);
  } else {
    printf("Usage: %s <kernel_num> <M> <N> <K>\n", argv[0]);
    printf("Usage with M=N=K: %s <kernel_num> <size> \n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Alloc
  float* A = (float*)_mm_malloc(sizeof(float) * M * K, MEM_ALIGN);
  float* B = (float*)_mm_malloc(sizeof(float) * K * N, MEM_ALIGN);
  float* C = (float*)_mm_malloc(sizeof(float) * M * N, MEM_ALIGN);
  float* C_val = (float*)_mm_malloc(sizeof(float) * M * N, MEM_ALIGN);

  // Initialize
  rand_init(A, M * K);
  rand_init(B, K * N);
  constant_init(C, M * N, 0.0f);
  constant_init(C_val, M * N, 0.0f);

  // Warmup run, generate ground truth data.
#ifdef DEBUG
  gemm_mkl(C, A, B, M, N, K);
  launch_kernel(kernel_num, C_val, A, B, M, N, K);
  allclose(C, C_val, M * N, 1e-3);
  printf("Results match, starting benchmark...\n");
#endif

  // Benchmark
  double work_gflops = (2.0 * M * N * K) * 1e-9;
  int repeats = (int)ceil(100.0 / work_gflops);

  constant_init(C, M * N, 0.0f);
  double mkl_gflops = benchmark_kernel(0, C, A, B, M, N, K, repeats);

  if (kernel_num == 0) {
    printf("[M = %4d, K = %4d, N = %4d] kernel 0 GFLOP/s: %.2f\n", M, K, N, mkl_gflops);
  } else {
    constant_init(C_val, M * N, 0.0f);
    double kernel_gflops = benchmark_kernel(kernel_num, C_val, A, B, M, N, K, repeats);
    printf("[M = %4d, K = %4d, N = %4d] kernel 0: %.2f GFLOP/s | kernel %d: %.2f GFLOP/s | factor: %.2f\n",
           M,
           K,
           N,
           mkl_gflops,
           kernel_num,
           kernel_gflops,
           kernel_gflops / mkl_gflops);
  }

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(C_val);
  return 0;
}