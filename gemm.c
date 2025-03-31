/**
 * Optimizing SGEMM in C.
 * ./gemm.py && clang -O3 -march=native gemm.c -o ./gemm && ./gemm
 */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

/**
 * Naive implementation of SGEMM that will act as ground truth.
 */
void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
  constant_init(C, M * N, 0.0f);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

int main(int argc, char **argv) {
  /**
   * Xeon 6258R
   * 2 AVX-512 FMA units
   * = 2 * 16 * 2 = 64 FLOP/cycle
   * = 2.7 * 64 = 172.8 GFLOP/s at 2.7GHz
   */
  
  // initialize
  int M, N, K;
#ifdef DEBUG
  M = N = K = 4;
#else
  M = N = K = 512;
#endif

  printf("Problem size M=%d, K=%d, N=%d\n", M, K, N);
  float *A = (float *)aligned_alloc(64, sizeof(float) * M * K);
  float *B = (float *)aligned_alloc(64, sizeof(float) * K * N);
  float *C = (float *)aligned_alloc(64, sizeof(float) * M * N);
  float *val = (float *)aligned_alloc(64, sizeof(float) * M * N);

  rand_init(A, M * K);
  rand_init(B, K * N);
  constant_init(C, M * N, 0.0f);
  constant_init(val, M * N, 0.0f);

  // Ground truth.
  gemm_naive(A, B, C, M, N, K);

#ifdef DEBUG
  print_matrix(A, M, K);
  print_matrix(B, K, N);
  print_matrix(C, M, N);
#endif

  // Benchmark
  int repeats = 4;
  for (int i = 0; i < repeats; i++) {
    uint64_t start = tick();
    gemm_naive(A, B, val, M, N, K);
    uint64_t stop = tick();
    uint64_t elapsed_time = (stop - start);
    printf("GFLOP/s: %.2f (%zu ms)\n", (2.0 * K * M * N * 1e-6) / elapsed_time, elapsed_time);
  }

  allclose(val, C, N * N, 1e-3f);
  printf("Match.\n");

  return 0;
}