/**
 * Optimizing SGEMM in C.
 * clang-18 -O2 -march=native gemm.c -o ./gemm && ./gemm 1024
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
  #pragma omp parallel for collapse(2)
  for (int j = 0; j < N; j++) {
    for (int k = 0; k < K; k++) {
      for (int i = 0; i < M; i++) {
        C[j * M + i] += A[k * M + i] * B[j * K + k];
      }
    }
  }
}

#define MR 16
#define NR 6

void kernel_16x6(const float *A, const float *B, float *C, int M, int K) {
  __m256 C_buffer[6][2];
  __m256 a0_vec, a1_vec;
  __m256 b_vec;
  // Load
  for (int j = 0; j < 6; j++) {
    C_buffer[j][0] = _mm256_load_ps(&C[j * M]);
    C_buffer[j][1] = _mm256_load_ps(&C[j * M + 8]);
  }

  // Compute
  for (int p = 0; p < K; p++) {
    a0_vec = _mm256_load_ps(&A[p * M]);
    a1_vec = _mm256_load_ps(&A[p * M + 8]);
    for (int j = 0; j < 6; j++) {
      b_vec = _mm256_broadcast_ss(&B[j * K + p]);
      C_buffer[j][0] = _mm256_fmadd_ps(a0_vec, b_vec, C_buffer[j][0]);
      C_buffer[j][1] = _mm256_fmadd_ps(a1_vec, b_vec, C_buffer[j][1]);
    }
  }

  // Store
  for (int j = 0; j < 6; j++) {
    _mm256_store_ps(&C[j * M], C_buffer[j][0]);
    _mm256_store_ps(&C[j * M + 8], C_buffer[j][1]);
  }
}

/**
 * AVX2 implementation using 16x6 micro-kernel.
 */
void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j+=NR) {
    for (int i = 0; i < M; i+=MR) {
      kernel_16x6(&A[i], &B[j * K], &C[j * M + i], M, K);
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
  if (argc > 1) {
    int size = atoi(argv[1]);
    M = N = K = size;
  } else {
    printf("Usage: %s <size>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
#endif

  printf("Problem size M=%d, K=%d, N=%d\n", M, K, N);
  float *A = (float *)_mm_malloc(sizeof(float) * M * K, 64);
  float *B = (float *)_mm_malloc(sizeof(float) * K * N, 64);
  float *C = (float *)_mm_malloc(sizeof(float) * M * N, 64);
  float *val = (float *)_mm_malloc(sizeof(float) * M * N, 64);

  rand_init(A, M * K);
  rand_init(B, K * N);
  constant_init(C, M * N, 0.0f);
  constant_init(val, M * N, 0.0f);

  // Ground truth.
  gemm_naive(A, B, C, M, N, K);
  printf("Naive SGEMM done.\n");

#ifdef DEBUG
  print_matrix(A, M, K);
  print_matrix(B, K, N);
  print_matrix(C, M, N);
#endif

  // Benchmark
  int repeats = 4;
  for (int i = 0; i < repeats; i++) {
    constant_init(val, M * N, 0.0f);
    double start = tick();
    gemm(A, B, val, M, N, K);
    double stop = tick();
    double elapsed_time = (stop - start);
    printf("-> GFLOP/s: %.2f (%.2f ms)\n", (2.0 * K * M * N * 1e-6f) / elapsed_time, elapsed_time);
  }

  allclose(val, C, N * N, 1e-3f);
  printf("Match.\n");

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(val);
  return 0;
}