/**
 * Optimizing SGEMM in C (Row-major layout).
 * clang -O2 -march=native gemm.c -o ./gemm && ./gemm 1024
 */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#define MEM_ALIGN 64

/**
 * Naive implementation of SGEMM that will act as ground truth.
 */
void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
  constant_init(C, M * N, 0.0f);
#pragma omp parallel for
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

#define MR 6
#define NR 16

/**
 * An MR x NR micro-kernel to compute a tile of C and update in-place in C.
 */
void kernel_6x16(const float *A, const float *B, float *C, int M, int N, int K) {
  __m256 a_vec;
  __m256 b0_vec, b1_vec;
  __m256 C_buffer[MR][NR/8];

  // Load.
  for (int i = 0; i < MR; i++) {
    C_buffer[i][0] = _mm256_loadu_ps(&C[i * N]);
    C_buffer[i][1] = _mm256_loadu_ps(&C[i * N + 8]);
  }

  // Compute.
  for (int p = 0; p < K; p++) {
    b0_vec = _mm256_load_ps(&B[p * N]);
    b1_vec = _mm256_load_ps(&B[p * N + 8]);
    for (int i = 0; i < MR; i++) {
      a_vec = _mm256_broadcast_ss(&A[i * K + p]);
      C_buffer[i][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[i][0]);
      C_buffer[i][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[i][1]);
    }
  }

  // Store.
  for (int i = 0; i < MR; i++) {
    _mm256_storeu_ps(&C[i * N], C_buffer[i][0]);
    _mm256_storeu_ps(&C[i * N + 8], C_buffer[i][1]);
  }
}

void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i+= MR) {
    for (int j = 0; j < N; j+= NR) {
      kernel_6x16(&A[i * K], &B[j], &C[i * N + j], M, N, K);
    }
  }
}

int main(int argc, char **argv) {
  /**
   * Xeon 6258R
   * 2 AVX-512 FMA units
   * = 2 * 16 * 2 = 64 FLOP/cycle
   * = 2.5 * 64 = 160 GFLOP/s at 2.5GHz
   * 
   * Equivalently, 80 GFLOP/s using AVX-256
   */

  // initialize
  int M, N, K;
#ifdef DEBUG
  M = N = K = 4;
#else
  if (argc > 3) {
    M = atoi(argv[1]);
    K = atoi(argv[2]);
    N = atoi(argv[3]);
  } else if (argc > 1) {
    M = K = N = atoi(argv[1]);
  } else {
    printf("Usage with custom sizes: %s <M> <K> <N>\n", argv[0]);
    printf("Usage with M=N=K: %s <size> \n", argv[0]);
    exit(EXIT_FAILURE);
  }
#endif

  printf("Problem size M=%d, K=%d, N=%d\n", M, K, N);
  float *A = (float *)_mm_malloc(sizeof(float) * M * K, MEM_ALIGN);
  float *B = (float *)_mm_malloc(sizeof(float) * K * N, MEM_ALIGN);
  float *C = (float *)_mm_malloc(sizeof(float) * M * N, MEM_ALIGN);
  float *val = (float *)_mm_malloc(sizeof(float) * M * N, MEM_ALIGN);

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

  allclose(val, C, M * N, 1e-3f);
  printf("Match.\n");

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(val);
  return 0;
}