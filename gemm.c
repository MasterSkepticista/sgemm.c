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

void pack_blockA(const float *A, float *blockA_packed, int m, int M, int K) {
  for (int k = 0; k < K; k++) {
    for (int i = 0; i < MR; i++) {
      *blockA_packed++ = (i < m) ? A[k * M + i] : 0.0f;
    }
  }
}

void pack_blockB(const float *B, float *blockB_packed, int n, int N, int K) {
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < NR; j++) {
      *blockB_packed++ = (j < n) ? B[j * K + k] : 0.0f;
    }
  }
}

void kernel_16x6(float *blockA_packed, float *blockB_packed, float *C, int m, int n, int M, int N, int K) {
  __m256 C_buffer[MR / 8][NR] = {};
  __m256 a0_packed;
  __m256 a1_packed;
  __m256 b_packed;
  __m256i masks[2];
  if (m != MR) {
    const unsigned int bit_mask = 65535;
    masks[0] = _mm256_setr_epi32(bit_mask << (m + 15), bit_mask << (m + 14), bit_mask << (m + 13), bit_mask << (m + 12),
                                 bit_mask << (m + 11), bit_mask << (m + 10), bit_mask << (m + 9), bit_mask << (m + 8));
    masks[1] = _mm256_setr_epi32(bit_mask << (m + 7), bit_mask << (m + 6), bit_mask << (m + 5), bit_mask << (m + 4),
                                 bit_mask << (m + 3), bit_mask << (m + 2), bit_mask << (m + 1), bit_mask << (m));
    for (int j = 0; j < n; j++) {
      C_buffer[0][j] = _mm256_maskload_ps(&C[j * M], masks[0]);
      C_buffer[1][j] = _mm256_maskload_ps(&C[j * M + 8], masks[1]);
    }
  } else {
    for (int j = 0; j < n; j++) {
      C_buffer[0][j] = _mm256_load_ps(&C[j * M]);
      C_buffer[1][j] = _mm256_load_ps(&C[j * M + 8]);
    }
  }
  // Compute
  for (int k = 0; k < K; k++) {
    a0_packed = _mm256_load_ps(blockA_packed);
    a1_packed = _mm256_load_ps(blockA_packed + 8);
    for (int j = 0; j < NR; j++) {
      b_packed = _mm256_set1_ps(blockB_packed[j]);
      C_buffer[0][j] = _mm256_fmadd_ps(a0_packed, b_packed, C_buffer[0][j]);
      C_buffer[1][j] = _mm256_fmadd_ps(a1_packed, b_packed, C_buffer[1][j]);
    }

    blockA_packed += MR;
    blockB_packed += NR;
  }

  // Store
  if (m != MR) {
    for (int j = 0; j < n; j++) {
      _mm256_maskstore_ps(&C[j * M], masks[0], C_buffer[0][j]);
      _mm256_maskstore_ps(&C[j * M + 8], masks[1], C_buffer[1][j]);
    }
  } else {
    for (int j = 0; j < n; j++) {
      _mm256_storeu_ps(&C[j * M], C_buffer[0][j]);
      _mm256_storeu_ps(&C[j * M + 8], C_buffer[1][j]);
    }
  }
}

void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
  float *blockA_packed = (float *)_mm_malloc(sizeof(float) * MR * K, 64);
  float *blockB_packed = (float *)_mm_malloc(sizeof(float) * K * NR, 64);

  for (int i = 0; i < M; i += MR) {
    int m = min(MR, M - i);
    pack_blockA(&A[i], blockA_packed, m, M, K);
    for (int j = 0; j < N; j += NR) {
      int n = min(NR, N - j);
      pack_blockB(&B[j * K], blockB_packed, n, N, K);
      kernel_16x6(blockA_packed, blockB_packed, &C[j * M + i], m, n, M, N, K);
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