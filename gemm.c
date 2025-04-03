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

#define MEM_ALIGN 64

/**
 * Naive implementation of SGEMM that will act as ground truth.
 */
void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
  constant_init(C, M * N, 0.0f);
#pragma omp parallel for
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

/**
 * Copies elements column-wise from A up to m. Pads m->MR with zeros.
 */
void maybe_pad_blockA(const float *A, float *padded_blockA, int m, int M, int K) {
  for (int p = 0; p < K; p++) {
    for (int i = 0; i < MR; i++) {
      *padded_blockA++ = (i < m) ? A[p * M + i] : 0.0f;
    }
  }
}

/**
 * Copies elements row-wise from B up to n. Pads n->NR with zeros.
 */
void maybe_pad_blockB(const float *B, float *padded_blockB, int n, int N, int K) {
  for (int p = 0; p < K; p++) {
    for (int j = 0; j < NR; j++) {
      *padded_blockB++ = (j < n) ? B[j * K + p] : 0.0f;
    }
  }
}

void kernel_16x6(const float *padded_blockA, const float *padded_blockB, float *C, int m, int n, int M, int K) {
  __m256 C_buffer[NR][MR / 8];
  __m256 a0_vec, a1_vec;
  __m256 b_vec;
  __m256i mask[2];

  // Load
  if (m < MR) {
    /**
     * Conditional load/store masks.
     * When rows of A or C are not multiples of MR, edge blocks may have <MR rows.
     * Lets assume m=9. In this case, we do not want to load more than 9 column elements
     * from A (or store more than 9 column elements to C).
     * We need a mask that says (1=yes) and (0=no) for each position in range(0, MR)
     * For this, we create two 8-size masks of 32-bit unsigned integers where MSB
     * dictates whether or not an element is to be loaded.
     * In case of m=9, we take a precomputed set of 16 indices: [0, 1, 2, 3, ..., 15]
     * and do a broadcasted compare.
     * cmp = [9, 9, ..., 9] > [0, 1, 2, 3, ..., 15]
     * cmp = [1, 1, (until 8)..., 0]
     * We move this result to the MSB bit.
     * mask = [1<<31, 1<<31, (until 8)..., 0<<31]
     */
    __m256i cmp0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(m), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    mask[0] = _mm256_slli_epi32(cmp0, 31);
    __m256i cmp1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(m), _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15));
    mask[1] = _mm256_slli_epi32(cmp1, 31);

    for (int j = 0; j < n; j++) {
      C_buffer[j][0] = _mm256_maskload_ps(&C[j * M], mask[0]);
      C_buffer[j][1] = _mm256_maskload_ps(&C[j * M + 8], mask[1]);
    }
  } else {
    for (int j = 0; j < n; j++) {
      C_buffer[j][0] = _mm256_loadu_ps(&C[j * M]);
      C_buffer[j][1] = _mm256_loadu_ps(&C[j * M + 8]);
    }
  }

  // Compute
  for (int p = 0; p < K; p++) {
    a0_vec = _mm256_loadu_ps(padded_blockA);
    a1_vec = _mm256_loadu_ps(padded_blockA + 8);
    // Since blocks are padded, we can be sure of NR iterations.
    for (int j = 0; j < NR; j++) {
      b_vec = _mm256_broadcast_ss(padded_blockB + j);
      C_buffer[j][0] = _mm256_fmadd_ps(a0_vec, b_vec, C_buffer[j][0]);
      C_buffer[j][1] = _mm256_fmadd_ps(a1_vec, b_vec, C_buffer[j][1]);
    }
    padded_blockA += MR;
    padded_blockB += NR;
  }

  // Store
  if (m < MR) {
    for (int j = 0; j < n; j++) {
      _mm256_maskstore_ps(&C[j * M], mask[0], C_buffer[j][0]);
      _mm256_maskstore_ps(&C[j * M + 8], mask[1], C_buffer[j][1]);
    }
  } else {
    for (int j = 0; j < n; j++) {
      _mm256_storeu_ps(&C[j * M], C_buffer[j][0]);
      _mm256_storeu_ps(&C[j * M + 8], C_buffer[j][1]);
    }
  }
}

/**
 * AVX2 implementation using 16x6 micro-kernel.
 */
void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
  float *padded_blockA = (float *)_mm_malloc(sizeof(float) * MR * K, MEM_ALIGN);
  float *padded_blockB = (float *)_mm_malloc(sizeof(float) * K * NR, MEM_ALIGN);

  for (int i = 0; i < M; i += MR) {
    const int m = min(MR, M - i);
    maybe_pad_blockA(&A[i], padded_blockA, m, M, K);
    for (int j = 0; j < N; j += NR) {
      const int n = min(NR, N - j);
      maybe_pad_blockB(&B[j * K], padded_blockB, n, N, K);
      kernel_16x6(padded_blockA, padded_blockB, &C[j * M + i], m, n, M, K);
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