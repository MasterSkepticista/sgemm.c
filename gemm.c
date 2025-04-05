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
#define NR 48

void maybe_pad_blockA(const float *A, float *padded_blockA, int m, int M, int K) {
  memcpy(padded_blockA, A, sizeof(float) * m * K);                 // Copy valid rows
  memset(padded_blockA + m * K, 0, sizeof(float) * (MR - m) * K);  // Zero pad
}

void maybe_pad_blockB(const float *B, float *padded_blockB, int n, int N, int K) {
  for (int p = 0; p < K; p++) {
    memcpy(padded_blockB, &B[p * N], n * sizeof(float));     // Copy valid columns
    memset(padded_blockB + n, 0, sizeof(float) * (NR - n));  // Pad remaining columns
    padded_blockB += NR;
  }
}

/**
 * An MR x NR micro-kernel to compute a tile of C and update in-place in C.
 */
void kernel_6x16(const float *padded_blockA, const float *padded_blockB, float *C, int m, int n, int M, int N, int K) {
  __m512 a_vec;
  __m512 b0_vec, b1_vec, b2_vec;
  __m512 C_buffer[MR][NR / 16];
  __mmask16 masks[3];

  // Load.
  if (n < NR) {
    masks[0] = _cvtu32_mask16((1 << (n > 16 ? 16 : n)) - 1);
    masks[1] = _cvtu32_mask16((1 << ((n > 16 ? n - 16 : 0) > 16 ? 16 : (n > 16 ? n - 16 : 0))) - 1);
    masks[2] = _cvtu32_mask16((1 << ((n > 32 ? n - 32 : 0) > 16 ? 16 : (n > 32 ? n - 32 : 0))) - 1);
    for (int i = 0; i < m; i++) {
      C_buffer[i][0] = _mm512_maskz_loadu_ps(masks[0], &C[i * N]);
      C_buffer[i][1] = _mm512_maskz_loadu_ps(masks[1], &C[i * N + 16]);
      C_buffer[i][2] = _mm512_maskz_loadu_ps(masks[2], &C[i * N + 32]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      C_buffer[i][0] = _mm512_loadu_ps(&C[i * N]);
      C_buffer[i][1] = _mm512_loadu_ps(&C[i * N + 16]);
      C_buffer[i][2] = _mm512_loadu_ps(&C[i * N + 32]);
    }
  }

  // Compute.
  for (int p = 0; p < K; p++) {
    b0_vec = _mm512_load_ps(&padded_blockB[p * NR]);
    b1_vec = _mm512_load_ps(&padded_blockB[p * NR + 16]);
    b2_vec = _mm512_load_ps(&padded_blockB[p * NR + 32]);

    a_vec = _mm512_set1_ps(padded_blockA[0 * K + p]);
    C_buffer[0][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[0][0]);
    C_buffer[0][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[0][1]);
    C_buffer[0][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[0][2]);

    a_vec = _mm512_set1_ps(padded_blockA[1 * K + p]);
    C_buffer[1][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[1][0]);
    C_buffer[1][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[1][1]);
    C_buffer[1][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[1][2]);

    a_vec = _mm512_set1_ps(padded_blockA[2 * K + p]);
    C_buffer[2][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[2][0]);
    C_buffer[2][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[2][1]);
    C_buffer[2][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[2][2]);

    a_vec = _mm512_set1_ps(padded_blockA[3 * K + p]);
    C_buffer[3][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[3][0]);
    C_buffer[3][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[3][1]);
    C_buffer[3][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[3][2]);

    a_vec = _mm512_set1_ps(padded_blockA[4 * K + p]);
    C_buffer[4][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[4][0]);
    C_buffer[4][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[4][1]);
    C_buffer[4][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[4][2]);

    a_vec = _mm512_set1_ps(padded_blockA[5 * K + p]);
    C_buffer[5][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[5][0]);
    C_buffer[5][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[5][1]);
    C_buffer[5][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[5][2]);
  }

  // Store.
  if (n < NR) {
    for (int i = 0; i < m; i++) {
      _mm512_mask_storeu_ps(&C[i * N], masks[0], C_buffer[i][0]);
      _mm512_mask_storeu_ps(&C[i * N + 16], masks[1], C_buffer[i][1]);
      _mm512_mask_storeu_ps(&C[i * N + 32], masks[2], C_buffer[i][2]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      _mm512_storeu_ps(&C[i * N], C_buffer[i][0]);
      _mm512_storeu_ps(&C[i * N + 16], C_buffer[i][1]);
      _mm512_storeu_ps(&C[i * N + 32], C_buffer[i][2]);
    }
  }
}

void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
  memset(C, 0, M * N * sizeof(float));
  float *padded_blockA = (float *)_mm_malloc(sizeof(float) * MR * K, MEM_ALIGN);
  float *padded_blockB = (float *)_mm_malloc(sizeof(float) * K * NR, MEM_ALIGN);

  for (int j = 0; j < N; j += NR) {
    const int n = min(NR, N - j);
    maybe_pad_blockB(&B[j], padded_blockB, n, N, K);
    for (int i = 0; i < M; i += MR) {
      const int m = min(MR, M - i);
      maybe_pad_blockA(&A[i * K], padded_blockA, m, M, K);
      kernel_6x16(padded_blockA, padded_blockB, &C[i * N + j], m, n, M, N, K);
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
  int M, N, K;
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

  // Initialize
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

  // Benchmark
  int repeats = 4;
  double total_gflops = 0.0;
  for (int i = 0; i < repeats; i++) {
    double start = tick();
    gemm(A, B, val, M, N, K);
    double stop = tick();
    allclose(val, C, M * N, 1e-3f);
    double elapsed_time = (stop - start);
    double gflops = (2.0 * K * M * N * 1e-6f) / elapsed_time;
    total_gflops += gflops;
  }
  double average_gflops = total_gflops / repeats;
  printf("[M = %4d, K = %4d, N = %4d] GFLOP/s: %.2f\n", M, K, N, average_gflops);

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(val);
  return 0;
}