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

#define MC 480
#define NC 480
#define KC 480

void pad_blockA(const float *A, float *blockA, int mc, int kc, int ldA) {
  for (int ir = 0; ir < mc; ir += MR) {
    const int m = min(MR, mc - ir);
    for (int p = 0; p < kc; p++) {
      for (int i = 0; i < MR; i++) {
        blockA[ir * kc + p * MR + i] = (i < m) ? A[(ir + i) * ldA + p] : 0.0f;
      }
    }
  }
}

void pad_blockB(const float *B, float *blockB, int nc, int kc, int ldB) {
  for (int jr = 0; jr < nc; jr += NR) {
    const int n = min(NR, nc - jr);
    for (int p = 0; p < kc; p++) {
      for (int j = 0; j < NR; j++) {
        blockB[jr * kc + p * NR + j] = (j < n) ? B[p * ldB + (jr + j)] : 0.0f;
      }
    }
  }
}

/**
 * An MR x NR micro-kernel to compute a tile of C and update in-place in C.
 */
void kernel_6x16(const float *blockA, const float *blockB, float *C, int m, int n, int k, int ldC) {
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
      C_buffer[i][0] = _mm512_maskz_loadu_ps(masks[0], &C[i * ldC]);
      C_buffer[i][1] = _mm512_maskz_loadu_ps(masks[1], &C[i * ldC + 16]);
      C_buffer[i][2] = _mm512_maskz_loadu_ps(masks[2], &C[i * ldC + 32]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      C_buffer[i][0] = _mm512_loadu_ps(&C[i * ldC]);
      C_buffer[i][1] = _mm512_loadu_ps(&C[i * ldC + 16]);
      C_buffer[i][2] = _mm512_loadu_ps(&C[i * ldC + 32]);
    }
  }

  // Compute.
  for (int p = 0; p < k; p++) {
    b0_vec = _mm512_load_ps(blockB);
    b1_vec = _mm512_load_ps(blockB + 16);
    b2_vec = _mm512_load_ps(blockB + 32);

    a_vec = _mm512_set1_ps(*(blockA + 0));
    C_buffer[0][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[0][0]);
    C_buffer[0][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[0][1]);
    C_buffer[0][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[0][2]);

    a_vec = _mm512_set1_ps(*(blockA + 1));
    C_buffer[1][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[1][0]);
    C_buffer[1][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[1][1]);
    C_buffer[1][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[1][2]);

    a_vec = _mm512_set1_ps(*(blockA + 2));
    C_buffer[2][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[2][0]);
    C_buffer[2][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[2][1]);
    C_buffer[2][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[2][2]);

    a_vec = _mm512_set1_ps(*(blockA + 3));
    C_buffer[3][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[3][0]);
    C_buffer[3][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[3][1]);
    C_buffer[3][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[3][2]);

    a_vec = _mm512_set1_ps(*(blockA + 4));
    C_buffer[4][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[4][0]);
    C_buffer[4][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[4][1]);
    C_buffer[4][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[4][2]);

    a_vec = _mm512_set1_ps(*(blockA + 5));
    C_buffer[5][0] = _mm512_fmadd_ps(a_vec, b0_vec, C_buffer[5][0]);
    C_buffer[5][1] = _mm512_fmadd_ps(a_vec, b1_vec, C_buffer[5][1]);
    C_buffer[5][2] = _mm512_fmadd_ps(a_vec, b2_vec, C_buffer[5][2]);

    blockA += MR;
    blockB += NR;
  }

  // Store.
  if (n < NR) {
    for (int i = 0; i < m; i++) {
      _mm512_mask_storeu_ps(&C[i * ldC], masks[0], C_buffer[i][0]);
      _mm512_mask_storeu_ps(&C[i * ldC + 16], masks[1], C_buffer[i][1]);
      _mm512_mask_storeu_ps(&C[i * ldC + 32], masks[2], C_buffer[i][2]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      _mm512_storeu_ps(&C[i * ldC], C_buffer[i][0]);
      _mm512_storeu_ps(&C[i * ldC + 16], C_buffer[i][1]);
      _mm512_storeu_ps(&C[i * ldC + 32], C_buffer[i][2]);
    }
  }
}

void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
  memset(C, 0, M * N * sizeof(float));
  float *blockA = (float *)_mm_malloc(sizeof(float) * KC * MC, MEM_ALIGN);
  float *blockB = (float *)_mm_malloc(sizeof(float) * KC * NC, MEM_ALIGN);

  for (int i = 0; i < M; i += MC) {
    const int mc = min(MC, M - i);
    for (int p = 0; p < K; p += KC) {
      const int kc = min(KC, K - p);
      pad_blockA(&A[i * K + p], blockA, mc, kc, K);
      for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        pad_blockB(&B[p * N + j], blockB, nc, kc, N);

        // Iterate over each (MR, NR) tile
        for (int jr = 0; jr < nc; jr += NR) {
          for (int ir = 0; ir < mc; ir += MR) {
            const int nr = min(NR, nc - jr);
            const int mr = min(MR, mc - ir);
            kernel_6x16(&blockA[ir * kc], &blockB[jr * kc], &C[(i + ir) * N + (j + jr)], mr, nr, kc, N);
          }
        }
      }
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