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
#define NR 16

#define MC 480
#define NC 128
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
 *
 * @param blockA: Ptr to block of size (MR, K).
 * @param blockB: Ptr to block of size (K, NR).
 * @param C: Ptr to C where (m, n) result values will be written.
 * @param m: Number of valid rows to write at given C ptr. m <= MR in all cases.
 * @param n: Number of valid cols to write at given C ptr. n <= NR in all cases.
 * @param k: Number of intermediate iterations to perform.
 * @param ldA: Leading dimension of blockA (or, number of columns in blockA).
 * @param ldB: Leading dimension of blockB (or, number of columns in blockB).
 * @param ldC: Leading dimension of C (or, number of columns in C).
 */
void kernel_6x16(const float *blockA, const float *blockB, float *C, int m, int n, int k, int ldC) {
  __m256 a_vec;
  __m256 b0_vec, b1_vec;
  __m256 C_buffer[MR][NR / 8];
  __m256i masks[2];

  // Load.
  if (n < NR) {
    const unsigned int bitmask = 65535;
    masks[0] = _mm256_setr_epi32(bitmask << (n + 15), bitmask << (n + 14), bitmask << (n + 13), bitmask << (n + 12),
                                 bitmask << (n + 11), bitmask << (n + 10), bitmask << (n + 9), bitmask << (n + 8));
    masks[1] = _mm256_setr_epi32(bitmask << (n + 7), bitmask << (n + 6), bitmask << (n + 5), bitmask << (n + 4),
                                 bitmask << (n + 3), bitmask << (n + 2), bitmask << (n + 1), bitmask << (n + 0));
    for (int i = 0; i < m; i++) {
      C_buffer[i][0] = _mm256_maskload_ps(&C[i * ldC], masks[0]);
      C_buffer[i][1] = _mm256_maskload_ps(&C[i * ldC + 8], masks[1]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      C_buffer[i][0] = _mm256_loadu_ps(&C[i * ldC]);
      C_buffer[i][1] = _mm256_loadu_ps(&C[i * ldC + 8]);
    }
  }

  // Compute partial gemm on entire padded (MR, K) @ (K, NR).
  for (int p = 0; p < k; p++) {
    b0_vec = _mm256_load_ps(blockB);
    b1_vec = _mm256_load_ps(blockB + 8);

    a_vec = _mm256_broadcast_ss(blockA + 0);
    C_buffer[0][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[0][0]);
    C_buffer[0][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[0][1]);

    a_vec = _mm256_broadcast_ss(blockA + 1);
    C_buffer[1][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[1][0]);
    C_buffer[1][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[1][1]);

    a_vec = _mm256_broadcast_ss(blockA + 2);
    C_buffer[2][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[2][0]);
    C_buffer[2][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[2][1]);

    a_vec = _mm256_broadcast_ss(blockA + 3);
    C_buffer[3][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[3][0]);
    C_buffer[3][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[3][1]);

    a_vec = _mm256_broadcast_ss(blockA + 4);
    C_buffer[4][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[4][0]);
    C_buffer[4][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[4][1]);

    a_vec = _mm256_broadcast_ss(blockA + 5);
    C_buffer[5][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[5][0]);
    C_buffer[5][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[5][1]);

    blockA += MR;
    blockB += NR;
  }

  // Store.
  if (n < NR) {
    for (int i = 0; i < m; i++) {
      _mm256_maskstore_ps(&C[i * ldC], masks[0], C_buffer[i][0]);
      _mm256_maskstore_ps(&C[i * ldC + 8], masks[1], C_buffer[i][1]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      _mm256_storeu_ps(&C[i * ldC], C_buffer[i][0]);
      _mm256_storeu_ps(&C[i * ldC + 8], C_buffer[i][1]);
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

        // Iterate over each MRxNR tile.
        for (int ir = 0; ir < mc; ir += MR) {
          for (int jr = 0; jr < nc; jr += NR) {
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

  // Warmup
  gemm(A, B, C, M, N, K);

  int repeats = 2;
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