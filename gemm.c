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

#define MC 120
#define NC 64

void maybe_pad_blockA(const float *A, float *padded_blockA, int mc, int M, int K) {
  memcpy(padded_blockA, A, sizeof(float) * mc * K);                  // Copy valid rows
  memset(padded_blockA + mc * K, 0, sizeof(float) * (MC - mc) * K);  // Zero pad
}

void maybe_pad_blockB(const float *B, float *padded_blockB, int nc, int N, int K) {
  for (int p = 0; p < K; p++) {
    memcpy(padded_blockB, &B[p * N], nc * sizeof(float));
    memset(padded_blockB + nc, 0, sizeof(float) * (NC - nc));
    padded_blockB += NC;
  }
}

/**
 * An MR x NR micro-kernel to compute a tile of C and update in-place in C.
 */
void kernel_6x16(const float *padded_blockA, const float *padded_blockB, float *C, int m, int n, int M, int N, int K) {
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
      C_buffer[i][0] = _mm256_maskload_ps(&C[i * N], masks[0]);
      C_buffer[i][1] = _mm256_maskload_ps(&C[i * N + 8], masks[1]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      C_buffer[i][0] = _mm256_loadu_ps(&C[i * N]);
      C_buffer[i][1] = _mm256_loadu_ps(&C[i * N + 8]);
    }
  }

  // Compute.
  for (int p = 0; p < K; p++) {
    b0_vec = _mm256_load_ps(&padded_blockB[p * NC]);
    b1_vec = _mm256_load_ps(&padded_blockB[p * NC + 8]);
    a_vec = _mm256_broadcast_ss(&padded_blockA[0 * K + p]);
    C_buffer[0][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[0][0]);
    C_buffer[0][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[0][1]);

    a_vec = _mm256_broadcast_ss(&padded_blockA[1 * K + p]);
    C_buffer[1][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[1][0]);
    C_buffer[1][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[1][1]);

    a_vec = _mm256_broadcast_ss(&padded_blockA[2 * K + p]);
    C_buffer[2][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[2][0]);
    C_buffer[2][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[2][1]);

    a_vec = _mm256_broadcast_ss(&padded_blockA[3 * K + p]);
    C_buffer[3][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[3][0]);
    C_buffer[3][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[3][1]);

    a_vec = _mm256_broadcast_ss(&padded_blockA[4 * K + p]);
    C_buffer[4][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[4][0]);
    C_buffer[4][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[4][1]);

    a_vec = _mm256_broadcast_ss(&padded_blockA[5 * K + p]);
    C_buffer[5][0] = _mm256_fmadd_ps(a_vec, b0_vec, C_buffer[5][0]);
    C_buffer[5][1] = _mm256_fmadd_ps(a_vec, b1_vec, C_buffer[5][1]);
  }

  // Store.
  if (n < NR) {
    for (int i = 0; i < m; i++) {
      _mm256_maskstore_ps(&C[i * N], masks[0], C_buffer[i][0]);
      _mm256_maskstore_ps(&C[i * N + 8], masks[1], C_buffer[i][1]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      _mm256_storeu_ps(&C[i * N], C_buffer[i][0]);
      _mm256_storeu_ps(&C[i * N + 8], C_buffer[i][1]);
    }
  }
}

void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
  memset(C, 0, M * N * sizeof(float));
  float *padded_blockA = (float *)_mm_malloc(sizeof(float) * MC * K, MEM_ALIGN);
  float *padded_blockB = (float *)_mm_malloc(sizeof(float) * K * NC, MEM_ALIGN);

  for (int j = 0; j < N; j += NC) {
    const int nc = min(NC, N - j);
    maybe_pad_blockB(&B[j], padded_blockB, nc, N, K);
    for (int i = 0; i < M; i += MC) {
      const int mc = min(MC, M - i);
      maybe_pad_blockA(&A[i * K], padded_blockA, mc, M, K);

      // Iterate over each MRxNR tile.
      for (int jr = 0; jr < nc; jr += NR) {
        for (int ir = 0; ir < mc; ir += MR) {
          const int nr = min(NR, nc - jr);
          const int mr = min(MR, mc - ir);
          kernel_6x16(&padded_blockA[ir * K], &padded_blockB[jr], &C[(i + ir) * N + (j + jr)], mr, nr, M, N, K);
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
  int repeats = 8;
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