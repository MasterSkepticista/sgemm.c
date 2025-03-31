/**
 * Optimizing SGEMM in C.
 * gcc -O3 -march=native gemm.c -o ./gemm && ./gemm 1024
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
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

#define BLOCK_Y 256
#define BLOCK_X 128
#define BLOCK_K 32

void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
  constant_init(C, M * N, 0.0f);
  for (int i = 0; i < M; i += BLOCK_Y) {
    for (int j = 0; j < N; j += BLOCK_X) {
      float acc[BLOCK_Y][BLOCK_X] = {};
      for (int k = 0; k < K; k += BLOCK_K) {
        for (int iy = 0; iy < BLOCK_Y; iy++) {
          for (int ik = 0; ik < BLOCK_K; ik++) {
            for (int jx = 0; jx < BLOCK_X; jx++) {
              acc[iy][jx] += A[(i + iy) * K + (k + ik)] * B[(k + ik) * N + (j + jx)];
            }
          }
        }
      }

      for (int iy = 0; iy < BLOCK_Y; iy++) {
        for (int jx = 0; jx < BLOCK_X; jx++) {
          C[(i + iy) * N + (j + jx)] += acc[iy][jx];
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

  return 0;
}