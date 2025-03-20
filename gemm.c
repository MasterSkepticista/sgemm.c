/**
 * Optimizing SGEMM in C.
 * ./gemm.py && clang -O3 -ffast-math -march=native gemm.c -o ./gemm && ./gemm
 */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#ifdef OMP
#include <omp.h>
#endif

#define FAST

#define BLOCK_Y 8
#define BLOCK_X 2
// #define BLOCK 8

#ifdef FAST

void swizzle(const float *B, float *Bs, int inners, int cols) {
  if (inners % 8 != 0) {
    printf("Error: inners must be a multiple of 8\n");
    exit(1);
  }
  for (int k = 0; k < inners; k += 8) {
    for (int x = 0; x < cols; x++) {
      for (int i = 0; i < 8; i++) {
        Bs[(k / 8) * cols * 8 + (x * 8) + i] = B[(k + i) * cols + x];
      }
    }
  }
}

// Reduce 8 floats to a single float.
inline float _reduce_sum(__m256 v) {
  __m128 lo = _mm256_extractf128_ps(v, 0);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return lo[0];
}

void gemm(const float *A, const float *B, float *C, int rows, int inners, int cols) {
  __m256 *Cm = (__m256 *)C;

  for (int y = 0; y < rows; y += BLOCK_Y) {
    for (int x = 0; x < cols / 8; x+=BLOCK_X) {
      __m256 Cv[BLOCK_Y][BLOCK_X] = {};
      __m256 Av[BLOCK_Y] = {};

      for (int k = 0; k < inners; k++) {
        
        for (int iy = 0; iy < BLOCK_Y; iy++) {
          // Load A
          Av[iy] = _mm256_broadcast_ss(&A[(y + iy) * inners + k]);

          for (int ix = 0; ix < BLOCK_X; ix++) {
            // Load B
            __m256 Bv = _mm256_load_ps(&B[(x + ix) * inners * 8 + k * 8]);
            // FMA
            Cv[iy][ix] = _mm256_fmadd_ps(Av[iy], Bv, Cv[iy][ix]);
          }
        }
      }

      for (int iy = 0; iy < BLOCK_Y; iy++) {
        for (int ix = 0; ix < BLOCK_X; ix++) {
          Cm[(y + iy) * cols / 8 + (x + ix)] += Cv[iy][ix];
        }
      }
    }
  }
}

#else
void gemm(const float *A, const float *B, float *C, int rows, int inners, int cols) {
  for (int by = 0; by < rows; by += BLOCK_Y) {
    for (int bx = 0; bx < cols; bx += BLOCK_X) {
      // Within a block.
      float tmp[BLOCK_Y][BLOCK_X] = {};

      // Compute
      for (int k = 0; k < inners; k++) {
        for (int y = 0; y < BLOCK_Y; y++) {
          for (int x = 0; x < BLOCK_X; x++) {
            tmp[y][x] += A[(by + y) * inners + k] * B[(bx + x) * inners + k];
          }
        }
      }

      // Store
      for (int y = 0; y < BLOCK_Y; y++) {
        for (int x = 0; x < BLOCK_X; x++) {
          C[(by + y) * cols + (bx + x)] = tmp[y][x];
        }
      }
    }
  }
}
#endif

#ifdef DEBUG
#define N 4
#else
#define N 768
#endif

float A[N * N] __attribute__((aligned(32)));
float B[N * N] __attribute__((aligned(32)));
float Bs[N * N] __attribute__((aligned(32)));
float C[N * N] __attribute__((aligned(32)));
float val[N * N] __attribute__((aligned(32)));

int main() {
  printf("Starting...\n");
  /**
   * Xeon 6258R
   * 2 AVX-512 FMA units
   * = 2 * 16 * 2 = 64 FLOP/cycle
   * = 2.7 * 64 = 172.8 GFLOP/s at 2.7GHz
   */

  // initialize
  FILE *file = fopen("/tmp/matmul", "rb");
  fread(A, 1, sizeof(float) * N * N, file);
  fread(B, 1, sizeof(float) * N * N, file);
  fread(C, 1, sizeof(float) * N * N, file);
  fclose(file);
  memset(val, 0, sizeof(float) * N * N);

  // Swizzle. B(N, N) -> Bs(N/8 * N, 8)
  swizzle(B, Bs, N, N);

  // Validate result
  gemm(A, Bs, val, N, N, N);
  allclose(val, C, N * N, 1e-3f);
  printf("Results verified, starting benchmarks...\n");

  // prints
  int repeats = 2;
  for (int i = 0; i < repeats; i++) {
    memset(val, 0, sizeof(float) * N * N);
    double start = tick();
    gemm(A, Bs, val, N, N, N);
    double stop = tick();
    allclose(val, C, N * N, 1e-3f);
    double elapsed_time = (stop - start) * 1e-3;
    printf("GFLOP/s: %f (%.2f ms)\n", (2.0 * N * N * N * 1e-9) / elapsed_time, elapsed_time * 1e3);
  }

#ifdef DEBUG
  print_matrix(A, N, N);
  print_matrix(B, N, N);
  print_matrix(C, N, N);
  print_matrix(val, N, N);
#endif

  return 0;
}