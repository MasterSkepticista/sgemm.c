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
void gemm(const float *A, const float *B, float *C, int rows, int inners, int cols) {
  
  for (int by = 0; by < rows; by += BLOCK_Y) {
    for (int bx = 0; bx < cols; bx += BLOCK_X) {
      // Within a single block.
      __m256 tmp[BLOCK_Y][BLOCK_X] = {};

      // Compute
      for (int k = 0; k < inners; k+=8) {
        for (int y = 0; y < BLOCK_Y; y++) {
          for (int x = 0; x < BLOCK_X; x++) {
            tmp[y][x] = _mm256_fmadd_ps(
              _mm256_load_ps(&A[(by + y) * inners + k]), 
              _mm256_load_ps(&B[(bx + x) * inners + k]),
            tmp[y][x]);
          }
        }
      }

      // Store
      for (int y = 0; y < BLOCK_Y; y++) {
        for (int x = 0; x < BLOCK_X; x++) {
          float sum = 0.0f;
          for (int i = 0; i < 8; i++) {
            sum += ((float *)&tmp[y][x])[i];
          }
          C[(by + y) * cols + (bx + x)] = sum;
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

  // Validate result
  gemm(A, B, val, N, N, N);
  allclose(val, C, N * N, 1e-3f);
  printf("Results verified, starting benchmarks...\n");

  // prints
  int repeats = 2;
  for (int i = 0; i < repeats; i++) {
    double start = tick();
    gemm(A, B, val, N, N, N);
    double stop = tick();
    double elapsed_time = (stop - start) * 1e-3;
    printf("GFLOP/s: %f\n", (2.0 * N * N * N * 1e-9) / elapsed_time);
  }

#ifdef DEBUG
  print_matrix(A, N, N);
  print_matrix(B, N, N);
  print_matrix(C, N, N);
  print_matrix(val, N, N);
#endif

  return 0;
}