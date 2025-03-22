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

#define WIDTH 16

#ifdef FAST

float *swizzle(const float *B, int inners, int cols) {
  float *Bs = (float*)malloc(sizeof(float) * inners * cols);
  if (cols % WIDTH != 0) {
    printf("Error: cols must be a multiple of WIDTH\n");
    exit(1);
  }
  #pragma omp parallel for shared(B)
  for (int x = 0; x < cols; x += WIDTH) {
    for (int k = 0; k < inners; k++) {
      for (int ix = 0; ix < WIDTH; ix++) {
        Bs[(x / WIDTH) * (inners * WIDTH) + (k * WIDTH) + ix] = B[k * cols + (x + ix)];
      }
    }
  }
  return Bs;
}

void gemm(const float *A, const float *B, float *C, int rows, int inners, int cols) {
  #pragma omp parallel for shared(A, B, C)
  for (int x = 0; x < cols; x += WIDTH * BLOCK_X) {
    for (int y = 0; y < rows; y += BLOCK_Y) {
      // Compute
      __m512 acc[BLOCK_Y][BLOCK_X] = {};
      for (int k = 0; k < inners; k++) {
        for (int iy = 0; iy < BLOCK_Y; iy++) {
          __m512 Av = _mm512_set1_ps(A[(y + iy) * inners + k]);
          for (int ix = 0; ix < BLOCK_X; ix++) {
            __m512 Bv = _mm512_load_ps(&B[((x + ix*WIDTH) / WIDTH) * (inners * WIDTH) + k * WIDTH]);
            acc[iy][ix] = _mm512_fmadd_ps(Av, Bv, acc[iy][ix]);
          }
        }
      }

      // Store
      for (int iy = 0; iy < BLOCK_Y; iy++) {
        for (int ix = 0; ix < BLOCK_X; ix++) {
          _mm512_store_ps(&C[(y + iy) * cols + (x + ix * WIDTH)], acc[iy][ix]);
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

#define N 4096

float A[N * N] __attribute__((aligned(64)));
float B[N * N] __attribute__((aligned(64)));
float C[N * N] __attribute__((aligned(64)));
float val[N * N] __attribute__((aligned(64)));

int main() {
  printf("Problem size [%d x %d]\n", N, N);
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

  // Benchmark
  int repeats = 400;
  for (int i = 0; i < repeats; i++) {
    double start = tick();
    gemm(A, swizzle(B, N, N), val, N, N, N);
    double stop = tick();
    double elapsed_time = (stop - start) * 1e-3;
    printf("GFLOP/s: %.2f (%.2f ms)\n", (2.0 * N * N * N * 1e-9) / elapsed_time, elapsed_time * 1e3);
  }

  // print_matrix(C, N, N);
  // print_matrix(val, N, N);
  allclose(val, C, N * N, 1e-3f);
  printf("Match.\n");

  return 0;
}