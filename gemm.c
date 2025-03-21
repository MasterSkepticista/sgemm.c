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

#ifdef FAST

void swizzle(const float *B, float *Bs, int rows, int cols) {
  if (rows % 16 != 0) {
    printf("Error: rows must be a multiple of 16\n");
    exit(1);
  }
  for (int y = 0; y < rows; y += 16) {
    for (int x = 0; x < cols; x++) {
      for (int iy = 0; iy < 16; iy++) {
        Bs[y * cols + x * 16 + iy] = B[(y + iy) * cols + x];
      }
    }
  }
}

void gemm(const float *A, const float *B, float *C, int rows, int inners, int cols) {
  for (int y = 0; y < rows; y += BLOCK_Y) {
    for (int x = 0; x < cols; x += 16) {
      // Compute
      __m512 acc[BLOCK_Y] = {};
      for (int k = 0; k < inners; k++) {
        // __m512 Bv = Bm[(x*cols + k * 16)/16];
        // __m512 Bv = {2.0f};
        __m512 Bv = _mm512_load_ps(&B[x * inners + k * 16]);
        for (int iy = 0; iy < BLOCK_Y; iy++) {
          __m512 Av = _mm512_set1_ps(A[(y + iy) * inners + k]);
          acc[iy] = _mm512_fmadd_ps(Av, Bv, acc[iy]);
        }
      }

      // Store
      for (int iy = 0; iy < BLOCK_Y; iy++) {
        _mm512_store_ps(&C[(y + iy) * cols + x], acc[iy]);
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

#define N 768

float A[N * N] __attribute__((aligned(64)));
float B[N * N] __attribute__((aligned(64)));
float Bs[N * N] __attribute__((aligned(64)));
float C[N * N] __attribute__((aligned(64)));
float val[N * N] __attribute__((aligned(64)));

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

  // Benchmark
  int repeats = 2;
  for (int i = 0; i < repeats; i++) {
    double start = tick();
    gemm(A, B, val, N, N, N);
    double stop = tick();
    double elapsed_time = (stop - start) * 1e-3;
    printf("GFLOP/s: %f (%.2f ms)\n", (2.0 * N * N * N * 1e-9) / elapsed_time, elapsed_time * 1e3);
  }
  
  allclose(val, C, N * N, 1e-3f);
  printf("Match.\n");
  
  return 0;
}