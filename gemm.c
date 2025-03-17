/**
 * Optimizing SGEMM in C.
 * clang -Ofast -ffast-math -march=native -fopenmp -lgomp gemm.c -o ./gemm.o && ./gemm
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "common.h"

#ifdef OMP
#include <omp.h>
#endif

void matmul(const float *left, const float *right, float *out, int rows, int inners, int cols) {
  // #pragma omp parallel for collapse(2) shared(left, right, out)
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      for (int k = 0; k < inners; k++) {
        out[y * cols + x] += left[y * inners + k] * right[x * inners + k];
      }
    }
  }
}

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
  matmul(A, B, val, N, N, N);
  allclose(val, C, N * N, 1e-3f);
  printf("Results verified, starting benchmarks...\n");

  // prints
  int repeats = 2;
  for (int i = 0; i < repeats; i++) {
    double start = tick();
    matmul(A, B, val, N, N, N);
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