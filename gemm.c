/**
 * Optimizing SGEMM in C using AVX (Row-major layout).
 * clang -O2 -march=native -mprefer-vector-width=512 gemm.c -o ./gemm && ./gemm 0 1024
 */

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include <mkl.h>
#include "common.h"

#define MEM_ALIGN 64

/** MKL-SGEMM as roofline. */
void gemm_mkl(float* __restrict C, 
               const float* __restrict A, 
               const float* __restrict B, 
               int M, 
               int N, 
               int K) {
  memset(C, 0, sizeof(float) * M * N);
  float alpha = 1.0f, beta = 1.0f;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K, alpha, A, K, B, N, beta, C, N);
}

/** Basic loop-reordered, pointwise GEMM kernel. */
void gemm_loop_reorder(float* __restrict C, 
                        const float* __restrict A, 
                        const float* __restrict B, 
                        int M, 
                        int N, 
                        int K) {
  memset(C, 0, M * N * sizeof(float));
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

/** Cache-blocking across dimensions. */
#define TILE_K 64
#define TILE_N 2048
#define TILE_M 1024

void gemm_cache_blocking(float* __restrict C, const float* __restrict A, const float* __restrict B, int M, int N, int K) {
  constant_init(C, M * N, 0.0f);

  // Tile across each dimension
  for (int i = 0; i < M; i += TILE_M) {
    const int mc = min(TILE_M, M - i);
    for (int k = 0; k < K; k += TILE_K) {
      const int kc = min(TILE_K, K - k);
      for (int j = 0; j < N; j += TILE_N) {
        const int nc = min(TILE_N, N - j);

        // Update partials on each tile
        for (int ir = 0; ir < mc; ir++) {
          for (int p = 0; p < kc; p++) {
            for (int jc = 0; jc < nc; jc++) {
              C[(i + ir) * N + (j + jc)] += A[(i + ir) * K + (k + p)] * B[(k + p) * N + (j + jc)];
            }
          }
        }
      }
    }
  }
}

#define MR 8
#define NR 8

void gemm_6x8(float* C, float* blockA, float* blockB, int m, int n, int k, int ldC) {
  __m256 a, b;
  __m256 c[MR];
	__m256i mask;

  // Load
  if (n < NR) {
    // Build mask.
    alignas(32) static const int32_t mask_table[9][8] = {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {-1, 0, 0, 0, 0, 0, 0, 0},
      {-1, -1, 0, 0, 0, 0, 0, 0},
      {-1, -1, -1, 0, 0, 0, 0, 0},
      {-1, -1, -1, -1, 0, 0, 0, 0},
      {-1, -1, -1, -1, -1, 0, 0, 0},
      {-1, -1, -1, -1, -1, -1, 0, 0},
      {-1, -1, -1, -1, -1, -1, -1, 0},
      {-1, -1, -1, -1, -1, -1, -1, -1},
    };
    mask = _mm256_load_si256((__m256i*)mask_table[n]);
    
    // Masked load
    for (int i = 0; i < m; i++) {
      c[i] = _mm256_maskload_ps(&C[i * ldC], mask);
    }
  } else {
    for (int i = 0; i < m; i++) {
      c[i] = _mm256_loadu_ps(&C[i * ldC]);
    }
  }

  // Compute
  for (int p = 0; p < k; p++) {
    b = _mm256_loadu_ps(blockB);

    a = _mm256_broadcast_ss(blockA);
    c[0] = _mm256_fmadd_ps(a, b, c[0]);
    a = _mm256_broadcast_ss(blockA + 1);
    c[1] = _mm256_fmadd_ps(a, b, c[1]);
    a = _mm256_broadcast_ss(blockA + 2);
    c[2] = _mm256_fmadd_ps(a, b, c[2]);
    a = _mm256_broadcast_ss(blockA + 3);
    c[3] = _mm256_fmadd_ps(a, b, c[3]);
    a = _mm256_broadcast_ss(blockA + 4);
    c[4] = _mm256_fmadd_ps(a, b, c[4]);
    a = _mm256_broadcast_ss(blockA + 5);
    c[5] = _mm256_fmadd_ps(a, b, c[5]);

    blockA += MR;
    blockB += NR;
  }

  // Store
  if (n < NR) {
    for (int i = 0; i < m; i++) {
      _mm256_maskstore_ps(&C[i * ldC], mask, c[i]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      _mm256_storeu_ps(&C[i * ldC], c[i]);
    }
  }
}

void pad_blockA(const float *A, float *blockA, int mr, int K) {
  for (int p = 0; p < K; p++) {
    for (int i = 0; i < MR; i++) {
      blockA[p * MR + i] = (i < mr) ? A[i * K + p] : 0.0f;
    }
  }
}

void pad_blockB(const float *B, float *blockB, int nr, int K, int ldB) {
  for (int p = 0; p < K; p++) {
    for (int j = 0; j < NR; j++) {
      blockB[p * NR + j] = (j < nr) ? B[p * ldB + j] : 0.0f;
    }
  }
}

void kernel2(float* __restrict C, const float* __restrict A, const float* __restrict B, int M, int N, int K) {
  constant_init(C, M * N, 0.0f);
  float *blockA = (float *)_mm_malloc(sizeof(float) * K * MR, MEM_ALIGN);
  float *blockB = (float *)_mm_malloc(sizeof(float) * K * NR, MEM_ALIGN);

  for (int j = 0; j < N; j += NR) {
    const int nr = min(NR, N - j);
    pad_blockB(&B[j], blockB, nr, K, N);
    for (int i = 0; i < M; i += MR) {
      const int mr = min(MR, M - i);
      pad_blockA(&A[i * K], blockA, mr, K);
      gemm_6x8(&C[i * N + j], blockA, blockB, mr, nr, K, N);
    }
  }
}

void launch_kernel(int kernel_num, float* C, float* A, float* B, int M, int N, int K) {
  switch (kernel_num) {
    case 0:
      gemm_mkl(C, A, B, M, N, K);
      break;
    case 1:
      gemm_loop_reorder(C, A, B, M, N, K);
      break;
    case 2:
      gemm_cache_blocking(C, A, B, M, N, K);
      break;
    case 3:
      kernel2(C, A, B, M, N, K);
      break;
    default:
      printf("Invalid kernel number `%d`\n", kernel_num);
      exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  int kernel_num, M, N, K;
  if (argc > 4) {
    kernel_num = atoi(argv[1]);
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
  } else if (argc > 2) {
    kernel_num = atoi(argv[1]);
    M = N = K = atoi(argv[2]);
  } else {
    printf("Usage: %s <kernel_num> <M> <N> <K>\n", argv[0]);
    printf("Usage with M=N=K: %s <kernel_num> <size> \n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Alloc
  float* A = (float*)_mm_malloc(sizeof(float) * M * K, MEM_ALIGN);
  float* B = (float*)_mm_malloc(sizeof(float) * K * N, MEM_ALIGN);
  float* C = (float*)_mm_malloc(sizeof(float) * M * N, MEM_ALIGN);
  float* C_val = (float*)_mm_malloc(sizeof(float) * M * N, MEM_ALIGN);

  // Initialize
  rand_init(A, M * K);
  rand_init(B, K * N);
  constant_init(C, M * N, 0.0f);
  constant_init(C_val, M * N, 0.0f);

  // Warmup run, generate ground truth data.
#ifdef DEBUG
  gemm_mkl(C, A, B, M, N, K);
  launch_kernel(kernel_num, C_val, A, B, M, N, K);
  allclose(C, C_val, M * N, 1e-2);
  printf("Results match, starting benchmark...\n");
#endif

  // Benchmark
  int repeats = 10;
  double gflops = (2.0 * M * N * K) * 1e-6;
  double total_gflops = 0.0;
  for (int i = 0; i < repeats; i++) {
    double start = tick();
    launch_kernel(kernel_num, C_val, A, B, M, N, K);
    double stop = tick();
    double elapsed_time = stop - start;
    total_gflops += gflops / elapsed_time;
  }
  printf("[M=%4d, N=%4d, K=%4d] GFLOP/s: %.2f\n", M, N, K, total_gflops / repeats);

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(C_val);
  return 0;
}