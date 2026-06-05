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

/** 0. MKL-SGEMM as roofline. */
void gemm_mkl(float* __restrict C, 
               const float* __restrict A, 
               const float* __restrict B, 
               int M, 
               int N, 
               int K) {
  float alpha = 1.0f, beta = 0.0f;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K, alpha, A, K, B, N, beta, C, N);
}

/** 1. Basic loop-reordered, pointwise GEMM kernel. */
void gemm_loop_reorder(float* __restrict C, 
                        const float* __restrict A, 
                        const float* __restrict B, 
                        int M, 
                        int N, 
                        int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

/** 2. Cache-blocking across dimensions. */
#define TK 128
#define TN 2048
#define TM 1024

void gemm_cache_blocked(float* __restrict C, 
                          const float* __restrict A, 
                          const float* __restrict B, 
                          int M, 
                          int N, 
                          int K) {
  // Tile across each dimension
  for (int i = 0; i < M; i += TM) {
    const int mc = min(TM, M - i);
    for (int k = 0; k < K; k += TK) {
      const int kc = min(TK, K - k);
      for (int j = 0; j < N; j += TN) {
        const int nc = min(TN, N - j);

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

/** Outer Product on MRxNR tiles */
#define MR 6
#define NR 16

const static int8_t mask[32]  __attribute__((aligned(64))) = 
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

void micro_gemm(float* __restrict C, 
                const float* __restrict blockA, 
                const float* __restrict blockB, 
                int m, 
                int n, 
                int k, 
                int ldC) {
  __m256 a, b0, b1;
  __m256 c[MR][2];
	__m256i masks[2];

  // Load
  if (n < NR) {
    // Build mask.
    masks[0] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - n]));
    masks[1] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - n + 8]));

    // Masked load
    for (int i = 0; i < m; i++) {
      c[i][0] = _mm256_maskload_ps(&C[i * ldC], masks[0]);
      c[i][1] = _mm256_maskload_ps(&C[i * ldC + 8], masks[1]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      c[i][0] = _mm256_loadu_ps(&C[i * ldC]);
      c[i][1] = _mm256_loadu_ps(&C[i * ldC + 8]);
    }
  }

  // Compute
  for (int p = 0; p < k; p++) {
    b0 = _mm256_load_ps(blockB);
    b1 = _mm256_load_ps(blockB + 8);

    a = _mm256_broadcast_ss(blockA);
    c[0][0] = _mm256_fmadd_ps(a, b0, c[0][0]);
    c[0][1] = _mm256_fmadd_ps(a, b1, c[0][1]);
    a = _mm256_broadcast_ss(blockA + 1);
    c[1][0] = _mm256_fmadd_ps(a, b0, c[1][0]);
    c[1][1] = _mm256_fmadd_ps(a, b1, c[1][1]);
    a = _mm256_broadcast_ss(blockA + 2);
    c[2][0] = _mm256_fmadd_ps(a, b0, c[2][0]);
    c[2][1] = _mm256_fmadd_ps(a, b1, c[2][1]);
    a = _mm256_broadcast_ss(blockA + 3);
    c[3][0] = _mm256_fmadd_ps(a, b0, c[3][0]);
    c[3][1] = _mm256_fmadd_ps(a, b1, c[3][1]);
    a = _mm256_broadcast_ss(blockA + 4);
    c[4][0] = _mm256_fmadd_ps(a, b0, c[4][0]);
    c[4][1] = _mm256_fmadd_ps(a, b1, c[4][1]);
    a = _mm256_broadcast_ss(blockA + 5);
    c[5][0] = _mm256_fmadd_ps(a, b0, c[5][0]);
    c[5][1] = _mm256_fmadd_ps(a, b1, c[5][1]);

    blockA += MR;
    blockB += NR;
  }

  // Store
  if (n < NR) {
    for (int i = 0; i < m; i++) {
      _mm256_maskstore_ps(&C[i * ldC], masks[0], c[i][0]);
      _mm256_maskstore_ps(&C[i * ldC + 8], masks[1], c[i][1]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      _mm256_storeu_ps(&C[i * ldC], c[i][0]);
      _mm256_storeu_ps(&C[i * ldC + 8], c[i][1]);
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

/** 3. Outer Product without Cache-Blocking. */
void gemm_outer_product(float* __restrict C, 
                        const float* __restrict A, 
                        const float* __restrict B, 
                        int M, 
                        int N, 
                        int K) {

  float *blockA = (float *)_mm_malloc(sizeof(float) * K * MR, MEM_ALIGN);
  float *blockB = (float *)_mm_malloc(sizeof(float) * K * NR, MEM_ALIGN);

  for (int j = 0; j < N; j += NR) {
    const int nr = min(NR, N - j);
    pad_blockB(&B[j], blockB, nr, K, N);
    for (int i = 0; i < M; i += MR) {
      const int mr = min(MR, M - i);
      pad_blockA(&A[i * K], blockA, mr, K);
      micro_gemm(&C[i * N + j], blockA, blockB, mr, nr, K, N);
    }
  }
}

/** 4. Outer Product with Cache-Blocking. */
#define KC 2048
#define NC 128
#define MC 1024

static float blockA[KC * MC] __attribute__((aligned(64)));
static float blockB[KC * NC] __attribute__((aligned(64)));

void pack_tileA(float * __restrict blockA, 
                const float * __restrict A, 
                int mc, 
                int kc, 
                int ldA) {
  for (int ir = 0; ir < mc; ir += MR) {
    const int m = min(MR, mc - ir);
    for (int p = 0; p < kc; p++) {
      for (int i = 0; i < MR; i++) {
        blockA[ir * kc + p * MR + i] = (i < m) ? A[(ir + i) * ldA + p] : 0.0f;
      }
    }
  }
}

void pack_tileB(float * __restrict blockB,
                const float * __restrict B,
                int nc,
                int kc,
                int ldB) {
  for (int jr = 0; jr < nc; jr += NR) {
    const int n = min(NR, nc - jr);
    for (int p = 0; p < kc; p++) {
      for (int j = 0; j < NR; j++) {
        blockB[jr * kc + p * NR + j] = (j < n) ? B[p * ldB + (jr + j)] : 0.0f;
      }
    }
  }
}

void gemm_outer_product_cache_blocking(float * __restrict C, 
                                      const float * __restrict A, 
                                      const float * __restrict B, 
                                      int M, 
                                      int N, 
                                      int K) {
  for (int i = 0; i < M; i += MC) {
    const int mc = min(MC, M - i);
    for (int p = 0; p < K; p += KC) {
      const int kc = min(KC, K - p);
      pack_tileA(blockA, &A[i * K + p], mc, kc, K);
      for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        pack_tileB(blockB, &B[p * N + j], nc, kc, N);
        for (int ir = 0; ir < mc; ir += MR) {
          for (int jr = 0; jr < nc; jr += NR) {
            const int mr = min(MR, mc - ir);
            const int nr = min(NR, nc - jr);
            micro_gemm(&C[(i + ir) * N + (j + jr)], 
                       &blockA[ir * kc], 
                       &blockB[jr * kc], 
                       mr, nr, kc, N);
          }
        }
      }
    }
  }
}

/** 5. 512-bit intrinsics for tiled outer product */
#define Z_MR 8
#define Z_NR 48

#define Z_MC Z_MR * 256
#define Z_KC Z_MR * 256
#define Z_NC 48

static float z_blockA[Z_KC * Z_MC] __attribute__((aligned(64)));
static float z_blockB[Z_KC * Z_NC] __attribute__((aligned(64)));

void z_pack_tileA(float * __restrict blockA, 
                const float * __restrict A, 
                int mc, 
                int kc, 
                int ldA) {
  for (int ir = 0; ir < mc; ir += Z_MR) {
    const int m = min(Z_MR, mc - ir);
    for (int p = 0; p < kc; p++) {
      for (int i = 0; i < Z_MR; i++) {
        blockA[ir * kc + p * Z_MR + i] = (i < m) ? A[(ir + i) * ldA + p] : 0.0f;
      }
    }
  }
}

void z_pack_tileB(float * __restrict blockB,
                const float * __restrict B,
                int nc,
                int kc,
                int ldB) {
  for (int jr = 0; jr < nc; jr += Z_NR) {
    const int n = min(Z_NR, nc - jr);
    for (int p = 0; p < kc; p++) {
      for (int j = 0; j < Z_NR; j++) {
        blockB[jr * kc + p * Z_NR + j] = (j < n) ? B[p * ldB + (jr + j)] : 0.0f;
      }
    }
  }
}

static inline int clamp16(int n) {
  if (n <= 0) return 0;
  if (n >= 16) return 16;
  return n;
}

void micro_gemm_512_fs(float* __restrict C, 
                const float* __restrict blockA, 
                const float* __restrict blockB, 
                int m, 
                int n, 
                int k, 
                int ldC) {
  __m512 a, b0, b1, b2;
  __m512 c[Z_MR][3] = {};
  __mmask16 masks[3];

  if (n < Z_NR) {
    masks[0] = _cvtu32_mask16((1 << clamp16(n)) - 1);
    masks[1] = _cvtu32_mask16((1 << clamp16(n - 16)) - 1);
    masks[2] = _cvtu32_mask16((1 << clamp16(n - 32)) - 1);
  }

  // Compute
  for (int p = 0; p < k; p++) {
    b0 = _mm512_load_ps(blockB);
    b1 = _mm512_load_ps(blockB + 16);
    b2 = _mm512_load_ps(blockB + 32);

    #pragma unroll
    for (int i = 0; i < Z_MR; i++) {
      a = _mm512_set1_ps(blockA[i]);
      c[i][0] = _mm512_fmadd_ps(a, b0, c[i][0]);
      c[i][1] = _mm512_fmadd_ps(a, b1, c[i][1]);
      c[i][2] = _mm512_fmadd_ps(a, b2, c[i][2]);
    }

    blockA += Z_MR;
    blockB += Z_NR;
  }

  // Store
  if (n < Z_NR) {
    for (int i = 0; i < m; i++) {
      _mm512_mask_store_ps(&C[i * ldC], masks[0], c[i][0]);
      _mm512_mask_store_ps(&C[i * ldC + 16], masks[1], c[i][1]);
      _mm512_mask_store_ps(&C[i * ldC + 32], masks[2], c[i][2]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      _mm512_storeu_ps(&C[i * ldC], c[i][0]);
      _mm512_storeu_ps(&C[i * ldC + 16], c[i][1]);
      _mm512_storeu_ps(&C[i * ldC + 32], c[i][2]);
    }
  }
}

void micro_gemm_512_lfs(float* __restrict C, 
                const float* __restrict blockA, 
                const float* __restrict blockB, 
                int m, 
                int n, 
                int k, 
                int ldC) {
  __m512 a, b0, b1, b2;
  __m512 c[Z_MR][3] = {};
  __mmask16 masks[3];

  // Load
  if (n < Z_NR) {
    masks[0] = _cvtu32_mask16((1 << clamp16(n)) - 1);
    masks[1] = _cvtu32_mask16((1 << clamp16(n - 16)) - 1);
    masks[2] = _cvtu32_mask16((1 << clamp16(n - 32)) - 1);

    for (int i = 0; i < m; i++) {
      c[i][0] = _mm512_maskz_loadu_ps(masks[0], &C[i * ldC]);
      c[i][1] = _mm512_maskz_loadu_ps(masks[1], &C[i * ldC + 16]);
      c[i][2] = _mm512_maskz_loadu_ps(masks[2], &C[i * ldC + 32]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      c[i][0] = _mm512_loadu_ps(&C[i * ldC]);
      c[i][1] = _mm512_loadu_ps(&C[i * ldC + 16]);
      c[i][2] = _mm512_loadu_ps(&C[i * ldC + 32]);
    }
  }

  // Compute
  for (int p = 0; p < k; p++) {
    b0 = _mm512_load_ps(blockB);
    b1 = _mm512_load_ps(blockB + 16);
    b2 = _mm512_load_ps(blockB + 32);

    #pragma unroll
    for (int i = 0; i < Z_MR; i++) {
      a = _mm512_set1_ps(blockA[i]);
      c[i][0] = _mm512_fmadd_ps(a, b0, c[i][0]);
      c[i][1] = _mm512_fmadd_ps(a, b1, c[i][1]);
      c[i][2] = _mm512_fmadd_ps(a, b2, c[i][2]);
    }

    blockA += Z_MR;
    blockB += Z_NR;
  }

  // Store
  if (n < Z_NR) {
    for (int i = 0; i < m; i++) {
      _mm512_mask_store_ps(&C[i * ldC], masks[0], c[i][0]);
      _mm512_mask_store_ps(&C[i * ldC + 16], masks[1], c[i][1]);
      _mm512_mask_store_ps(&C[i * ldC + 32], masks[2], c[i][2]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      _mm512_storeu_ps(&C[i * ldC], c[i][0]);
      _mm512_storeu_ps(&C[i * ldC + 16], c[i][1]);
      _mm512_storeu_ps(&C[i * ldC + 32], c[i][2]);
    }
  }
}

void gemm_outer_product_cache_blocking_512(float * __restrict C, 
                                      const float * __restrict A, 
                                      const float * __restrict B, 
                                      int M, 
                                      int N, 
                                      int K) {
  for (int i = 0; i < M; i += Z_MC) {
    const int mc = min(Z_MC, M - i);
    for (int p = 0; p < K; p += Z_KC) {
      const int kc = min(Z_KC, K - p);
      z_pack_tileA(z_blockA, &A[i * K + p], mc, kc, K);
      for (int j = 0; j < N; j += Z_NC) {
        const int nc = min(Z_NC, N - j);
        z_pack_tileB(z_blockB, &B[p * N + j], nc, kc, N);
        for (int ir = 0; ir < mc; ir += Z_MR) {
          for (int jr = 0; jr < nc; jr += Z_NR) {
            const int mr = min(Z_MR, mc - ir);
            const int nr = min(Z_NR, nc - jr);
            if (p == 0) {
              micro_gemm_512_fs(&C[(i + ir) * N + (j + jr)], 
                         &z_blockA[ir * kc], 
                         &z_blockB[jr * kc], 
                         mr, nr, kc, N);
            } else {
              micro_gemm_512_lfs(&C[(i + ir) * N + (j + jr)], 
                          &z_blockA[ir * kc], 
                          &z_blockB[jr * kc], 
                          mr, nr, kc, N);
            }
          }
        }
      }
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
      gemm_cache_blocked(C, A, B, M, N, K);
      break;
    case 3:
      gemm_outer_product(C, A, B, M, N, K);
      break;
    case 4:
      gemm_outer_product_cache_blocking(C, A, B, M, N, K);
      break;
    case 5:
      gemm_outer_product_cache_blocking_512(C, A, B, M, N, K);
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
    K = atoi(argv[3]);
    N = atoi(argv[4]);
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
  allclose(C, C_val, M * N, 1e-3);
  printf("Results match, starting benchmark...\n");
#endif

  // Benchmark
  double gflops = (2.0 * M * N * K) * 1e-9;
  double total_time = 0.0;
  int repeats = (int)ceil(100.0 / gflops);
  for (int i = 0; i < repeats; i++) {
    double start = tick();
    launch_kernel(kernel_num, C_val, A, B, M, N, K);
    double stop = tick();
    double elapsed_time = stop - start;
    total_time += elapsed_time;
  }
  printf("[M = %4d, K = %4d, N = %4d] GFLOP/s: %.2f\n", M, K, N, gflops / (total_time / repeats));

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(C_val);
  return 0;
}