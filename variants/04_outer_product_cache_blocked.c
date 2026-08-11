#include <immintrin.h>

#include "../common.h"
#include "variants.h"

/** 4. Outer Product with Cache-Blocking. */
#define MR 6
#define NR 16

#define MC MR * 4
#define KC 256
#define NC 2048

static float blockA[KC * MC] __attribute__((aligned(64)));
static float blockB[KC * NC] __attribute__((aligned(64)));

static const int8_t mask[32]  __attribute__((aligned(64))) = 
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

static inline __attribute__((always_inline))
void accumulate_6x16(__m256 c[MR][2],
                     const float* __restrict blockA,
                     const float* __restrict blockB,
                     int k) {
  #pragma unroll 4
  for (int p = 0; p < k; p++) {
    __m256 a, b0, b1;
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
}

static void micro_gemm_6x16(float* __restrict C,
                const float* __restrict blockA,
                const float* __restrict blockB,
                int m,
                int n,
                int k,
                int ldC) {
  __m256 c[MR][2] = {};

  // Start fetching C into L1.
  for (int i = 0; i < MR; i++) {
    _mm_prefetch(&C[i * ldC], _MM_HINT_T0);
  }

  // Compute all but the final 32 K values.
  accumulate_6x16(c, blockA, blockB, k);

  // Update C.
  #pragma unroll 6
  for (int i = 0; i < MR; i++) {
    _mm256_store_ps(&C[i * ldC], _mm256_add_ps(c[i][0], _mm256_load_ps(&C[i * ldC])));
    _mm256_store_ps(&C[i * ldC + 8], _mm256_add_ps(c[i][1], _mm256_load_ps(&C[i * ldC + 8])));
  }
}


static void micro_gemm_edge(float* __restrict C, 
                const float* __restrict blockA, 
                const float* __restrict blockB, 
                int m, 
                int n, 
                int k, 
                int ldC) {
  __m256 c[MR][2] = {};

  // Start fetching C into L1.
  for (int i = 0; i < MR; i++) {
    _mm_prefetch(&C[i * ldC], _MM_HINT_T0);
  }

  // Compute
  accumulate_6x16(c, blockA, blockB, k);

  // Masked update for tail mr/nr.
	__m256i masks[2];
  masks[0] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - n]));
  masks[1] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - n + 8]));
  for (int i = 0; i < m; i++) {
    _mm256_maskstore_ps(
      &C[i * ldC], 
      masks[0], 
      _mm256_add_ps(c[i][0], _mm256_maskload_ps(&C[i * ldC], masks[0]))
    );
    _mm256_maskstore_ps(
      &C[i * ldC + 8], 
      masks[1], 
      _mm256_add_ps(c[i][1], _mm256_maskload_ps(&C[i * ldC + 8], masks[1]))
    );
  }
}


static void pack_tileA(float * __restrict blockA, 
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

static void pack_tileB(float * __restrict blockB,
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
  for (int j = 0; j < N; j += NC) {
    const int nc = min(NC, N - j);
    for (int p = 0; p < K; p += KC) {
      const int kc = min(KC, K - p);
      pack_tileB(blockB, &B[p * N + j], nc, kc, N);
      for (int i = 0; i < M; i += MC) {
        const int mc = min(MC, M - i);
        pack_tileA(blockA, &A[i * K + p], mc, kc, K);
        for (int jr = 0; jr < nc; jr += NR) {
          for (int ir = 0; ir < mc; ir += MR) {
            const int mr = min(MR, mc - ir);
            const int nr = min(NR, nc - jr);
            if (nr == NR && mr == MR && (N % 8) == 0) {
              micro_gemm_6x16(
                              &C[(i + ir) * N + (j + jr)],
                              &blockA[ir * kc],
                              &blockB[jr * kc],
                              mr, nr, kc, N);
            } else {
              micro_gemm_edge(&C[(i + ir) * N + (j + jr)], 
                              &blockA[ir * kc], 
                              &blockB[jr * kc], 
                              mr, nr, kc, N);
            }
          }
        }
      }
    }
  }
}
