#include <immintrin.h>

#include "../common.h"
#include "variants.h"

/** 5. 512-bit intrinsics for tiled outer product */
#define MR 8
#define NR 48

#define MC MR * 256
#define KC MR * 256
#define NC 48

static float blockA[KC * MC] __attribute__((aligned(64)));
static float blockB[KC * NC] __attribute__((aligned(64)));

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

static inline int clamp16(int n) {
  if (n <= 0) return 0;
  if (n >= 16) return 16;
  return n;
}

static void micro_gemm_512_8x48(float* __restrict C, 
                                const float* __restrict blockA, 
                                const float* __restrict blockB, 
                                int m, 
                                int n, 
                                int k, 
                                int ldC) {
  __m512 a, b0, b1, b2;
  __m512 c[MR][3] = {};
  __mmask16 masks[3];

  // Compute
  for (int p = 0; p < k; p++) {
    b0 = _mm512_load_ps(blockB);
    b1 = _mm512_load_ps(blockB + 16);
    b2 = _mm512_load_ps(blockB + 32);

    #pragma unroll
    for (int i = 0; i < MR; i++) {
      a = _mm512_set1_ps(blockA[i]);
      c[i][0] = _mm512_fmadd_ps(a, b0, c[i][0]);
      c[i][1] = _mm512_fmadd_ps(a, b1, c[i][1]);
      c[i][2] = _mm512_fmadd_ps(a, b2, c[i][2]);
    }

    blockA += MR;
    blockB += NR;
  }

  // Load, update and store fused
  if (n < NR) {
    masks[0] = _cvtu32_mask16((1 << clamp16(n)) - 1);
    masks[1] = _cvtu32_mask16((1 << clamp16(n - 16)) - 1);
    masks[2] = _cvtu32_mask16((1 << clamp16(n - 32)) - 1);

    for (int i = 0; i < m; i++) {
      __m512 tmp0 = _mm512_maskz_loadu_ps(masks[0], &C[i * ldC]);
      __m512 tmp1 = _mm512_maskz_loadu_ps(masks[1], &C[i * ldC + 16]);
      __m512 tmp2 = _mm512_maskz_loadu_ps(masks[2], &C[i * ldC + 32]);

      tmp0 = _mm512_add_ps(tmp0, c[i][0]);
      tmp1 = _mm512_add_ps(tmp1, c[i][1]);
      tmp2 = _mm512_add_ps(tmp2, c[i][2]);

      _mm512_mask_store_ps(&C[i * ldC], masks[0], tmp0);
      _mm512_mask_store_ps(&C[i * ldC + 16], masks[1], tmp1);
      _mm512_mask_store_ps(&C[i * ldC + 32], masks[2], tmp2);
    }
  } else {
    for (int i = 0; i < m; i++) {
      __m512 tmp0 = _mm512_loadu_ps(&C[i * ldC]);
      __m512 tmp1 = _mm512_loadu_ps(&C[i * ldC + 16]);
      __m512 tmp2 = _mm512_loadu_ps(&C[i * ldC + 32]);

      tmp0 = _mm512_add_ps(tmp0, c[i][0]);
      tmp1 = _mm512_add_ps(tmp1, c[i][1]);
      tmp2 = _mm512_add_ps(tmp2, c[i][2]);

      _mm512_storeu_ps(&C[i * ldC], tmp0);
      _mm512_storeu_ps(&C[i * ldC + 16], tmp1);
      _mm512_storeu_ps(&C[i * ldC + 32], tmp2);
    }
  }
}

void gemm_outer_product_cache_blocking_512(float * __restrict C, 
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
            micro_gemm_512_8x48(&C[(i + ir) * N + (j + jr)], 
                                &blockA[ir * kc], 
                                &blockB[jr * kc], 
                                mr, nr, kc, N);
          }
        }
      }
    }
  }
}
