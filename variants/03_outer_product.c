#include <immintrin.h>

#include "../common.h"
#include "variants.h"

/** Outer Product on MRxNR tiles */
#define MR 6
#define NR 16

static const int8_t mask[32]  __attribute__((aligned(64))) = 
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

static void micro_gemm_6x16(float* __restrict C,
                            const float* __restrict A,
                            const float* __restrict B,
                            int k,
                            int ldA,
                            int ldB,
                            int ldC) {
  __m256 c[MR][2] = {};

  // Compute
  for (int p = 0; p < k; p++) {
    __m256 a, b0, b1;
    b0 = _mm256_loadu_ps(B);
    b1 = _mm256_loadu_ps(B + 8);

    a = _mm256_broadcast_ss(&A[p]);
    c[0][0] = _mm256_fmadd_ps(a, b0, c[0][0]);
    c[0][1] = _mm256_fmadd_ps(a, b1, c[0][1]);
    a = _mm256_broadcast_ss(&A[ldA + p]);
    c[1][0] = _mm256_fmadd_ps(a, b0, c[1][0]);
    c[1][1] = _mm256_fmadd_ps(a, b1, c[1][1]);
    a = _mm256_broadcast_ss(&A[2 * ldA + p]);
    c[2][0] = _mm256_fmadd_ps(a, b0, c[2][0]);
    c[2][1] = _mm256_fmadd_ps(a, b1, c[2][1]);
    a = _mm256_broadcast_ss(&A[3 * ldA + p]);
    c[3][0] = _mm256_fmadd_ps(a, b0, c[3][0]);
    c[3][1] = _mm256_fmadd_ps(a, b1, c[3][1]);
    a = _mm256_broadcast_ss(&A[4 * ldA + p]);
    c[4][0] = _mm256_fmadd_ps(a, b0, c[4][0]);
    c[4][1] = _mm256_fmadd_ps(a, b1, c[4][1]);
    a = _mm256_broadcast_ss(&A[5 * ldA + p]);
    c[5][0] = _mm256_fmadd_ps(a, b0, c[5][0]);
    c[5][1] = _mm256_fmadd_ps(a, b1, c[5][1]);

    B += ldB;
  }

  // Update
  for (int i = 0; i < MR; i++) {
    c[i][0] += _mm256_loadu_ps(&C[i * ldC]);
    c[i][1] += _mm256_loadu_ps(&C[i * ldC + 8]);
    _mm256_storeu_ps(&C[i * ldC], c[i][0]);
    _mm256_storeu_ps(&C[i * ldC + 8], c[i][1]);
  }
}

static void micro_gemm_edge(float* __restrict C,
                            const float* __restrict A,
                            const float* __restrict B,
                            int m,
                            int n,
                            int k,
                            int ldA,
                            int ldB,
                            int ldC) {
  __m256 c[MR][2] = {};
  __m256i masks[2];

  masks[0] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - n]));
  masks[1] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - n + 8]));

  // Compute
  for (int p = 0; p < k; p++) {
    const __m256 b0 = _mm256_maskload_ps(B, masks[0]);
    const __m256 b1 = _mm256_maskload_ps(B + 8, masks[1]);

    for (int i = 0; i < m; i++) {
      const __m256 a = _mm256_broadcast_ss(&A[i * ldA + p]);
      c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
      c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
    }

    B += ldB;
  }

  // Masked update
  for (int i = 0; i < m; i++) {
    c[i][0] += _mm256_maskload_ps(&C[i * ldC], masks[0]);
    c[i][1] += _mm256_maskload_ps(&C[i * ldC + 8], masks[1]);
    _mm256_maskstore_ps(&C[i * ldC], masks[0], c[i][0]);
    _mm256_maskstore_ps(&C[i * ldC + 8], masks[1], c[i][1]);
  }
}

/** 3. Outer Product without Cache-Blocking. */
void gemm_outer_product(float* __restrict C, 
                        const float* __restrict A, 
                        const float* __restrict B, 
                        int M, 
                        int N, 
                        int K) {
  for (int j = 0; j < N; j += NR) {
    const int nr = min(NR, N - j);
    for (int i = 0; i < M; i += MR) {
      const int mr = min(MR, M - i);
      if (mr == MR && nr == NR) {
        micro_gemm_6x16(&C[i * N + j], &A[i * K], &B[j], K, K, N, N);
      } else {
        micro_gemm_edge(&C[i * N + j], &A[i * K], &B[j],
                        mr, nr, K, K, N, N);
      }
    }
  }
}
