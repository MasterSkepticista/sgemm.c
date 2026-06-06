#include "../common.h"
#include "variants.h"

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
