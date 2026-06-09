#include <cblas.h>

#include "variants.h"

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
