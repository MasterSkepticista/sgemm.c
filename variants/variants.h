#pragma once

void gemm_mkl(float* __restrict C,
              const float* __restrict A,
              const float* __restrict B,
              int M,
              int N,
              int K);

void gemm_loop_reorder(float* __restrict C,
                       const float* __restrict A,
                       const float* __restrict B,
                       int M,
                       int N,
                       int K);

void gemm_cache_blocked(float* __restrict C,
                        const float* __restrict A,
                        const float* __restrict B,
                        int M,
                        int N,
                        int K);

void gemm_outer_product(float* __restrict C,
                        const float* __restrict A,
                        const float* __restrict B,
                        int M,
                        int N,
                        int K);

void gemm_outer_product_cache_blocking(float* __restrict C,
                                       const float* __restrict A,
                                       const float* __restrict B,
                                       int M,
                                       int N,
                                       int K);

void gemm_outer_product_cache_blocking_512(float* __restrict C,
                                           const float* __restrict A,
                                           const float* __restrict B,
                                           int M,
                                           int N,
                                           int K);
