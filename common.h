#pragma once
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

static inline double tick() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec + ts.tv_nsec / 1e9);
}

static inline void print_matrix(float *m, int rows, int cols) {
  printf("_____________\n");
  for (int i = 0; i < rows; i++) {
    printf("[");
    for (int j = 0; j < cols; j++) {
      printf("%8.3f", m[i * cols + j]);
    }
    printf("]\n");
  }
  printf("_____________\n");
}

/** Frobenius norm relative error check */
static inline void allclose(float *a, float *b, int numel, float rtol) {
  double sum_diff = 0.0;
  double sum_b = 0.0;
  for (int i = 0; i < numel; i++) {
    double diff = a[i] - b[i];
    sum_diff += (double)diff * diff;
    sum_b += (double)b[i] * b[i];
  }
  double frobenius_diff = sqrt(sum_diff);
  double frobenius_b = sqrt(sum_b);
  double relative_error = frobenius_diff / frobenius_b;
  if (relative_error > rtol) {
    printf("Results do not match! Relative Error: %f\n", relative_error);
    exit(1);
  }
}

static inline void rand_init(float *m, int numel) {
  for (int i = 0; i < numel; i++) {
    m[i] = (float)rand() / (float)RAND_MAX;
  }
}

static inline void constant_init(float *m, int numel, float val) {
  for (int i = 0; i < numel; i++) {
    m[i] = val;
  }
}