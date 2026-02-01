#pragma once
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

double tick() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec + ts.tv_nsec / 1e9);
}

void print_matrix(float *m, int rows, int cols) {
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

void allclose(float *a, float *b, int numel, float rtol) {
  for (int i = 0; i < numel; i++) {
    if (fabsf(a[i] - b[i]) > rtol) {
      printf("mismatch at idx %d, %f != %f \n", i, a[i], b[i]);
      exit(1);
    }
  }
}

void rand_init(float *m, int numel) {
  for (int i = 0; i < numel; i++) {
    m[i] = (float)rand() / (float)RAND_MAX;
  }
}

void constant_init(float *m, int numel, float val) {
  for (int i = 0; i < numel; i++) {
    m[i] = val;
  }
}