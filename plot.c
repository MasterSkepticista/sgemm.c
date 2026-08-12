#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <immintrin.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "common.h"
#include "variants/variants.h"

#define MEM_ALIGN 64
#define MIN_SIZE 64
#define MAX_SIZE 4096
#define SIZE_STEP 64
#define MAX_KERNEL_COUNT 6
#define MIN_BENCHMARK_SECONDS 0.2
#define DATA_FILE "output/sgemm_gflops.dat"

static const char* kernel_names[MAX_KERNEL_COUNT] = {
    "0:OpenBLAS",
    "1:Loop reorder",
    "2:Cache blocked",
    "3:Outer product",
    "4:Outer product cache blocked",
    "5:Outer product AVX-512",
};

static void launch_kernel(int kernel_num,
                          float* C,
                          const float* A,
                          const float* B,
                          int size) {
  switch (kernel_num) {
    case 0:
      gemm_mkl(C, A, B, size, size, size);
      break;
    case 1:
      gemm_loop_reorder(C, A, B, size, size, size);
      break;
    case 2:
      gemm_cache_blocked(C, A, B, size, size, size);
      break;
    case 3:
      gemm_outer_product(C, A, B, size, size, size);
      break;
    case 4:
      gemm_outer_product_cache_blocking(C, A, B, size, size, size);
      break;
    case 5:
      gemm_outer_product_cache_blocking_512(C, A, B, size, size, size);
      break;
  }
}

static double benchmark_kernel(int kernel_num,
                               float* C,
                               const float* A,
                               const float* B,
                               int size) {
  int repeats = 1;

  launch_kernel(kernel_num, C, A, B, size);
  for (;;) {
    const double start = tick();
    for (int repeat = 0; repeat < repeats; repeat++) {
      launch_kernel(kernel_num, C, A, B, size);
    }
    const double elapsed = tick() - start;

    if (elapsed >= MIN_BENCHMARK_SECONDS || repeats > INT_MAX / 2) {
      const double operations = 2.0 * size * size * size * repeats;
      return operations / elapsed * 1e-9;
    }
    repeats *= 2;
  }
}

static int generate_plot(int kernel_count) {
  FILE* gnuplot = popen("gnuplot", "w");
  if (gnuplot == NULL) {
    perror("gnuplot");
    return EXIT_FAILURE;
  }

  fprintf(gnuplot, "set terminal pngcairo size 1280,720\n");
  fprintf(gnuplot, "set title 'SGEMM performance by kernel'\n");
  fprintf(gnuplot, "set xlabel 'Matrix size (M = N = K)'\n");
  fprintf(gnuplot, "set ylabel 'GFLOP/s'\n");
  fprintf(gnuplot, "set xrange [%d:%d]\n", MIN_SIZE, MAX_SIZE);
  fprintf(gnuplot, "set grid\n");
  fprintf(gnuplot, "set key inside right center\n");

  for (int last_kernel = 1; last_kernel < kernel_count; last_kernel++) {
    fprintf(gnuplot,
            "set output 'output/sgemm_gflops_0_%d.png'\nplot ",
            last_kernel);
    for (int kernel = 0; kernel <= last_kernel; kernel++) {
      fprintf(gnuplot,
              "%s'" DATA_FILE "' using 1:%d with linespoints title '%s'",
              kernel == 0 ? "" : ", ",
              kernel + 2,
              kernel_names[kernel]);
    }
    fprintf(gnuplot, "\n");
  }
  fprintf(gnuplot, "unset output\n");

  if (pclose(gnuplot) != 0) {
    fprintf(stderr,
            "gnuplot failed; the benchmark data is still available in "
            DATA_FILE "\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main(void) {
  if (system("gnuplot --version >/dev/null 2>&1") != 0) {
    fprintf(stderr, "gnuplot is required to generate the benchmark plots\n");
    return EXIT_FAILURE;
  }
  if (mkdir("output", 0755) != 0 && errno != EEXIST) {
    perror("output");
    return EXIT_FAILURE;
  }
  int kernel_count = MAX_KERNEL_COUNT;
  if (!__builtin_cpu_supports("avx512f")) {
    kernel_count--;
    fprintf(stderr, "Skipping kernel 5: CPU does not support AVX-512F\n");
  }

  FILE* data = fopen(DATA_FILE, "w");
  if (data == NULL) {
    perror(DATA_FILE);
    return EXIT_FAILURE;
  }

  fprintf(data, "# size");
  for (int kernel = 0; kernel < kernel_count; kernel++) {
    fprintf(data, " kernel_%d", kernel);
  }
  fprintf(data, "\n");

  for (int size = MIN_SIZE; size <= MAX_SIZE; size += SIZE_STEP) {
    const size_t elements = (size_t)size * size;
    float* A = _mm_malloc(sizeof(*A) * elements, MEM_ALIGN);
    float* B = _mm_malloc(sizeof(*B) * elements, MEM_ALIGN);
    float* C = _mm_malloc(sizeof(*C) * elements, MEM_ALIGN);
    if (A == NULL || B == NULL || C == NULL) {
      fprintf(stderr, "Allocation failed for size %d\n", size);
      _mm_free(A);
      _mm_free(B);
      _mm_free(C);
      fclose(data);
      return EXIT_FAILURE;
    }

    rand_init(A, (int)elements);
    rand_init(B, (int)elements);
    fprintf(data, "%d", size);
    printf("size %4d:", size);
    for (int kernel = 0; kernel < kernel_count; kernel++) {
      constant_init(C, (int)elements, 0.0f);
      const double gflops = benchmark_kernel(kernel, C, A, B, size);
      fprintf(data, " %.6f", gflops);
      printf(" k%d=%8.2f", kernel, gflops);
      fflush(stdout);
    }
    fprintf(data, "\n");
    fflush(data);
    printf(" GFLOP/s\n");

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
  }

  if (fclose(data) != 0) {
    perror(DATA_FILE);
    return EXIT_FAILURE;
  }
  if (generate_plot(kernel_count) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  printf("Wrote " DATA_FILE " and %d incremental plots in output/\n",
         kernel_count - 1);
  return EXIT_SUCCESS;
}
