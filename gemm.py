#!/usr/bin/python3
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np

N = 4096
np.random.seed(42)
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

for i in range(4):
  t0 = time.monotonic()
  C = A @ B
  dt = (time.monotonic() - t0) * 1e9
  print(f"GFLOPS: {2.0 * N ** 3 / dt:.3f} ({dt * 1e-6:.2f} ms)")

with open("/tmp/matmul", "wb") as f:
  f.write(A.data)
  f.write(B.data)
  f.write(C.data)
