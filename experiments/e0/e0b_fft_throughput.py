#!/usr/bin/env python3
"""E0b-pme(微基准): 批量 cuFFT 128^3 fp64 吞吐, batch {1,4,8,16,48} 的每-transform 成本标度。
回答: η_batch(FFT) 的上界 —— 批量是否摊薄了 per-transform 成本。
GPU 有竞争时拒绝运行。
"""
import sys, time, subprocess
import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit(f"GPU busy ({util}%), timing invalid.")

import cupy as cp

n = 128
flop_per = 5 * n**3 * np.log2(n**3) * 2  # fwd+inv
bytes_per = n**3 * 16 * 2 * 2            # in+out, fwd+inv (下界,实际更高)

print(f"{'batch':>6} {'ms/iter':>10} {'ms/transform':>14} {'GFLOP/s':>10} {'GB/s(lower)':>12}")
for bs in (1, 4, 8, 16, 48):
    free = cp.cuda.Device().mem_info[0]
    if bs * n**3 * 16 * 3 > free * 0.8:
        print(f"{bs:>6}  skip (mem)")
        continue
    x = cp.random.standard_normal((bs, n, n, n)) + 1j*cp.random.standard_normal((bs, n, n, n))
    # warmup + plan creation
    for _ in range(3):
        y = cp.fft.fftn(x, axes=(1,2,3)); z = cp.fft.ifftn(y, axes=(1,2,3))
    cp.cuda.Stream.null.synchronize()
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        y = cp.fft.fftn(x, axes=(1,2,3)); z = cp.fft.ifftn(y, axes=(1,2,3))
    cp.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / reps
    print(f"{bs:>6} {dt*1e3:>10.3f} {dt/bs*1e3:>14.3f} {bs*flop_per/dt/1e9:>10.0f} {bs*bytes_per/dt/1e9:>12.0f}")
    del x, y, z
    cp.get_default_memory_pool().free_all_blocks()
print("done. 注: A100 80G HBM2e 带宽 ~2 TB/s, fp64 peak 9.7 TF.")
