#!/usr/bin/env python3
"""E0e: cuFFT fp64 位级行为测试 —— 直接检验 spec C2/M15 的前提。

测试矩阵:
  T1: 同一 plan 重复执行 -> 位级一致? (M15 前提)
  T2: batch=R 一次批量 vs R 次 batch=1 -> 位级一致? (M1b 前提)
  T3: 不同 batch size {1,4,8,48} 两两比较
  T4: in-place vs out-of-place
报告: 每个比较给出 bitwise-equal? / max abs diff / max rel diff / 不同 ulp 的元素比例
"""
import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    sys.exit("cupy not available")

def fft_batch(data, batch, n):
    """data: (batch, n, n, n) complex128, out-of-place forward FFT."""
    return cp.fft.fftn(data, axes=(1, 2, 3))

def compare(a, b, label):
    a64, b64 = a.view(np.int64), b.view(np.int64)
    bit_eq = np.array_equal(a64, b64)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, diff / denom, 0.0)
    n_ulp_diff = np.count_nonzero(a64 != b64)
    print(f"{label:58s} bitwise={'EQ ' if bit_eq else 'DIFF'} "
          f"maxabs={diff.max():.3e} maxrel={rel.max():.3e} "
          f"diff_elems={n_ulp_diff}/{a.size} ({100*n_ulp_diff/a.size:.4f}%)")
    return bit_eq

rng = np.random.default_rng(42)

for n in (64, 128):
    print(f"\n=== grid {n}^3 complex128 ===")
    R = 48
    free = cp.cuda.Device().mem_info[0]
    need = R * n**3 * 16 * 2  # in + out
    if need > free * 0.8:
        print(f"skip R=48 at {n}^3: need {need/2**30:.1f} GiB, free {free/2**30:.1f} GiB")
        R = 16
    base = [rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))
            for _ in range(R)]

    # T1: same batch plan, repeated
    d = cp.asarray(np.stack(base))
    r1 = cp.asnumpy(fft_batch(d, R, n)); cp.cuda.Stream.null.synchronize()
    r2 = cp.asnumpy(fft_batch(d, R, n)); cp.cuda.Stream.null.synchronize()
    compare(r1, r2, f"T1 same-plan repeated (batch={R})")
    del d, r1, r2; cp.get_default_memory_pool().free_all_blocks()

    # T2/T3: batch vs singles, and across batch sizes
    singles = None
    batched = {}
    for bs in (1, 4, 8, R):
        if bs == 1:
            out = [cp.asnumpy(fft_batch(cp.asarray(b[None]), 1, n)[0]) for b in base]
            singles = np.stack(out)
            batched[1] = singles
            del out
        else:
            res = []
            for i in range(0, R, bs):
                chunk = np.stack(base[i:i+bs])
                res.append(cp.asnumpy(fft_batch(cp.asarray(chunk), len(chunk), n)))
                del chunk
            batched[bs] = np.concatenate(res)
            del res
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    for bs in (4, 8, R):
        if bs in batched and bs != 1:
            compare(singles, batched[bs], f"T2 batch=1 x{R} vs batch={bs} chunks")
    compare(batched[4], batched[8], "T3 batch=4 vs batch=8")

    # T4: in-place vs out-of-place (batch=1)
    d1 = cp.asarray(base[0])
    oop = cp.asnumpy(cp.fft.fftn(d1))
    try:
        d2 = d1.copy()
        # cupy fftn always copies; use cufft directly for in-place
        from cupyx.scipy import fft as cpxfft
        ip = cpxfft.fftn(d2, overwrite_x=True)
        cp.cuda.Stream.null.synchronize()
        compare(oop, cp.asnumpy(ip), "T4 out-of-place vs in-place (batch=1)")
    except Exception as e:
        print(f"T4 skipped: {e}")
    del d1, oop; cp.get_default_memory_pool().free_all_blocks()

print("\ndone.")
