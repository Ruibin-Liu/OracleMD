#!/usr/bin/env python3
"""B1 前提验证: fp64 atomicAdd scatter 的非确定性 vs int64 定点累加的确定性。

模拟 PME 电荷铺展模式: N 个原子各向 64 个网格点 scatter 加权电荷。
  K1: atomicAdd(double) 两个不同 block 配置 -> 预期位级不同 (B1 的前提)
  K2: atomicAdd(double) 同配置重复多次 -> 检查 run-to-run
  K3: 定点 int64 (Q16.48 风格) 两个不同 block 配置 -> 预期位级相同
"""
import numpy as np
try:
    import cupy as cp
except ImportError:
    raise SystemExit("cupy not available")

N_ATOMS = 60000
GRID = 128
G3 = GRID**3

kernel_src = r"""
extern "C" __global__
void spread_f64(const double* __restrict__ q, const int* __restrict__ idx,
                const double* __restrict__ w, double* __restrict__ grid, int n, int nw) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    double qa = q[a];
    for (int j = 0; j < nw; ++j) {
        atomicAdd(&grid[idx[a*nw + j]], qa * w[a*nw + j]);
    }
}
extern "C" __global__
void spread_i64(const double* __restrict__ q, const int* __restrict__ idx,
                const double* __restrict__ w, long long* __restrict__ grid, int n, int nw,
                double scale) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    double qa = q[a];
    for (int j = 0; j < nw; ++j) {
        long long fx = llrint(qa * w[a*nw + j] * scale);
        atomicAdd((unsigned long long*)&grid[idx[a*nw + j]], (unsigned long long)fx);
    }
}
"""
mod = cp.RawModule(code=kernel_src)
kf = mod.get_function("spread_f64")
ki = mod.get_function("spread_i64")

rng = np.random.default_rng(7)
NW = 64
q = cp.asarray(rng.uniform(-1, 1, N_ATOMS))
idx = cp.asarray(rng.integers(0, G3, N_ATOMS * NW).astype(np.int32))
w = cp.asarray(rng.random(N_ATOMS * NW))

def run_f64(block):
    g = cp.zeros(G3, dtype=cp.float64)
    kf(((N_ATOMS + block - 1)//block,), (block,), (q, idx, w, g, N_ATOMS, NW))
    cp.cuda.Stream.null.synchronize()
    return g

def run_i64(block):
    g = cp.zeros(G3, dtype=cp.int64)
    ki(((N_ATOMS + block - 1)//block,), (block,), (q, idx, w, g, N_ATOMS, NW, 2.0**48))
    cp.cuda.Stream.null.synchronize()
    return g

def report(name, a, b):
    if a.dtype == np.int64:
        eq = np.array_equal(cp.asnumpy(a), cp.asnumpy(b))
        nd = np.count_nonzero(cp.asnumpy(a) != cp.asnumpy(b))
        print(f"{name:44s} bitwise={'EQ' if eq else 'DIFF'} diff_elems={nd}/{a.size}")
    else:
        an, bn = cp.asnumpy(a), cp.asnumpy(b)
        eq = np.array_equal(an.view(np.int64), bn.view(np.int64))
        nd = np.count_nonzero(an.view(np.int64) != bn.view(np.int64))
        rel = np.abs(an-bn)/np.maximum(np.maximum(np.abs(an),np.abs(bn)),1e-300)
        print(f"{name:44s} bitwise={'EQ' if eq else 'DIFF'} diff_elems={nd}/{a.size} maxrel={rel.max():.3e}")

# K1: fp64, different block sizes (order changes)
g1 = run_f64(128); g2 = run_f64(512)
report("K1 fp64 atomicAdd block=128 vs 512", g1, g2)

# K2: fp64, same config, repeated 5x (run-to-run under scheduling jitter)
base = run_f64(256)
allsame = True
for i in range(5):
    gi = run_f64(256)
    same = np.array_equal(cp.asnumpy(base).view(np.int64), cp.asnumpy(gi).view(np.int64))
    allsame &= same
    report(f"K2 fp64 run-to-run rep{i+1} vs rep0", base, gi)
print(f"K2 summary: run-to-run bitwise stable = {allsame}")

# K3: fixed-point int64, different block sizes
h1 = run_i64(128); h2 = run_i64(512)
report("K3 Q16.48 int64 block=128 vs 512", h1, h2)

# K4: int64 vs fp64 数值一致性 (应差在量化分辨率 ~2^-48 * 值域)
ref = cp.asnumpy(h1) / 2.0**48
got = cp.asnumpy(g1)
relerr = np.abs(ref-got)[np.abs(ref)>1e-12] / np.abs(ref)[np.abs(ref)>1e-12]
print(f"K4 fixed-point vs fp64 数值: max rel err = {relerr.max():.3e} (量化分辨率 2^-48={2**-48:.2e})")
print("done.")
