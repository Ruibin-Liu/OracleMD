#!/usr/bin/env python3
"""E0b-direct / E0d-direct: 直空间对循环微基准, 模拟生产内核的关键结构:
  - replica 最内层布局 x[a][r][3]
  - 跨副本共享邻居列表 (nlist[a] 广播, x[j][r] 合并读取)
  - fp64 (及 fp32 对照 = E0d) LJ + Ewald 实空间(erfc)
回答: R=1 vs R=48 的 achieved 吞吐比 (eta_serial vs eta_batch 的直空间分量),
     以及 f32/f64 比 (E0d 的直空间分量)。
GPU 有竞争时拒绝运行。
"""
import sys, time, subprocess
import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit(f"GPU busy ({util}%), timing invalid.")

import cupy as cp

SRC = r"""
extern "C" __global__ void pairs_f64(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    double* __restrict__ F, int N, int R, int maxnb, double rc2, double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    double xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    double qi = q[a], si = sig[a], ei = eps[a];
    double fx=0., fy=0., fz=0.;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        double dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            double invr2 = 1.0/r2, invr = 1.0/sqrt(r2);
            double s = 0.5*(si+sig[j]), e = sqrt(ei*eps[j]);
            double sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            double flj = 24.0*e*(2.0*sr6*sr6 - sr6)*invr2;
            double fc = qi*q[j]*invr2*(erfc(alpha*invr)*invr
                        + 2.0*alpha*0.5641895835477563*exp(-alpha*alpha*r2));
            double f = flj + fc;
            fx += f*dx; fy += f*dy; fz += f*dz;
        }
    }
    F[(a*R+r)*3] = fx; F[(a*R+r)*3+1] = fy; F[(a*R+r)*3+2] = fz;
}
extern "C" __global__ void pairs_f32(
    const float* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const float* __restrict__ q, const float* __restrict__ sig, const float* __restrict__ eps,
    float* __restrict__ F, int N, int R, int maxnb, float rc2, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    float xi = x[(a*R+r)*3], yi = x[(a*R+r)*3+1], zi = x[(a*R+r)*3+2];
    float qi = q[a], si = sig[a], ei = eps[a];
    float fx=0.f, fy=0.f, fz=0.f;
    int nb = ncount[a];
    for (int k = 0; k < nb; ++k) {
        int j = nlist[a*maxnb + k];
        float dx = x[(j*R+r)*3] - xi, dy = x[(j*R+r)*3+1] - yi, dz = x[(j*R+r)*3+2] - zi;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < rc2) {
            float invr2 = 1.0f/r2, invr = rsqrtf(r2);
            float s = 0.5f*(si+sig[j]), e = sqrtf(ei*eps[j]);
            float sr2 = s*s*invr2, sr6 = sr2*sr2*sr2;
            float flj = 24.0f*e*(2.0f*sr6*sr6 - sr6)*invr2;
            float fc = qi*q[j]*invr2*(erfcf(alpha*invr)*invr
                       + 2.0f*alpha*0.56418958f*__expf(-alpha*alpha*r2));
            float f = flj + fc;
            fx += f*dx; fy += f*dy; fz += f*dz;
        }
    }
    F[(a*R+r)*3] = fx; F[(a*R+r)*3+1] = fy; F[(a*R+r)*3+2] = fz;
}
"""

mod = cp.RawModule(code=SRC)
kf64 = mod.get_function("pairs_f64")
kf32 = mod.get_function("pairs_f32")

N = 60000
NB = 409          # 半列表, 与 roofline 一致
FLOP_PER_PAIR = 60  # spec roofline 的约定, 结果以此换算

rng = np.random.default_rng(3)
x0 = (rng.standard_normal((N, 48, 3)) * 5).cumsum(axis=0) / N**0.5  # 简单相关构象
nlist = rng.integers(0, N, (N, NB)).astype(np.int32)
ncount = np.full(N, NB, dtype=np.int32)
q = rng.uniform(-1, 1, N)
sig = rng.uniform(0.3, 0.4, N)
eps = rng.uniform(0.3, 0.7, N)

d_nlist = cp.asarray(nlist.ravel())
d_ncount = cp.asarray(ncount)

def bench(R, prec, iters=20):
    xp = np.float64 if prec == "f64" else np.float32
    x = cp.asarray(x0[:, :R].copy(), dtype=xp)
    F = cp.zeros(N*R*3, dtype=xp)
    dq, ds, de = (cp.asarray(a, dtype=xp) for a in (q, sig, eps))
    k = kf64 if prec == "f64" else kf32
    rc2, alpha = xp(1.0e9), xp(3.0)  # rc2 巨大 => 全部走计算路径
    grid = ((N*R + 255)//256,)
    k(grid, (256,), (x, d_nlist, d_ncount, dq, ds, de, F, N, R, NB, rc2, alpha))
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        k(grid, (256,), (x, d_nlist, d_ncount, dq, ds, de, F, N, R, NB, rc2, alpha))
    cp.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / iters
    pairs = N * NB * R
    # 换算: 该 kernel 每步若按此速度, 每副本 ns/day @ 2fs
    nsday_per_rep = 2e-15 / dt * 86400
    return dt, pairs/dt, pairs/dt*FLOP_PER_PAIR, nsday_per_rep

print(f"{'prec':>5} {'R':>3} {'ms/iter':>9} {'Gpair/s':>8} {'GFLOP/s*':>9} {'ns/day/rep':>10} {'%peak*':>7}")
res = {}
for prec in ("f64", "f32"):
    for R in (1, 8, 48):
        dt, pps, flops, nsday = bench(R, prec)
        peak = 9.7e12 if prec == "f64" else 19.5e12
        res[(prec, R)] = (pps, flops)
        print(f"{prec:>5} {R:>3} {dt*1e3:>9.3f} {pps/1e9:>8.2f} {flops/1e9:>9.0f} {nsday:>10.1f} {100*flops/peak:>6.1f}%")

r64 = res[("f64", 48)][0] / res[("f64", 1)][0]
r32 = res[("f32", 48)][0] / res[("f32", 1)][0]
rfd = res[("f32", 48)][0] / res[("f64", 48)][0]
print(f"\neta 比(直空间分量, 对数吞吐): batch48/serial1 f64 = {r64:.2f}x, f32 = {r32:.2f}x")
print(f"E0d(直空间分量): f32/f64 @ batch48 = {rfd:.2f}x")
print("* GFLOP/s 与 %peak 按 60 FLOP/pair 约定换算, 仅用于相对比较; 绝对值以 Gpair/s 为准")
print("注: erfc 为本征函数(慢路径), 生产内核若用查表/多项式, 绝对数会上移, R 标度比才是本测试的目的")
