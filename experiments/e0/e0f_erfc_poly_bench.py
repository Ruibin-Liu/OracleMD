#!/usr/bin/env python3
"""E0f: 查表/多项式 erfc 直空间内核微基准(M2 第一件,Q-002 替换点)。

对照 E0b-direct(intrinsic erfc,f64 R1/R48 = 14.0/29.5 Gpair/s):
  - erfc(x) = exp(-x^2)*W(x),W = 3 段 Chebyshev(t 空间 Horner)
    [0,1.5] deg18 / [1.5,3] deg14 / [3,6.5] deg18,max_rel <= 3.2e-14
    (scipy double erfc 为参照;A1 容差 rel 1e-10,余量 3+ 量级)
  - exp(-x^2) 与力公式第二项共享,一次计算两用
  - 附:poly vs intrinsic 的力输出全量交叉验证(max rel diff)
GPU util > 20% 拒跑。
"""
import sys, time, subprocess
import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit(f"GPU busy ({util}%), timing invalid.")

import cupy as cp

SRC = r"""
#define RSQRTPI 0.5641895835477563

__device__ __forceinline__ double W0(double t) {
    return ((((((((((((((((((+2.19935009363420147e-10*t-1.19435620325330841e-09)*t+4.42788930256360589e-09)*t-1.86014220734415395e-08)*t+8.07600742878595086e-08)*t-3.35144993111066006e-07)*t+1.34243296404639601e-06)*t-5.20962084392019835e-06)*t+1.95266080384681256e-05)*t-7.04679827391670090e-05)*t+2.44039031762325503e-04)*t-8.07782029369062682e-04)*t+2.54317035818231582e-03)*t-7.56936980337291881e-03)*t+2.11329450976001504e-02)*t-5.47745886538915511e-02)*t+1.29913948997565865e-01)*t-2.75979518741850893e-01)*t+5.06937650293145192e-01);
}
__device__ __forceinline__ double W1(double t) {
    return ((((((((((((((+1.82218060187137474e-10*t-1.00905589840545899e-09)*t+4.71420328519759667e-09)*t-2.43522908366651598e-08)*t+1.24068122897698662e-07)*t-6.12088052588045740e-07)*t+2.93863938978228677e-06)*t-1.37115593164665698e-05)*t+6.20318158827869387e-05)*t-2.71412140420430772e-04)*t+1.14507274914490368e-03)*t-4.64149438073509744e-03)*t+1.79958529184971079e-02)*t-6.63648771065850906e-02)*t+2.31087258730392209e-01);
}
__device__ __forceinline__ double W2(double t) {
    return ((((((((((((((((((+1.16099213085226341e-10*t-4.42548124312728053e-10)*t+1.04009121870080483e-09)*t-3.52776855919964062e-09)*t+1.32084281919475480e-08)*t-4.53816976253245234e-08)*t+1.52452698965931662e-07)*t-5.09310289079644697e-07)*t+1.68164518958108709e-06)*t-5.47956945116423911e-06)*t+1.76184602562209627e-05)*t-5.58730192110227247e-05)*t+1.74667247329627305e-04)*t-5.37951714317812420e-04)*t+1.63125725782422670e-03)*t-4.86684252617713281e-03)*t+1.42753120049775029e-02)*t-4.11310350467865293e-02)*t+1.16302707210247463e-01);
}

// erfc(x) via 3-piece polynomial; emx2 = exp(-x*x) precomputed (shared with force term))
__device__ __forceinline__ double erfc_poly(double x, double emx2) {
    double w;
    if (x < 1.5)      { double t = x*1.3333333333333333 - 1.0;              w = W0(t); }
    else if (x < 3.0) { double t = (x-1.5)*1.3333333333333333 - 1.0;        w = W1(t); }
    else              { double xc = x > 6.5 ? 6.5 : x;
                        double t = (xc-3.0)*0.5714285714285714 - 1.0;      w = W2(t); }
    return emx2 * w;
}

extern "C" __global__ void pairs_f64(
    const double* __restrict__ x, const int* __restrict__ nlist, const int* __restrict__ ncount,
    const double* __restrict__ q, const double* __restrict__ sig, const double* __restrict__ eps,
    double* __restrict__ F, int N, int R, int maxnb, double rc2, double alpha, int use_poly)
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
            double emx2 = exp(-alpha*alpha*r2);
            double xr = alpha/invr;   // = alpha*r (Ewald real-space argument)
            double ec = use_poly ? erfc_poly(xr, emx2) : erfc(xr);
            // F = qiqj*KE*[erfc(ar)/r^2 + (2a/sqrt(pi)) e^{-a^2 r^2} / r]
            double fc = qi*q[j]*(ec*invr2 + 2.0*alpha*RSQRTPI*emx2*invr);
            double f = flj + fc;
            fx += f*dx; fy += f*dy; fz += f*dz;
        }
    }
    F[(a*R+r)*3] = fx; F[(a*R+r)*3+1] = fy; F[(a*R+r)*3+2] = fz;
}
"""

mod = cp.RawModule(code=SRC)
k = mod.get_function("pairs_f64")

N = 60000
NB = 409
rng = np.random.default_rng(3)
x0 = (rng.standard_normal((N, 48, 3)) * 5).cumsum(axis=0) / N**0.5
nlist = rng.integers(0, N, (N, NB)).astype(np.int32)
ncount = np.full(N, NB, dtype=np.int32)
q = rng.uniform(-1, 1, N)
sig = rng.uniform(0.3, 0.4, N)
eps = rng.uniform(0.3, 0.7, N)
d_nlist, d_ncount = cp.asarray(nlist.ravel()), cp.asarray(ncount)


def bench(R, use_poly, iters=20):
    x = cp.asarray(x0[:, :R].copy())
    F = cp.zeros(N * R * 3)
    dq, ds, de = (cp.asarray(a) for a in (q, sig, eps))
    rc2, alpha = 1.0e9, 3.0
    grid = ((N * R + 255) // 256,)
    args = (x, d_nlist, d_ncount, dq, ds, de, F, N, R, NB, rc2, alpha, use_poly)
    k(grid, (256,), args)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        k(grid, (256,), args)
    cp.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt, N * NB * R / dt, F


print(f"{'variant':>10} {'R':>3} {'ms/iter':>9} {'Gpair/s':>8} {'vs_intrinsic':>12}")
res = {}
for R in (1, 8, 48):
    dt_i, pps_i, F_i = bench(R, use_poly=0)
    dt_p, pps_p, F_p = bench(R, use_poly=1)
    res[R] = (pps_i, pps_p)
    # 数值交叉验证: max rel diff(以 intrinsic 分量幅值归一)
    Fi, Fp = F_i.get(), F_p.get()
    ok = np.isfinite(Fi) & np.isfinite(Fp) & (np.abs(Fi) > 1e-6 * np.nanmax(np.abs(Fi)))
    dmax = (np.abs(Fp - Fi)[ok] / np.abs(Fi)[ok]).max() if ok.any() else float("nan")
    print(f"{'intrinsic':>10} {R:>3} {dt_i*1e3:>9.3f} {pps_i/1e9:>8.2f}")
    print(f"{'poly':>10} {R:>3} {dt_p*1e3:>9.3f} {pps_p/1e9:>8.2f} {pps_p/pps_i:>11.2f}x   F-xcheck rel={dmax:.2e}")

r48 = res[48][1] / res[1][1]
print(f"\npoly 批量红利: R48/R1 = {r48:.2f}x(intrinsic 对照 2.10x)")
print(f"Q-002 口径注: poly 绝对吞吐 {res[1][1]/1e9:.1f}(R1)/ {res[48][1]/1e9:.1f}(R48) Gpair/s")

# ---- 数值交叉验证(物理几何: 大基准的随机游走 r~10-40nm, 库仑项双路径都下溢为 0,
#      xcheck 空转——必须用真实尺度的盒子)
def xcheck_small():
    Ns, NBs, R = 4000, 60, 8
    rng = np.random.default_rng(0)
    xs_ = cp.asarray(rng.uniform(0, 3.0, (Ns, R, 3)))
    nls = cp.asarray(rng.integers(0, Ns, (Ns * NBs,)).astype(np.int32))
    ncs = cp.asarray(np.full(Ns, NBs, dtype=np.int32))
    qs, ss, es_ = (cp.asarray(rng.uniform(*r_, Ns)) for r_ in ((-1, 1), (0.25, 0.45), (0.3, 0.7)))
    grid = ((Ns * R + 255) // 256,)
    outs = []
    for up in (0, 1):
        Fs = cp.zeros(Ns * R * 3)
        k(grid, (256,), (xs_, nls, ncs, qs, ss, es_, Fs, Ns, R, NBs, 1.21, 3.5, up))
        cp.cuda.Stream.null.synchronize()
        outs.append(Fs.get())
    Fi, Fp = outs
    ok = np.isfinite(Fi) & np.isfinite(Fp)
    mag = np.abs(Fi)[ok]
    thr = 1e-6 * np.percentile(mag, 99)
    sel = ok & (np.abs(Fi) > thr) & (np.abs(Fi) < 1e6)  # 近奇异随机对非物理(G3b/Q-008 包络外),剔除
    rel = np.abs(Fp - Fi)[sel] / np.abs(Fi)[sel]
    absg = np.abs(Fp - Fi)[sel] / np.percentile(mag, 99)
    dmax = rel.max()
    print(f"  rel: p50={np.median(rel):.1e} p99={np.percentile(rel,99):.1e} max={rel.max():.1e}")
    print(f"  abs(以 p99 幅值归一, 免对消放大): max={absg.max():.1e}")
    # erfc 活跃对占比(coulomb 非零分量)供参考
    print(f"xcheck(物理几何, rc2=1.21 即 rc=1.1, alpha=3.5): max_rel = {dmax:.2e}"
          f"  (n_sel={int(sel.sum())}; 要求 <=1e-11, 预期 ~1e-13)")

xcheck_small()
