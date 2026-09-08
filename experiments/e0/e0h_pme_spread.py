#!/usr/bin/env python3
"""E0h: PME 铺展/回插微基准(M2 分量地板表待实测项;柱 2 语义:Q16.48 定点 + int64 atomicAdd)。

语义(order-4 cardinal B-spline;权重值与 opus/pme.py 参考精确一致 2.8e-16,
  锚点约定不同:本基准中心式 k-1..k+2,参考 support(0,p) 式 k-3..k——
  生产内核须与参考同约定以使网格逐点可比,吞吐不受约定影响):
  - 铺展:每原子 64 个网格点(4^3),deposit = llround(q * wx*wy*wz * 2^48),
    int64 atomicAdd(Q16.48;整数加法交换结合 ⇒ 位级确定,与线程调度无关)
  - 回插:每原子 64 次网格读取 × 同权重(力插值代表;每原子独立读,无原子)
  - 网格:48 副本 × 128^3(805 MB);每步 clear(memset)
维度:N=60000, R=48, alpha=3.5 nm^-1(修正后), grid 128^3(0.066 nm 间距@8.43 nm 盒)
验证:铺展跑两遍位级比对(柱 2);对总电荷守恒(Q16.48 求和 ≈ Σq×2^48)。
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
#define NG 128

__device__ __forceinline__ void weights4(double frac, int* anchor, double w[4]) {
    // order-4 cardinal B-spline, support [-2,2]; partition of unity per dim
    // atom at grid k+s (s in [0,1)); weights onto k-1..k+2:
    //   c0 = (1-s)^3/6, c1 = 2/3 - s^2 + s^3/2,
    //   c2 = 1/6 + s/2 + s^2/2 - s^3/2, c3 = s^3/6
    double x = frac * NG;
    int k = (int)x;
    anchor[0] = k - 1; anchor[1] = k; anchor[2] = k + 1; anchor[3] = k + 2;
    double s = x - k;
    double om = 1.0 - s;
    w[0] = om * om * om * (1.0 / 6.0);
    w[1] = (2.0 / 3.0) - s * s + 0.5 * s * s * s;
    w[2] = (1.0 / 6.0) + 0.5 * s + 0.5 * s * s - 0.5 * s * s * s;
    w[3] = s * s * s * (1.0 / 6.0);
}

extern "C" __global__ void spread(
    const double* __restrict__ x, const double* __restrict__ q,
    long long* __restrict__ grid,     // (R, NG, NG, NG) Q16.48
    int N, int R, double inv_h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a * R;
    if (a >= N) return;
    int ix = (int)(x[(a*R+r)*3]   * inv_h);
    int iy = (int)(x[(a*R+r)*3+1] * inv_h);
    int iz = (int)(x[(a*R+r)*3+2] * inv_h);
    double fx = x[(a*R+r)*3]   * inv_h - ix;
    double fy = x[(a*R+r)*3+1] * inv_h - iy;
    double fz = x[(a*R+r)*3+2] * inv_h - iz;
    int ax[4], ay[4], az[4];
    double wx[4], wy[4], wz[4];
    weights4(fx, ax, wx); weights4(fy, ay, wy); weights4(fz, az, wz);
    long long* g = grid + (long long)r * NG * NG * NG;
    double qi = q[a];
    for (int dz = 0; dz < 4; ++dz)
        for (int dy = 0; dy < 4; ++dy) {
            double wyz = wy[dy] * wz[dz] * qi;
            int yy = (ay[dy]) & (NG - 1), zz = (az[dz]) & (NG - 1);
            for (int dx = 0; dx < 4; ++dx) {
                int xx = ax[dx] & (NG - 1);
                long long dep = (long long)((wx[dx] * wyz) * 281474976710656.0);
                atomicAdd(reinterpret_cast<unsigned long long*>(
                              &g[((long long)yy * NG + zz) * NG + xx]),
                          (unsigned long long)dep);
            }
        }
}

extern "C" __global__ void interp(
    const double* __restrict__ x, const double* __restrict__ q,
    const long long* __restrict__ grid,   // Q16.48 potential proxy (reuse spread result)
    double* __restrict__ F, int N, int R, double inv_h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a * R;
    if (a >= N) return;
    int ix = (int)(x[(a*R+r)*3]   * inv_h);
    int iy = (int)(x[(a*R+r)*3+1] * inv_h);
    int fz_i = (int)(x[(a*R+r)*3+2] * inv_h);
    double fx = x[(a*R+r)*3]   * inv_h - ix;
    double fy = x[(a*R+r)*3+1] * inv_h - iy;
    double fz = x[(a*R+r)*3+2] * inv_h - fz_i;
    int ax[4], ay[4], az[4];
    double wx[4], wy[4], wz[4];
    weights4(fx, ax, wx); weights4(fy, ay, wy); weights4(fz, az, wz);
    const long long* g = grid + (long long)r * NG * NG * NG;
    double acc = 0.0;
    for (int dz = 0; dz < 4; ++dz)
        for (int dy = 0; dy < 4; ++dy) {
            double wyz = wy[dy] * wz[dz];
            int yy = (ay[dy]) & (NG - 1), zz = (az[dz]) & (NG - 1);
            for (int dx = 0; dx < 4; ++dx) {
                int xx = ax[dx] & (NG - 1);
                acc += (double)g[((long long)yy * NG + zz) * NG + xx] * (wx[dx] * wyz);
            }
        }
    F[a * R + r] = acc * q[a] * (1.0 / 281474976710656.0);
}
"""

mod = cp.RawModule(code=SRC)
k_spread = mod.get_function("spread")
k_interp = mod.get_function("interp")

N, R, NG = 60000, 48, 128
rng = np.random.default_rng(7)
L = 8.43
x = cp.asarray(rng.uniform(0, L, (N, R, 3)))
q = cp.asarray(rng.uniform(-1, 1, N))
grid = cp.zeros(R * NG * NG * NG, dtype=cp.int64)
F = cp.zeros(N * R)
inv_h = NG / L

grid_shape_bytes = R * NG**3 * 8
print(f"N={N} R={R} grid={R}x{NG}^3 ({grid_shape_bytes/1e6:.0f} MB int64)")


def run(k, iters=10, label=""):
    grid.fill(0)
    kgrid = ((N * R + 255) // 256,)
    args_sp = (x, q, grid, N, R, inv_h)
    k_spread(kgrid, (256,), args_sp)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        k_spread(kgrid, (256,), args_sp)
    cp.cuda.Stream.null.synchronize()
    dt_sp = (time.perf_counter() - t0) / iters
    return dt_sp


# ---- spread timing (含 clear) ----
for block in (256, 128):
    kgrid = ((N * R + block - 1) // block,)
    args = (x, q, grid, N, R, inv_h)
    # warmup
    k_spread(kgrid, (block,), args); cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        grid.fill(0)
        k_spread(kgrid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print(f"spread+clear (block {block}): {dt*1e3:.3f} ms/步  "
          f"({N*R*64/dt/1e9:.2f} G-atomic/s)", flush=True)

# ---- determinism (柱 2): 两遍铺展位级比对 ----
grid.fill(0)
k_spread(((N * R + 255) // 256,), (256,), (x, q, grid, N, R, inv_h))
g1 = grid.copy()
grid.fill(0)
k_spread(((N * R + 255) // 256,), (256,), (x, q, grid, N, R, inv_h))
g2 = grid.copy()
same = cp.array_equal(g1, g2)
print(f"determinism (int64 atomicAdd x2): bitwise identical = {same}")
# 电荷守恒: Σ grid / 2^48 ≈ R * Σ q (权重和 = 1/原子/维... 每原子 Σ(wx*wy*wz)=1)
tot = g1.sum() / 281474976710656.0
target = R * float(q.sum())
print(f"charge conservation: {tot:.6f} vs {target:.6f} (rel {abs(tot-target)/abs(target):.2e})")

# ---- interp timing ----
for block in (256,):
    kgrid = ((N * R + block - 1) // block,)
    args = (x, q, grid, F, N, R, inv_h)
    k_interp(kgrid, (block,), args); cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        k_interp(kgrid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print(f"interp (block {block}): {dt*1e3:.3f} ms/步  ({N*R*64/dt/1e9:.2f} G-read/s)")

print("\nE0h done. 分量地板表回填:铺展+回插(ms/步@R=48/N=60k/128^3/Q16.48)")
