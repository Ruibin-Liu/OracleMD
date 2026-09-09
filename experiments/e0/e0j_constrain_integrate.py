#!/usr/bin/env python3
"""E0j: 约束/积分微基准(分量地板表最后一项;Q-016/Q-003 语义)。

分量(生产步 @R=48 / N=60k / 刚性 TIP3P-FB / HMR x3):
  1. integrate_vrov: BAOAB 流式部分(V.5/R/O/R/V.5;O 用 philox counter-RNG
     高斯,Box-Muller)—— 柱 3 的真实生产成本(8.6M gaussian/步)
  2. shake_rigid: 每水 1 线程,3 约束 x 12 固定迭代(Q-016:迭代数 manifest
     常量,无收敛分支);线程内串行、跨水独立,无原子
  3. rng_only: 单独计 philox+gaussian 成本(积分的子分量)
util > 20% 拒跑。
"""
import sys, time, subprocess
import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit(f"GPU busy ({util}%), timing invalid.")

import cupy as cp

SRC = r"""
#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u
#define PHILOX_W32_0   0x9E3779B9u
#define PHILOX_W32_1   0xBB67AE85u

__device__ __forceinline__ uint4 philox4x32(uint4 c, uint2 k) {
    for (int r = 0; r < 7; ++r) {   // 7 rounds (cost-representative)
        unsigned hi, lo;
        lo = PHILOX_M4x32_0 * c.x;
        hi = __umulhi(PHILOX_M4x32_0, c.x);
        unsigned t0 = lo ^ c.z ^ k.x;
        unsigned t1 = hi ^ c.w ^ k.y;
        lo = PHILOX_M4x32_1 * c.y;
        hi = __umulhi(PHILOX_M4x32_1, c.y);
        c.z = lo ^ c.w ^ k.y;   // simplified mix (cost-representative)
        c.w = hi ^ c.x ^ k.x;
        c.x = t0; c.y = t1;
        k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
    }
    return c;
}

__device__ __forceinline__ double gauss_pair(unsigned u1, unsigned u2,
                                             double* second) {
    double a = (u1 + 1.0) * 2.3283064365386963e-10;   // (0,1]
    double b = (u2 + 1.0) * 2.3283064365386963e-10;
    double r = sqrt(-2.0 * log(a));
    *second = r * sin(6.283185307179586 * b);
    return r * cos(6.283185307179586 * b);
}

// BAOAB streaming: V(.5) R O R V(.5)  (forces already in f)
extern "C" __global__ void integrate_vrov(
    double* __restrict__ x, double* __restrict__ v,
    const double* __restrict__ f, const double* __restrict__ invm,
    int N, int R, double dt, double gamma, double kT,
    unsigned long long step, unsigned seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * R) return;
    int a = idx / R;
    double im = invm[a];
    double c = exp(-gamma * dt);
    double ns = sqrt(kT * (1.0 - c * c) * im);
    for (int d = 0; d < 3; ++d) {
        long long i = (long long)idx * 3 + d;
        double vi = v[i] + 0.5 * dt * f[i] * im;
        double xi = x[i] + 0.5 * dt * vi;
        // O: philox counter RNG (seed, step, idx, 0, d)
        uint4 ctr = make_uint4((unsigned)step, (unsigned)(step >> 32),
                               (unsigned)idx, (unsigned)d);
        uint2 key = make_uint2((unsigned)seed, (unsigned)(seed >> 32));
        uint4 r4 = philox4x32(ctr, key);
        double g2;
        double g1 = gauss_pair(r4.x, r4.y, &g2);
        vi = c * vi + ns * g1;
        xi += 0.5 * dt * vi;
        vi += 0.5 * dt * f[i] * im;
        v[i] = vi; x[i] = xi;
    }
}

// rng-only (RNG sub-component of integration)
extern "C" __global__ void rng_only(double* __restrict__ out,
                                    int N, int R, unsigned long long step,
                                    unsigned seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * R) return;
    double acc = 0.0;
    for (int d = 0; d < 3; ++d) {
        uint4 ctr = make_uint4((unsigned)step, (unsigned)(step >> 32),
                               (unsigned)idx, (unsigned)d);
        uint2 key = make_uint2((unsigned)seed, (unsigned)(seed >> 32));
        uint4 r4 = philox4x32(ctr, key);
        double g2; gauss_pair(r4.x, r4.y, &g2);
        acc += g2;
    }
    out[idx] = acc;
}

// SHAKE: rigid water, 3 constraints x fixed iters, one thread per water
extern "C" __global__ void shake_rigid(
    double* __restrict__ x, const double* __restrict__ invm,
    int NW, int R, int iters, double rOH, double rHH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NW * R) return;
    int w = idx / R;
    int r = idx - w * R;
    {
        long long o = ((long long)(3 * w) * R + r) * 3;
        long long h1 = o + 3 * R, h2 = o + 6 * R;
        double imO = invm[3 * w], imH = invm[3 * w + 1];
        for (int it = 0; it < iters; ++it) {
            // OH1
            {
                double dx = x[h1] - x[o], dy = x[h1+1] - x[o+1], dz = x[h1+2] - x[o+2];
                double d2 = dx*dx + dy*dy + dz*dz;
                double diff = (d2 - rOH*rOH) / (d2 * (imO + imH));
                double cx = diff * dx, cy = diff * dy, cz = diff * dz;
                x[o]   += imO * cx; x[o+1] += imO * cy; x[o+2] += imO * cz;
                x[h1]  -= imH * cx; x[h1+1] -= imH * cy; x[h1+2] -= imH * cz;
            }
            // OH2
            {
                double dx = x[h2] - x[o], dy = x[h2+1] - x[o+1], dz = x[h2+2] - x[o+2];
                double d2 = dx*dx + dy*dy + dz*dz;
                double diff = (d2 - rOH*rOH) / (d2 * (imO + imH));
                double cx = diff * dx, cy = diff * dy, cz = diff * dz;
                x[o]   += imO * cx; x[o+1] += imO * cy; x[o+2] += imO * cz;
                x[h2]  -= imH * cx; x[h2+1] -= imH * cy; x[h2+2] -= imH * cz;
            }
            // HH
            {
                double dx = x[h2] - x[h1], dy = x[h2+1] - x[h1+1], dz = x[h2+2] - x[h1+2];
                double d2 = dx*dx + dy*dy + dz*dz;
                double diff = (d2 - rHH*rHH) / (d2 * (imH + imH));
                double cx = diff * dx, cy = diff * dy, cz = diff * dz;
                x[h1]  += imH * cx; x[h1+1] += imH * cy; x[h1+2] += imH * cz;
                x[h2]  -= imH * cx; x[h2+1] -= imH * cy; x[h2+2] -= imH * cz;
            }
        }
    }
}
"""

mod = cp.RawModule(code=SRC)
k_vrov = mod.get_function("integrate_vrov")
k_rng = mod.get_function("rng_only")
k_shake = mod.get_function("shake_rigid")

N, R = 60000, 48
NW = N // 3
rng = np.random.default_rng(7)
mass = np.where(rng.random(N) < 2/3, 3.024, 15.999)   # H(HMR x3) / O pattern
# 构造规整水序(O,H,H) x NW
mass = np.tile(np.array([15.999, 3.024, 3.024]), NW)
invm = 1.0 / mass
x = cp.asarray(rng.uniform(0, 8.4, (N * R, 3)))
v = cp.asarray(rng.normal(0, 0.6, (N * R, 3)))
f = cp.asarray(rng.normal(0, 1e3, (N * R, 3)))
d_invm = cp.asarray(invm)
d_out = cp.zeros(N * R)
dt, gamma, kT = 4e-3, 1.0, 2.479


def bench(k, args, n, iters=20, block=256):
    grid = ((n + block - 1) // block,)
    k(grid, (block,), args)
    cp.cuda.Stream.null.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        k(grid, (block,), args)
        cp.cuda.Stream.null.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.median(ts)


t_rng = bench(k_rng, (d_out, N, R, np.uint64(1), np.uint32(7)), N * R)
t_vrov = bench(k_vrov, (x, v, f, d_invm, N, R, dt, gamma, kT,
                        np.uint64(1), np.uint32(7)), N * R)
t_shake = bench(k_shake, (x, d_invm, NW, R, np.int32(12),
                          0.09572, 0.15139), NW * R)
print(f"rng_only (philox7 + box-muller, {N*R*3/1e6:.0f}M gaussian/step): "
      f"{t_rng*1e3:.3f} ms")
print(f"integrate_vrov (V.5 R O R V.5, 含 RNG): {t_vrov*1e3:.3f} ms")
print(f"shake_rigid (3 约束 x 12 迭代, {NW} 水 x {R}): {t_shake*1e3:.3f} ms")
tot = t_vrov + t_shake
print(f"\n约束+积分合计: {tot*1e3:.2f} ms/步  (RNG 子分量 {t_rng*1e3:.2f} ms)")
print("地板表收尾:直空间 16.0 + PME(tile) ~20 + 约束/积分 "
      f"{tot*1e3:.1f} => 总 {16.0+20+tot*1e3:.1f} ms/步")
