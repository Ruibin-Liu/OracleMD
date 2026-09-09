#!/usr/bin/env python3
"""E0i: 直空间 ∥ PME 链双流重叠(M2 总步时下一杠杆;Q-002)。

结构:同一步的直空间(只吃坐标)与 PME 链(铺展→rFFT→scale→irFFT→回插)在
数据依赖上独立 ⇒ 双流并行。问题:同卡 SM/带宽竞争下的实际收益。
  - 直空间 = e0g k2_prefold(合规形态,真实物理表,~16 ms)
  - PME 链 = spread_v1(Q16.48 原子铺展,工作负载代表;生产为 tile 4.7 ms)
            + cupy.fft rfftn/ifftn(48x128^3 f64)
            + scale(elementwise,代表影响函数乘法)+ interp(e0h 形式)
对照:同工作单流串行 vs 双流重叠,各 10 次取中位。
"""
import sys, time, subprocess
from pathlib import Path

import numpy as np

util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if util > 20:
    sys.exit(f"GPU busy ({util}%), timing invalid.")

import cupy as cp

# ---------- 直空间 kernel: 直接读已部署且经过验证的 e0g_kernels.cu ----------
cu = Path("/root/e0g_kernels.cu").read_text(encoding="ascii")
assert "k2_prefold" in cu

SRC = cu + r"""
#define NG 128

__device__ __forceinline__ void weights4(double xg, int* anchor, double w[4]) {
    int k = (int)xg;
    anchor[0] = k - 1; anchor[1] = k; anchor[2] = k + 1; anchor[3] = k + 2;
    double s = xg - k;
    double om = 1.0 - s;
    w[0] = om * om * om * (1.0 / 6.0);
    w[1] = (2.0 / 3.0) - s * s + 0.5 * s * s * s;
    w[2] = (1.0 / 6.0) + 0.5 * s + 0.5 * s * s - 0.5 * s * s * s;
    w[3] = s * s * s * (1.0 / 6.0);
}

extern "C" __global__ void spread_v1(
    const double* __restrict__ x, const double* __restrict__ q,
    long long* __restrict__ grid, int N, int R, double inv_h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    int ax[4], ay[4], az[4];
    double wx[4], wy[4], wz[4];
    weights4(x[((long long)a * R + r) * 3] * inv_h, ax, wx);
    weights4(x[((long long)a * R + r) * 3 + 1] * inv_h, ay, wy);
    weights4(x[((long long)a * R + r) * 3 + 2] * inv_h, az, wz);
    long long* g = grid + (long long)r * NG * NG * NG;
    double qi = q[a];
    for (int dz = 0; dz < 4; ++dz)
        for (int dy = 0; dy < 4; ++dy) {
            double wyz = wy[dy] * wz[dz] * qi;
            int yy = ay[dy] & (NG-1), zz = az[dz] & (NG-1);
            for (int dx = 0; dx < 4; ++dx) {
                long long dep = (long long)((wx[dx] * wyz) * 281474976710656.0);
                atomicAdd(reinterpret_cast<unsigned long long*>(
                              &g[((long long)yy * NG + zz) * NG + (ax[dx] & (NG-1))]),
                          (unsigned long long)dep);
            }
        }
}

extern "C" __global__ void interp_v1(
    const double* __restrict__ x, const double* __restrict__ q,
    const double* __restrict__ pot,   // (R, NG, NG, NG) double, potential proxy
    double* __restrict__ F, int N, int R, double inv_h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    int ax[4], ay[4], az[4];
    double wx[4], wy[4], wz[4];
    weights4(x[((long long)a * R + r) * 3] * inv_h, ax, wx);
    weights4(x[((long long)a * R + r) * 3 + 1] * inv_h, ay, wy);
    weights4(x[((long long)a * R + r) * 3 + 2] * inv_h, az, wz);
    const double* g = pot + (long long)r * NG * NG * NG;
    double acc = 0.0;
    for (int dz = 0; dz < 4; ++dz)
        for (int dy = 0; dy < 4; ++dy) {
            double wyz = wy[dy] * wz[dz];
            int yy = ay[dy] & (NG-1), zz = az[dz] & (NG-1);
            for (int dx = 0; dx < 4; ++dx) {
                acc += g[((long long)yy * NG + zz) * NG + (ax[dx] & (NG-1))]
                       * (wx[dx] * wyz);
            }
        }
    F[a * R + r] = acc * q[a];
}
"""

mod = cp.RawModule(code=SRC)
k_direct = mod.get_function("k2_prefold")
k_spread = mod.get_function("spread_v1")
k_interp = mod.get_function("interp_v1")

# ---------- 物理数据(e0g 同法:抖动格子 + 真实 1.35 nm 表) ----------
N, R, NG = 60000, 48, 128
RHO, RC, SKIN, ALPHA = 100.0, 1.0, 0.35, 3.5
LR = RC + SKIN
rng = np.random.default_rng(7)
L = (N / RHO) ** (1 / 3)
m = int(np.ceil(L / 0.215))
pos = np.stack(np.unravel_index(np.arange(N), (m, m, m)), 1).astype(float) * (L / m)
pos += rng.uniform(0, 0.06, pos.shape)
pos %= L
cell = np.floor(pos / LR).astype(int)
ncell = int(np.ceil(L / LR))
cellid = (cell[:, 0] * ncell + cell[:, 1]) * ncell + cell[:, 2]
order = np.argsort(cellid)
sorted_cid = cellid[order]
cell_start = np.searchsorted(sorted_cid, np.arange(ncell ** 3))
cell_end = np.searchsorted(sorted_cid, np.arange(ncell ** 3) + 1)
idx_sorted = order
lists = np.zeros((N, 1200), dtype=np.int32)
ncnt = np.zeros(N, dtype=np.int32)
maxnb = 0
CH = 2000
for a0 in range(0, N, CH):
    a1 = min(N, a0 + CH)
    cand, masks = [], []
    cx, cy, cz = cell[a0:a1, 0], cell[a0:a1, 1], cell[a0:a1, 2]
    for dx_ in (-1, 0, 1):
        for dy_ in (-1, 0, 1):
            for dz_ in (-1, 0, 1):
                gx, gy, gz = (cx + dx_) % ncell, (cy + dy_) % ncell, (cz + dz_) % ncell
                cid = (gx * ncell + gy) * ncell + gz
                s, e = cell_start[cid], cell_end[cid]
                cnt = e - s
                mc = int(cnt.max()) if len(cnt) else 0
                if mc == 0:
                    continue
                ar = np.arange(mc)
                offs = s[:, None] + ar[None, :]
                mk = ar[None, :] < cnt[:, None]
                cand.append(idx_sorted[np.where(mk, offs, 0)])
                masks.append(mk)
    C = np.concatenate(cand, 1)
    M = np.concatenate(masks, 1)
    d = pos[C] - pos[a0:a1, None, :]
    d -= np.round(d / L) * L
    r2 = (d * d).sum(-1)
    keep = M & (r2 <= LR * LR) & (C > a0 + np.arange(a1 - a0)[:, None])
    for a in range(a1 - a0):
        js = C[a][keep[a]]
        lists[a0 + a, :len(js)] = js
        ncnt[a0 + a] = len(js)
        maxnb = max(maxnb, len(js))
print(f"N={N} maxnb={maxnb} avg={ncnt.mean():.0f}")

q = rng.uniform(-1, 1, N)
sig = rng.uniform(0.3, 0.4, N)
eps = rng.uniform(0.3, 0.7, N)
x0 = np.repeat(pos[:, None, :], R, axis=1) + rng.normal(0, 0.01, (N, R, 3))
x0 %= L

d_x = cp.asarray(x0)
d_list = cp.asarray(lists[:, :maxnb].ravel())
d_ncnt = cp.asarray(ncnt)
d_q, d_s, d_e = cp.asarray(q), cp.asarray(sig), cp.asarray(eps)
d_se = cp.asarray(np.sqrt(eps))
F = cp.zeros(N * R * 3)
Fp = cp.zeros(N * R)
grid_ll = cp.zeros(R * NG ** 3, dtype=cp.int64)
grid_f = cp.zeros((R, NG, NG, NG))          # spread -> double (代表)
pot = cp.zeros((R, NG, NG, NG))
inv_h = NG / L
grid_shape = (R, NG, NG, NG)


def pme_chain(stream=None):
    with stream or cp.cuda.Stream.null:
        grid_ll.fill(0)
        k_spread(((N * R + 255) // 256,), (256,),
                 (d_x, d_q, grid_ll, N, R, inv_h))
        grid_f[:] = grid_ll.reshape(grid_shape) * (1.0 / 281474976710656.0)
        G = cp.fft.rfftn(grid_f, axes=(1, 2, 3))
        G *= 0.37                                     # 代表影响函数乘法
        pot[:] = cp.fft.irfftn(G, s=(NG, NG, NG), axes=(1, 2, 3))
        k_interp(((N * R + 255) // 256,), (256,),
                 (d_x, d_q, pot, Fp, N, R, inv_h))


def direct(stream=None):
    with stream or cp.cuda.Stream.null:
        k_direct(((N * R + 127) // 128,), (128,),
                 (d_x, d_list, d_ncnt, d_q, d_s, d_e, d_se, F, N, R, maxnb,
                  RC * RC, ALPHA))


# warmup(含 cuFFT plan)
direct(); pme_chain()
cp.cuda.Stream.null.synchronize()

# 分量单独计时
for name, fn in (("direct", direct), ("pme_chain", pme_chain)):
    ts = []
    for _ in range(10):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        ts.append(time.perf_counter() - t0)
    print(f"{name}: {np.median(ts)*1e3:.2f} ms")

# 串行
ts = []
for _ in range(10):
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    direct(); pme_chain()
    cp.cuda.Stream.null.synchronize()
    ts.append(time.perf_counter() - t0)
t_serial = np.median(ts)
print(f"serial (direct+pme): {t_serial*1e3:.2f} ms")

# 双流重叠
s1, s2 = cp.cuda.Stream(), cp.cuda.Stream()
ts = []
for _ in range(10):
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    direct(s1)
    pme_chain(s2)
    s1.synchronize(); s2.synchronize()
    ts.append(time.perf_counter() - t0)
t_ov = np.median(ts)
print(f"two-stream overlap: {t_ov*1e3:.2f} ms  (serial {t_serial*1e3:.2f}; "
      f"节省 {(t_serial-t_ov)*1e3:.2f} ms; vs max(分量) "
      f"{max(16.0, 0)*1e3 if False else ''}{''})"
      f"")
print(f"\n总步时图景:重叠后 direct∥PME = {t_ov*1e3:.1f} ms + 约束/积分(待测)"
      f" => Q-002 更新")
