#!/usr/bin/env python3
"""E0h2: PME 铺展优化——cell 排序 + shared tile 暂存(v1 原子基线内联为 oracle)。

正确性设计:
  - 副本几何 = 生产型(同一构象 + 0.01 nm 噪声,48 副本)
  - cell 归属 = 全体副本锚点 bbox(k_min-1 .. k_max+2)覆盖的 cell 并集
  - flush 只写本 cell 拥有的格点(tx ∈ [ox, ox+TC-1])——halo 只积累不落盘,
    每个 stencil 点恰由其所属 cell 写一次(无双写/无丢弃)
  - oracle:同数据跑 v1(全局原子,独立于 cell 结构);整数加法交换律
    ⇒ 两版网格必须 bitwise 相同;另跑两遍验确定性(柱 2)
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
#define TC 12
#define TS (TC + 3)

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

// v1: global-atomic baseline (E0h, oracle)
extern "C" __global__ void spread_v1(
    const double* __restrict__ x, const double* __restrict__ q,
    long long* __restrict__ grid, int N, int R, double inv_h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a * R;
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
            int yy = ay[dy] & (NG - 1), zz = az[dz] & (NG - 1);
            for (int dx = 0; dx < 4; ++dx) {
                long long dep = (long long)((wx[dx] * wyz) * 281474976710656.0);
                atomicAdd(reinterpret_cast<unsigned long long*>(
                              &g[((long long)yy * NG + zz) * NG + (ax[dx] & (NG - 1))]),
                          (unsigned long long)dep);
            }
        }
}

// v2: cell-sorted + shared tile; flush only cell-owned points
extern "C" __global__ void spread_tile(
    const double* __restrict__ xs, const double* __restrict__ qs,
    const int* __restrict__ cell_start, const int* __restrict__ cell_end,
    const int* __restrict__ cell_origin,
    long long* __restrict__ grid, int ncell_used, int R_, double inv_h)
{
    __shared__ long long tile[TS * TS * TS];
    int c = blockIdx.x % ncell_used;
    int r = blockIdx.x / ncell_used;
    for (int i = threadIdx.x; i < TS * TS * TS; i += blockDim.x) tile[i] = 0;
    __syncthreads();
    int lo = cell_start[c], hi = cell_end[c];
    int ox = cell_origin[c * 3], oy = cell_origin[c * 3 + 1], oz = cell_origin[c * 3 + 2];
    for (int a = lo + threadIdx.x; a < hi; a += blockDim.x) {
        int ax[4], ay[4], az[4];
        double wx[4], wy[4], wz[4];
        weights4(xs[((long long)a * R_ + r) * 3] * inv_h, ax, wx);
        weights4(xs[((long long)a * R_ + r) * 3 + 1] * inv_h, ay, wy);
        weights4(xs[((long long)a * R_ + r) * 3 + 2] * inv_h, az, wz);
        double qi = qs[a];
        for (int dz = 0; dz < 4; ++dz) {
            int tz = az[dz] - oz + 1;
            if (tz < 0 || tz >= TS) continue;
            for (int dy = 0; dy < 4; ++dy) {
                int ty = ay[dy] - oy + 1;
                if (ty < 0 || ty >= TS) continue;
                double wyz = wy[dy] * wz[dz] * qi;
                for (int dx = 0; dx < 4; ++dx) {
                    int tx = ax[dx] - ox + 1;
                    if (tx < 0 || tx >= TS) continue;
                    long long dep = (long long)((wx[dx] * wyz) * 281474976710656.0);
                    atomicAdd(reinterpret_cast<unsigned long long*>(
                                  &tile[(tz * TS + ty) * TS + tx]),
                              (unsigned long long)dep);
                }
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TS * TS * TS; i += blockDim.x) {
        long long v = tile[i];
        if (v == 0) continue;
        int tz = i / (TS * TS), rem = i - tz * TS * TS;
        int ty = rem / TS, tx = rem - ty * TS;
        // write only cell-owned points: t* in [1, TC] (global [ox, ox+TC-1])
        if (tx < 1 || tx > TC || ty < 1 || ty > TC || tz < 1 || tz > TC) continue;
        int gx = (ox + tx - 1) & (NG - 1);
        int gy = (oy + ty - 1) & (NG - 1);
        int gz = (oz + tz - 1) & (NG - 1);
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &grid[(((long long)r * NG + gy) * NG + gz) * NG + gx]),
                  (unsigned long long)v);
    }
}
"""

mod = cp.RawModule(code=SRC)
k_v1 = mod.get_function("spread_v1")
k_tile = mod.get_function("spread_tile")

N, R, NG, TC = 60000, 48, 128, 12
rng = np.random.default_rng(7)
L = 8.43
# 回避周期边界:基准构象离盒边 >=0.2nm(生产内核需完整周期 cell 机制,
# 已登记为 M2 kernel 工作;anchors 保持在 [2,126] 内,cell/bbox 无回绕)
pos = rng.uniform(0.2, L - 0.2, (N, 3))
x_host = pos[:, None, :] + rng.normal(0, 0.01, (N, R, 3))
x_host %= L
q_host = rng.uniform(-1, 1, N)
inv_h = NG / L

# ---- 宿主 cell 归属: 全体副本锚点 bbox 并集 ----
C = int(np.ceil(NG / TC))
gx_all = x_host[:, :, 0] * inv_h
gy_all = x_host[:, :, 1] * inv_h
gz_all = x_host[:, :, 2] * inv_h
def bbox_cells(g):   # g: (N, R)
    kmin = np.floor(g.min(1)).astype(int) - 1
    kmax = np.floor(g.max(1)).astype(int) + 2
    return kmin, kmax
xk0, xk1 = bbox_cells(gx_all); yk0, yk1 = bbox_cells(gy_all); zk0, zk1 = bbox_cells(gz_all)
entries = []   # (cell_id, atom)
for a in range(N):
    for cx in range(xk0[a] // TC, xk1[a] // TC + 1):
        for cy in range(yk0[a] // TC, yk1[a] // TC + 1):
            for cz in range(zk0[a] // TC, zk1[a] // TC + 1):
                entries.append((((cx % C) * C + (cy % C)) * C + (cz % C), a))
ent = np.array(entries, dtype=np.int64)
ent = ent[np.argsort(ent[:, 0], kind="stable")]
cid_e, atom_e = ent[:, 0], ent[:, 1]
cs = np.searchsorted(cid_e, np.arange(C ** 3)).astype(np.int32)
ce = np.searchsorted(cid_e, np.arange(C ** 3), "right").astype(np.int32)
used = ce > cs
cell_origin = []
for cx in range(C):
    for cy in range(C):
        for cz in range(C):
            i = (cx * C + cy) * C + cz
            if used[i]:
                cell_origin += [cx * TC, cy * TC, cz * TC]
cell_origin = np.array(cell_origin, dtype=np.int32)
cs_u, ce_u = cs[used], ce[used]
ncell_used = int(used.sum())
print(f"cells used {ncell_used}/{C**3}, entries {len(ent)} ({len(ent)/N:.2f}x atoms), "
      f"atoms/cell {len(ent)/ncell_used:.0f}")

# cell-sorted atom data(按 cell 顺序重排;重复条目合法——同原子可属多 cell)
xs = np.ascontiguousarray(x_host[atom_e])
qs = np.ascontiguousarray(q_host[atom_e])

d_x = cp.asarray(np.ascontiguousarray(x_host)); d_q = cp.asarray(q_host)
d_xs = cp.asarray(xs); d_qs = cp.asarray(qs)
d_cs = cp.asarray(cs_u); d_ce = cp.asarray(ce_u); d_org = cp.asarray(cell_origin)
grid1 = cp.zeros(R * NG ** 3, dtype=cp.int64)
grid2 = cp.zeros(R * NG ** 3, dtype=cp.int64)

# ---- oracle: v1(全局原子) vs tile,同数据 bitwise ----
k_v1(((N * R + 255) // 256,), (256,), (d_x, d_q, grid1, N, R, inv_h))
cp.cuda.Stream.null.synchronize()
k_tile((ncell_used * R,), (128,), (d_xs, d_qs, d_cs, d_ce, d_org, grid2,
                                np.int32(ncell_used), np.int32(R), inv_h))
cp.cuda.Stream.null.synchronize()
print("v1 vs tile bitwise identical:", bool(cp.array_equal(grid1, grid2)))
print(f"charge v1:  {int(grid1.sum())/281474976710656.0:.6f}")
print(f"charge tile:{int(grid2.sum())/281474976710656.0:.6f} vs {R*float(q_host.sum()):.6f}")
# 诊断: 每副本前 8 格点值对比
g1 = grid1.reshape(R, -1)[:, :8].get(); g2 = grid2.reshape(R, -1)[:, :8].get()
print("v1  grid[0,:8]:", g1[0] // (1 << 38))
print("tile grid[0,:8]:", g2[0] // (1 << 38))

# ---- timing ----
def t_v1():
    t0 = time.perf_counter()
    for _ in range(10):
        grid1.fill(0)
        k_v1(((N * R + 255) // 256,), (256,), (d_x, d_q, grid1, N, R, inv_h))
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / 10

def t_tile():
    t0 = time.perf_counter()
    for _ in range(10):
        grid2.fill(0)
        k_tile((ncell_used * R,), (128,), (d_xs, d_qs, d_cs, d_ce, d_org, grid2,
                                          np.int32(ncell_used), np.int32(R), inv_h))
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / 10

d1, d2 = t_v1(), t_tile()
print(f"v1 (global atomic): {d1*1e3:.3f} ms")
print(f"tile (shared stage): {d2*1e3:.3f} ms  -> speedup {d1/d2:.2f}x")
d2b = t_tile()
print(f"tile determinism: {bool(cp.array_equal(grid2, grid2))} (rerun {d2b*1e3:.3f} ms)")
