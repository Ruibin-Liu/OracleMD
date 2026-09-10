#!/usr/bin/env python3
"""E0l: CUDA Graph + K 窗口装配原型(M2 装配阶段;§7 graph 语义, I-023, Q-003)。

结构:
  步序列(生产序, 全部用已验证内核):
    1 spread_v1 (Q16.48) → [图外: int64->f64, rfftn, scale, irfftn]
    2 interp_v1 (recip 力 proxy)
    3 k2_prefold (直空间)
    4 step_counter++ (设备端, §7) + integrate_vrov(步数从设备缓冲读)
    5 shake_rigid
    6 c1_flag (重原子 max|dx|>0.14 → device flag; 裁决 a 语义)
  Graph = 1..6 中除 FFT 链;K=25 次重放 vs 25 次串行。
测量:
  a) 每步启动开销(串行总时 - 图总时)/K —— Q-003 定标
  b) 位级等价: 串行 vs 图重放的 (x, v, grid, flag) 逐位比对 —— M2 原型级
  c) K 窗口语义: 25 步后 host 检查 flag(裁决 a: 重原子 d=0.14)
FFT 不入图的原因登记: cupy fft 每次调用分配输出(捕获期禁止);
生产全图化 = cufft exec 节点(cufftExec 可捕获), 属生产 kernel 工程项。
"""
import subprocess, time
import numpy as np


def _excl():
    try:
        mem = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"]).strip())
        return mem < 100
    except Exception:
        return False


EXCLUSIVE = _excl()
util = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).strip())
if not EXCLUSIVE and util > 20:
    raise SystemExit(f"GPU busy ({util}%), abort.")

import cupy as cp
from pathlib import Path

CU = Path("/root/e0g_kernels.cu").read_text(encoding="ascii")

SRC = CU + r"""
#define NG 128
#define PHILOX_W0 0x9E3779B9u

__device__ __forceinline__ void weights4(double xg, int* anchor, double w[4]) {
    int k = (int)xg;
    anchor[0] = k - 1; anchor[1] = k; anchor[2] = k + 1; anchor[3] = k + 2;
    double s = xg - k;
    double om = 1.0 - s;
    w[0] = om*om*om*(1.0/6.0);
    w[1] = (2.0/3.0) - s*s + 0.5*s*s*s;
    w[2] = (1.0/6.0) + 0.5*s + 0.5*s*s - 0.5*s*s*s;
    w[3] = s*s*s*(1.0/6.0);
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
    weights4(x[((long long)a*R+r)*3]*inv_h, ax, wx);
    weights4(x[((long long)a*R+r)*3+1]*inv_h, ay, wy);
    weights4(x[((long long)a*R+r)*3+2]*inv_h, az, wz);
    long long* g = grid + (long long)r*NG*NG*NG;
    double qi = q[a];
    for (int dz = 0; dz < 4; ++dz)
        for (int dy = 0; dy < 4; ++dy) {
            double wyz = wy[dy]*wz[dz]*qi;
            int yy = ay[dy] & (NG-1), zz = az[dz] & (NG-1);
            for (int dx = 0; dx < 4; ++dx) {
                long long dep = (long long)((wx[dx]*wyz) * 281474976710656.0);
                atomicAdd(reinterpret_cast<unsigned long long*>(
                              &g[((long long)yy*NG+zz)*NG + (ax[dx] & (NG-1))]),
                          (unsigned long long)dep);
            }
        }
}

extern "C" __global__ void interp_v1(
    const double* __restrict__ x, const double* __restrict__ q,
    const double* __restrict__ pot, double* __restrict__ F,
    int N, int R, double inv_h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx / R, r = idx - a*R;
    if (a >= N) return;
    int ax[4], ay[4], az[4];
    double wx[4], wy[4], wz[4];
    weights4(x[((long long)a*R+r)*3]*inv_h, ax, wx);
    weights4(x[((long long)a*R+r)*3+1]*inv_h, ay, wy);
    weights4(x[((long long)a*R+r)*3+2]*inv_h, az, wz);
    const double* g = pot + (long long)r*NG*NG*NG;
    double acc = 0.0;
    for (int dz = 0; dz < 4; ++dz)
        for (int dy = 0; dy < 4; ++dy) {
            double wyz = wy[dy]*wz[dz];
            int yy = ay[dy] & (NG-1), zz = az[dz] & (NG-1);
            for (int dx = 0; dx < 4; ++dx)
                acc += g[((long long)yy*NG+zz)*NG + (ax[dx] & (NG-1))] * (wx[dx]*wyz);
        }
    F[a*R + r] = acc * q[a];
}

// device-side step counter (spec 7) + integrate reading it
extern "C" __global__ void step_inc(unsigned long long* step) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *step += 1ULL;
}

extern "C" __global__ void integrate_g(
    const unsigned long long* __restrict__ step_ptr,
    double* __restrict__ x, double* __restrict__ v,
    const double* __restrict__ f, const double* __restrict__ invm,
    int N, int R, double dt, double gamma, double kT, unsigned seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N*R) return;
    unsigned long long step = *step_ptr;
    int a = idx / R;
    double im = invm[a];
    double c = exp(-gamma*dt);
    double ns = sqrt(kT*(1.0-c*c)*im);
    // philox-lite counter mix (cost-representative; determinism is the property)
    for (int d = 0; d < 3; ++d) {
        long long i = (long long)idx*3 + d;
        double vi = v[i] + 0.5*dt*f[i]*im;
        double xi = x[i] + 0.5*dt*vi;
        unsigned h = (unsigned)(step*0x9E3779B97F4A7C15ULL) ^ (unsigned)(idx*3+d) ^ seed;
        h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
        double g1 = ((h & 0xFFFF) * (1.0/65536.0) - 0.5) * 2.0 * ns;
        xi += 0.5*dt*(vi = c*vi + g1);
        v[i] = vi + 0.5*dt*f[i]*im;
        x[i] = xi;
    }
}

extern "C" __global__ void shake_rigid(
    double* __restrict__ x, const double* __restrict__ invm,
    int NW, int R, int iters, double rOH, double rHH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NW * R) return;
    int w = idx / R;
    int r = idx - w * R;
    long long o = ((long long)(3*w)*R + r)*3;
    long long h1 = o + 3*R, h2 = o + 6*R;
    double imO = invm[3*w], imH = invm[3*w+1];
    for (int it = 0; it < iters; ++it) {
        {
            double dx = x[h1]-x[o], dy = x[h1+1]-x[o+1], dz = x[h1+2]-x[o+2];
            double d2 = dx*dx+dy*dy+dz*dz;
            double df = (d2 - rOH*rOH)/(d2*(imO+imH));
            x[o] += imO*df*dx; x[o+1] += imO*df*dy; x[o+2] += imO*df*dz;
            x[h1] -= imH*df*dx; x[h1+1] -= imH*df*dy; x[h1+2] -= imH*df*dz;
        }
        {
            double dx = x[h2]-x[o], dy = x[h2+1]-x[o+1], dz = x[h2+2]-x[o+2];
            double d2 = dx*dx+dy*dy+dz*dz;
            double df = (d2 - rOH*rOH)/(d2*(imO+imH));
            x[o] += imO*df*dx; x[o+1] += imO*df*dy; x[o+2] += imO*df*dz;
            x[h2] -= imH*df*dx; x[h2+1] -= imH*df*dy; x[h2+2] -= imH*df*dz;
        }
        {
            double dx = x[h2]-x[h1], dy = x[h2+1]-x[h1+1], dz = x[h2+2]-x[h1+2];
            double d2 = dx*dx+dy*dy+dz*dz;
            double df = (d2 - rHH*rHH)/(d2*(imH+imH));
            x[h1] += imH*df*dx; x[h1+1] += imH*df*dy; x[h1+2] += imH*df*dz;
            x[h2] -= imH*df*dx; x[h2+1] -= imH*df*dy; x[h2+2] -= imH*df*dz;
        }
    }
}

// C1 flag (ruling a): heavy atoms, d=0.14 vs window-start reference
extern "C" __global__ void c1_flag(
    const double* __restrict__ x, const double* __restrict__ x_ref,
    const int* __restrict__ heavy, int n_heavy, int R,
    double d2_thresh, int* __restrict__ flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heavy * R) return;
    int h = heavy[idx / R], r = idx - (idx / R) * R;
    long long i = ((long long)h * R + r) * 3;
    double dx = x[i]-x_ref[i], dy = x[i+1]-x_ref[i+1], dz = x[i+2]-x_ref[i+2];
    if (dx*dx + dy*dy + dz*dz > d2_thresh)
        atomicOr(reinterpret_cast<unsigned int*>(flag), 1u);
}
"""

mod = cp.RawModule(code=SRC)
K = {n: mod.get_function(n) for n in
     ("k2_prefold", "spread_v1", "interp_v1", "step_inc",
      "integrate_g", "shake_rigid", "c1_flag")}

# ---------------- data ----------------
N, R, NG = 60000, 48, 128
RHO, RC, SKIN, ALPHA = 100.0, 1.0, 0.35, 3.5
LR = RC + SKIN
rng = np.random.default_rng(7)
L = (N / RHO) ** (1 / 3)
mm = int(np.ceil(L / 0.215))
pos = np.stack(np.unravel_index(np.arange(N), (mm, mm, mm)), 1).astype(float) * (L / mm)
pos += rng.uniform(0, 0.06, pos.shape)
pos %= L
cell = np.floor(pos / LR).astype(int)
ncl = int(np.ceil(L / LR))
cid_ = (cell[:, 0] * ncl + cell[:, 1]) * ncl + cell[:, 2]
order = np.argsort(cid_)
scid = cid_[order]
cs = np.searchsorted(scid, np.arange(ncl ** 3))
ce = np.searchsorted(scid, np.arange(ncl ** 3) + 1)
lists = np.zeros((N, 1200), dtype=np.int32)
ncnt = np.zeros(N, dtype=np.int32)
maxnb = 0
for a0 in range(0, N, 2000):
    a1 = min(N, a0 + 2000)
    cand, masks = [], []
    cx, cy, cz = cell[a0:a1, 0], cell[a0:a1, 1], cell[a0:a1, 2]
    for dx_ in (-1, 0, 1):
        for dy_ in (-1, 0, 1):
            for dz_ in (-1, 0, 1):
                c2 = (((cx+dx_) % ncl)*ncl + (cy+dy_) % ncl)*ncl + (cz+dz_) % ncl
                s, e = cs[c2], ce[c2]
                cnt = e - s
                mc = int(cnt.max()) if len(cnt) else 0
                if mc == 0:
                    continue
                ar = np.arange(mc)
                offs = s[:, None] + ar[None, :]
                mk = ar[None, :] < cnt[:, None]
                cand.append(order[np.where(mk, offs, 0)])
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
print(f"N={N} maxnb={maxnb} exclusive={EXCLUSIVE}")

mass = np.tile(np.array([15.999, 3.024, 3.024]), N // 3)
invm = 1.0 / mass
heavy = np.where(mass > 3.5)[0]
q = rng.uniform(-1, 1, N)
sig = rng.uniform(0.3, 0.4, N)
eps = rng.uniform(0.3, 0.7, N)
x0 = np.repeat(pos[:, None, :], R, axis=1) + rng.normal(0, 0.008, (N, R, 3))
x0 %= L
v0 = rng.normal(0, 0.5, (N * R, 3))
f0 = rng.normal(0, 1e2, (N * R, 3))

d_x = cp.asarray(x0.reshape(-1, 3))
d_v = cp.asarray(v0)
d_f = cp.asarray(f0)
d_invm = cp.asarray(invm)
d_q, d_s, d_e = cp.asarray(q), cp.asarray(sig), cp.asarray(eps)
d_se = cp.asarray(np.sqrt(eps))
d_heavy = cp.asarray(heavy.astype(np.int32))
d_list = cp.asarray(lists[:, :maxnb].ravel())
d_ncnt = cp.asarray(ncnt)
grid = cp.zeros(R * NG ** 3, dtype=cp.int64)
d_frec = cp.zeros(N * R)
F = cp.zeros(N * R * 3)
d_step = cp.zeros(1, dtype=cp.uint64)
d_flag = cp.zeros(1, dtype=cp.int32)
x_ref = d_x.copy()
inv_h = NG / L
NW = N // 3
dt, gamma, kT = 4e-3, 1.0, 2.479
KK = 25

grid_shape = (R, NG, NG, NG)
G = cp.empty((R, NG, NG, NG // 2 + 1), dtype=cp.complex128)
pot = cp.empty(grid_shape)


def fft_chain():
    # pot/G 预分配稳定缓冲(图捕获冻结指针); irfftn 无 out=, 拷回
    gf = grid.reshape(R, NG, NG, NG).astype(cp.float64)
    G[...] = cp.fft.rfftn(gf, axes=(1, 2, 3))
    cp.multiply(G, 0.37, out=G)
    pot[...] = cp.fft.irfftn(G, s=(NG, NG, NG), axes=(1, 2, 3))


def step_p1():
    """graph part 1: spread."""
    grid.fill(0)
    K["spread_v1"](((N * R + 255) // 256,), (256,),
                   (d_x, d_q, grid, N, R, inv_h))


def step_p2():
    """graph part 2: interp .. c1 (FFT between p1 and p2)."""
    K["interp_v1"](((N * R + 255) // 256,), (256,),
                   (d_x, d_q, pot, d_frec, N, R, inv_h))
    K["k2_prefold"](((N * R + 127) // 128,), (128,),
                    (d_x, d_list, d_ncnt, d_q, d_s, d_e, d_se, F, N, R, maxnb,
                     RC * RC, ALPHA))
    K["step_inc"]((1,), (1,), (d_step,))
    K["integrate_g"](((N * R + 255) // 256,), (256,),
                     (d_step, d_x, d_v, F, d_invm, N, R, dt, gamma, kT,
                      np.uint32(7)))
    K["shake_rigid"](((NW * R + 255) // 256,), (256,),
                     (d_x, d_invm, NW, R, np.int32(12), 0.09572, 0.15139))
    K["c1_flag"](((len(heavy) * R + 255) // 256,), (256,),
                 (d_x, x_ref, d_heavy, np.int32(len(heavy)), R,
                  0.14 * 0.14, d_flag))


def run_serial(nsteps):
    for _ in range(nsteps):
        step_p1()
        fft_chain()
        step_p2()


# ---- warmup (plans, jit) ----
step_p1()
fft_chain()
step_p2()
cp.cuda.Stream.null.synchronize()

def reset_state():
    d_x[:] = cp.asarray(x0.reshape(-1, 3)); d_v[:] = cp.asarray(v0)
    d_step[:] = 0; d_flag[:] = 0; grid[:] = 0
    pot[:] = 0.0; G[:] = 0; F[:] = 0.0; d_frec[:] = 0.0
    cp.cuda.Stream.null.synchronize()


# ---- serial reference ----
reset_state()
t0 = time.perf_counter()
run_serial(KK)
cp.cuda.Stream.null.synchronize()
t_serial = time.perf_counter() - t0
x_ser, v_ser = d_x.copy(), d_v.copy()
grid_ser, flag_ser = grid.copy(), int(d_flag.get()[0])

# 自证: 串行第二遍(全状态重置后应 bitwise)
reset_state()
run_serial(KK)
cp.cuda.Stream.null.synchronize()
print("serial-vs-serial bitwise:",
      bool(cp.array_equal(x_ser.view(cp.int64), d_x.view(cp.int64))),
      bool(cp.array_equal(v_ser.view(cp.int64), d_v.view(cp.int64))),
      bool(cp.array_equal(grid_ser, grid)))

# ---- graph (capture 1 step, replay K) ----
s = cp.cuda.Stream()
with s:
    step_p1(); fft_chain(); step_p2()
    s.synchronize()
    reset_state()
    s.begin_capture()
    step_p1()
    g1 = s.end_capture()
    s.begin_capture()
    step_p2()
    g2 = s.end_capture()
print("captured graphs: g1(spread) + g2(rest)")
with s:
    reset_state()
    t0 = time.perf_counter()
    for _ in range(KK):
        g1.launch()
        s.synchronize()
        fft_chain()
        g2.launch()
        s.synchronize()
    t_graph = time.perf_counter() - t0
x_gr, v_gr = d_x.copy(), d_v.copy()
grid_gr, flag_gr = grid.copy(), int(d_flag.get()[0])

# ---- results ----
# 位级比较(int64 视图;NaN==NaN 同位模式算等 —— 教训:NaN!=NaN 曾造成
# 假阳性"非确定性",一个合成力场下爆炸的水引发两轮追查,2026-09-10)
same_x = bool(cp.array_equal(x_ser.view(cp.int64), x_gr.view(cp.int64)))
same_v = bool(cp.array_equal(v_ser.view(cp.int64), v_gr.view(cp.int64)))
same_g = bool(cp.array_equal(grid_ser, grid_gr))
print(f"K={KK} steps: serial {t_serial*1e3:.1f} ms, graph+fft {t_graph*1e3:.1f} ms")
print(f"per-step: serial {t_serial/KK*1e3:.2f} ms, graph {t_graph/KK*1e3:.2f} ms")
print(f"launch overhead recovered: {(t_serial-t_graph)/KK*1e6:.1f} us/step "
      f"({100*(t_serial-t_graph)/t_serial:.1f}% of step; Q-003 预估 3-6%)")
print(f"bitwise: x={same_x} v={same_v} grid={same_g}")
print(f"C1 flag after 25 steps: serial={flag_ser} graph={flag_gr} (d=0.14 重原子语义)")
